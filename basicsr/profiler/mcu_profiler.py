"""
MCU Hardware Profiler for LUMEN.

Since we cannot physically plug in a Jetson tegrastats cable to an STM32,
we estimate performance using:

1. MAC counting per layer (via torch hooks)
2. CMSIS-NN throughput tables (cycles/MAC for each op type on each MCU)
3. STM32 power models (mW per MHz from datasheet)
4. Activation memory tracking (SRAM peak)

Reference CMSIS-NN throughput benchmarks (cycles per MAC):
  Cortex-M4 (no DSP):  DW-Conv ~2.0, PW-Conv ~1.4, FC ~1.3
  Cortex-M4F (DSP):    DW-Conv ~1.0, PW-Conv ~0.7, FC ~0.6
  Cortex-M7 (FPU):     DW-Conv ~0.5, PW-Conv ~0.4, FC ~0.3
  Cortex-M33 (DSP):    DW-Conv ~1.0, PW-Conv ~0.7, FC ~0.6
  Cortex-M55 (MVE):    DW-Conv ~0.25, PW-Conv ~0.15, FC ~0.12

Reference STM32 power models (active power at max freq):
  STM32F4 (Cortex-M4F, 168MHz):  ~100 mW active, ~1.8 mA/MHz
  STM32H7 (Cortex-M7, 480MHz):   ~300 mW active, ~0.9 mA/MHz at 3.3V
  STM32U5 (Cortex-M33, 160MHz):  ~26 mW active (ultra-low power)
  STM32H5 (Cortex-M33, 250MHz):  ~55 mW active
  STM32N6 (Cortex-M55, 800MHz):  ~500 mW active (has NPU)

Sources:
  - CMSIS-NN paper (Lai et al., 2018) and benchmarks
  - STM32 datasheets, power calculators
  - ARM Cortex-M Performance Analysis Guide
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# MCU hardware profiles (realistic STM32 targets)
# --------------------------------------------------------------------------

@dataclass
class STM32Profile:
    """Hardware profile for a specific STM32 MCU variant."""
    name: str
    core: str              # e.g. 'Cortex-M7'
    freq_mhz: int          # max clock frequency
    flash_kb: int          # Flash memory
    sram_kb: int           # SRAM (for activations + weights)
    active_power_mw: float # typical active power at max freq
    # Cycles per MAC for each operation type (from CMSIS-NN benchmarks)
    cycles_per_mac_dw: float    # depthwise conv
    cycles_per_mac_pw: float    # pointwise (1x1) conv
    cycles_per_mac_fc: float    # fully connected / GAP
    cycles_per_mac_conv: float  # regular conv (3x3+)
    # INT8 support (CMSIS-NN requires INT8 for acceleration)
    has_dsp: bool = False
    has_fpu: bool = False
    has_mve: bool = False  # Helium SIMD (M55)
    has_npu: bool = False


# Pre-defined STM32 profiles
STM32_PROFILES = {
    'stm32f4': STM32Profile(
        name='STM32F411 (Cortex-M4F)',
        core='Cortex-M4F', freq_mhz=100, flash_kb=512, sram_kb=128,
        active_power_mw=100.0,
        cycles_per_mac_dw=1.0, cycles_per_mac_pw=0.7,
        cycles_per_mac_fc=0.6, cycles_per_mac_conv=0.9,
        has_dsp=True, has_fpu=True,
    ),
    'stm32h7': STM32Profile(
        name='STM32H743 (Cortex-M7)',
        core='Cortex-M7', freq_mhz=480, flash_kb=2048, sram_kb=1024,
        active_power_mw=350.0,
        cycles_per_mac_dw=0.5, cycles_per_mac_pw=0.4,
        cycles_per_mac_fc=0.3, cycles_per_mac_conv=0.5,
        has_dsp=True, has_fpu=True,
    ),
    'stm32u5': STM32Profile(
        name='STM32U585 (Cortex-M33, ultra-low power)',
        core='Cortex-M33', freq_mhz=160, flash_kb=4096, sram_kb=786,
        active_power_mw=26.0,
        cycles_per_mac_dw=1.0, cycles_per_mac_pw=0.7,
        cycles_per_mac_fc=0.6, cycles_per_mac_conv=0.9,
        has_dsp=True, has_fpu=True,
    ),
    'stm32h5': STM32Profile(
        name='STM32H573 (Cortex-M33)',
        core='Cortex-M33', freq_mhz=250, flash_kb=2048, sram_kb=640,
        active_power_mw=55.0,
        cycles_per_mac_dw=1.0, cycles_per_mac_pw=0.7,
        cycles_per_mac_fc=0.6, cycles_per_mac_conv=0.9,
        has_dsp=True, has_fpu=True,
    ),
    'stm32n6': STM32Profile(
        name='STM32N657 (Cortex-M55 + NPU)',
        core='Cortex-M55', freq_mhz=800, flash_kb=4096, sram_kb=4224,
        active_power_mw=500.0,
        cycles_per_mac_dw=0.25, cycles_per_mac_pw=0.15,
        cycles_per_mac_fc=0.12, cycles_per_mac_conv=0.20,
        has_dsp=True, has_fpu=True, has_mve=True, has_npu=True,
    ),
}


# --------------------------------------------------------------------------
# MAC counter via forward hooks
# --------------------------------------------------------------------------

class _MACCounter:
    """Counts MACs per layer type using PyTorch forward hooks."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.layer_macs: Dict[str, dict] = {}  # name -> {macs, type, shape_in, shape_out}
        self._hooks = []

    def _hook_conv2d(self, name: str):
        def hook(module: nn.Conv2d, inp, out):
            B, C_in, H_in, W_in = inp[0].shape
            B, C_out, H_out, W_out = out.shape
            kH, kW = module.kernel_size
            groups = module.groups

            # MACs = C_out * H_out * W_out * (C_in/groups * kH * kW)
            macs = C_out * H_out * W_out * (C_in // groups) * kH * kW

            # Determine op type
            is_dw = (groups == C_in and groups == C_out)
            is_pw = (kH == 1 and kW == 1 and not is_dw)

            if is_dw:
                op_type = 'depthwise'
            elif is_pw:
                op_type = 'pointwise'
            else:
                op_type = 'conv'

            self.layer_macs[name] = {
                'macs': macs,
                'type': op_type,
                'shape_in': tuple(inp[0].shape),
                'shape_out': tuple(out.shape),
                'params': sum(p.numel() for p in module.parameters()),
            }
        return hook

    def _hook_linear(self, name: str):
        def hook(module: nn.Linear, inp, out):
            B = inp[0].shape[0] if inp[0].dim() > 1 else 1
            macs = B * module.in_features * module.out_features
            self.layer_macs[name] = {
                'macs': macs,
                'type': 'fc',
                'shape_in': tuple(inp[0].shape),
                'shape_out': tuple(out.shape),
                'params': sum(p.numel() for p in module.parameters()),
            }
        return hook

    def _hook_avgpool(self, name: str):
        def hook(module, inp, out):
            B, C, H_in, W_in = inp[0].shape
            # GAP: each output channel needs H*W additions
            macs = B * C * H_in * W_in
            self.layer_macs[name] = {
                'macs': macs,
                'type': 'pool',
                'shape_in': tuple(inp[0].shape),
                'shape_out': tuple(out.shape),
                'params': 0,
            }
        return hook

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                h = module.register_forward_hook(self._hook_conv2d(name))
                self._hooks.append(h)
            elif isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._hook_linear(name))
                self._hooks.append(h)
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                h = module.register_forward_hook(self._hook_avgpool(name))
                self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def total_macs(self) -> int:
        return sum(v['macs'] for v in self.layer_macs.values())

    def macs_by_type(self) -> Dict[str, int]:
        result = {}
        for v in self.layer_macs.values():
            t = v['type']
            result[t] = result.get(t, 0) + v['macs']
        return result


# --------------------------------------------------------------------------
# Activation memory tracker
# --------------------------------------------------------------------------

class _MemoryTracker:
    """Tracks peak activation memory during forward pass — MCU ping-pong model.

    On MCU with CMSIS-NN, activations are processed layer-by-layer with a
    ping-pong (double-buffer) scheme: only the current layer's input + output
    need to be live simultaneously. Peak SRAM for activations is therefore:
        max over all layers of (input_bytes + output_bytes)

    We also track the single largest intermediate tensor (max_single_bytes)
    as a conservative lower bound.
    """

    def __init__(self):
        self.peak_bytes = 0        # peak ping-pong pair: max(in + out)
        self.max_single_bytes = 0  # largest single activation tensor
        self._hooks = []

    def _hook(self, module, inp, out):
        # Only track Conv2d-level ops (leaf ops that map to CMSIS-NN kernels)
        if not isinstance(module, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d)):
            return
        if isinstance(out, torch.Tensor):
            out_bytes = out.numel() * out.element_size()
            self.max_single_bytes = max(self.max_single_bytes, out_bytes)
            # Input activation bytes (first input tensor only)
            in_bytes = 0
            if inp and isinstance(inp[0], torch.Tensor):
                in_bytes = inp[0].numel() * inp[0].element_size()
            # Ping-pong peak = input buffer + output buffer
            pair_bytes = in_bytes + out_bytes
            self.peak_bytes = max(self.peak_bytes, pair_bytes)

    def register(self, model: nn.Module):
        for module in model.modules():
            h = module.register_forward_hook(self._hook)
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# --------------------------------------------------------------------------
# Main profiler
# --------------------------------------------------------------------------

@dataclass
class ProfileResult:
    """Complete profiling result for LUMEN on a target MCU."""
    # Model stats
    total_params: int
    total_macs: int
    macs_by_type: Dict[str, int]

    # Memory
    param_memory_fp32_kb: float
    param_memory_int8_kb: float
    peak_activation_fp32_kb: float
    peak_activation_int8_kb: float
    total_sram_int8_kb: float   # params + activations in INT8

    # MCU target
    mcu_name: str
    mcu_freq_mhz: int
    mcu_sram_kb: int

    # Timing estimates
    estimated_cycles: int
    estimated_latency_ms: float

    # Power & energy
    mcu_active_power_mw: float
    estimated_energy_uj: float  # microjoules per inference

    # Feasibility
    fits_in_sram: bool
    sram_utilization_pct: float

    # Throughput
    throughput_fps: float

    def __str__(self) -> str:
        lines = [
            f"{'='*60}",
            f"LUMEN MCU Profile — {self.mcu_name}",
            f"{'='*60}",
            f"",
            f"[Model]",
            f"  Parameters:          {self.total_params:>12,}",
            f"  Total MACs:          {self.total_macs:>12,}  ({self.total_macs/1e6:.2f} M)",
            f"  MACs breakdown:",
        ]
        for t, m in sorted(self.macs_by_type.items(), key=lambda x: -x[1]):
            pct = 100 * m / max(self.total_macs, 1)
            lines.append(f"    {t:<14}:  {m:>12,}  ({pct:.1f}%)")
        lines += [
            f"",
            f"[Memory]",
            f"  Weights (FP32):      {self.param_memory_fp32_kb:>10.1f} KB",
            f"  Weights (INT8):      {self.param_memory_int8_kb:>10.1f} KB",
            f"  Peak activations:    {self.peak_activation_fp32_kb:>10.1f} KB (FP32)",
            f"                       {self.peak_activation_int8_kb:>10.1f} KB (INT8)",
            f"  Total SRAM (INT8):   {self.total_sram_int8_kb:>10.1f} KB",
            f"  MCU SRAM:            {self.mcu_sram_kb:>10} KB",
            f"  SRAM utilization:    {self.sram_utilization_pct:>10.1f}%",
            f"  Fits in SRAM:        {'YES ✓' if self.fits_in_sram else 'NO ✗':>10}",
            f"",
            f"[Performance @ {self.mcu_freq_mhz} MHz]",
            f"  Est. cycles:         {self.estimated_cycles:>12,}  ({self.estimated_cycles/1e6:.1f} M)",
            f"  Est. latency:        {self.estimated_latency_ms:>10.1f} ms per image",
            f"  Throughput:          {self.throughput_fps:>10.1f} FPS",
            f"",
            f"[Power & Energy]",
            f"  MCU active power:    {self.mcu_active_power_mw:>10.0f} mW",
            f"  Energy/inference:    {self.estimated_energy_uj:>10.1f} µJ",
            f"  Energy/inference:    {self.estimated_energy_uj/1000:>10.3f} mJ",
            f"{'='*60}",
        ]
        return '\n'.join(lines)


class MCUProfiler:
    """Profile LUMEN for deployment on STM32 Cortex-M targets.

    Performs:
    1. MAC counting via forward hooks
    2. Peak activation memory estimation
    3. Latency estimation using CMSIS-NN throughput tables
    4. Energy estimation using STM32 datasheet power figures
    5. SRAM feasibility check

    Usage:
        profiler = MCUProfiler(model, mcu='stm32h7')
        result = profiler.profile(input_size=(1, 3, 32, 32))
        print(result)
    """

    def __init__(self, model: nn.Module, mcu: str = 'stm32h7'):
        self.model = model
        if mcu not in STM32_PROFILES:
            raise ValueError(f"Unknown MCU '{mcu}'. Available: {list(STM32_PROFILES.keys())}")
        self.hw = STM32_PROFILES[mcu]

    def profile(self, input_size: tuple = (1, 3, 32, 32),
                dtype: torch.dtype = torch.float32,
                verbose: bool = True) -> ProfileResult:
        """Run profiling.

        Args:
            input_size: (B, C, H, W) — use B=1 for single-image MCU inference.
            dtype: Input dtype (float32 for PyTorch profiling; INT8 estimates derived).
            verbose: Print result to stdout.

        Returns:
            ProfileResult with all metrics.
        """
        self.model.eval()
        mac_counter = _MACCounter()
        mem_tracker = _MemoryTracker()
        mac_counter.register(self.model)
        mem_tracker.register(self.model)

        dummy = torch.zeros(input_size, dtype=dtype)
        with torch.no_grad():
            _ = self.model(dummy)

        mac_counter.remove()
        mem_tracker.remove()

        # ---- MACs ----
        total_macs = mac_counter.total_macs()
        macs_by_type = mac_counter.macs_by_type()

        # ---- Parameters & weight memory ----
        total_params = sum(p.numel() for p in self.model.parameters())
        param_memory_fp32_kb = total_params * 4 / 1024
        param_memory_int8_kb = total_params * 1 / 1024

        # ---- Activation memory (ping-pong MCU model) ----
        # On MCU with CMSIS-NN: only one layer's (input + output) buffers are live
        # at once (ping-pong scheme). peak_bytes already reflects this.
        peak_act_fp32_kb = mem_tracker.peak_bytes / 1024
        # INT8 activations are 4x smaller (INT8 quantization)
        peak_act_int8_kb = peak_act_fp32_kb / 4.0

        # ---- Total SRAM in INT8 mode ----
        # Weights in Flash (read-only), activations in SRAM
        # So SRAM needed = activation buffers only (weights loaded from Flash)
        total_sram_int8_kb = peak_act_int8_kb

        # ---- Cycle estimation using CMSIS-NN tables ----
        hw = self.hw
        cycles = 0
        for name, info in mac_counter.layer_macs.items():
            t = info['type']
            m = info['macs']
            if t == 'depthwise':
                cycles += m * hw.cycles_per_mac_dw
            elif t == 'pointwise':
                cycles += m * hw.cycles_per_mac_pw
            elif t == 'fc':
                cycles += m * hw.cycles_per_mac_fc
            elif t == 'pool':
                cycles += m * 0.1  # very cheap
            else:  # regular conv
                cycles += m * hw.cycles_per_mac_conv

        # Add overhead: BN, activation, memory accesses (~15% overhead)
        cycles = int(cycles * 1.15)

        # ---- Latency ----
        freq_hz = hw.freq_mhz * 1e6
        latency_ms = (cycles / freq_hz) * 1000.0

        # ---- Power & energy ----
        power_mw = hw.active_power_mw
        energy_uj = power_mw * latency_ms  # mW * ms = µJ

        # ---- Feasibility ----
        fits = total_sram_int8_kb <= hw.sram_kb
        sram_pct = 100.0 * total_sram_int8_kb / hw.sram_kb
        fps = 1000.0 / max(latency_ms, 1e-6)

        result = ProfileResult(
            total_params=total_params,
            total_macs=total_macs,
            macs_by_type=macs_by_type,
            param_memory_fp32_kb=param_memory_fp32_kb,
            param_memory_int8_kb=param_memory_int8_kb,
            peak_activation_fp32_kb=peak_act_fp32_kb,
            peak_activation_int8_kb=peak_act_int8_kb,
            total_sram_int8_kb=total_sram_int8_kb,
            mcu_name=hw.name,
            mcu_freq_mhz=hw.freq_mhz,
            mcu_sram_kb=hw.sram_kb,
            estimated_cycles=cycles,
            estimated_latency_ms=latency_ms,
            mcu_active_power_mw=power_mw,
            estimated_energy_uj=energy_uj,
            fits_in_sram=fits,
            sram_utilization_pct=sram_pct,
            throughput_fps=fps,
        )

        if verbose:
            print(result)

        return result

    def compare_all_mcus(self, input_size: tuple = (1, 3, 32, 32)) -> Dict[str, ProfileResult]:
        """Profile the model on all available MCU targets and print comparison table."""
        results = {}
        for mcu_key in STM32_PROFILES:
            self.hw = STM32_PROFILES[mcu_key]
            results[mcu_key] = self.profile(input_size, verbose=False)

        # Print comparison table
        print(f"\n{'MCU':<32} {'Latency':>10} {'Energy':>10} {'SRAM':>10} {'Fits':>6} {'FPS':>8}")
        print('-' * 80)
        for key, r in results.items():
            fits_str = 'YES' if r.fits_in_sram else 'NO'
            print(f"{r.mcu_name:<32} {r.estimated_latency_ms:>8.1f}ms "
                  f"{r.estimated_energy_uj:>8.1f}µJ "
                  f"{r.total_sram_int8_kb:>8.1f}KB "
                  f"{fits_str:>6} "
                  f"{r.throughput_fps:>7.1f}")
        print()
        return results
