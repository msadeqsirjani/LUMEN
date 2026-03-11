# LUMEN

**Ultra-Lightweight Super-Resolution for STM32 Cortex-M MCUs**

LUMEN is a compact SR architecture purpose-designed for deployment on resource-constrained embedded systems while remaining competitive with SOTA ultra-lightweight SR models (~141K params, ×2 and ×4 upscaling).

## Architecture

```
Input (LR) → Conv3×3 → N × LumenGroup → Conv3×3 → PixelShuffle → Output (HR)
                                                  ↑_________global skip_________|
```

Each **LumenGroup** contains a stack of **LumenBlocks** with a group-level residual. Each LumenBlock consists of:

- **MSDM** (Multi-Scale Depthwise Mixer): captures local (3×3, 5×5 DW) and semi-global (1×7, 7×1 strip DW) context in parallel — O(C·HW), no softmax, CMSIS-NN compatible
- **LCR** (Lightweight Channel Recalibrator): SE-style global pooling with hard-sigmoid (no exp) for channel recalibration at O(C²) cost
- **LumenFFN**: inverted bottleneck (PW expand → DW 3×3 → LCR → PW project) with ReLU6 — INT8-quantizable

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

```bash
python scripts/download_datasets.py   # downloads DIV2K + Benchmarks
```

Expected layout:

```
datasets/
├── DIV2K/
│   ├── HR/Train/
│   └── LR_bicubic/Train/X4/
└── Benchmarks/
    ├── Set5/
    ├── Set14/
    ├── BSDS100/
    ├── Urban100/
    └── Manga109/
```

## Training

```bash
python basicsr/train.py -opt options/train/train_LUMEN_x4_DIV2K.yml
```

## Evaluation

```bash
python basicsr/test.py -opt options/test/test_LUMEN_x4.yml
```

## Benchmark

```bash
python scripts/benchmark.py
python scripts/benchmark.py --input-size 1 3 64 64
```

## Export

```bash
# ONNX + TorchScript
python scripts/export.py --weights pretrained/LUMEN_x4.pth

# Low-rank compression
python scripts/compress.py --weights pretrained/LUMEN_x4.pth --rank 4
```

### STM32 Deployment

1. Export to ONNX via `scripts/export.py`
2. Open STM32CubeMX → Add X-CUBE-AI middleware
3. Import `LUMEN_x4.onnx` — X-CUBE-AI handles INT8 quantization and C code generation
4. Flash to STM32 and run

## Why LUMEN?

Lumen is the SI unit of luminous flux — a measure of light output. Just like the model: it takes dim, low-resolution inputs and illuminates them with sharp, high-resolution details. Fast, efficient, and purpose-built for embedded deployment.
