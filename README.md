# LUMEN

**Lightweight Multi-Scale Network for Efficient Super-Resolution**

LUMEN is a GPU-efficient SR architecture combining re-parameterizable convolutions, three-scale partial depthwise mixing, and parameter-free attention. Designed for competitive quality at minimal inference cost.

## Architecture

```
Input (LR) --> Conv3x3 --> N x LumenBlock --> Conv3x3 + skip --> PixelShuffle --> Output (HR)
```

Each **LumenBlock** consists of:

- **RepDCCM** — Re-parameterizable Double Conv Channel Mixer (3x3+1x1+identity branches during training, collapses to single 3x3 at inference)
- **EPMSDM** — Enhanced Partial Multi-Scale Depthwise Mixer (fine 3x3 + medium 5x5 + global Kx1/1xK on partial channels)
- **PFA** — Parameter-Free Attention (`x * sigmoid(|x|)`, zero extra parameters)
- **StarReLU** — Efficient activation (`s * ReLU(x)^2 + b`, 4 FLOPs vs GELU's 14)

### Model Variants

| Variant | embed_dim | blocks | partial_ch | large_kernel | Params (fused) |
|---------|-----------|--------|------------|--------------|----------------|
| LUMEN-tiny | 32 | 8 | 8 | 17 | ~174K |
| LUMEN | 48 | 16 | 12 | 21 | ~581K |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Download Datasets

```bash
python scripts/download_datasets.py
```

Expected layout:

```
datasets/
├── DIV2K/
│   ├── HR/Train/
│   └── LR_bicubic/Train/X{2,3,4}/
├── Flickr2K/                          # optional, for DF2K training
│   ├── HR/
│   └── LR_bicubic/X{2,3,4}/
└── Benchmarks/
    ├── Set5/{HR, LR_bicubic/X{2,3,4}}/
    ├── Set14/
    ├── BSDS100/
    ├── Urban100/
    └── Manga109/
```

### 3. Train

```bash
# Train x2 from scratch (primary model)
python basicsr/train.py -opt options/train/train_LUMEN_x2_DIV2K.yml

# Train x3 and x4 (same procedure)
python basicsr/train.py -opt options/train/train_LUMEN_x3_DIV2K.yml
python basicsr/train.py -opt options/train/train_LUMEN_x4_DIV2K.yml
```

### 4. Evaluate

```bash
# Test on all 5 benchmarks (Set5, Set14, BSDS100, Urban100, Manga109)
python basicsr/test.py -opt options/test/test_LUMEN_x4.yml
python basicsr/test.py -opt options/test/test_LUMEN_x2.yml
python basicsr/test.py -opt options/test/test_LUMEN_x3.yml
```

### 5. Benchmark (Params / MACs / Latency / MGO)

```bash
# Default: all scales, HD (1280x720) output
python scripts/benchmark.py

# With GPU latency and memory measurement
python scripts/benchmark.py --latency

# Single scale
python scripts/benchmark.py --upscale 4 --latency
```

---

## Experiment Reproduction (ICML)

### Step 1: Run Ablation Studies

All ablation configs are in `options/train/ablation/`. Each trains at x2 scale on DIV2K for 500K iterations.

```bash
# Preview all ablation configs (no training)
python scripts/ablation.py --dry-run

# Run all ablations
python scripts/ablation.py --gpu 0

# Run a specific group
python scripts/ablation.py --filter kernel    # kernel size: K=7,11,13,17,21
python scripts/ablation.py --filter partial   # partial channels: p=4,6,8,10
python scripts/ablation.py --filter loss      # loss functions: L1, L1+SSIM, L1+FFT

# Collect results from completed experiments
python scripts/ablation.py --collect-only
```

**Component ablation (Table 4 in paper):**

| Config | What changes |
|--------|-------------|
| `ablation_a_baseline` | Plain DCCM + 2-scale PMSDM + no attention + GELU |
| `ablation_b_reparam` | + RepDCCM (re-parameterization) |
| `ablation_c_epmsdm` | + EPMSDM (3-scale spatial mixing) |
| `ablation_d_pfa` | + PFA (parameter-free attention) |
| `ablation_e_starrelu` | + StarReLU (= full LUMEN) |

### Step 2: Knowledge Distillation (optional)

Train a larger teacher first, then distill into LUMEN-tiny:

```bash
# Train teacher (LUMEN with embed_dim=48, 16 blocks)
# ... (modify config accordingly)

# Distill into student
python scripts/train_distill.py options/train/train_LUMEN_x4_DIV2K.yml \
    --teacher-weights pretrained/LUMEN_teacher_x4.pth \
    --alpha 0.8
```

### Step 3: Analysis and Visualization

#### LAM (Local Attribution Map)

Shows which input pixels influence the output — demonstrates effective receptive field.

```bash
python scripts/lam_analysis.py \
    --checkpoint pretrained/LUMEN_x4.pth \
    --image datasets/Benchmarks/Urban100/LR_bicubic/X4/img_004.png \
    --scale 4 \
    --output results/lam_analysis.png

# Compare multiple models
python scripts/lam_analysis.py \
    --checkpoint pretrained/LUMEN_x4.pth pretrained/PLKSR_x4.pth \
    --labels LUMEN PLKSR-tiny \
    --image datasets/Benchmarks/Urban100/LR_bicubic/X4/img_004.png \
    --scale 4
```

#### Fourier Feature Visualization

Shows frequency content captured at different network depths.

```bash
python scripts/fourier_viz.py \
    --checkpoint pretrained/LUMEN_x4.pth \
    --image datasets/Benchmarks/Urban100/LR_bicubic/X4/img_004.png \
    --scale 4 \
    --output results/fourier_viz.png

# Compare two models at specific blocks
python scripts/fourier_viz.py \
    --checkpoint pretrained/LUMEN_x4.pth pretrained/PLKSR_x4.pth \
    --labels LUMEN PLKSR-tiny \
    --image img.png --scale 4 --block-idx 0 3 7
```

#### Visual Comparison

Generates side-by-side crop comparisons with per-crop PSNR.

```bash
python scripts/visual_compare.py \
    --gt datasets/Benchmarks/Urban100/HR/img_004.png \
    --lr datasets/Benchmarks/Urban100/LR_bicubic/X4/img_004.png \
    --checkpoint pretrained/LUMEN_x4.pth pretrained/PLKSR_x4.pth \
    --labels LUMEN PLKSR-tiny \
    --scale 4 \
    --output results/visual_compare.png

# With custom crop region (HR coordinates)
python scripts/visual_compare.py \
    --gt img_hr.png --lr img_lr.png \
    --checkpoint model.pth --labels LUMEN --scale 4 \
    --crop 200 300 128 128
```

#### Pareto Plot

Generates latency-vs-PSNR scatter with built-in baseline data.

```bash
python scripts/pareto_plot.py --output results/pareto_plot.png

# With custom data
python scripts/pareto_plot.py --data results/benchmark_data.json
```

### Step 4: Export for Deployment

```bash
# ONNX + TorchScript
python scripts/export.py --weights pretrained/LUMEN_x4.pth

# Low-rank compression
python scripts/compress.py --weights pretrained/LUMEN_x4.pth --rank 4
```

---

## Project Structure

```
LUMEN/
├── basicsr/
│   ├── archs/lumen/
│   │   ├── model.py          # LUMEN model (main entry)
│   │   ├── lumen_block.py    # LumenBlock with ablation flags
│   │   ├── rep_dccm.py       # Re-parameterizable channel mixer
│   │   ├── rep_conv.py       # Re-parameterizable 3x3 convolution
│   │   ├── dccm.py           # Plain channel mixer (ablation baseline)
│   │   ├── epmsdm.py         # 3-scale partial depthwise mixer
│   │   ├── pmsdm.py          # 2-scale partial depthwise (ablation baseline)
│   │   ├── pfa.py            # Parameter-free attention
│   │   └── star_relu.py      # StarReLU activation
│   ├── losses/losses.py      # L1, FFT, SSIM losses
│   ├── models/sr_model.py    # Training loop (SRModel)
│   ├── distill/loss.py       # Knowledge distillation loss
│   ├── train.py              # Training entry point
│   └── test.py               # Evaluation entry point
├── scripts/
│   ├── benchmark.py           # Params, MACs, latency, MGO
│   ├── ablation.py            # Automated ablation runner
│   ├── lam_analysis.py        # Local Attribution Map
│   ├── fourier_viz.py         # Fourier feature visualization
│   ├── visual_compare.py      # Side-by-side crop comparison
│   ├── pareto_plot.py         # Latency vs PSNR scatter
│   ├── train_distill.py       # Knowledge distillation training
│   ├── export.py              # ONNX / TorchScript export
│   └── compress.py            # SVD compression
├── options/
│   ├── train/
│   │   ├── train_LUMEN_x{2,3,4}_DIV2K.yml
│   │   └── ablation/          # 15 ablation configs
│   └── test/
│       └── test_LUMEN_x{2,3,4}.yml
└── docs/
    └── main.tex               # Paper LaTeX source
```

## Re-parameterization

LUMEN uses re-parameterizable convolutions that have different topologies at training vs inference:

```
Training:   ┌─ Conv3x3+BN ─┐
            ├─ Conv1x1+BN ─┤ → Sum → StarReLU → ...
            └─ Identity+BN ─┘

Inference:  Conv3x3 → StarReLU → ...    (single fused conv)
```

To fuse for deployment:

```python
from basicsr.archs.lumen import LUMEN

model = LUMEN(upscale=4)
model.load_state_dict(torch.load("pretrained/LUMEN_x4.pth"))
model.fuse()   # collapse re-param branches
model.eval()
# Now ready for inference / export
```

---

## Why LUMEN?

Lumen is the SI unit of luminous flux — a measure of light output. Just like the model: it takes dim, low-resolution inputs and illuminates them with sharp, high-resolution details.
