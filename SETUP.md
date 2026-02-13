# Setup Guide

## Quick Setup with UV (Recommended)

This project uses **uv** for fast, reproducible Python dependency management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Create Virtual Environment

```bash
# Navigate to project
cd quranNemoASR

# Create Python 3.10 venv
uv venv -p 3.10

# Activate (optional, scripts use .venv/bin/python directly)
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA 12.4

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchaudio torchvision
```

### 4. Install NeMo and Dependencies

```bash
# Downgrade numpy to avoid ABI conflicts with NeMo
uv pip install "numpy<2"

# Install NeMo toolkit with ASR extensions
uv pip install "nemo_toolkit[asr]" datasets soundfile tqdm librosa jiwer
```

### 5. Verify Installation

```bash
.venv/bin/python - << 'PY'
import torch
import nemo
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NeMo version: {nemo.__version__}")
PY
```

## RunPod / Cloud GPU Setup

### Quick Start (Copy-Paste)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone repo
git clone <your-repo-url> quranNemoASR
cd quranNemoASR

# Setup environment
uv venv -p 3.10
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio torchvision
uv pip install "numpy<2"
uv pip install "nemo_toolkit[asr]" datasets soundfile tqdm librosa jiwer

# Verify
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Start training
bash train_nemo_finetune.sh
```

## Environment Versions

| Package | Version | Index |
|---------|---------|-------|
| Python | 3.10.x | |
| torch | 2.6.0+cu124 | PyTorch (CUDA 12.4) |
| torchaudio | 2.6.0+cu124 | PyTorch (CUDA 12.4) |
| torchvision | 0.21.0+cu124 | PyTorch (CUDA 12.4) |
| numpy | 1.26.4 | PyPI (pinned <2) |
| nemo-toolkit | 2.6.2 | PyPI |
| lightning | 2.4.0 | PyPI |

## Troubleshooting

### NumPy ABI Error

If you see:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Fix:
```bash
uv pip install --force-reinstall "numpy<2"
```

### Missing SequenceParallel

If you see:
```
cannot import name 'SequenceParallel' from 'torch.distributed.tensor.parallel'
```

Fix: Upgrade torch to 2.3.1+
```bash
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchaudio torchvision
```

### CUDA Not Available

Verify CUDA version matches PyTorch build:
```bash
nvidia-smi  # Check CUDA driver version
python -c "import torch; print(torch.version.cuda)"  # Should show 12.4
```

## Dataset Preparation

```bash
# Download and prepare Quran dataset
python prepare_dataset.py \
  --dataset_name hifzyml/quran_dataset_v0 \
  --output_dir data \
  --copy_audio

# This creates:
# - data/manifests/train.json
# - data/manifests/val.json
# - data/manifests/test.json
# - data/audio/{train,val,test}/*.wav
```

## Fine-tuning

```bash
# Train using the main script
bash train_nemo_finetune.sh

# Monitor with TensorBoard
tensorboard --logdir nemo_experiments --port 6006
```

The training script will:
1. Load pretrained Arabic FastConformer
2. Fine-tune on Quran dataset (50 epochs)
3. Save checkpoints to `nemo_experiments/`
4. Keep top-3 models by val_wer

## Hardware Requirements

- **Minimum:** 1x GPU with 24GB VRAM (RTX 3090, RTX 4090, A5000)
- **Recommended:** 2x GPUs for faster training with DDP
- **RAM:** 32GB+ system memory
- **Storage:** 50GB+ free space (dataset + models + checkpoints)

## Training Time Estimates

| Hardware | Batch Size | Time per Epoch | Total (50 epochs) |
|----------|------------|----------------|-------------------|
| 1x RTX 3090 | 16 | ~30-40 min | ~25-30 hours |
| 2x RTX 3090 | 32 (16 per GPU) | ~15-20 min | ~12-15 hours |
| 1x A100 | 32 | ~15-20 min | ~12-15 hours |

## Next Steps

After training completes:
1. Check `nemo_experiments/{run_name}/checkpoints/` for best model
2. Use best checkpoint for inference
3. Evaluate on test set with normalized text for accurate WER

## Pretrained Model Download

The pretrained Arabic model is downloaded automatically from Hugging Face:
- Model: `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
- Size: ~460MB
- Cached in: `pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo`

If download fails, manually download from [Hugging Face](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) and place in `pretrained_models/`.
