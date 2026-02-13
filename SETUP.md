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
# Clone/navigate to project
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
uv pip install "nemo_toolkit[asr]" datasets soundfile tqdm librosa
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
uv pip install "nemo_toolkit[asr]" datasets soundfile tqdm librosa

# Verify
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Start training
bash train_nemo_with_diacritics.sh
```

### Reproducible Setup with pyproject.toml

If you want to lock exact versions:

```bash
# Generate uv.lock file
uv pip freeze > requirements.lock.txt

# On new machine, install from lock
uv pip install -r requirements.lock.txt
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

## Training

### Fine-tune Arabic Model (Recommended)

The pretrained Arabic FastConformer already knows Arabic phonetics and diacritics, including Quranic recitation!

```bash
# Step 1: Download pretrained Arabic model
bash download_arabic_model.sh

# Step 2: Fine-tune on your Quran dataset  
bash train_nemo_finetune.sh

# Step 3: Monitor with TensorBoard
python launch_tensorboard.py --logdir nemo_experiments --port 6006
```

**Why this is better:**
- ✅ Pretrained on 1100h Arabic speech (including 390h Quranic)
- ✅ Baseline WER: 6.55% on Quran (already excellent!)
- ✅ Training time: 12-24 hours (vs 48-72h from scratch)
- ✅ Expected final WER: < 5%
- ✅ Already supports diacritical marks
- ✅ Uses model's pretrained tokenizer (no custom tokenizer needed)

## File Structure

```
quranNemoASR/
├── .venv/                     # UV virtual environment
├── pyproject.toml             # Project metadata and uv config
├── requirements.txt           # Dependency list
├── SETUP.md                   # This file
├── train_nemo_with_diacritics.sh  # Training script
├── data/manifests/            # Train/val/test JSON manifests
├── tokenizer/                 # Regenerated tokenizer (with diacritics)
└── nemo_experiments/          # Training outputs
```
