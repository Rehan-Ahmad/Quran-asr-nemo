# Quran ASR Fine-tuning with NeMo

Fine-tune NVIDIA's Arabic FastConformer model on the Quran dataset using the NeMo toolkit.

## Why Fine-tuning?

Instead of training from scratch, we use NVIDIA's pretrained Arabic model:
- ✅ **Already trained on 1100h Arabic speech** (including 390h Quran)
- ✅ **Baseline WER: 6.55%** on Quranic test set
- ✅ **Supports diacritical marks** natively
- ✅ **12-24 hours training** (vs 48-72h from scratch)
- ✅ **Expected final WER: < 5%**

## Quick Start

### 1) Setup Environment

See [SETUP.md](SETUP.md) for detailed instructions. Quick version:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv and install dependencies
uv venv -p 3.10
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio torchvision
uv pip install "numpy<2"
uv pip install "nemo_toolkit[asr]" datasets soundfile tqdm librosa jiwer
```

### 2) Prepare Dataset

```bash
python prepare_dataset.py \
  --dataset_name hifzyml/quran_dataset_v0 \
  --output_dir data \
  --copy_audio
```

### 3) Fine-tune on Quran

```bash
bash train_nemo_finetune.sh
```

### 4) Monitor Training

```bash
tensorboard --logdir nemo_experiments --port 6006
```

## RunPod Automation (uv)

This repository includes a fully automated RunPod script that:
- Creates a uv virtual environment
- Installs CUDA PyTorch + dependencies
- Downloads the pretrained .nemo model from Hugging Face
- Prepares the dataset from Hugging Face
- Trains a custom tokenizer
- Fine-tunes the model
- Evaluates the best checkpoint
- Optionally starts TensorBoard

### One-command RunPod Start

```bash
bash runpod_automation.sh
```

### Optional Environment Overrides

```bash
export HF_DATASET="hifzyml/quran_dataset_v0"
export HF_MODEL_REPO="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
export HF_MODEL_FILENAME="stt_ar_fastconformer_hybrid_large_pcd.nemo"
export EXP_NAME="FastConformer-Custom-Tokenizer"
export TB_PORT=6006
export RUN_TENSORBOARD=1
# Optional for private repos
export HF_TOKEN="<your_hf_token>"
```

## Model Details

- **Base Model:** `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
- **Architecture:** FastConformer Hybrid RNNT-CTC (115M params)
- **Vocab Size:** 1024 (SentencePiece BPE)
- **Pretrained on:**
  - MASC: 690h
  - Common Voice: 65h
  - Fleurs: 5h
  - TarteelAI Everyayah: 390h (Quranic recitation!)
- **Baseline WER on Quran:** 6.55% (measured on test set: 70.56% with strict diacritics, 6-8% normalized)

## Project Structure

```
quranNemoASR/
├── .venv/                     # UV virtual environment
├── train_nemo_finetune.sh     # Main fine-tuning script
├── prepare_dataset.py         # Dataset preparation
├── data/manifests/            # Train/val/test JSON manifests
├── nemo_scripts/              # Training config and scripts
│   ├── fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml
│   └── speech_to_text_hybrid_rnnt_ctc_bpe.py
├── pretrained_models/         # Downloaded .nemo models
├── tokenizer/                 # BPE tokenizer (1024 vocab)
└── nemo_experiments/          # Training outputs & checkpoints
```

## Expected Results

| Stage | WER (Normalized) | Time |
|-------|------------------|------|
| Pretrained (no fine-tuning) | ~6-8% | 0h |
| After 25 epochs | ~4-6% | 6-12h |
| After 50 epochs | **< 4%** | 12-24h |

*Note: Strict diacritics matching gives 70% WER, but normalized (without diacritics) shows actual model quality.*

## Training Configuration

The fine-tuning uses optimized hyperparameters:

- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealing (warmup=1000 steps, min_lr=1e-6)
- **Batch Size:** 32 per GPU
- **Precision:** bf16-mixed (faster + lower memory)
- **Strategy:** DDP (multi-GPU)
- **Epochs:** 50
- **Checkpointing:** Top-3 models by val_wer

## References

- [NeMo ASR documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/asr_dataset.html)
- [FastConformer model card](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)
- [Quran dataset](https://huggingface.co/datasets/hifzyml/quran_dataset_v0)
