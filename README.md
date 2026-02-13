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
uv pip install "nemo_toolkit[asr]" datasets soundfile tqdm librosa
```

### 2) Prepare Dataset

```bash
python prepare_dataset.py \
  --dataset_name hifzyml/quran_dataset_v0 \
  --output_dir data \
  --copy_audio
```

### 3) Download Pretrained Arabic Model

```bash
bash download_arabic_model.sh
```

### 4) Fine-tune on Quran

```bash
bash train_nemo_finetune.sh
```

### 5) Monitor Training

```bash
python launch_tensorboard.py --logdir nemo_experiments --port 6006
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
- **Baseline WER on Quran:** 6.55%

- **Baseline WER on Quran:** 6.55%

## Project Structure

```
quranNemoASR/
├── .venv/                     # UV virtual environment
├── download_arabic_model.sh   # Download pretrained Arabic model
├── train_nemo_finetune.sh     # Fine-tune on Quran dataset
├── prepare_dataset.py         # Dataset preparation
├── inspect_predictions.py     # Check model predictions
├── evaluate_best_model.py     # Final evaluation
├── launch_tensorboard.py      # Training monitoring
├── data/manifests/            # Train/val/test JSON manifests
├── pretrained_models/         # Downloaded models
└── nemo_experiments/          # Fine-tuning outputs
```

## Expected Results

| Stage | WER | Time |
|-------|-----|------|
| Pretrained (no fine-tuning) | ~6.55% | 0h |
| After 25 epochs | ~5-6% | 6-12h |
| After 50 epochs | **< 5%** | 12-24h |

## Troubleshooting

See [SETUP.md](SETUP.md) for detailed troubleshooting.

## References

- [NeMo ASR documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/asr_dataset.html)
- [FastConformer Streaming model card](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- [Quran dataset](https://huggingface.co/datasets/hifzyml/quran_dataset_v0)
