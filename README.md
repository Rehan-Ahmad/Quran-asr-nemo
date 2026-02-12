# Quran ASR Fine-tuning with NeMo

Fine-tune NVIDIA's FastConformer Hybrid model on the Quran dataset using the NeMo toolkit.

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** NeMo requires a CUDA-capable GPU. Ensure your environment has `torch` with CUDA support:
> ```bash
> conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
> ```

### 2) Prepare dataset

The `prepare_dataset.py` script:
- Downloads the Quran dataset from Hugging Face
- Resamples all audio to **16 kHz mono** (required by the model)
- Validates and skips invalid samples
- Splits into train/val/test
- Outputs NeMo-compliant JSONL manifests

```bash
python prepare_dataset.py \
  --dataset_name hifzyml/quran_dataset_v0 \
  --output_dir data \
  --copy_audio
```

**Options:**
- `--copy_audio`: Copy and resample audio into `data/audio/` (recommended for stable paths)
- `--num_samples N`: Limit to first N samples (useful for testing)
- `--val_ratio 0.05`: Validation split ratio
- `--test_ratio 0.05`: Test split ratio

**Outputs:**
- `data/manifests/{train,val,test}.json` — NeMo manifests
- `data/vocab.txt` — Character vocabulary
- `data/audio/{train,val,test}/` — 16 kHz mono WAV files (if `--copy_audio`)

### 3) Fine-tune

The NeMo training script and config are included in `nemo_scripts/`:

```bash
export DATA_DIR="$PWD/data"
bash train_nemo.sh
```

To train on CPU (slow):

```bash
bash train_nemo.sh trainer.devices=1 trainer.accelerator=cpu
```

## Model Details

- **Model:** `stt_en_fastconformer_hybrid_large_streaming_multi` (NVIDIA)
- **Audio:** 16 kHz mono WAV (resampled automatically)
- **Training approach:** Hybrid Transducer-CTC
- **Decoders:** Transducer (RNNT) or CTC (configurable)

## Project Structure

```
quranNemoASR/
├── prepare_dataset.py           # Dataset preparation script
├── train_nemo.sh                # Training launcher
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── nemo_scripts/                # NeMo official scripts (downloaded from GitHub)
│   ├── speech_to_text_hybrid_rnnt_ctc_bpe.py
│   └── fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml
│   └── process_asr_text_tokenizer.py
└── data/                        # Dataset (created during preparation)
    ├── manifests/               # Train/val/test JSON manifests
    ├── audio/                   # 16 kHz mono WAV files
    └── vocab.txt                # Character vocabulary
```

## Troubleshooting

### CUDA not available
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, install CUDA-enabled PyTorch:
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Driver/library version mismatch
Update NVIDIA driver and reboot:
```bash
sudo apt update && sudo apt upgrade nvidia-driver-*
sudo reboot
```

Then reinstall torch with CUDA.

## References

- [NeMo ASR documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/asr_dataset.html)
- [FastConformer Streaming model card](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- [Quran dataset](https://huggingface.co/datasets/hifzyml/quran_dataset_v0)
