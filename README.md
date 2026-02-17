# Quran ASR Project

Fine-tune NVIDIA NeMo's FastConformer model on Quranic Arabic speech with custom BPE tokenization.

## Quick Start

### 1. Prepare Data
```bash
python prepare_dataset.py
```
Expected data structure:
```
data/
├── audio/
│   ├── train/    # Training audio files
│   ├── val/      # Validation audio files
│   └── test/     # Test audio files
└── manifests/
    ├── train.json
    ├── val.json
    └── test.json
```

Manifest format (JSON lines):
```json
{"audio_filepath": "path/to/audio.wav", "duration": 5.0, "text": "النص المكتوب"}
```

### 2. Build Tokenizer
```bash
python train_custom_tokenizer.py
```
Creates: `tokenizer/quran_tokenizer_bpe_v1024/`

### 3. Fine-tune Model
```bash
python finetune_multilingual_simple.py
```
Or:
```bash
./train_nemo_finetune.sh
```
Outputs: `nemo_experiments/FastConformer-Streaming-Custom-Tokenizer-*/`

### 4. Evaluate
```bash
python evaluate.py
```

The evaluation script is **interactive**:
1. Automatically discovers all trained models in `nemo_experiments/`
2. Shows available models with type (STREAMING 🌊 or STANDARD 📶)
3. You select which model to evaluate (enter 1, 2, 3, etc.)
4. You choose how many samples to test (default 20, max 3,745)
5. System adapts evaluation based on model type
6. Shows results with accuracy percentage

Example:
```
🎙️  Quran ASR Model Evaluator

================================================================================
📦 AVAILABLE MODELS
================================================================================
 1. [📶 STANDARD] FastConformer-Custom-Tokenizer/...nemo
    Size: 438.5 MB
 2. [🌊 STREAMING] FastConformer-Streaming-Custom-Tokenizer/...nemo
    Size: 439.1 MB

================================================================================
Select model (1-2): 1

✓ Selected: FastConformer-Custom-Tokenizer
✓ Type: STANDARD

How many samples to evaluate? (default 20): 50

================================================================================
📊 EVALUATING (Standard Mode)
================================================================================

Results: 45/50 correct (90.0% accuracy)
```

## Configuration

Edit hyperparameters in `pyproject.toml`:
- `learning_rate`: Typically 5e-4
- `batch_size`: Depends on GPU memory
- `num_epochs`: Training epochs
- `model_path`: Pretrained model checkpoint
- Data paths

## File Overview

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Create JSON manifests from audio + text |
| `train_custom_tokenizer.py` | Build SentencePiece BPE tokenizer (1024 tokens) |
| `finetune_multilingual_simple.py` | Main training script (NeMo + PyTorch Lightning) |
| `train_nemo_finetune.sh` | Shell wrapper for training |
| `evaluate.py` | Inference on validation samples |
| `pyproject.toml` | Configuration file |
| `requirements.txt` | Python dependencies |

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `nemo-toolkit[asr]`
- `torch`, `pytorch-lightning`
- `librosa`, `soundfile`, `jiwer`

## Notes

- Uses streaming FastConformer architecture
- Custom Arabic BPE tokenizer (1024 tokens)
- Hybrid RNNT/CTC loss combination
- Validation WER tracked per epoch
