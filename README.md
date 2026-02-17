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

The evaluation script uses **NVIDIA NeMo's official evaluation pattern**:
- **Proper WER/CER Metrics**: Computes word/character error rates correctly
- **Model Discovery**: Automatically finds all trained `.nemo` checkpoints
- **Interactive Selection**: Browse and select models with numbered menu
- **Text Processing**: Handles punctuation, case normalization
- **Per-sample Analysis**: Detailed results for each sample (saved to JSON)
- **Streaming Support**: Evaluates streaming models with proper context sizes

Example workflow:
```
1. Script discovers all models in nemo_experiments/
2. Shows available models: [STREAMING] or [STANDARD]
3. You select model by number (1, 2, 3, etc.)
4. Script transcribes validation samples
5. Computes WER and CER metrics
6. Results: evaluation_results.json
```

Example output:
```
================================================================================
Quran ASR Evaluation Script
================================================================================
✓ Model loaded
✓ Tokenizer applied
✓ Model moved to GPU
Loading manifest from data/manifests/val.json
Loaded 3745 samples from manifest
Transcribing 3745 samples...
  Progress: 375/3745
  Progress: 750/3745
  ...
Transcription complete
Computing metrics...
================================================================================
EVALUATION RESULTS
================================================================================
Total samples: 3745
WER: 0.1542 (15.42%)
CER: 0.0823 (8.23%)
Output: evaluation_results.json
================================================================================
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
