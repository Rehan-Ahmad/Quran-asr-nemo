# Quran NeMo Experiments - Complete Overview

## Folder Structure

```
nemo_experiments/
├── [ARCHITECTURE NAME]/
│   ├── 2026-MM-DD_HH-MM-SS/        (Latest Run Folder by Timestamp)
│   │   ├── checkpoints/
│   │   │   └── best_model.nemo    (Best 3 models, latest saved first)
│   │   ├── hparams.yaml           (Configuration: streaming, decoder, etc.)
│   │   ├── nemo_log_...txt        (Training/evaluation logs)
│   │   └── events.out.tfevents    (TensorBoard events)
│   │
│   ├── 2026-MM-DD_HH-MM-SS/        (Previous Runs)
│   └── ...
│
└── [NESTED ARCHITECTURE]/
    └── [EXPERIMENT NAME]/
        └── 2026-MM-DD_HH-MM-SS/
            ├── checkpoints/
            └── ...
```

## Discovered Architectures

### 1. **FastConformer-Custom-Tokenizer** 📶 STANDARD
- **Latest Run**: 2026-02-14_08-36-37
- **Best Model**: FastConformer-Custom-Tokenizer.nemo (438.0 MB)
- **Path**: `nemo_experiments/FastConformer-Custom-Tokenizer/2026-02-14_08-36-37/checkpoints/FastConformer-Custom-Tokenizer.nemo`
- **Type**: Standard (non-streaming)
- **Status**: ✅ Ready for evaluation

### 2. **FastConformer-Hybrid-Transducer-CTC-BPE-Streaming** 🌊 STREAMING
- **Latest Run**: 2026-02-14_00-14-11
- **Best Model**: FastConformer-Hybrid-Transducer-CTC-BPE-Streaming.nemo (438.0 MB)
- **Path**: `nemo_experiments/FastConformer-Hybrid-Transducer-CTC-BPE-Streaming/2026-02-14_00-14-11/checkpoints/FastConformer-Hybrid-Transducer-CTC-BPE-Streaming.nemo`
- **Type**: Streaming, Hybrid RNNT/CTC
- **Config Parameters**:
  - `decoder_type`: Can select 'ctc' or 'rnnt' for decoding
  - `att_context_size`: For streaming attention cache
- **Status**: ✅ Ready for evaluation

### 3. **FastConformer-Streaming-Custom-Tokenizer** 🌊 STREAMING
- **Latest Run**: 2026-02-15_13-35-12
- **Best Model**: FastConformer-Streaming-Custom-Tokenizer.nemo (438.5 MB)
- **Path**: `nemo_experiments/FastConformer-Streaming-Custom-Tokenizer/2026-02-15_13-35-12/checkpoints/FastConformer-Streaming-Custom-Tokenizer.nemo`
- **Type**: Streaming
- **Tokenizer**: Custom BPE (1024 tokens)
- **Status**: ✅ Ready for evaluation

### 4. **FastConformer-Streaming-Custom-Tokenizer-Official-Seed** 🌊 STREAMING
- **Latest Run**: 2026-02-16_12-24-08
- **Best Model**: finetune_streaming_quran.nemo (438.4 MB)
- **Path**: `nemo_experiments/FastConformer-Streaming-Custom-Tokenizer-Official-Seed/finetune_streaming_quran/2026-02-16_12-24-08/checkpoints/finetune_streaming_quran.nemo`
- **Type**: Streaming (Official pre-trained seed)
- **Note**: Uses nested structure with experiment name `finetune_streaming_quran`
- **Warning**: ⚠️ Earlier evaluation showed 0% accuracy - kernel tokens, overfitting after epoch 43
- **Status**: ❌ Not recommended for production

### 5. **FastConformer-CTC-BPE-Streaming-Quran** 🌊 STREAMING
- **Latest Run**: 2026-02-17_05-27-58
- **Best Model**: None (no checkpoints saved)
- **Status**: ⚠️ Incomplete - no checkpoints found

### 6. **FastConformer-English-Quran-Tokenizer** 📶 STANDARD
- **Latest Run**: 2026-02-17_12-11-54
- **Best Model**: None (no checkpoints saved)
- **Status**: ⚠️ Incomplete - no checkpoints found

### 7. **FastConformer-Streaming-From-Pretrained** 🌊 STREAMING
- **Latest Run**: 2026-02-15_21-34-26
- **Best Model**: None (no checkpoints saved)
- **Status**: ⚠️ Incomplete - no checkpoints found

## Evaluating Models

### Interactive Evaluation
```bash
python evaluate.py
```

The script will:
1. **Discover** all model architectures in nemo_experiments/
2. **Find** latest run per architecture by timestamp
3. **Select** best checkpoint from each run
4. **Display** architecture details (type, model name, size)
5. **Let you choose** which architecture to evaluate
6. **Load** the best model
7. **Apply** custom tokenizer if available
8. **Transcribe** validation samples
9. **Compute** WER (Word Error Rate) and CER (Character Error Rate)
10. **Save** results to evaluation_results.json

### Configuration Options

Edit evaluate.py or pass via command line:

```python
# Model selection
model_path: str              # Auto-discovered if None
decoder_type: Optional[str]  # 'ctc' or 'rnnt' for hybrid models

# Data
dataset_manifest: str = "data/manifests/val.json"
batch_size: int = 32

# Metrics
use_cer: bool = False        # Use CER instead of WER
do_lowercase: bool = False   # Normalize text to lowercase
rm_punctuation: bool = False # Remove punctuation before computing metrics

# Streaming specific
att_context_size: Optional[List[int]] = None  # Context window for streaming

# Tolerance
tolerance: Optional[float] = None  # Fail evaluation if metrics exceed this

# Device
cuda: bool = True           # Use GPU if available
```

## Key Insights

### Streaming Detection
- Models with "streaming" in name are streaming models
- Streaming models support `att_context_size` parameter for attention cache
- Check `hparams.yaml` for `att_context_size: [<left>, <right>]` config

### Hybrid Models
- `FastConformer-Hybrid-Transducer-CTC-BPE-Streaming` supports both RNNT and CTC decoding
- Use `decoder_type` parameter to select:
  - `decoder_type='rnnt'` for Transducer decoding (default)
  - `decoder_type='ctc'` for CTC decoding

### Custom Tokenizer
- Located in `tokenizer/quran_tokenizer_bpe_v1024/`
- Automatically applied during evaluation
- BPE vocabulary: 1024 tokens (optimized for Quranic Arabic)

### Metrics Computed
- **WER** (Word Error Rate): % of words that differ from reference
- **CER** (Character Error Rate): % of characters that differ from reference
- Lower is better (0% = perfect match)

## Model Performance Reference

### Official-Seed 50-Epoch Model
- **Run**: 2026-02-16_12-24-08
- **Epochs**: 50
- **Issue**: Overfitting after epoch 43
- **Val WER**: Degraded from 0.2100 → 0.2128
- **Test Accuracy**: 0% (token repetition loops)
- **Recommendation**: ❌ Do not use

### Recommended for Production
1. **FastConformer-Streaming-Custom-Tokenizer** (Latest, good tokenizer)
2. **FastConformer-Hybrid-Transducer-CTC-BPE-Streaming** (Flexible decoders)
3. **FastConformer-Custom-Tokenizer** (Baseline standard model)

## Running Evaluations

### Evaluate Latest Streaming Model
```bash
python evaluate.py
# Then select: FastConformer-Streaming-Custom-Tokenizer (option 3)
```

### Evaluate All Models
Run evaluate.py multiple times, selecting different architectures:
```bash
for arch in 1 2 3 4; do
  echo "$arch" | python evaluate.py
done
```

### Check Results
```bash
cat evaluation_results.json | head -5
```

Output format (JSON lines):
```json
{
  "audio_filepath": "data/audio/val/file.wav",
  "duration": 5.2,
  "text": "النص الأصلي",
  "pred_text": "النص الأصلي"
}
```

Then runs WER/CER comparison and logs to console:
```
================================================================================
EVALUATION RESULTS
================================================================================
Total samples: 3745
WER: 0.1234 (12.34%)
CER: 0.0567 (5.67%)
Output: evaluation_results.json
================================================================================
```

## Folder Navigation Logic

The evaluate.py script implements:

1. **parse_timestamp()**: Parses `2026-02-17_05-27-32` format
2. **find_latest_run()**: Finds max timestamp across:
   - direct folders: `2026-02-05_HH-MM-SS`
   - nested folders: `experiment_name/2026-02-05_HH-MM-SS`
3. **find_best_checkpoint()**: Finds latest .nemo in `checkpoints/`
4. **check_streaming()**: Detects from:
   - `hparams.yaml` config: `att_context_size`, `streaming` keywords
   - Folder name: `streaming` substring

This handles both flat and nested architecture layouts automatically.

## Summary

| Architecture | Type | Status | Latest Run | Model Size |
|---|---|---|---|---|
| FastConformer-Custom-Tokenizer | 📶 Standard | ✅ Ready | 2026-02-14 | 438.0 MB |
| FastConformer-Hybrid-Transducer-CTC-BPE-Streaming | 🌊 Streaming | ✅ Ready | 2026-02-14 | 438.0 MB |
| FastConformer-Streaming-Custom-Tokenizer | 🌊 Streaming | ✅ Ready | 2026-02-15 | 438.5 MB |
| FastConformer-Streaming-Custom-Tokenizer-Official-Seed | 🌊 Streaming | ❌ Issues | 2026-02-16 | 438.4 MB |
| FastConformer-CTC-BPE-Streaming-Quran | 🌊 Streaming | ⚠️ Incomplete | 2026-02-17 | — |
| FastConformer-English-Quran-Tokenizer | 📶 Standard | ⚠️ Incomplete | 2026-02-17 | — |
| FastConformer-Streaming-From-Pretrained | 🌊 Streaming | ⚠️ Incomplete | 2026-02-15 | — |

**Total**: 7 architectures, 4 ready for evaluation, 3 incomplete
