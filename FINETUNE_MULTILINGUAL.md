# Fine-tune Multilingual FastConformer on Quran Data

Fine-tune the official **stt_en_fastconformer_hybrid_large_streaming_multi.nemo** model on your Quran ASR dataset using the official NeMo streaming configuration.

## Overview

| Property | Value |
|----------|-------|
| **Base Model** | stt_en_fastconformer_hybrid_large_streaming_multi.nemo |
| **Architecture** | FastConformer Hybrid RNNT/CTC |
| **Parameters** | ~115M |
| **Config** | Official NeMo cache-aware streaming |
| **Tokenizer** | Custom BPE (1024 vocab, vocab.txt) |
| **Dataset** | Quran ASR (Arabic speech) |
| **Streaming** | Enabled (att_context_size: [70, 13], ~1.04s latency) |

## Quick Start

### 1. Verify Files
```bash
# Check model exists
ls -lh pretrained_models/stt_en_fastconformer_hybrid_large_streaming_multi.nemo

# Check datasets
ls data/manifests/{train,val,test}.json
ls data/vocab.txt
```

### 2. Run Training

**Option A: Shell Script (Recommended)**
```bash
./train_multilingual_streaming.sh
```

**Option B: Direct Python (with Hydra)**
```bash
python finetune_multilingual_streaming_hydra.py
```

**Option C: Direct Python (simple)**
```bash
python finetune_multilingual_streaming.py
```

## Configuration Details

### Model Architecture
- **Encoder**: ConformerEncoder (17 layers, 512 hidden, 8 heads)
- **Streaming**: Cache-aware with att_context_size [70, 13]
  - Left context: 70 frames (640ms)
  - Right context (lookahead): 13 frames (104ms)
  - Total latency: ~1.04s
- **Subsampling**: 8x (via dw_striding)
- **Loss**: CTC (for hybrid model training)

### Training Configuration
- **Optimizer**: AdamW (lr=2.0)
- **Scheduler**: Noam annealing (warmup: 10k steps, min_lr: 1e-6)
- **Batch size**: 16
- **Max epochs**: 100
- **Precision**: fp32
- **Devices**: All available GPUs (DDP strategy)

### Data Paths (from config)
- Train: `data/manifests/train.json`
- Val: `data/manifests/val.json`
- Test: `data/manifests/test.json`
- Tokenizer: `data/vocab.txt`

## Monitoring & Evaluation

### View Training Progress
```bash
tensorboard --logdir=nemo_experiments/FastConformer-CTC-BPE-Streaming-Quran
```

### Evaluate Best Checkpoint
```bash
python evaluate_model.py --nemo-model nemo_experiments/FastConformer-CTC-BPE-Streaming-Quran/*/checkpoints/best.nemo
```

### View Logs
```bash
# Full training log
tail -f nemo_experiments/FastConformer-CTC-BPE-Streaming-Quran/*/nemo_log_*.txt

# Error log (if any)
cat nemo_experiments/FastConformer-CTC-BPE-Streaming-Quran/*/nemo_error_log.txt
```

## Key Differences from Arabic Streaming Model

| Aspect | Arabic Model | Multilingual Model |
|--------|-------------|-------------------|
| **Base** | stt_ar_fastconformer_hybrid_large_pcd.nemo | stt_en_fastconformer_hybrid_large_streaming_multi.nemo |
| **Languages** | Arabic only | Multi-language (English, Spanish, German, etc.) |
| **Tokenizer** | Custom Quran BPE | Pre-trained multilingual |
| **Fine-tune Strategy** | Convert → stream → fine-tune | Direct fine-tune on streaming |

## Files Created

1. **nemo_scripts/fastconformer_ctc_bpe_streaming_quran.yaml**
   - Configuration file (official NeMo config adapted for Quran data)
   - Paths to your manifests and tokenizer

2. **finetune_multilingual_streaming.py**
   - Simple Python fine-tuning script
   - Direct model loading and training

3. **finetune_multilingual_streaming_hydra.py**
   - Hydra-based fine-tuning (NeMo standard approach)
   - Recommended for production use

4. **train_multilingual_streaming.sh**
   - Shell script wrapper for validation and execution

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size in config:
```yaml
model:
  train_ds:
    batch_size: 8  # or 4 for smaller GPUs
```

### Slow Training
- Increase num_workers: `num_workers: 16`
- Enable gradient accumulation: `trainer.accumulate_grad_batches: 2`
- Use mixed precision: `trainer.precision: 16` (faster but less stable)

### Model Not Loading
Check absolute paths in config are correct and all files exist:
```bash
python -c "
from pathlib import Path
from omegaconf import OmegaConf

config = OmegaConf.load('nemo_scripts/fastconformer_ctc_bpe_streaming_quran.yaml')
print('Train:', config.model.train_ds.manifest_filepath)
print('Val:', config.model.validation_ds.manifest_filepath)
print('Tokenizer:', config.model.tokenizer.dir)
"
```

## References

- [Official Config](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_ctc_bpe_streaming.yaml)
- [NeMo FastConformer Docs](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#fast-conformer)
- [Cache-Aware Streaming](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#cache-aware-streaming-conformer)

## Next Steps

After fine-tuning:
1. Evaluate using `evaluate_model.py` (auto-detects streaming config)
2. Compare WER with baseline
3. Export to TorchScript or ONNX if needed
4. Deploy to production
