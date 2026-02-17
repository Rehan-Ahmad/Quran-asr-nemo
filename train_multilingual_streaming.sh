#!/bin/bash
# Fine-tune multilingual English FastConformer streaming model on Quran data
# Uses official NeMo config: https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_ctc_bpe_streaming.yaml

set -euo pipefail

WORKSPACE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========================================="
echo "Multilingual FastConformer Fine-tuning"
echo "========================================="
echo "Model:     stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
echo "Config:    Official NeMo streaming config (cache-aware)"
echo "Dataset:   Quran ASR (Arabic speech)"
echo "Tokenizer: Custom BPE (1024 vocab) - vocab.txt"
echo "Streaming: Enabled (att_context_size: [70, 13], latency: ~1.04s)"
echo "========================================="

# Validate paths
MODEL_PATH="${WORKSPACE_ROOT}/pretrained_models/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
TRAIN_MANIFEST="${WORKSPACE_ROOT}/data/manifests/train.json"
VAL_MANIFEST="${WORKSPACE_ROOT}/data/manifests/val.json"
VOCAB_FILE="${WORKSPACE_ROOT}/data/vocab.txt"

echo ""
echo "Validating required files..."
for file in "$MODEL_PATH" "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$VOCAB_FILE"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing: $file"
    exit 1
  fi
  echo "✓ $(basename "$file")"
done

echo ""
echo "Running fine-tuning..."
echo "Log: nemo_experiments/FastConformer-CTC-BPE-Streaming-Quran/*/nemo_log_*.txt"
echo "TensorBoard: tensorboard --logdir nemo_experiments"
echo ""

# Run training
cd "$WORKSPACE_ROOT"
$PYTHON_BIN finetune_multilingual_streaming.py

echo ""
echo "========================================="
echo "Fine-tuning complete!"
echo "========================================="
echo "View results:"
echo "  tensorboard --logdir=nemo_experiments/FastConformer-CTC-BPE-Streaming-Quran"
echo "Evaluate best model:"
echo "  python evaluate_model.py --nemo-model nemo_experiments/FastConformer-CTC-BPE-Streaming-Quran/*/checkpoints/best.nemo"
echo "========================================="
