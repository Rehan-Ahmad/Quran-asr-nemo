#!/bin/bash
# Fine-tune Quran ASR with streaming (cache-aware) constraints

set -euo pipefail

WORKSPACE_ROOT=$(pwd)
DATA_DIR="${WORKSPACE_ROOT}/data/manifests"
TOKENIZER_DIR="${WORKSPACE_ROOT}/tokenizer/quran_tokenizer_bpe_v1024"
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"${WORKSPACE_ROOT}/pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo"}
PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"
EXP_NAME="FastConformer-Streaming-Custom-Tokenizer"

echo "========================================="
echo "Quran ASR Streaming Fine-tuning"
echo "========================================="
echo "Pretrained: Arabic FastConformer (115M)"
echo "Pretrained model path: $PRETRAINED_MODEL"
echo "Tokenizer: Quran custom BPE"
echo "Streaming: cache-aware (chunked_limited)"
echo "========================================="

# Validate paths
if [ ! -f "$PRETRAINED_MODEL" ]; then
  echo "❌ Pretrained model not found: $PRETRAINED_MODEL"
  exit 1
fi

if [ ! -f "${DATA_DIR}/train.json" ]; then
  echo "❌ Training manifest not found: ${DATA_DIR}/train.json"
  exit 1
fi

if [ ! -f "$TOKENIZER_DIR/tokenizer.model" ]; then
  echo "❌ Tokenizer not found: $TOKENIZER_DIR"
  exit 1
fi

echo "✓ All required files found"

echo ""
echo "Starting streaming fine-tuning..."

echo "-----------------------------------------"
"$PYTHON_BIN" finetune_with_custom_tokenizer.py \
  init_from_nemo_model="$PRETRAINED_MODEL" \
  model.tokenizer.dir="$TOKENIZER_DIR" \
  model.train_ds.manifest_filepath="$DATA_DIR/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/test.json" \
  model.preprocessor.normalize=NA \
  model.encoder.causal_downsampling=true \
  model.encoder.att_context_style=chunked_limited \
  model.encoder.att_context_size='[140,27]' \
  model.encoder.conv_context_size=causal \
  exp_manager.exp_dir="$WORKSPACE_ROOT/nemo_experiments" \
  exp_manager.name="$EXP_NAME"

echo ""
echo "========================================="
echo "Streaming fine-tuning complete!"
echo "========================================="
echo "Experiment: nemo_experiments/$EXP_NAME"
echo "TensorBoard: python launch_tensorboard.py"
echo "Evaluate: python evaluate_model.py"
echo "========================================="
