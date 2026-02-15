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
echo "Tokenizer: Quran custom BPE (1024 tokens)"
echo "Streaming: cache-aware (chunked_limited)"
echo "Attention context: [70, 13] frames"
echo "Look-ahead latency: ~1040ms (right_ctx × subsampling × stride)"
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
echo "Step 1: Converting pretrained model to streaming..."
echo "-----------------------------------------"

STREAMING_MODEL="pretrained_models/stt_ar_fastconformer_hybrid_large_streaming.nemo"

if [ -f "$STREAMING_MODEL" ]; then
  echo "✓ Streaming model already exists: $STREAMING_MODEL"
  echo "  (delete it to regenerate with updated tokenizer)"
else
  echo "Creating streaming variant from pretrained model..."
  "$PYTHON_BIN" convert_model_to_streaming.py \
    --pretrained-model=pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo \
    --output-model="$STREAMING_MODEL" \
    --custom-tokenizer="$TOKENIZER_DIR"
fi

echo ""
echo "Step 2: Fine-tuning with streaming config and custom tokenizer..."
echo "-----------------------------------------"

# Use streaming config file for training with the streaming-enabled model
# Note: Use +init_from_nemo_model to add it to the config
"$PYTHON_BIN" finetune_with_custom_tokenizer.py \
  --config-path=nemo_scripts \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe_streaming \
  '+init_from_nemo_model=pretrained_models/stt_ar_fastconformer_hybrid_large_streaming.nemo' \
  model.tokenizer.dir="$TOKENIZER_DIR" \
  model.train_ds.manifest_filepath="data/manifests/train.json" \
  model.validation_ds.manifest_filepath="data/manifests/val.json" \
  model.test_ds.manifest_filepath="data/manifests/test.json" \
  exp_manager.exp_dir="nemo_experiments" \
  exp_manager.name="$EXP_NAME"

echo ""
echo "========================================="
echo "Streaming fine-tuning complete!"
echo "========================================="
echo "Experiment: nemo_experiments/$EXP_NAME"
echo "TensorBoard: python launch_tensorboard.py"
echo "Evaluate: python evaluate_model.py"
echo "========================================="
