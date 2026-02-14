#!/bin/bash
# Train custom tokenizer and fine-tune model with it

set -euo pipefail

WORKSPACE_ROOT=$(pwd)
DATA_DIR="${WORKSPACE_ROOT}/data/manifests"
TOKENIZER_DIR="${WORKSPACE_ROOT}/tokenizer/quran_tokenizer_bpe_v1024"
PRETRAINED_MODEL="${WORKSPACE_ROOT}/pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo"
PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"
EXP_NAME="FastConformer-Custom-Tokenizer"

echo "========================================="
echo "Quran ASR Fine-tuning with Custom Tokenizer"
echo "========================================="
echo "Pretrained: Arabic FastConformer (115M)"
echo "Custom Tokenizer: Trained on Quran corpus"
echo "Dataset: Quran recitations (~67k samples)"
echo "Hardware: Multi-GPU with DDP + bf16"
echo "========================================="

# Step 1: Train custom tokenizer (if not already trained)
if [ ! -d "$TOKENIZER_DIR" ] || [ ! -f "$TOKENIZER_DIR/tokenizer.model" ]; then
  echo ""
  echo "Step 1: Training custom tokenizer..."
  echo "-----------------------------------------"
  "$PYTHON_BIN" train_custom_tokenizer.py
  
  if [ ! -f "$TOKENIZER_DIR/tokenizer.model" ]; then
    echo "❌ Tokenizer training failed"
    exit 1
  fi
  
  echo "✓ Custom tokenizer trained successfully"
else
  echo ""
  echo "Step 1: Custom tokenizer already exists"
  echo "-----------------------------------------"
  echo "✓ Using existing tokenizer: $TOKENIZER_DIR"
fi

# Step 2: Validate paths
echo ""
echo "Step 2: Validating paths..."
echo "-----------------------------------------"

if [ ! -f "$PRETRAINED_MODEL" ]; then
  echo "❌ Pretrained model not found: $PRETRAINED_MODEL"
  exit 1
fi

if [ ! -f "${DATA_DIR}/train.json" ]; then
  echo "❌ Training manifest not found: ${DATA_DIR}/train.json"
  exit 1
fi

echo "✓ All required files found"

# Step 3: Fine-tune with custom tokenizer
echo ""
echo "Step 3: Fine-tuning model with custom tokenizer..."
echo "-----------------------------------------"

"$PYTHON_BIN" finetune_with_custom_tokenizer.py \
  init_from_nemo_model="$PRETRAINED_MODEL" \
  model.tokenizer.dir="$TOKENIZER_DIR" \
  model.train_ds.manifest_filepath="$DATA_DIR/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/test.json" \
  exp_manager.exp_dir="$WORKSPACE_ROOT/nemo_experiments" \
  exp_manager.name="$EXP_NAME"

echo ""
echo "========================================="
echo "Training complete!"
echo "========================================="
echo "Experiment: nemo_experiments/$EXP_NAME"
echo "TensorBoard: python launch_tensorboard.py"
echo "Evaluate: python evaluate_model.py"
echo "========================================="
