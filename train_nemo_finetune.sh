#!/usr/bin/env bash
set -euo pipefail

# FINE-TUNING ARABIC FASTCONFORMER ON QURAN DATASET
# Uses pretrained Arabic model with diacritics support

WORKSPACE_ROOT=$(pwd)
DATA_DIR="${WORKSPACE_ROOT}/data/manifests"
PRETRAINED_MODEL="${WORKSPACE_ROOT}/pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo"
PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"
EXP_NAME="FastConformer-Hybrid-Transducer-CTC-BPE-Streaming"

echo "========================================="
echo "QURAN ASR FINE-TUNING"
echo "========================================="
echo "Pretrained: Arabic FastConformer (115M)"
echo "Dataset: Quran recitations (~67k samples)"
echo "Hardware: Multi-GPU with DDP + bf16"
echo "Note: Using tokenizer from pretrained model"
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

echo "✓ All required files found"
echo ""

# Run fine-tuning with Hydra overrides
"$PYTHON_BIN" finetune_quran_asr.py \
  init_from_nemo_model="$PRETRAINED_MODEL" \
  model.train_ds.manifest_filepath="$DATA_DIR/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/test.json" \
  exp_manager.exp_dir="$WORKSPACE_ROOT/nemo_experiments" \
  exp_manager.name="$EXP_NAME"

echo ""
echo "========================================="
echo "✅ Fine-tuning complete!"
echo "Checkpoints: ${WORKSPACE_ROOT}/nemo_experiments/${EXP_NAME}"
echo "========================================="
