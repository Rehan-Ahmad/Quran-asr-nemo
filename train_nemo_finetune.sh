#!/usr/bin/env bash
set -euo pipefail

# FINE-TUNING ARABIC FASTCONFORMER ON QURAN DATASET
# Uses pretrained Arabic model with diacritics support
# Much faster and better results than training from scratch!

DATA_DIR=${DATA_DIR:-$(pwd)/data/manifests}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-$(pwd)/pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo}
SCRIPT_PATH=$(pwd)/nemo_scripts/speech_to_text_hybrid_rnnt_ctc_bpe.py
CONFIG_DIR=$(pwd)/nemo_scripts
CONFIG_NAME="fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml"
PYTHON_BIN=$(pwd)/.venv/bin/python

echo "========================================="
echo "QURAN ASR FINE-TUNING"
echo "Using Arabic FastConformer (Pretrained)"
echo "========================================="
echo "Base Model: Arabic FastConformer (115M params)"
echo "  - Already trained on 1100h Arabic speech"
echo "  - Includes 390h Quranic recitation (TarteelAI)"
echo "  - Baseline WER on Quran: 6.55%"
echo "  - Supports diacritical marks"
echo ""
echo "Fine-tuning on: Your 67k Quran samples"
echo "Expected: WER < 5% (better than baseline)"
echo "Training time: ~12-24 hours (vs 48-72 from scratch)"
echo "Hardware: 2x RTX 3090 with DDP + bf16"
echo "========================================="

if [ ! -f "$PRETRAINED_MODEL" ]; then
  echo "❌ Pretrained model not found at: $PRETRAINED_MODEL"
  echo "Run: bash download_arabic_model.sh"
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --config-path="$CONFIG_DIR" \
  --config-name="$CONFIG_NAME" \
  \
  model.train_ds.manifest_filepath="$DATA_DIR/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/test.json" \
  \
  model.train_ds.num_workers=8 \
  model.validation_ds.num_workers=8 \
  model.test_ds.num_workers=8 \
  \
  +init_from_pretrained_model="$PRETRAINED_MODEL" \
  \
  trainer.max_epochs=50 \
  trainer.precision=bf16-mixed \
  trainer.devices=2 \
  trainer.accelerator=gpu \
  \
  model.train_ds.batch_size=32 \
  \
  model.optim.lr=0.0001 \
  model.optim.sched.warmup_steps=1000 \
  \
  exp_manager.exp_dir=nemo_experiments \
  exp_manager.name=FastConformer-Arabic-Quran-Finetuned \
  exp_manager.checkpoint_callback_params.monitor=val_wer \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.save_top_k=5 \
  exp_manager.checkpoint_callback_params.always_save_nemo=true
