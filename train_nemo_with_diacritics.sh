#!/usr/bin/env bash
set -euo pipefail

# OPTION B: TRAINING WITH DIACRITICS (PROPER FOR QURAN)
# Keeps: Full Quranic text with all diacritical marks
# Uses: Original manifests + regenerated tokenizer with BOS/EOS
# Training: From scratch to avoid encoder/decoder mismatch

DATA_DIR=${DATA_DIR:-$(pwd)/data/manifests}
TOKENIZER_DIR=${TOKENIZER_DIR:-$(pwd)/tokenizer/tokenizer_spe_bpe_v1024_diacritics_bos_eos}
SCRIPT_PATH=$(pwd)/nemo_scripts/speech_to_text_hybrid_rnnt_ctc_bpe.py
CONFIG_DIR=$(pwd)/nemo_scripts
CONFIG_NAME="fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml"
PYTHON_BIN=$(pwd)/.venv/bin/python

echo "========================================="
echo "PROPER QURANIC ASR TRAINING"
echo "With Full Diacritics (as required)"
echo "========================================="
echo "Data: Original with diacritics (لَيْلَةُ ٱلْقَدْرِ)"
echo "Tokenizer: Regenerated on diacritical text + BOS/EOS"
echo "Mode: Training from scratch"
echo "Expected: Proper Quranic reading with pronunciation"
echo "Dataset: 67k train, 3.7k val, 3.7k test"
echo "Hardware: 2x RTX 3090 with DDP + bf16"
echo "========================================="

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --config-path="$CONFIG_DIR" \
  --config-name="$CONFIG_NAME" \
  model.train_ds.manifest_filepath="$DATA_DIR/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/test.json" \
  \
  model.tokenizer.dir="$TOKENIZER_DIR" \
  \
  model.train_ds.num_workers=8 \
  model.validation_ds.num_workers=8 \
  model.test_ds.num_workers=8 \
  \
  trainer.max_epochs=120 \
  trainer.precision=bf16-mixed \
  trainer.devices=2 \
  trainer.accelerator=gpu \
  \
  model.train_ds.batch_size=32 \
  \
  model.optim.lr=1.0 \
  model.optim.sched.warmup_steps=3000 \
  \
  model.ctc_loss_weight=0.3 \
  model.rnnt.decoding.strategy=greedy \
  model.rnnt.decoding.greedy.max_symbols_per_step=30 \
  \
  exp_manager.exp_dir=nemo_experiments \
  exp_manager.name=FastConformer-Hybrid-Transducer-CTC-BPE-Streaming-With-Diacritics \
  exp_manager.checkpoint_callback_params.monitor=val_wer \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.save_top_k=10 \
  exp_manager.checkpoint_callback_params.always_save_nemo=true
