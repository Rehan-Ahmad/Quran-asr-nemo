#!/usr/bin/env bash
set -euo pipefail

# FINE-TUNING ARABIC FASTCONFORMER ON QURAN DATASET
# Uses pretrained Arabic model with diacritics support

WORKSPACE_ROOT=$(pwd)
DATA_DIR="${WORKSPACE_ROOT}/data/manifests"
TOKENIZER_DIR="${WORKSPACE_ROOT}/tokenizer/tokenizer_spe_bpe_v1024_bos_eos"
PRETRAINED_MODEL="${WORKSPACE_ROOT}/pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo"
CONFIG_FILE="${WORKSPACE_ROOT}/nemo_scripts/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml"
PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"
EXP_NAME="FastConformer-Hybrid-Transducer-CTC-BPE-Streaming"

echo "========================================="
echo "QURAN ASR FINE-TUNING"
echo "========================================="
echo "Pretrained: Arabic FastConformer (115M)"
echo "Dataset: Quran recitations (~67k samples)"
echo "Hardware: Multi-GPU with DDP + bf16"
echo "========================================="

# Validate paths
if [ ! -f "$PRETRAINED_MODEL" ]; then
  echo "❌ Pretrained model not found: $PRETRAINED_MODEL"
  exit 1
fi

if [ ! -d "$TOKENIZER_DIR" ]; then
  echo "❌ Tokenizer not found: $TOKENIZER_DIR"
  exit 1
fi

if [ ! -f "${DATA_DIR}/train.json" ]; then
  echo "❌ Training manifest not found: ${DATA_DIR}/train.json"
  exit 1
fi

echo "✓ All required files found"
echo ""

# Fine-tune using NeMo script with YAML config
"$PYTHON_BIN" nemo_scripts/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path="${WORKSPACE_ROOT}/nemo_scripts" \
  --config-name="fastconformer_hybrid_transducer_ctc_bpe_streaming" \
  model.train_ds.manifest_filepath="${DATA_DIR}/train.json" \
  model.validation_ds.manifest_filepath="${DATA_DIR}/val.json" \
  model.test_ds.manifest_filepath="${DATA_DIR}/test.json" \
  model.tokenizer.dir="${TOKENIZER_DIR}" \
  model.tokenizer.type="bpe" \
  ~model.train_ds.is_tarred \
  ~model.train_ds.tarred_audio_filepaths \
  +init_from_pretrained_model="${PRETRAINED_MODEL}" \
  model.optim.name="adamw" \
  model.optim.lr=0.0001 \
  model.optim.betas=[0.9,0.999] \
  model.optim.weight_decay=0.0001 \
  model.optim.sched.name="CosineAnnealing" \
  model.optim.sched.warmup_steps=1000 \
  model.optim.sched.min_lr=1e-6 \
  trainer.devices=-1 \
  trainer.accelerator="gpu" \
  trainer.strategy="ddp" \
  trainer.max_epochs=50 \
  trainer.val_check_interval=1.0 \
  trainer.precision="bf16-mixed" \
  trainer.log_every_n_steps=50 \
  trainer.gradient_clip_val=1.0 \
  exp_manager.exp_dir="${WORKSPACE_ROOT}/nemo_experiments" \
  exp_manager.name="${EXP_NAME}" \
  exp_manager.create_tensorboard_logger=true \
  exp_manager.create_checkpoint_callback=true \
  exp_manager.checkpoint_callback_params.monitor="val_wer" \
  exp_manager.checkpoint_callback_params.mode="min" \
  exp_manager.checkpoint_callback_params.save_top_k=3 \
  exp_manager.checkpoint_callback_params.always_save_nemo=true

echo ""
echo "========================================="
echo "✅ Fine-tuning complete!"
echo "Checkpoints: ${WORKSPACE_ROOT}/nemo_experiments/${EXP_NAME}"
echo "========================================="
