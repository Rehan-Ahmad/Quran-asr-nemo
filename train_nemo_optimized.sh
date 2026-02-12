#!/usr/bin/env bash
set -euo pipefail

# OPTIMIZED TRAINING WITH PRETRAINED INITIALIZATION
# Combines transfer learning + optimized hyperparameters for best results
# Based on dataset: 67k train, 3.7k val, 3.7k test samples
# Target: WER < 0.15 (15% error rate) in 100 epochs

DATA_DIR=${DATA_DIR:-$(pwd)/data}
TOKENIZER_DIR=${TOKENIZER_DIR:-$(pwd)/tokenizer/tokenizer_spe_bpe_v1024_bos_eos}
SCRIPT_PATH=$(pwd)/nemo_scripts/speech_to_text_hybrid_rnnt_ctc_bpe.py
CONFIG_DIR=$(pwd)/nemo_scripts
CONFIG_NAME="fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml"

# Pretrained model for transfer learning (faster convergence, better results)
# Comment out to train from scratch (not recommended - takes 200+ epochs)
PRETRAINED_MODEL="stt_en_fastconformer_transducer_large"

# Download pretrained model if needed
if [ -n "${PRETRAINED_MODEL:-}" ]; then
  PRETRAINED_PATH="./pretrained_models/${PRETRAINED_MODEL}.nemo"
  if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Downloading pretrained model: $PRETRAINED_MODEL"
    mkdir -p pretrained_models
    conda run -n quranASR python -c "
from nemo.collections.asr.models import EncDecRNNTBPEModel
import os
model = EncDecRNNTBPEModel.from_pretrained('${PRETRAINED_MODEL}')
os.makedirs('pretrained_models', exist_ok=True)
model.save_to('${PRETRAINED_PATH}')
print('Pretrained model downloaded to: ${PRETRAINED_PATH}')
"
  else
    echo "Using cached pretrained model: $PRETRAINED_PATH"
  fi
fi

echo "========================================="
echo "OPTIMIZED TRAINING CONFIGURATION"
echo "========================================="
if [ -n "${PRETRAINED_MODEL:-}" ]; then
  echo "Mode: Transfer Learning (RECOMMENDED)"
  echo "Pretrained: $PRETRAINED_MODEL"
  echo "Expected: WER < 0.2 in 50 epochs, < 0.15 in 100 epochs"
else
  echo "Mode: Training from scratch"
  echo "Expected: WER < 0.3 in 100 epochs (slower convergence)"
fi
echo "Dataset: 67k train, 3.7k val, 3.7k test samples"
echo "Hardware: 2x RTX 3090 with DDP + mixed precision"
echo "========================================="

python "$SCRIPT_PATH" \
  --config-path="$CONFIG_DIR" \
  --config-name="$CONFIG_NAME" \
  model.train_ds.manifest_filepath="$DATA_DIR/manifests/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/manifests/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/manifests/test.json" \
  \
  `# ========== PRETRAINED INITIALIZATION (Transfer Learning) ========== ` \
  ${PRETRAINED_PATH:++init_from_pretrained_model="$PRETRAINED_PATH"} \
  \
  `# ========== DATA LOADING OPTIMIZED ========== ` \
  model.train_ds.num_workers=8 \
  model.validation_ds.num_workers=8 \
  model.test_ds.num_workers=8 \
  model.train_ds.batch_size=32 \
  model.validation_ds.batch_size=32 \
  model.test_ds.batch_size=32 \
  model.train_ds.pin_memory=true \
  model.validation_ds.pin_memory=true \
  model.test_ds.pin_memory=true \
  \
  `# ========== CRITICAL: INCREASE TRAINING DURATION ========== ` \
  trainer.max_epochs=100 \
  trainer.val_check_interval=1.0 \
  trainer.check_val_every_n_epoch=1 \
  \
  `# ========== OPTIMIZER: TUNED FOR FINE-TUNING ========== ` \
  model.optim.name=adamw \
  model.optim.lr=1.5 \
  model.optim.betas=[0.9,0.98] \
  model.optim.weight_decay=1e-3 \
  model.optim.sched.name=NoamAnnealing \
  model.optim.sched.warmup_steps=3000 \
  model.optim.sched.min_lr=1e-6 \
  \
  `# ========== GRADIENT ACCUMULATION FOR EFFECTIVE BATCH SIZE ========== ` \
  trainer.accumulate_grad_batches=2 \
  trainer.gradient_clip_val=1.0 \
  \
  `# ========== PRECISION: USE MIXED PRECISION ========== ` \
  trainer.precision=bf16-mixed \
  \
  `# ========== AUGMENTATION: ENABLE SPEC AUGMENT ========== ` \
  model.spec_augment.freq_masks=2 \
  model.spec_augment.time_masks=10 \
  model.spec_augment.freq_width=27 \
  model.spec_augment.time_width=0.05 \
  \
  `# ========== REGULARIZATION ========== ` \
  model.encoder.dropout=0.1 \
  model.encoder.dropout_emb=0.1 \
  model.encoder.dropout_att=0.1 \
  model.decoder.prednet.dropout=0.2 \
  model.joint.jointnet.dropout=0.2 \
  \
  `# ========== LOSS CONFIGURATION ========== ` \
  model.aux_ctc.ctc_loss_weight=0.3 \
  model.loss.warprnnt_numba_kwargs.fastemit_lambda=0.001 \
  model.joint.fused_batch_size=8 \
  \
  `# ========== DECODING SETTINGS ========== ` \
  +model.decoding.greedy.use_cuda_graph_decoder=false \
  model.aux_ctc.decoding.strategy=greedy_batch \
  model.decoding.strategy=greedy_batch \
  \
  `# ========== LOGGING ========== ` \
  model.log_prediction=false \
  trainer.log_every_n_steps=10 \
  \
  `# ========== HARDWARE SETTINGS ========== ` \
  trainer.devices=2 \
  trainer.accelerator=gpu \
  trainer.strategy=ddp \
  \
  `# ========== CHECKPOINT SETTINGS ========== ` \
  exp_manager.checkpoint_callback_params.save_top_k=10 \
  exp_manager.checkpoint_callback_params.monitor=val_wer \
  exp_manager.checkpoint_callback_params.mode=min \
  \
  `# ========== TOKENIZER ========== ` \
  model.tokenizer.dir="$TOKENIZER_DIR" \
  model.tokenizer.type="bpe"

