#!/usr/bin/env bash
set -euo pipefail

# FINE-TUNING ARABIC FASTCONFORMER ON QURAN DATASET
# Uses pretrained Arabic model with diacritics support
# Much faster and better results than training from scratch!

DATA_DIR=${DATA_DIR:-$(pwd)/data/manifests}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-$(pwd)/pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo}
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

"$PYTHON_BIN" << 'FINETUNE'
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Load pretrained Arabic model
print("Loading pretrained Arabic FastConformer model...")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    "${PRETRAINED_MODEL}"
)

# Update dataset paths
model.setup_training_data(train_data_config={
    'manifest_filepath': '${DATA_DIR}/train.json',
    'sample_rate': 16000,
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 8,
})

model.setup_validation_data(val_data_config={
    'manifest_filepath': '${DATA_DIR}/val.json',
    'sample_rate': 16000,
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 8,
})

# Configure optimizer for fine-tuning (lower LR)
model.cfg.optim.lr = 0.0001
model.cfg.optim.weight_decay = 0.001
model.cfg.optim.sched.warmup_steps = 1000

# Setup trainer
trainer = pl.Trainer(
    accelerator='gpu',
    devices=2,
    strategy='ddp',
    max_epochs=50,
    precision='bf16-mixed',
    log_every_n_steps=50,
    val_check_interval=0.5,
    default_root_dir='nemo_experiments/FastConformer-Arabic-Quran-Finetuned',
)

# Fine-tune
print("Starting fine-tuning...")
trainer.fit(model)

# Save fine-tuned model
output_path = 'nemo_experiments/quran_finetuned_model.nemo'
model.save_to(output_path)
print(f"✅ Fine-tuned model saved to: {output_path}")
FINETUNE
