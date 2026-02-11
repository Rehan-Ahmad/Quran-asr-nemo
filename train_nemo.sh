#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-$(pwd)/data}
TOKENIZER_DIR=${TOKENIZER_DIR:-$(pwd)/tokenizer/tokenizer_spe_bpe_v1024_bos_eos}
SCRIPT_PATH=$(pwd)/nemo_scripts/speech_to_text_hybrid_rnnt_ctc_bpe.py
CONFIG_DIR=$(pwd)/nemo_scripts
CONFIG_NAME="fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml"

python "$SCRIPT_PATH" \
  --config-path="$CONFIG_DIR" \
  --config-name="$CONFIG_NAME" \
  model.train_ds.manifest_filepath="$DATA_DIR/manifests/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/manifests/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/manifests/test.json" \
  model.train_ds.num_workers=0 \
  model.validation_ds.num_workers=0 \
  model.test_ds.num_workers=0 \
  model.train_ds.batch_size=8 \
  model.validation_ds.batch_size=8 \
  model.test_ds.batch_size=8 \
  model.tokenizer.dir="$TOKENIZER_DIR" \
  model.tokenizer.type="bpe" \
  trainer.devices=2 \
  trainer.accelerator=gpu \
  trainer.strategy=ddp \
  trainer.max_epochs=10 \
  trainer.log_every_n_steps=5

