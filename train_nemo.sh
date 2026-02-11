#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-$(pwd)/data}
TOKENIZER_DIR=${TOKENIZER_DIR:-$(pwd)/tokenizer/tokenizer_spe_bpe_v1024_bos_eos}
SCRIPT_PATH=$(pwd)/nemo_scripts/speech_to_text_hybrid_rnnt_ctc_bpe.py
CONFIG_PATH=$(pwd)/nemo_scripts/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml

python "$SCRIPT_PATH" \
  --config-path="$(dirname "$CONFIG_PATH")" \
  --config-name="$(basename "$CONFIG_PATH")" \
  model.train_ds.manifest_filepath="$DATA_DIR/manifests/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/manifests/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/manifests/test.json" \
  model.tokenizer.dir="$TOKENIZER_DIR" \
  model.tokenizer.type="bpe" \
  model.pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_multi" \
  trainer.devices=1 \
  trainer.accelerator=gpu

