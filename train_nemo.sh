#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-$(pwd)/data}
CACHE_DIR=${CACHE_DIR:-.nemo_cache}
mkdir -p "$CACHE_DIR"

# NeMo repo URLs
NEMO_REPO="https://raw.githubusercontent.com/NVIDIA/NeMo/main"
SCRIPT_URL="$NEMO_REPO/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py"
CONFIG_URL="$NEMO_REPO/examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml"

# Download script
SCRIPT_PATH="$CACHE_DIR/speech_to_text_hybrid_rnnt_ctc_bpe.py"
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Downloading NeMo training script..."
  curl -fsSL "$SCRIPT_URL" -o "$SCRIPT_PATH"
fi

# Download config
CONFIG_PATH="$CACHE_DIR/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Downloading NeMo training config..."
  curl -fsSL "$CONFIG_URL" -o "$CONFIG_PATH"
fi

python "$SCRIPT_PATH" \
  --config-path="$(dirname "$CONFIG_PATH")" \
  --config-name="$(basename "$CONFIG_PATH")" \
  model.train_ds.manifest_filepath="$DATA_DIR/manifests/train.json" \
  model.validation_ds.manifest_filepath="$DATA_DIR/manifests/val.json" \
  model.test_ds.manifest_filepath="$DATA_DIR/manifests/test.json" \
  model.tokenizer.dir="$DATA_DIR" \
  model.tokenizer.type="char" \
  model.pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_multi" \
  trainer.devices=1 \
  trainer.accelerator=gpu
