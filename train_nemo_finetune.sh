#!/usr/bin/env bash
set -euo pipefail

# FINE-TUNING ARABIC FASTCONFORMER ON QURAN DATASET
# Uses pretrained Arabic model with diacritics support

WORKSPACE_ROOT=$(pwd)
DATA_DIR="${WORKSPACE_ROOT}/data/manifests"
PRETRAINED_MODEL="https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
PRETRAINED_LOCAL_DIR="${WORKSPACE_ROOT}/pretrained_models"
PRETRAINED_LOCAL_PATH="${PRETRAINED_LOCAL_DIR}/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
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

# Validate paths: if PRETRAINED_MODEL is a URL, we'll download it later
if [[ "$PRETRAINED_MODEL" != http* ]]; then
  if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "❌ Pretrained model not found: $PRETRAINED_MODEL"
    exit 1
  fi
fi

if [ ! -f "${DATA_DIR}/train.json" ]; then
  echo "❌ Training manifest not found: ${DATA_DIR}/train.json"
  exit 1
fi

echo "✓ All required files found"
echo ""

# Run fine-tuning with Hydra overrides
# If PRETRAINED_MODEL is a URL (huggingface repo), attempt to download the .nemo artifact
if [[ "$PRETRAINED_MODEL" == http* ]]; then
  mkdir -p "$PRETRAINED_LOCAL_DIR"
  echo "Downloading pretrained model from Hugging Face repo: $PRETRAINED_MODEL"
  # Ensure huggingface_hub is available in the venv
  "$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null || "$PYTHON_BIN" -m pip install --upgrade huggingface_hub
  "$PYTHON_BIN" - <<PY
from huggingface_hub import HfApi, hf_hub_download
import sys
repo = "$PRETRAINED_MODEL".split('huggingface.co/')[-1]
api = HfApi()
files = api.list_repo_files(repo)
nemo_files = [f for f in files if f.endswith('.nemo')]
if not nemo_files:
    print('No .nemo file found in repo:', repo)
    sys.exit(2)
fname = nemo_files[0]
print('Found .nemo file:', fname)
path = hf_hub_download(repo_id=repo, filename=fname, cache_dir='${PRETRAINED_LOCAL_DIR}')
print('Downloaded to', path)
print(path)
PY
  # Set local path to the downloaded file if present
  if [ -f "$PRETRAINED_LOCAL_PATH" ]; then
    PRETRAINED_MODEL="$PRETRAINED_LOCAL_PATH"
  else
    # try to discover the downloaded .nemo file in the local dir
    dl=$(ls ${PRETRAINED_LOCAL_DIR}/*.nemo 2>/dev/null | head -n 1 || true)
    if [ -n "$dl" ]; then
      PRETRAINED_MODEL="$dl"
    else
      echo "❌ Failed to download .nemo from $PRETRAINED_MODEL; please install huggingface_hub and retry or provide a local .nemo file"
      exit 1
    fi
  fi
fi

# Export selected model path to be used by the Python fine-tune entrypoint
export NEMO_MODEL="$PRETRAINED_MODEL"

"$PYTHON_BIN" finetune_multilingual_simple.py \
  
  # The script reads manifest and tokenizer directories from env vars or defaults
  

echo ""
echo "========================================="
echo "✅ Fine-tuning complete!"
echo "Checkpoints: ${WORKSPACE_ROOT}/nemo_experiments/${EXP_NAME}"
echo "========================================="
