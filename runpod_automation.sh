#!/bin/bash
# RunPod end-to-end automation (uv-based) for Quran ASR fine-tuning

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKDIR=${RUNPOD_WORKDIR:-$SCRIPT_DIR}

PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}

HF_DATASET=${HF_DATASET:-hifzyml/quran_dataset_v0}
HF_MODEL_REPO=${HF_MODEL_REPO:-nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0}
HF_MODEL_FILENAME=${HF_MODEL_FILENAME:-stt_ar_fastconformer_hybrid_large_pcd.nemo}

EXP_NAME=${EXP_NAME:-FastConformer-Custom-Tokenizer}
TB_PORT=${TB_PORT:-6006}
RUN_TENSORBOARD=${RUN_TENSORBOARD:-1}

PYTHON_BIN="$WORKDIR/.venv/bin/python"

cd "$WORKDIR"

echo "========================================="
echo "RunPod automation: Quran ASR (uv)"
echo "========================================="
echo "Workdir: $WORKDIR"
echo "Python: $PYTHON_VERSION"
echo "Dataset: $HF_DATASET"
echo "Model repo: $HF_MODEL_REPO"
echo "Model file: $HF_MODEL_FILENAME"
echo "Experiment: $EXP_NAME"
echo "TensorBoard port: $TB_PORT"
echo "========================================="

# 1) Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Create venv (if missing)
if [ ! -f "$WORKDIR/.venv/bin/python" ]; then
  echo "Creating uv venv..."
  uv venv .venv --python "$PYTHON_VERSION"
fi

# 3) Install dependencies
echo "Installing dependencies via uv..."
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
uv pip install -r requirements.txt
uv pip install huggingface_hub

# 4) Download pretrained model from HF
mkdir -p "$WORKDIR/pretrained_models"
if [ ! -f "$WORKDIR/pretrained_models/$HF_MODEL_FILENAME" ]; then
  echo "Downloading pretrained .nemo from Hugging Face..."
  "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ.get("HF_MODEL_REPO")
filename = os.environ.get("HF_MODEL_FILENAME")
token = os.environ.get("HF_TOKEN")

path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir="pretrained_models",
    local_dir_use_symlinks=False,
    token=token,
)
print(f"Downloaded: {path}")
PY
else
  echo "Pretrained model already present: pretrained_models/$HF_MODEL_FILENAME"
fi

# 5) Prepare dataset (HF -> local manifests/audio)
if [ ! -f "$WORKDIR/data/manifests/train.json" ]; then
  echo "Preparing dataset..."
  "$PYTHON_BIN" prepare_dataset.py \
    --dataset_name "$HF_DATASET" \
    --output_dir data \
    --copy_audio
else
  echo "Dataset manifests already present. Skipping prepare_dataset.py"
fi

# 6) Train custom tokenizer (if missing)
if [ ! -f "$WORKDIR/tokenizer/quran_tokenizer_bpe_v1024/tokenizer.model" ]; then
  echo "Training custom tokenizer..."
  "$PYTHON_BIN" train_custom_tokenizer.py
else
  echo "Custom tokenizer already present. Skipping training."
fi

# 7) Fine-tune model with custom tokenizer
export EVAL_MODEL_NAME="$EXP_NAME"
export HF_MODEL_REPO
export HF_MODEL_FILENAME

# Ensure pretrained model path matches script expectations
if [ -f "$WORKDIR/pretrained_models/$HF_MODEL_FILENAME" ]; then
  ln -sf "$WORKDIR/pretrained_models/$HF_MODEL_FILENAME" "$WORKDIR/pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo"
fi

echo "Starting fine-tuning..."
./train_with_custom_tokenizer.sh

# 8) Evaluate best checkpoint (non-interactive)
echo "Running evaluation..."
EVAL_MODEL_NAME="$EXP_NAME" "$PYTHON_BIN" evaluate_model.py

# 9) Launch TensorBoard (optional)
if [ "$RUN_TENSORBOARD" = "1" ]; then
  echo "Launching TensorBoard on port $TB_PORT..."
  nohup "$PYTHON_BIN" launch_tensorboard.py --logdir "nemo_experiments/$EXP_NAME" --port "$TB_PORT" > tensorboard.log 2>&1 &
  echo "TensorBoard PID: $! (log: tensorboard.log)"
fi

echo "========================================="
echo "RunPod automation complete"
echo "========================================="
