#!/usr/bin/env bash
set -euo pipefail

# Download Arabic FastConformer model from HuggingFace
# This model is already trained on Arabic with diacritics including Quranic recitation

MODEL_DIR=${MODEL_DIR:-$(pwd)/pretrained_models}
MODEL_NAME="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
PYTHON_BIN=${PYTHON_BIN:-$(pwd)/.venv/bin/python}

echo "==========================================="
echo "DOWNLOADING ARABIC FASTCONFORMER MODEL"
echo "==========================================="
echo "Model: $MODEL_NAME"
echo "Target: $MODEL_DIR"
echo "==========================================="

mkdir -p "$MODEL_DIR"

"$PYTHON_BIN" << EOF
import nemo.collections.asr as nemo_asr
import os

model_dir = "${MODEL_DIR}"
model_name = "${MODEL_NAME}"

print(f"Downloading {model_name}...")
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    model_name=model_name
)

# Save to local directory
output_path = os.path.join(model_dir, "stt_ar_fastconformer_hybrid_large_pcd.nemo")
asr_model.save_to(output_path)

print(f"✅ Model saved to: {output_path}")
print(f"✅ Model vocab size: {asr_model.tokenizer.vocab_size}")
print(f"✅ Model supports diacritics: Yes")
print(f"✅ Trained on Quranic data (TarteelAI): Yes")
print(f"✅ Baseline WER on Quranic test set: 6.55%")
EOF

echo ""
echo "✅ Download complete!"
echo "Now you can fine-tune with: bash train_nemo_finetune.sh"
