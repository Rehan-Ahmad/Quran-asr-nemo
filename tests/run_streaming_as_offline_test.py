#!/usr/bin/env python3
import os
import sys
import pathlib
import torch

# Ensure repo root is on sys.path so local modules (evaluate.py) can be imported
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nemo.collections.asr.models import ASRModel
from evaluate import transcribe_offline

MODEL_PATH = "pretrained_models/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
TOKENIZER_DIR = "tokenizer/quran_tokenizer_bpe_v1024"

SAMPLES = [
    "data/audio/test/000000.wav",
    "data/audio/test/000001.wav",
]
REFS = [
    "فِيهِمَا عَيْنَانِ نَضَّاخَتَانِ",
    "وَإِنَّهُ لَتَنزِيلُ رَبِّ الْعَالَمِينَ",
]


def main():
    print("Loading model:", MODEL_PATH)
    model = ASRModel.restore_from(MODEL_PATH)
    print("Model loaded.")

    # Apply Quran BPE tokenizer if available
    try:
        model.change_vocabulary(new_tokenizer_dir=TOKENIZER_DIR, new_tokenizer_type="bpe")
        print("Applied tokenizer:", TOKENIZER_DIR)
    except Exception as e:
        print("Tokenizer change failed:", e)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
        print("Moved model to CUDA")
    else:
        print("Running on CPU")

    # Run offline transcription (batch_size=1 for quick test)
    print("Running offline transcription on samples:", SAMPLES)
    hyps = transcribe_offline(model, SAMPLES, batch_size=1)

    print("\nResults:\n")
    for i, (fp, ref, hyp) in enumerate(zip(SAMPLES, REFS, hyps)):
        print(f"Sample {i}: {fp}")
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}")
        print()


if __name__ == '__main__':
    main()
