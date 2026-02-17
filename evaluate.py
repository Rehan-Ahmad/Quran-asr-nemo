#!/usr/bin/env python3
"""
Simple evaluation script for the fine-tuned Quran ASR model.
Transcribes audio samples and compares against references.
"""

import json
import sys
from pathlib import Path

from nemo.collections.asr.models import ASRModel


def main():
    # Model checkpoint
    model_path = Path("nemo_experiments/FastConformer-Streaming-Custom-Tokenizer-Official-Seed/finetune_streaming_quran/2026-02-16_12-24-08/checkpoints/finetune_streaming_quran.nemo")
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        sys.exit(1)
    
    # Tokenizer path
    tokenizer_dir = Path("tokenizer/quran_tokenizer_bpe_v1024")
    
    if not tokenizer_dir.exists():
        print(f"❌ Tokenizer not found at {tokenizer_dir}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {model_path.name}...")
    model = ASRModel.restore_from(str(model_path))
    print("✓ Model loaded")
    
    # Apply custom tokenizer
    print(f"Applying tokenizer from {tokenizer_dir.name}...")
    model.change_vocabulary(new_tokenizer_dir=str(tokenizer_dir), new_tokenizer_type="bpe")
    model = model.cuda()
    print("✓ Tokenizer applied")
    
    # Load validation data
    manifest_path = Path("data/manifests/val.json")
    with open(manifest_path) as f:
        val_samples = [json.loads(line) for line in f]
    
    print(f"\nLoaded {len(val_samples)} validation samples")
    
    # Evaluate on first N samples
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    num_samples = min(num_samples, len(val_samples))
    
    print(f"\nEvaluating on {num_samples} samples...")
    print("=" * 80)
    
    matches = 0
    for idx, sample in enumerate(val_samples[:num_samples], 1):
        audio_path = sample['audio_filepath']
        reference = sample['text']
        
        predictions = model.transcribe([audio_path], batch_size=1)
        prediction = predictions[0].text if predictions and hasattr(predictions[0], 'text') else ""
        
        is_match = prediction.strip() == reference.strip()
        if is_match:
            matches += 1
        
        status = "✓" if is_match else "✗"
        print(f"\n[{idx}] {status}")
        print(f"  Ref:  {reference}")
        print(f"  Pred: {prediction}")
    
    accuracy = (matches / num_samples) * 100
    print("\n" + "=" * 80)
    print(f"Results: {matches}/{num_samples} correct ({accuracy:.1f}% accuracy)")


if __name__ == "__main__":
    main()
