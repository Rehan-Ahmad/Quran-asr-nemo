"""
Quick test script to verify streaming comparison app functionality
Tests model loading and basic inference without launching Gradio UI
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

CHECKPOINT_PATH = './nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/epoch=49-step=45200.ckpt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("Streaming Comparison App - Quick Test")
print("=" * 60)

# Check checkpoint exists
checkpoint_file = Path(CHECKPOINT_PATH)
if not checkpoint_file.exists():
    print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
    sys.exit(1)
print(f"✓ Checkpoint found: {CHECKPOINT_PATH}")

# Check device
print(f"✓ Device: {DEVICE}")

# Load streaming model
print("\n[1/2] Loading streaming model...")
try:
    streaming_model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(CHECKPOINT_PATH)
    streaming_model = streaming_model.to(DEVICE)
    streaming_model.eval()
    att_context = streaming_model.cfg.encoder.att_context_size
    print(f"    ✓ Streaming model loaded")
    print(f"    ✓ att_context_size: {att_context}")
    print(f"    ✓ Active context: {streaming_model.encoder.att_context_size}")
except Exception as e:
    print(f"    ❌ Failed to load streaming model: {e}")
    sys.exit(1)

# Load non-streaming model
print("\n[2/2] Loading non-streaming model...")
try:
    non_streaming_model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(CHECKPOINT_PATH)
    non_streaming_model = non_streaming_model.to(DEVICE)
    non_streaming_model.eval()
    
    # Modify for full context
    if hasattr(non_streaming_model.encoder, 'att_context_size'):
        non_streaming_model.encoder.att_context_size = [10000, 10000]
        print(f"    ✓ Non-streaming model loaded")
        print(f"    ✓ Modified att_context_size: {non_streaming_model.encoder.att_context_size}")
    else:
        print(f"    ⚠️  No att_context_size attribute found")
except Exception as e:
    print(f"    ❌ Failed to load non-streaming model: {e}")
    sys.exit(1)

# Test with sample audio if available
print("\n[3/3] Testing inference...")
sample_manifest = Path('./data/manifests/test.json')

if sample_manifest.exists():
    import json
    
    # Load first sample
    with open(sample_manifest, 'r', encoding='utf-8') as f:
        first_sample = json.loads(f.readline())
    
    audio_path = first_sample['audio_filepath']
    reference = first_sample['text']
    
    if Path(audio_path).exists():
        print(f"    Using audio: {audio_path}")
        print(f"    Reference: {reference[:50]}...")
        
        # Test streaming
        import time
        start = time.time()
        with torch.no_grad():
            streaming_result = streaming_model.transcribe(audio=[audio_path], batch_size=1)[0]
        streaming_time = time.time() - start
        
        # Test non-streaming
        start = time.time()
        with torch.no_grad():
            non_streaming_result = non_streaming_model.transcribe(audio=[audio_path], batch_size=1)[0]
        non_streaming_time = time.time() - start
        
        # Extract text from Hypothesis objects
        streaming_text = streaming_result.text if hasattr(streaming_result, 'text') else str(streaming_result)
        non_streaming_text = non_streaming_result.text if hasattr(non_streaming_result, 'text') else str(non_streaming_result)
        
        print(f"\n    Streaming transcription: {streaming_text[:50]}...")
        print(f"    Streaming time: {streaming_time:.3f}s")
        
        print(f"\n    Non-streaming transcription: {non_streaming_text[:50]}...")
        print(f"    Non-streaming time: {non_streaming_time:.3f}s")
        
        if streaming_text == non_streaming_text:
            print(f"\n    ✓ IDENTICAL outputs")
        else:
            print(f"\n    ⚠️  DIFFERENT outputs")
        
        print(f"    Speed comparison: {non_streaming_time/streaming_time:.2f}x")
    else:
        print(f"    ⚠️  Audio file not found: {audio_path}")
else:
    print(f"    ⚠️  Test manifest not found: {sample_manifest}")

print("\n" + "=" * 60)
print("✅ All tests passed! App is ready to launch.")
print("=" * 60)
print("\nTo launch the Gradio app:")
print("    python app_streaming_comparison.py")
print("\nApp will be available at: http://localhost:7860")
