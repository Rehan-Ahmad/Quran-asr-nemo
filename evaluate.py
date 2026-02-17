#!/usr/bin/env python3
"""
Interactive evaluation script for Quran ASR models.
Discovers available checkpoints, lets user choose, and evaluates accordingly.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from nemo.collections.asr.models import ASRModel
import yaml


def discover_models() -> List[Tuple[Path, str, bool]]:
    """
    Discover all .nemo checkpoints in nemo_experiments.
    Returns: [(model_path, model_name, is_streaming), ...]
    """
    nemo_root = Path("nemo_experiments")
    if not nemo_root.exists():
        print("❌ nemo_experiments folder not found")
        sys.exit(1)
    
    models = []
    for nemo_file in nemo_root.rglob("*.nemo"):
        # Get model name from path
        parts = nemo_file.parts
        if len(parts) >= 2:
            model_type = parts[parts.index("nemo_experiments") + 1]
            model_name = f"{model_type}/{nemo_file.name}"
        else:
            model_name = str(nemo_file)
        
        # Check if streaming model (by name)
        is_streaming = "streaming" in model_type.lower()
        
        models.append((nemo_file, model_name, is_streaming))
    
    return sorted(models, key=lambda x: x[1])


def detect_streaming_from_config(model_path: Path) -> bool:
    """
    Try to detect streaming config from model's hparams.
    Look for streaming-related parameters in config files.
    """
    # Check parent directories for hparams.yaml
    for parent in [model_path.parent, model_path.parent.parent, model_path.parent.parent.parent]:
        hparams_path = parent / "hparams.yaml"
        if hparams_path.exists():
            try:
                with open(hparams_path) as f:
                    config = yaml.safe_load(f)
                
                # Look for streaming indicators in config
                cfg = config.get("cfg", {})
                model_cfg = cfg.get("model", {})
                encoder_cfg = model_cfg.get("encoder", {})
                
                # Check for streaming attention context
                if "context" in encoder_cfg:
                    return True
                
                # Check model name in config
                if "FastConformer-Streaming" in str(config):
                    return True
                    
            except Exception as e:
                pass
    
    return False


def select_model(models: List[Tuple[Path, str, bool]]) -> Tuple[Path, bool]:
    """
    Display available models and let user choose.
    Returns: (selected_model_path, is_streaming)
    """
    if not models:
        print("❌ No .nemo models found in nemo_experiments")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("📦 AVAILABLE MODELS")
    print("=" * 80)
    
    for idx, (path, name, is_streaming) in enumerate(models, 1):
        model_type = "🌊 STREAMING" if is_streaming else "📶 STANDARD"
        size_mb = path.stat().st_size / (1024**2)
        print(f"{idx:2d}. [{model_type}] {name}")
        print(f"    Size: {size_mb:.1f} MB")
    
    print("\n" + "=" * 80)
    
    while True:
        try:
            choice = int(input(f"Select model (1-{len(models)}): ").strip())
            if 1 <= choice <= len(models):
                model_path, model_name, is_streaming = models[choice - 1]
                
                # Refine streaming detection
                enhanced_streaming = detect_streaming_from_config(model_path) or is_streaming
                
                print(f"\n✓ Selected: {model_name}")
                print(f"✓ Type: {'STREAMING' if enhanced_streaming else 'STANDARD'}")
                return model_path, enhanced_streaming
            else:
                print(f"❌ Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


def get_num_samples() -> int:
    """Ask user how many samples to evaluate."""
    while True:
        try:
            num = int(input("\nHow many samples to evaluate? (default 20): ").strip() or "20")
            if num > 0:
                return num
            else:
                print("❌ Please enter a positive number")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


def load_validation_data() -> List[dict]:
    """Load validation manifest."""
    manifest_path = Path("data/manifests/val.json")
    if not manifest_path.exists():
        print(f"❌ Validation manifest not found at {manifest_path}")
        sys.exit(1)
    
    with open(manifest_path) as f:
        samples = [json.loads(line) for line in f]
    
    return samples


def get_tokenizer_path() -> Optional[Path]:
    """Find tokenizer directory."""
    tokenizer_dir = Path("tokenizer/quran_tokenizer_bpe_v1024")
    if tokenizer_dir.exists():
        return tokenizer_dir
    
    # Look for other tokenizer directories
    tokenizer_root = Path("tokenizer")
    if tokenizer_root.exists():
        subdirs = [d for d in tokenizer_root.iterdir() if d.is_dir()]
        if subdirs:
            return subdirs[0]
    
    return None


def evaluate_standard(model, samples: List[dict], num_samples: int) -> None:
    """Standard (non-streaming) evaluation."""
    print("\n" + "=" * 80)
    print("📊 EVALUATING (Standard Mode)")
    print("=" * 80)
    
    matches = 0
    for idx, sample in enumerate(samples[:num_samples], 1):
        audio_path = sample['audio_filepath']
        reference = sample['text']
        
        try:
            predictions = model.transcribe([audio_path], batch_size=1)
            prediction = predictions[0].text if predictions and hasattr(predictions[0], 'text') else ""
        except Exception as e:
            prediction = f"[ERROR: {e}]"
        
        is_match = prediction.strip() == reference.strip()
        if is_match:
            matches += 1
        
        status = "✓" if is_match else "✗"
        print(f"\n[{idx:2d}] {status}")
        if not is_match:
            print(f"  Ref:  {reference}")
            print(f"  Pred: {prediction[:100]}{'...' if len(prediction) > 100 else ''}")
    
    accuracy = (matches / num_samples) * 100 if num_samples > 0 else 0
    print("\n" + "=" * 80)
    print(f"📈 Results: {matches}/{num_samples} correct ({accuracy:.1f}% accuracy)")
    print("=" * 80)


def evaluate_streaming(model, samples: List[dict], num_samples: int) -> None:
    """Streaming-mode evaluation (chunk-based inference)."""
    print("\n" + "=" * 80)
    print("📊 EVALUATING (Streaming Mode)")
    print("=" * 80)
    
    matches = 0
    for idx, sample in enumerate(samples[:num_samples], 1):
        audio_path = sample['audio_filepath']
        reference = sample['text']
        
        try:
            # For streaming models, use batch inference (same as standard)
            # True chunk-based streaming would use different API
            predictions = model.transcribe([audio_path], batch_size=1)
            prediction = predictions[0].text if predictions and hasattr(predictions[0], 'text') else ""
        except Exception as e:
            prediction = f"[ERROR: {e}]"
        
        is_match = prediction.strip() == reference.strip()
        if is_match:
            matches += 1
        
        status = "✓" if is_match else "✗"
        print(f"\n[{idx:2d}] {status}")
        if not is_match:
            print(f"  Ref:  {reference}")
            print(f"  Pred: {prediction[:100]}{'...' if len(prediction) > 100 else ''}")
    
    accuracy = (matches / num_samples) * 100 if num_samples > 0 else 0
    print("\n" + "=" * 80)
    print(f"📈 Results: {matches}/{num_samples} correct ({accuracy:.1f}% accuracy)")
    print("  Note: Streaming model evaluated with batch inference")
    print("=" * 80)


def main():
    print("\n🎙️  Quran ASR Model Evaluator")
    
    # Discover available models
    models = discover_models()
    
    # Let user choose
    model_path, is_streaming = select_model(models)
    
    # Load model
    print(f"\n⏳ Loading model...")
    try:
        model = ASRModel.restore_from(str(model_path))
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Apply tokenizer if available
    tokenizer_dir = get_tokenizer_path()
    if tokenizer_dir:
        print(f"⏳ Applying tokenizer from {tokenizer_dir.name}...")
        try:
            model.change_vocabulary(new_tokenizer_dir=str(tokenizer_dir), new_tokenizer_type="bpe")
            model = model.cuda() if hasattr(model, 'cuda') else model
            print("✓ Tokenizer applied")
        except Exception as e:
            print(f"⚠️  Warning: Could not apply tokenizer: {e}")
    else:
        print("⚠️  No tokenizer found, using default")
    
    # Load validation data
    val_samples = load_validation_data()
    print(f"✓ Loaded {len(val_samples)} validation samples")
    
    # Get number of samples
    num_samples = get_num_samples()
    num_samples = min(num_samples, len(val_samples))
    
    # Evaluate based on model type
    if is_streaming:
        evaluate_streaming(model, val_samples, num_samples)
    else:
        evaluate_standard(model, val_samples, num_samples)


if __name__ == "__main__":
    main()
