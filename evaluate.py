#!/usr/bin/env python3
"""
Proper ASR evaluation script for Quran models.
Based on NVIDIA NeMo's speech_to_text_eval.py pattern.

Computes WER/CER metrics properly instead of simple string matching.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from omegaconf import MISSING, OmegaConf, open_dict
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Model selection
    model_path: Optional[str] = None
    decoder_type: Optional[str] = None  # 'ctc' or 'rnnt' for hybrid models
    
    # Data
    dataset_manifest: str = "data/manifests/val.json"
    batch_size: int = 32
    
    # Output
    output_filename: str = "evaluation_results.json"
    
    # Metrics
    use_cer: bool = False  # Use CER instead of WER
    scores_per_sample: bool = False
    
    # Text processing
    do_lowercase: bool = False
    rm_punctuation: bool = False
    
    # Streaming specific
    att_context_size: Optional[List[int]] = None  # For streaming models
    
    # Tolerance check
    tolerance: Optional[float] = None
    
    # Device
    cuda: bool = True


def discover_models() -> List[tuple]:
    """Find all .nemo models."""
    nemo_root = Path("nemo_experiments")
    models = []
    
    for nemo_file in nemo_root.rglob("*.nemo"):
        parts = nemo_file.parts
        if len(parts) >= 2:
            model_type = parts[parts.index("nemo_experiments") + 1]
            model_name = f"{model_type}/{nemo_file.name}"
        else:
            model_name = str(nemo_file)
        
        is_streaming = "streaming" in model_type.lower()
        size_mb = nemo_file.stat().st_size / (1024**2)
        
        models.append((nemo_file, model_name, is_streaming, size_mb))
    
    return sorted(models, key=lambda x: x[1])


def select_model_interactive() -> str:
    """Let user select model interactively."""
    models = discover_models()
    
    if not models:
        logging.error("No .nemo models found in nemo_experiments/")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("📦 AVAILABLE MODELS")
    print("=" * 80)
    
    for idx, (path, name, is_streaming, size_mb) in enumerate(models, 1):
        model_type = "🌊 STREAMING" if is_streaming else "📶 STANDARD"
        print(f"{idx:2d}. [{model_type}] {name}")
        print(f"    Size: {size_mb:.1f} MB")
    
    print("\n" + "=" * 80)
    
    while True:
        try:
            choice = int(input(f"Select model (1-{len(models)}): ").strip())
            if 1 <= choice <= len(models):
                selected_path, selected_name, _, _ = models[choice - 1]
                print(f"\n✓ Selected: {selected_name}")
                return str(selected_path)
            else:
                print(f"❌ Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


def load_manifest(manifest_path: str) -> List[dict]:
    """Load JSON manifest file."""
    logging.info(f"Loading manifest from {manifest_path}")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    logging.info(f"Loaded {len(samples)} samples from manifest")
    return samples


def get_tokenizer_path() -> Optional[Path]:
    """Find tokenizer directory."""
    tokenizer_dir = Path("tokenizer/quran_tokenizer_bpe_v1024")
    if tokenizer_dir.exists():
        return tokenizer_dir
    
    tokenizer_root = Path("tokenizer")
    if tokenizer_root.exists():
        subdirs = [d for d in tokenizer_root.iterdir() if d.is_dir()]
        if subdirs:
            return subdirs[0]
    
    return None


def transcribe_dataset(model: ASRModel, cfg: EvaluationConfig, samples: List[dict]) -> List[dict]:
    """Transcribe all samples in dataset."""
    logging.info(f"Transcribing {len(samples)} samples...")
    
    # Transcribe in batches
    audio_paths = [s['audio_filepath'] for s in samples]
    
    # Create output manifest with predictions
    output_samples = []
    
    for idx, sample in enumerate(samples):
        audio_path = sample['audio_filepath']
        
        try:
            # Transcribe with appropriate decoder
            if cfg.decoder_type:
                predictions = model.transcribe(
                    [audio_path], 
                    batch_size=1,
                    decoder_type=cfg.decoder_type  # type: ignore
                )
            else:
                predictions = model.transcribe([audio_path], batch_size=1)
            
            # Extract text from Hypothesis object
            if predictions and hasattr(predictions[0], 'text'):
                pred_text = predictions[0].text
            else:
                pred_text = str(predictions[0]) if predictions else ""
            
        except Exception as e:
            logging.warning(f"Failed to transcribe {audio_path}: {e}")
            pred_text = ""
        
        # Add prediction to sample
        output_samples.append({
            **sample,
            'pred_text': pred_text
        })
        
        if (idx + 1) % max(1, len(samples) // 10) == 0:
            logging.info(f"  Progress: {idx + 1}/{len(samples)}")
    
    logging.info("Transcription complete")
    return output_samples


def compute_wer_cer(predictions: List[str], references: List[str], use_cer: bool = False) -> tuple:
    """Compute WER and CER metrics."""
    from nemo.collections.asr.metrics.wer import word_error_rate
    
    cer = word_error_rate(hypotheses=predictions, references=references, use_cer=True)
    wer = word_error_rate(hypotheses=predictions, references=references, use_cer=False)
    
    return wer, cer


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig) -> EvaluationConfig:
    """Main evaluation routine."""
    logging.info("=" * 80)
    logging.info("Quran ASR Evaluation Script")
    logging.info("=" * 80)
    
    # Disable gradients
    torch.set_grad_enabled(False)
    
    # Convert to OmegaConf if needed
    if not isinstance(cfg, type(OmegaConf.create({}))):
        cfg = OmegaConf.structured(cfg)
    
    # Select model if not specified
    if cfg.model_path is None:
        cfg.model_path = select_model_interactive()
    
    # Load model
    logging.info(f"Loading model from {cfg.model_path}")
    model = ASRModel.restore_from(str(cfg.model_path))
    logging.info("✓ Model loaded")
    
    # Apply custom tokenizer if available
    tokenizer_dir = get_tokenizer_path()
    if tokenizer_dir:
        logging.info(f"Applying tokenizer from {tokenizer_dir}")
        try:
            model.change_vocabulary(
                new_tokenizer_dir=str(tokenizer_dir),
                new_tokenizer_type="bpe"
            )
            logging.info("✓ Tokenizer applied")
        except Exception as e:
            logging.warning(f"Could not apply tokenizer: {e}")
    
    # Move to GPU if available
    if cfg.cuda and torch.cuda.is_available():
        model = model.cuda()
        logging.info("✓ Model moved to GPU")
    else:
        model = model.eval()
    
    # Load validation data
    samples = load_manifest(cfg.dataset_manifest)
    
    # Transcribe all samples
    output_samples = transcribe_dataset(model, cfg, samples)
    
    # Write transcriptions to output file
    logging.info(f"Writing transcriptions to {cfg.output_filename}")
    with open(cfg.output_filename, 'w') as f:
        for sample in output_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Compute metrics
    logging.info("Computing metrics...")
    references = [s.get('text', '') for s in output_samples]
    predictions = [s.get('pred_text', '') for s in output_samples]
    
    try:
        wer, cer = compute_wer_cer(predictions, references, use_cer=cfg.use_cer)
        
        # Log results
        logging.info("=" * 80)
        logging.info(f"EVALUATION RESULTS")
        logging.info("=" * 80)
        logging.info(f"Total samples: {len(samples)}")
        logging.info(f"WER: {wer:.4f} ({wer*100:.2f}%)")
        logging.info(f"CER: {cer:.4f} ({cer*100:.2f}%)")
        logging.info(f"Output: {cfg.output_filename}")
        logging.info("=" * 80)
        
        # Check tolerance if specified
        if cfg.tolerance is not None:
            metric_value = cer if cfg.use_cer else wer
            if metric_value > cfg.tolerance:
                raise ValueError(
                    f"Metric {metric_value:.4f} exceeds tolerance {cfg.tolerance}"
                )
            logging.info(f"✓ Metric within tolerance ({cfg.tolerance})")
        
        # Store metrics in config
        with open_dict(cfg):
            cfg.wer = float(wer)
            cfg.cer = float(cer)
        
    except Exception as e:
        logging.error(f"Error computing metrics: {e}")
        return cfg
    
    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
