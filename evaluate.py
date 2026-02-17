#!/usr/bin/env python3
"""
Proper ASR evaluation script for Quran models.
Based on NVIDIA NeMo's speech_to_text_eval.py pattern.

Explores nemo_experiments folder structure:
- Each first-level directory = model architecture
- Latest run per architecture = newest timestamp folder
- Best checkpoint = latest saved .nemo file in checkpoints/
- Detects streaming vs standard from config/name
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import torch
import yaml
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


def parse_timestamp(name: str) -> datetime:
    """Parse timestamp from folder name like 2026-02-17_05-27-32."""
    try:
        return datetime.strptime(name, '%Y-%m-%d_%H-%M-%S')
    except:
        return datetime.min


def find_latest_run(arch_path: Path) -> Optional[Path]:
    """Find latest run folder by date for an architecture."""
    runs = []
    
    # Check direct subdirectories with dates (202*)
    for item in arch_path.iterdir():
        if item.is_dir() and item.name[0].isdigit():
            runs.append(item)
    
    # Also check nested subdirectories (e.g., finetune/2026-*)
    for item in arch_path.iterdir():
        if item.is_dir() and not item.name[0].isdigit():
            for subitem in item.iterdir():
                if subitem.is_dir() and subitem.name[0].isdigit():
                    runs.append(subitem)
    
    if not runs:
        return None
    
    return max(runs, key=lambda x: parse_timestamp(x.name))


def find_best_checkpoint(run_path: Path) -> Optional[Path]:
    """Find best (latest) checkpoint in run."""
    checkpoints = run_path / "checkpoints"
    
    if not checkpoints.exists():
        return None
    
    nemo_files = list(checkpoints.glob("*.nemo"))
    if not nemo_files:
        return None
    
    # Sort by modification time (latest first)
    nemo_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return nemo_files[0]


def check_streaming(run_path: Path, arch_name: str) -> bool:
    """Check if model is streaming from config or name."""
    # Check hparams.yaml
    hparams_path = run_path / "hparams.yaml"
    if not hparams_path.exists():
        hparams_path = run_path.parent / "hparams.yaml"
    
    if hparams_path.exists():
        try:
            with open(hparams_path) as f:
                config = yaml.safe_load(f)
            
            config_str = json.dumps(config, default=str).lower()
            if "streaming" in config_str or "att_context_size" in config_str:
                return True
        except:
            pass
    
    # Fallback to name-based detection
    return "streaming" in arch_name.lower()


def get_model_architectures() -> Dict[str, Dict]:
    """Discover all model architectures and their best runs."""
    nemo_root = Path("nemo_experiments")
    architectures = {}
    
    for arch_path in sorted(nemo_root.iterdir()):
        if not arch_path.is_dir():
            continue
        
        arch_name = arch_path.name
        latest_run = find_latest_run(arch_path)
        
        if latest_run is None:
            continue
        
        best_ckpt = find_best_checkpoint(latest_run)
        
        if best_ckpt is None:
            continue  # Skip architectures without checkpoints
        
        is_streaming = check_streaming(latest_run, arch_name)
        
        architectures[arch_name] = {
            'latest_run': latest_run.name,
            'latest_run_path': str(latest_run),
            'best_ckpt': str(best_ckpt),
            'best_ckpt_name': best_ckpt.name,
            'is_streaming': is_streaming,
            'size_mb': best_ckpt.stat().st_size / (1024**2)
        }
    
    return architectures


def discover_models() -> List[tuple]:
    """Find all .nemo models (for backward compatibility)."""
    architectures = get_model_architectures()
    models = []
    
    for arch_name, info in architectures.items():
        nemo_file = Path(info['best_ckpt'])
        model_name = f"{arch_name}/{info['best_ckpt_name']}"
        is_streaming = info['is_streaming']
        size_mb = info['size_mb']
        
        models.append((nemo_file, model_name, is_streaming, size_mb))
    
    return sorted(models, key=lambda x: x[1])


def select_model_interactive() -> Tuple[str, Dict]:
    """Let user select model interactively with architecture details."""
    architectures = get_model_architectures()
    
    if not architectures:
        logging.error("No .nemo models found in nemo_experiments/")
        sys.exit(1)
    
    models_list = list(architectures.items())
    
    print("\n" + "=" * 90)
    print("📦 AVAILABLE MODEL ARCHITECTURES")
    print("=" * 90)
    
    for idx, (arch_name, info) in enumerate(models_list, 1):
        model_type = "🌊 STREAMING" if info['is_streaming'] else "📶 STANDARD"
        print(f"{idx:2d}. [{model_type}] {arch_name}")
        print(f"    Latest Run: {info['latest_run']}")
        print(f"    Best Model: {info['best_ckpt_name']}")
        print(f"    Size: {info['size_mb']:.1f} MB")
    
    print("\n" + "=" * 90)
    
    while True:
        try:
            choice = int(input(f"Select architecture (1-{len(models_list)}): ").strip())
            if 1 <= choice <= len(models_list):
                arch_name, info = models_list[choice - 1]
                print(f"\n✓ Selected: {arch_name}")
                print(f"✓ Type: {'STREAMING' if info['is_streaming'] else 'STANDARD'}")
                print(f"✓ Model: {info['best_ckpt']}")
                return info['best_ckpt'], info
            else:
                print(f"❌ Please enter a number between 1 and {len(models_list)}")
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
        cfg.model_path, model_info = select_model_interactive()
    else:
        model_info = {}
    
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
        
        if model_info:
            logging.info(f"Model Type: {'STREAMING' if model_info.get('is_streaming') else 'STANDARD'}")
        
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
