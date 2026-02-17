#!/usr/bin/env python3
"""
Proper ASR evaluation script for Quran models.
Based on NVIDIA NeMo's official evaluation patterns:
- speech_to_text_eval.py for metrics computation
- transcribe_speech.py for transcription

Supports:
- Interactive model selection from nemo_experiments/
- WER/CER metrics computation
- Streaming and standard models
- Text processing (punctuation, case normalization)
- Per-sample metrics
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import torch
from omegaconf import MISSING, OmegaConf, open_dict
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.transcribe_utils import (
    PunctuationCapitalization,
    TextProcessingConfig,
    compute_metrics_per_sample,
)
from nemo.collections.common.metrics.punct_er import DatasetPunctuationErrorRate
from nemo.core.config import hydra_runner
from nemo.utils import logging
import yaml


@dataclass
class EvaluationConfig:
    """Configuration for evaluation, inheriting from standard ASR patterns."""
    # Model selection
    model_path: Optional[str] = None
    
    # Data
    dataset_manifest: str = "data/manifests/val.json"
    batch_size: int = 32
    
    # Output
    output_filename: str = "evaluation_results.json"
    
    # Metrics
    use_cer: bool = False
    use_punct_er: bool = False
    tolerance: Optional[float] = None
    only_score_manifest: bool = False
    scores_per_sample: bool = False
    
    # Model-specific
    decoder_type: Optional[str] = None  # 'ctc' or 'rnnt' for hybrid models
    att_context_size: Optional[List[int]] = None  # For streaming models
    
    # Text processing
    text_processing: Optional[TextProcessingConfig] = field(
        default_factory=lambda: TextProcessingConfig(
            punctuation_marks=".,?",
            separate_punctuation=False,
            do_lowercase=False,
            rm_punctuation=False,
        )
    )
    
    # Ground truth field name
    gt_text_attr_name: str = "text"
    
    # Device
    cuda: bool = True


def parse_timestamp(name: str) -> datetime:
    """Parse timestamp from folder name like 2026-02-15_13-35-12."""
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
    
    if not nemo_root.exists():
        logging.error("nemo_experiments folder not found")
        return {}
    
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


def select_model_interactive() -> str:
    """Let user select model interactively with architecture details."""
    architectures = get_model_architectures()
    
    if not architectures:
        logging.error("No .nemo models found in nemo_experiments/")
        sys.exit(1)
    
    models_list = list(architectures.items())
    
    print("\n" + "=" * 100)
    print("📦 AVAILABLE MODEL ARCHITECTURES")
    print("=" * 100)
    
    for idx, (arch_name, info) in enumerate(models_list, 1):
        model_type = "🌊 STREAMING" if info['is_streaming'] else "📶 STANDARD"
        print(f"\n{idx:2d}. [{model_type}] {arch_name}")
        print(f"    Latest Run:    {info['latest_run']}")
        print(f"    Best Model:    {info['best_ckpt_name']}")
        print(f"    Size:          {info['size_mb']:.1f} MB")
    
    print("\n" + "=" * 100)
    
    while True:
        try:
            choice = int(input(f"Select architecture (1-{len(models_list)}): ").strip())
            if 1 <= choice <= len(models_list):
                arch_name, info = models_list[choice - 1]
                print(f"\n✓ Selected: {arch_name}")
                print(f"✓ Type:     {'STREAMING' if info['is_streaming'] else 'STANDARD'}")
                print(f"✓ Model:    {info['best_ckpt']}")
                return info['best_ckpt']
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
    
    # Look for other tokenizer directories
    tokenizer_root = Path("tokenizer")
    if tokenizer_root.exists():
        subdirs = [d for d in tokenizer_root.iterdir() if d.is_dir()]
        if subdirs:
            return subdirs[0]
    
    return None


def transcribe_manifest(model: ASRModel, cfg: EvaluationConfig, samples: List[dict]) -> List[dict]:
    """Transcribe all samples in batches and add predictions to manifest.

    Uses `cfg.batch_size` to group audio files into batches. If a batch
    transcription fails, falls back to per-sample transcription for that
    batch to maximize robustness.
    """
    total = len(samples)
    logging.info(f"Transcribing {total} samples in batches (batch_size={cfg.batch_size})...")

    output_samples: List[dict] = []

    # Helper to transcribe a list of audio paths and append results
    def _transcribe_and_append(audio_paths: List[str], batch_samples: List[dict]):
        try:
            if cfg.decoder_type:
                preds = model.transcribe(audio_paths, batch_size=len(audio_paths), decoder_type=cfg.decoder_type)  # type: ignore
            else:
                preds = model.transcribe(audio_paths, batch_size=len(audio_paths))

            for sample_obj, hyp in zip(batch_samples, preds):
                if hyp is None:
                    text = ""
                elif hasattr(hyp, 'text'):
                    text = hyp.text
                else:
                    text = str(hyp)

                output_samples.append({**sample_obj, 'pred_text': text})

        except KeyboardInterrupt:
            logging.info("Transcription interrupted by user")
            sys.exit(0)
        except Exception as e:
            logging.warning(f"Batch transcription failed ({len(audio_paths)} items): {e}. Falling back to per-sample.")
            # Fallback: transcribe items one-by-one
            for single_path, single_sample in zip(audio_paths, batch_samples):
                try:
                    if cfg.decoder_type:
                        single_preds = model.transcribe([single_path], batch_size=1, decoder_type=cfg.decoder_type)  # type: ignore
                    else:
                        single_preds = model.transcribe([single_path], batch_size=1)

                    hyp = single_preds[0] if single_preds else None
                    text = hyp.text if (hyp is not None and hasattr(hyp, 'text')) else (str(hyp) if hyp is not None else "")
                except Exception as ex:
                    logging.warning(f"Failed to transcribe {single_path}: {ex}")
                    text = ""

                output_samples.append({**single_sample, 'pred_text': text})

    # Process in batches
    batch_size = max(1, int(cfg.batch_size))
    progress_interval = max(1, total // 10)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = samples[start:end]
        audio_paths = [s['audio_filepath'] for s in batch]

        _transcribe_and_append(audio_paths, batch)

        # Progress reporting
        if end % progress_interval == 0 or end == total:
            logging.info(f"  Progress: {end}/{total}")

    logging.info("Transcription complete")
    return output_samples


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig) -> EvaluationConfig:
    """Main evaluation routine following NeMo official pattern."""
    logging.info("=" * 100)
    logging.info("Quran ASR Evaluation Script")
    logging.info("=" * 100)
    
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
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to GPU if available
    if cfg.cuda and torch.cuda.is_available():
        model = model.cuda()
        logging.info("✓ Model moved to GPU")
    else:
        logging.info("✓ Using CPU for inference")
    
    # Load validation data
    samples = load_manifest(cfg.dataset_manifest)
    
    # Transcribe all samples and create output manifest
    if not cfg.only_score_manifest:
        output_samples = transcribe_manifest(model, cfg, samples)
        
        # Write transcriptions to output file
        logging.info(f"Writing transcriptions to {cfg.output_filename}")
        with open(cfg.output_filename, 'w') as f:
            for sample in output_samples:
                f.write(json.dumps(sample) + '\n')
        
        manifest_to_evaluate = cfg.output_filename
    else:
        logging.info(f"Using existing manifest for scoring: {cfg.dataset_manifest}")
        manifest_to_evaluate = cfg.dataset_manifest
    
    # Load the manifest to evaluate
    with open(manifest_to_evaluate, 'r') as f:
        evaluation_samples = [json.loads(line) for line in f if line.strip()]
    
    # Extract ground truth and predictions
    ground_truth_text = []
    predicted_text = []
    
    for data in evaluation_samples:
        if "pred_text" not in data:
            logging.error(
                f"Manifest {manifest_to_evaluate} does not contain 'pred_text' field. "
                "Please transcribe first or use only_score_manifest=False"
            )
            sys.exit(1)
        
        ground_truth_text.append(data.get(cfg.gt_text_attr_name, ""))
        predicted_text.append(data.get("pred_text", ""))
    
    # Apply text processing
    logging.info("Applying text processing...")
    pc = PunctuationCapitalization(cfg.text_processing.punctuation_marks)
    
    if cfg.text_processing.separate_punctuation:
        ground_truth_text = pc.separate_punctuation(ground_truth_text)
        predicted_text = pc.separate_punctuation(predicted_text)
    
    if cfg.text_processing.do_lowercase:
        ground_truth_text = pc.do_lowercase(ground_truth_text)
        predicted_text = pc.do_lowercase(predicted_text)
    
    if cfg.text_processing.rm_punctuation:
        ground_truth_text = pc.rm_punctuation(ground_truth_text)
        predicted_text = pc.rm_punctuation(predicted_text)
    
    # Compute WER and CER (NeMo may return percentages > 1.0)
    logging.info("Computing metrics...")
    cer_raw = word_error_rate(hypotheses=predicted_text, references=ground_truth_text, use_cer=True)
    wer_raw = word_error_rate(hypotheses=predicted_text, references=ground_truth_text, use_cer=False)

    # Normalize to fractional form (0..1). If the returned value is >1,
    # assume it's a percent (e.g. 74.5) and divide by 100.
    def _to_frac(x: float) -> float:
        try:
            return x / 100.0 if x > 1.0 else x
        except Exception:
            return float(x)

    cer = _to_frac(cer_raw)
    wer = _to_frac(wer_raw)
    
    # Punctuation Error Rate (optional)
    if cfg.use_punct_er:
        dper_obj = DatasetPunctuationErrorRate(
            hypotheses=predicted_text,
            references=ground_truth_text,
            punctuation_marks=list(cfg.text_processing.punctuation_marks),
        )
        dper_obj.compute()
    
    # Per-sample metrics (optional)
    if cfg.scores_per_sample:
        logging.info("Computing per-sample metrics...")
        metrics_to_compute = ["wer", "cer"]
        if cfg.use_punct_er:
            metrics_to_compute.append("punct_er")
        
        compute_metrics_per_sample(
            manifest_path=manifest_to_evaluate,
            reference_field=cfg.gt_text_attr_name,
            hypothesis_field="pred_text",
            metrics=metrics_to_compute,
            punctuation_marks=cfg.text_processing.punctuation_marks,
            output_manifest_path=cfg.output_filename,
        )
    
    # Determine which metric to report (use fractional values for tolerance)
    metric_name = 'CER' if cfg.use_cer else 'WER'
    metric_value = cer if cfg.use_cer else wer

    # Log results (show percent and fractional forms)
    logging.info("=" * 100)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 100)
    logging.info(f"Total samples: {len(evaluation_samples)}")
    logging.info(f"WER: {wer*100:.2f}% ({wer:.4f} fraction)")
    logging.info(f"CER: {cer*100:.2f}% ({cer:.4f} fraction)")
    
    if cfg.use_punct_er:
        dper_obj.print()
    
    logging.info(f"Output: {cfg.output_filename}")
    logging.info("=" * 100)
    
    # Check tolerance if specified
    if cfg.tolerance is not None:
        if metric_value > cfg.tolerance:
            raise ValueError(
                f"Got {metric_name} of {metric_value:.4f}, which was higher than tolerance={cfg.tolerance}"
            )
        logging.info(f"✓ Got {metric_name} of {metric_value:.4f}. Tolerance was {cfg.tolerance}")
    
    # Store metrics in config
    with open_dict(cfg):
        cfg.wer = float(wer)
        cfg.cer = float(cer)
        cfg.metric_name = metric_name
        cfg.metric_value = float(metric_value)
    
    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
