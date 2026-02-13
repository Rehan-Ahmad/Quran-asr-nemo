#!/usr/bin/env python3
"""Print a few model predictions vs references to diagnose WER > 1."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from omegaconf import open_dict
import librosa
import soundfile as sf
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

VAL_WER_RE = re.compile(r"val_wer=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ASR predictions vs references")
    parser.add_argument(
        "--manifest",
        default="data/manifests/val.json",
        help="Path to manifest JSONL (default: data/manifests/val.json)",
    )
    parser.add_argument(
        "--exp_root",
        default="nemo_experiments",
        help="Root directory where NeMo experiments are stored.",
    )
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Specific run directory (contains checkpoints/) to use.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Explicit checkpoint path (.ckpt or .nemo). Overrides exp_dir/exp_root.",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to inspect")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def find_latest_run(exp_root: Path) -> Path:
    runs = [p for p in exp_root.rglob("*") if p.is_dir() and (p / "checkpoints").exists()]
    if not runs:
        raise FileNotFoundError(f"No runs found under {exp_root}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def find_checkpoints(checkpoints_dir: Path) -> Iterable[Path]:
    return checkpoints_dir.glob("*.ckpt")


def parse_val_wer(path: Path) -> Optional[float]:
    match = VAL_WER_RE.search(path.name)
    if not match:
        return None
    return float(match.group(1))


def pick_best_ckpt(checkpoints_dir: Path) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_score: float = float("inf")

    for ckpt in find_checkpoints(checkpoints_dir):
        score = parse_val_wer(ckpt)
        if score is None:
            continue
        if score < best_score:
            best_score = score
            best_path = ckpt

    if best_path is not None:
        return best_path

    nemo_files = sorted(checkpoints_dir.glob("*.nemo"), key=lambda p: p.stat().st_mtime, reverse=True)
    return nemo_files[0] if nemo_files else None


def load_model(ckpt_path: Path) -> EncDecHybridRNNTCTCBPEModel:
    if ckpt_path.suffix == ".ckpt":
        return EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(str(ckpt_path))
    return EncDecHybridRNNTCTCBPEModel.restore_from(str(ckpt_path))


def sample_manifest(manifest_path: Path, num_samples: int, seed: int) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    random.seed(seed)
    if num_samples >= len(lines):
        return lines
    return random.sample(lines, num_samples)


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    audio, sr = sf.read(path, always_2d=False)
    if audio is None:
        raise ValueError(f"Failed to load audio: {path}")

    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return torch.tensor(audio, dtype=torch.float32)


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    ckpt_path: Optional[Path] = Path(args.ckpt) if args.ckpt else None
    if ckpt_path is None:
        run_dir = Path(args.exp_dir) if args.exp_dir else find_latest_run(Path(args.exp_root))
        ckpt_path = pick_best_ckpt(run_dir / "checkpoints")
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

    print(f"Using checkpoint: {ckpt_path}")

    model = load_model(ckpt_path)
    model.eval()

    # Force-disable Lhotse to avoid DynamicCutSampler errors
    if hasattr(model, "cfg"):
        with open_dict(model.cfg):
            for ds_name in ("train_ds", "validation_ds", "test_ds"):
                if hasattr(model.cfg, ds_name):
                    getattr(model.cfg, ds_name)["use_lhotse"] = False

    if torch.cuda.is_available():
        model = model.cuda()

    samples = sample_manifest(manifest_path, args.num_samples, args.seed)
    audio_paths = [s["audio_filepath"] for s in samples]
    refs = [s["text"] for s in samples]

    print("\n=== Predictions vs References ===\n")
    sample_rate = model.cfg.sample_rate if hasattr(model, "cfg") else 16000
    audio_tensors = [load_audio(path, sample_rate) for path in audio_paths]

    transcribe_cfg = {
        "batch_size": 1,
        "num_workers": 0,
    }
    preds = model.transcribe(audio_tensors, **transcribe_cfg)

    for i, (ref, pred, path) in enumerate(zip(refs, preds, audio_paths), start=1):
        print(f"[{i}] Audio: {path}")
        print(f"REF : {ref}")
        print(f"PRED: {pred}")
        print("-" * 80)


if __name__ == "__main__":
    main()
