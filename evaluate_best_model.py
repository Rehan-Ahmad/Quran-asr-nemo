#!/usr/bin/env python
"""Evaluate the best NeMo checkpoint and print training/eval summary.

- Finds the latest run directory under nemo_experiments (unless --exp_dir given)
- Lists checkpoints with val_wer in filename and picks the lowest
- Runs test on a single GPU (or CPU if no GPU)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import lightning.pytorch as pl
import torch

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel


VAL_WER_RE = re.compile(r"val_wer=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best NeMo checkpoint on test set")
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Path to a specific experiment run directory (contains checkpoints/).",
    )
    parser.add_argument(
        "--exp_root",
        default="nemo_experiments",
        help="Root directory where NeMo experiments are stored.",
    )
    parser.add_argument(
        "--test_manifest",
        default="data/manifests/test.json",
        help="Path to test manifest JSONL.",
    )
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


def pick_best_ckpt(checkpoints_dir: Path) -> Tuple[Optional[Path], list[tuple[str, float]]]:
    scored: list[tuple[str, float]] = []
    best_path: Optional[Path] = None
    best_score: float = float("inf")

    for ckpt in find_checkpoints(checkpoints_dir):
        score = parse_val_wer(ckpt)
        if score is None:
            continue
        scored.append((ckpt.name, score))
        if score < best_score:
            best_score = score
            best_path = ckpt

    scored.sort(key=lambda x: x[1])
    return best_path, scored


def load_best_model(ckpt_path: Optional[Path], checkpoints_dir: Path) -> EncDecHybridRNNTCTCBPEModel:
    if ckpt_path is not None:
        return EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(str(ckpt_path))

    # Fallback to .nemo if no scored ckpts
    nemo_files = sorted(checkpoints_dir.glob("*.nemo"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not nemo_files:
        raise FileNotFoundError(f"No .ckpt or .nemo files found in {checkpoints_dir}")
    return EncDecHybridRNNTCTCBPEModel.restore_from(str(nemo_files[0]))


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root)
    run_dir = Path(args.exp_dir) if args.exp_dir else find_latest_run(exp_root)
    checkpoints_dir = run_dir / "checkpoints"

    print(f"Run directory: {run_dir}")
    print(f"Checkpoints dir: {checkpoints_dir}")

    best_ckpt, scored = pick_best_ckpt(checkpoints_dir)
    if scored:
        print("\nTop val_wer checkpoints:")
        for name, score in scored[:5]:
            print(f"  {score:.5f}  {name}")
    else:
        print("\nNo val_wer checkpoints found; will use latest .nemo file.")

    if best_ckpt:
        print(f"\nBest checkpoint: {best_ckpt.name}")

    model = load_best_model(best_ckpt, checkpoints_dir)

    test_manifest = Path(args.test_manifest)
    if not test_manifest.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")

    # Set test manifest with full config
    test_config = {
        "manifest_filepath": str(test_manifest),
        "sample_rate": 16000,
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
    }
    model.setup_test_data(test_data_config=test_config)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(devices=1, accelerator=device)
    print(f"\nRunning test on {device}...")

    if model.prepare_test(trainer):
        trainer.test(model)


if __name__ == "__main__":
    main()
