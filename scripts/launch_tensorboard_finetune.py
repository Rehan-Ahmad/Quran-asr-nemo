#!/usr/bin/env python3
"""Launch TensorBoard pointed at the latest finetune run under `nemo_experiments`.

Usage:
  scripts/launch_tensorboard_finetune.py [--root PATH] [--arch NAME] [--port PORT] [--dry-run]

If --dry-run is set the script prints the discovered logdir and exits.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from shutil import which


def _parse_timestamp(name: str) -> datetime:
    for fmt in ('%Y-%m-%d_%H-%M-%S', '%Y-%m-%d_%H-%M-%S_%f'):
        try:
            return datetime.strptime(name, fmt)
        except Exception:
            continue
    return datetime.min


def find_latest_finetune_run(root: Path, arch: str | None = None) -> Path | None:
    root = root.expanduser()
    if not root.exists():
        return None
    candidates = []
    arch_dirs = [p for p in root.iterdir() if p.is_dir()]
    if arch:
        arch_dirs = [p for p in arch_dirs if p.name == arch]
    for a in arch_dirs:
        # Prefer nested `finetune/<timestamp>` or any timestamped child
        fin = a / 'finetune'
        if fin.exists() and fin.is_dir():
            for c in fin.iterdir():
                if c.is_dir():
                    candidates.append((c, _parse_timestamp(c.name)))
        # fallback: direct timestamped children under arch dir
        for c in a.iterdir():
            if c.is_dir() and c.name[:4].isdigit():
                candidates.append((c, _parse_timestamp(c.name)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='nemo_experiments', help='Base experiments dir')
    p.add_argument('--arch', default=None, help='Optional architecture folder to restrict search')
    p.add_argument('--port', type=int, default=6006, help='TensorBoard port')
    p.add_argument('--dry-run', action='store_true', help='Only print discovered logdir')
    p.add_argument('--tensorboard-bin', default=None, help='Override tensorboard binary')
    args = p.parse_args(argv)

    root = Path(args.root)
    run_dir = find_latest_finetune_run(root, arch=args.arch)
    if run_dir is None:
        print(f'No finetune runs found under: {root}', file=sys.stderr)
        return 2

    print(f'Selected run: {run_dir}')
    # Usually logs are under the run dir or its lightning_logs subdir
    tb_candidates = [run_dir, run_dir / 'lightning_logs', run_dir / 'lightning_logs' / 'version_0']
    logdir = None
    for c in tb_candidates:
        if c.exists():
            logdir = c
            break
    if logdir is None:
        # fallback to run_dir anyway
        logdir = run_dir

    print(f'TensorBoard logdir: {logdir}')
    if args.dry_run:
        return 0

    tb_bin = args.tensorboard_bin or which('tensorboard')
    if not tb_bin:
        print('tensorboard binary not found on PATH. Install with `pip install tensorboard`', file=sys.stderr)
        return 3

    cmd = [tb_bin, '--logdir', str(logdir), '--port', str(args.port), '--bind_all']
    print('Starting TensorBoard: ' + ' '.join(cmd))
    try:
        # Replace current process with tensorboard for convenience
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
