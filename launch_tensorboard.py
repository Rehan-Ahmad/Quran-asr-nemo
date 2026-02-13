#!/usr/bin/env python3
"""Launch TensorBoard for NeMo training logs."""

import argparse
import socket
import subprocess
from pathlib import Path


def find_free_port(default_port: int) -> int:
    """Return default_port if free, otherwise find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", default_port))
            return default_port
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch TensorBoard for NeMo runs")
    parser.add_argument(
        "--logdir",
        type=str,
        default="nemo_experiments",
        help="Path to NeMo experiment logs (default: nemo_experiments)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Preferred TensorBoard port (default: 6006)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        raise FileNotFoundError(f"Log directory not found: {logdir}")

    port = find_free_port(args.port)

    cmd = [
        "tensorboard",
        f"--logdir={logdir}",
        f"--host={args.host}",
        f"--port={port}",
    ]

    print("Launching TensorBoard...")
    print(f"Logdir: {logdir}")
    print(f"URL: http://{args.host}:{port}")
    print("Press Ctrl+C to stop.")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
