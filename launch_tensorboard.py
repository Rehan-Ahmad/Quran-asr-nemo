#!/usr/bin/env python3
"""
Launch TensorBoard for monitoring NeMo training experiments.
This script provides a simple interface to start TensorBoard with the correct log directory.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_tensorboard_installed():
    """Check if TensorBoard is installed."""
    try:
        import tensorboard
        return True
    except ImportError:
        return False


def check_setuptools_installed():
    """Check if setuptools is installed (required for TensorBoard 2.20+)."""
    try:
        import setuptools
        return True
    except ImportError:
        return False


def find_experiment_dirs():
    """Find available experiment directories."""
    experiments_root = Path("nemo_experiments")
    if not experiments_root.exists():
        return []
    
    # Get all subdirectories
    exp_dirs = [d for d in experiments_root.iterdir() if d.is_dir()]
    return exp_dirs


def launch_tensorboard(logdir: str, port: int = 6006, bind_all: bool = True):
    """
    Launch TensorBoard with the specified log directory.
    
    Args:
        logdir: Path to TensorBoard log directory
        port: Port number for TensorBoard server (default: 6006)
        bind_all: If True, bind to all network interfaces (default: True)
    """
    # Verify log directory exists
    if not os.path.exists(logdir):
        print(f"Error: Log directory not found: {logdir}")
        sys.exit(1)
    
    # Check dependencies
    if not check_tensorboard_installed():
        print("Error: TensorBoard is not installed.")
        print("Install it with: pip install tensorboard")
        sys.exit(1)
    
    if not check_setuptools_installed():
        print("Warning: setuptools not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "setuptools"], check=True)
    
    # Build command
    cmd = [
        "tensorboard",
        f"--logdir={logdir}",
        f"--port={port}"
    ]
    
    if bind_all:
        cmd.append("--bind_all")
    
    # Print info
    print(f"\n{'='*70}")
    print(f"Launching TensorBoard")
    print(f"{'='*70}")
    print(f"Log directory: {logdir}")
    print(f"Port: {port}")
    print(f"Bind to all interfaces: {bind_all}")
    print(f"{'='*70}\n")
    
    # Get hostname
    hostname = subprocess.check_output(["hostname"]).decode().strip()
    print(f"TensorBoard will be available at:")
    print(f"  - http://localhost:{port}/")
    print(f"  - http://{hostname}:{port}/")
    print(f"\nPress Ctrl+C to stop TensorBoard\n")
    print(f"{'='*70}\n")
    
    # Launch TensorBoard
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nTensorBoard stopped.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for NeMo training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with default settings (auto-detect experiment directory)
  python launch_tensorboard.py
  
  # Launch with custom log directory
  python launch_tensorboard.py --logdir nemo_experiments/MyExperiment
  
  # Launch on custom port
  python launch_tensorboard.py --port 6007
  
  # Launch on localhost only (not accessible from other machines)
  python launch_tensorboard.py --no-bind-all
        """
    )
    
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Path to TensorBoard log directory (default: auto-detect from nemo_experiments/)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port number for TensorBoard server (default: 6006)"
    )
    
    parser.add_argument(
        "--no-bind-all",
        action="store_true",
        help="Only bind to localhost (not accessible from other machines)"
    )
    
    args = parser.parse_args()
    
    # Determine log directory
    if args.logdir:
        logdir = args.logdir
    else:
        # Auto-detect experiment directory
        exp_dirs = find_experiment_dirs()
        
        if not exp_dirs:
            print("Error: No experiment directories found in nemo_experiments/")
            print("\nPlease specify a log directory with --logdir")
            sys.exit(1)
        
        if len(exp_dirs) == 1:
            logdir = str(exp_dirs[0])
            print(f"Auto-detected experiment directory: {logdir}\n")
        else:
            print("Multiple experiment directories found:")
            for i, exp_dir in enumerate(exp_dirs, 1):
                print(f"  {i}. {exp_dir}")
            
            # For now, just use the first one or prompt user
            logdir = str(exp_dirs[0])
            print(f"\nUsing: {logdir}")
            print("(Specify a different directory with --logdir)\n")
    
    # Launch TensorBoard
    launch_tensorboard(
        logdir=logdir,
        port=args.port,
        bind_all=not args.no_bind_all
    )


if __name__ == "__main__":
    main()
