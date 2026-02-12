#!/usr/bin/env python3
"""
Plot training and validation results from NeMo training run.
Extracts metrics from TensorBoard logs and creates comprehensive visualizations.
"""

import argparse
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def find_latest_run(base_dir: Path) -> Path:
    """Find the most recent training run directory."""
    run_dirs = sorted(base_dir.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    
    latest = run_dirs[0]
    print(f"Using latest run: {latest.name}")
    return latest


def load_tensorboard_metrics(tb_dir: Path):
    """Load all metrics from TensorBoard event files."""
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {tb_dir}")
    
    print(f"Loading metrics from {len(event_files)} event file(s)...")
    
    # Use event accumulator to parse TensorBoard logs
    ea = event_accumulator.EventAccumulator(str(tb_dir))
    ea.Reload()
    
    # Get all available scalar metrics
    scalar_tags = ea.Tags().get('scalars', [])
    print(f"Found {len(scalar_tags)} scalar metrics")
    
    metrics = defaultdict(lambda: {'steps': [], 'values': []})
    
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        for event in events:
            metrics[tag]['steps'].append(event.step)
            metrics[tag]['values'].append(event.value)
    
    return metrics, scalar_tags


def plot_training_curves(metrics, output_dir: Path):
    """Create comprehensive training visualization plots."""
    
    # Define metric groups to plot
    metric_groups = {
        'WER': ['val_wer', 'train_wer'],
        'Loss': ['val_loss', 'train_loss', 'loss'],
        'Learning Rate': ['lr-AdamW'],
        'RNNT Loss': ['train_loss_rnnt', 'val_loss_rnnt'],
        'CTC Loss': ['train_loss_ctc', 'val_loss_ctc'],
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot each metric group
    plots_created = 0
    
    for group_name, metric_names in metric_groups.items():
        # Check which metrics exist
        available_metrics = [m for m in metric_names if m in metrics]
        
        if not available_metrics:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for metric_name in available_metrics:
            steps = np.array(metrics[metric_name]['steps'])
            values = np.array(metrics[metric_name]['values'])
            
            # Plot with markers for validation, solid line for training
            if 'val' in metric_name:
                ax.plot(steps, values, 'o-', label=metric_name, linewidth=2, markersize=6)
            else:
                ax.plot(steps, values, '-', label=metric_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel(group_name, fontsize=12)
        ax.set_title(f'Training Progress: {group_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        filename = f"{group_name.lower().replace(' ', '_')}.png"
        filepath = output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plots_created += 1
        plt.close()
    
    # Create comprehensive overview plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: WER comparison
    ax = axes[0, 0]
    for metric in ['val_wer', 'train_wer']:
        if metric in metrics:
            steps = np.array(metrics[metric]['steps'])
            values = np.array(metrics[metric]['values'])
            marker = 'o-' if 'val' in metric else '-'
            ax.plot(steps, values, marker, label=metric, linewidth=2, markersize=5)
    ax.set_xlabel('Step')
    ax.set_ylabel('WER')
    ax.set_title('Word Error Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss comparison
    ax = axes[0, 1]
    for metric in ['val_loss', 'train_loss', 'loss']:
        if metric in metrics:
            steps = np.array(metrics[metric]['steps'])
            values = np.array(metrics[metric]['values'])
            marker = 'o-' if 'val' in metric else '-'
            ax.plot(steps, values, marker, label=metric, linewidth=2, markersize=5, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RNNT vs CTC Loss
    ax = axes[1, 0]
    for metric in ['train_loss_rnnt', 'train_loss_ctc', 'val_loss_rnnt', 'val_loss_ctc']:
        if metric in metrics:
            steps = np.array(metrics[metric]['steps'])
            values = np.array(metrics[metric]['values'])
            marker = 'o-' if 'val' in metric else '-'
            label = metric.replace('train_', '').replace('_', ' ').upper()
            ax.plot(steps, values, marker, label=label, linewidth=2, markersize=5, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('RNNT vs CTC Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    ax = axes[1, 1]
    if 'lr-AdamW' in metrics:
        steps = np.array(metrics['lr-AdamW']['steps'])
        values = np.array(metrics['lr-AdamW']['values'])
        ax.plot(steps, values, '-', linewidth=2, color='red')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    overview_path = output_dir / 'training_overview.png'
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {overview_path}")
    plots_created += 1
    plt.close()
    
    return plots_created


def print_summary_statistics(metrics):
    """Print summary statistics from training."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    
    # Helper to get best value
    def get_best(metric_name, minimize=True):
        if metric_name not in metrics:
            return None
        values = np.array(metrics[metric_name]['values'])
        steps = np.array(metrics[metric_name]['steps'])
        if minimize:
            idx = np.argmin(values)
        else:
            idx = np.argmax(values)
        return values[idx], steps[idx]
    
    # Validation WER
    if 'val_wer' in metrics:
        best_wer, step = get_best('val_wer', minimize=True)
        final_wer = metrics['val_wer']['values'][-1]
        print(f"\nValidation WER:")
        print(f"  Best:  {best_wer:.4f} (at step {step})")
        print(f"  Final: {final_wer:.4f}")
    
    # Validation Loss
    if 'val_loss' in metrics:
        best_loss, step = get_best('val_loss', minimize=True)
        final_loss = metrics['val_loss']['values'][-1]
        print(f"\nValidation Loss:")
        print(f"  Best:  {best_loss:.4f} (at step {step})")
        print(f"  Final: {final_loss:.4f}")
    
    # Training Loss
    if 'train_loss' in metrics:
        final_train_loss = metrics['train_loss']['values'][-1]
        print(f"\nTraining Loss:")
        print(f"  Final: {final_train_loss:.4f}")
    elif 'loss' in metrics:
        final_train_loss = metrics['loss']['values'][-1]
        print(f"\nTraining Loss:")
        print(f"  Final: {final_train_loss:.4f}")
    
    # Learning rate
    if 'lr-AdamW' in metrics:
        final_lr = metrics['lr-AdamW']['values'][-1]
        print(f"\nLearning Rate:")
        print(f"  Final: {final_lr:.2e}")
    
    # Total steps
    max_steps = max(max(metrics[m]['steps']) for m in metrics if metrics[m]['steps'])
    print(f"\nTotal Training Steps: {max_steps}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot NeMo ASR training results')
    parser.add_argument(
        '--run-dir',
        type=str,
        help='Path to specific run directory (auto-detect latest if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='training_plots',
        help='Directory to save plots (default: training_plots)'
    )
    
    args = parser.parse_args()
    
    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        base_dir = Path('nemo_experiments/FastConformer-Hybrid-Transducer-CTC-BPE-Streaming')
        run_dir = find_latest_run(base_dir)
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    print(f"\nAnalyzing run: {run_dir}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Load metrics from TensorBoard
    tb_dir = run_dir
    metrics, tags = load_tensorboard_metrics(tb_dir)
    
    print(f"\nAvailable metrics: {', '.join(sorted(tags))}\n")
    
    # Print summary statistics
    print_summary_statistics(metrics)
    
    # Create plots
    output_dir = Path(args.output_dir)
    plots_created = plot_training_curves(metrics, output_dir)
    
    print(f"\n✓ Created {plots_created} visualization plots in {output_dir}/")
    print(f"✓ View training_overview.png for a comprehensive summary\n")


if __name__ == '__main__':
    main()
