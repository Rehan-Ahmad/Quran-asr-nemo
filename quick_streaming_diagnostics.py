#!/usr/bin/env python3
"""
Quick diagnostic script to measure streaming model metrics.
Run this after training completes to validate the streaming conversion.
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
from pathlib import Path

import nemo.collections.asr as nemo_asr
from nemo.utils import logging


def quick_metrics_check(model_path, audio_duration=5.0):
    """
    Quick check of streaming model metrics.
    """
    print("\n" + "="*70)
    print("STREAMING MODEL METRICS CHECK")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    
    # Load model
    print(f"\n→ Loading model: {Path(model_path).name}")
    model = nemo_asr.models.ASRModel.restore_from(model_path)
    model = model.eval()
    model = model.to(device)
    print(f"✓ Model type: {type(model).__name__}")
    
    # Check config
    print(f"\n→ Checking Encoder Config:")
    try:
        enc_cfg = model.cfg.encoder
        print(f"  ✓ causal_downsampling: {enc_cfg.get('causal_downsampling', False)}")
        print(f"  ✓ att_context_size: {enc_cfg.get('att_context_size', 'N/A')}")
        print(f"  ✓ att_context_style: {enc_cfg.get('att_context_style', 'N/A')}")
        print(f"  ✓ conv_context_size: {enc_cfg.get('conv_context_size', 'N/A')}")
        
        # Verify streaming config
        is_streaming = (
            enc_cfg.get('causal_downsampling', False) and
            enc_cfg.get('att_context_style') == 'chunked_limited'
        )
        
        if is_streaming:
            print(f"\n✓✓✓ MODEL IS CONFIGURED FOR STREAMING ✓✓✓")
        else:
            print(f"\n⚠ WARNING: Model may not be properly configured for streaming")
            
    except Exception as e:
        print(f"⚠ Could not fully check encoder config: {e}")
    
    # Warm up
    print(f"\n→ Warming up model...")
    sample_rate = model.cfg.sample_rate
    dummy_audio = torch.randn(int(0.5 * sample_rate))
    with torch.no_grad():
        _ = model.transcribe([dummy_audio.cpu().numpy()])
    print(f"✓ Warm-up complete")
    
    # Measure latency
    print(f"\n→ Measuring latency ({audio_duration}s synthetic audio):")
    audio = torch.randn(int(audio_duration * sample_rate))
    
    # Synchronize before timing
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        pred = model.transcribe([audio.cpu().numpy()])
    
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    rtf = elapsed / audio_duration
    
    print(f"  Audio duration: {audio_duration}s")
    print(f"  Processing time: {elapsed:.3f}s")
    print(f"  Real-Time Factor (RTF): {rtf:.4f}")
    print(f"  Speed: {audio_duration/elapsed:.1f}x real-time")
    
    # Assess RTF
    if rtf < 0.1:
        print(f"  ✓✓✓ Excellent! RTF < 0.1 (10x+ real-time)")
    elif rtf < 0.5:
        print(f"  ✓ Good! RTF < 0.5 (2-10x real-time)")
    elif rtf < 1.0:
        print(f"  ⚠ Marginal: RTF < 1.0 (barely real-time)")
    else:
        print(f"  ✗ Poor: RTF > 1.0 (cannot handle real-time)")
    
    # Estimated algorithm latency
    print(f"\n→ Estimated Algorithm Latency:")
    left_context = enc_cfg.get('att_context_size', [70, 13])[0]
    subsampling = enc_cfg.get('subsampling_factor', 8)
    window_stride = model.cfg.preprocessor.get('window_stride', 0.01)
    
    alg_latency_s = (left_context / subsampling) * window_stride
    alg_latency_ms = alg_latency_s * 1000
    
    print(f"  Left context frames: {left_context}")
    print(f"  Subsampling factor: {subsampling}")
    print(f"  Window stride: {window_stride}s")
    print(f"  → Algorithm latency: {alg_latency_ms:.1f}ms")
    
    if alg_latency_ms < 100:
        print(f"  ✓ Good algorithm latency (<100ms)")
    elif alg_latency_ms < 200:
        print(f"  ✓ Acceptable algorithm latency (<200ms)")
    else:
        print(f"  ⚠ High algorithm latency (>200ms)")
    
    # Memory measurement
    print(f"\n→ Measuring memory:")
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        
        # Get baseline
        baseline_mem = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"  Baseline GPU memory: {baseline_mem:.1f} MB")
        
        with torch.no_grad():
            _ = model.transcribe([audio.cpu().numpy()])
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        used_mem = peak_mem - baseline_mem
        
        print(f"  Peak GPU memory: {peak_mem:.1f} MB")
        print(f"  Used for inference: {used_mem:.1f} MB")
    else:
        print(f"  Running on CPU (memory measurement not available)")
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"✓ Model Type: {type(model).__name__}")
    print(f"✓ Streaming Enabled: {is_streaming}")
    print(f"✓ RTF: {rtf:.4f}")
    print(f"✓ Algorithm Latency: {alg_latency_ms:.1f}ms")
    print(f"✓ Speed: {audio_duration/elapsed:.1f}x real-time")
    
    if is_streaming and rtf < 0.1:
        print(f"\n✓✓✓ STREAMING MODEL IS READY FOR DEPLOYMENT ✓✓✓")
    elif is_streaming and rtf < 0.5:
        print(f"\n✓ Streaming model is functional (RTF < 0.5)")
    else:
        print(f"\n⚠ Review streaming configuration")
    
    print("\n" + "="*70)


def compare_checkpoints(checkpoint_dir):
    """
    Compare metrics across training checkpoints.
    """
    print("\n" + "="*70)
    print("CHECKPOINT COMPARISON")
    print("="*70)
    
    checkpoint_pattern = Path(checkpoint_dir).glob("*.nemo")
    checkpoints = sorted(checkpoint_pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    
    for cp in checkpoints[:3]:  # Show first 3
        print(f"\n→ {cp.name}")
        quick_metrics_check(str(cp), audio_duration=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick streaming model diagnostics")
    parser.add_argument('--model', default='pretrained_models/stt_ar_fastconformer_hybrid_large_streaming.nemo',
                       help='Path to streaming model')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Audio duration for latency test (seconds)')
    parser.add_argument('--compare-checkpoints', help='Compare metrics across checkpoint directory')
    
    args = parser.parse_args()
    
    if args.compare_checkpoints:
        compare_checkpoints(args.compare_checkpoints)
    else:
        quick_metrics_check(args.model, args.duration)
