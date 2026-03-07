#!/usr/bin/env python3
"""
Convert NeMo checkpoint (.ckpt) to NeMo package (.nemo)

This script properly converts PyTorch Lightning checkpoints to deployable
NeMo packages. The conversion preserves model configuration, weights, and
tokenizer information.

Usage:
    python convert_ckpt_to_nemo.py --ckpt <path_to_ckpt> --output <output_path> [--model-class <class>]

Example:
    python convert_ckpt_to_nemo.py \
        --ckpt ./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/epoch=49-step=45200.ckpt \
        --output ./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/FastConformer-English-Quran-Tokenizer-streaming.nemo \
        --model-class EncDecHybridRNNTCTCBPEModel
"""

import argparse
import sys
from pathlib import Path
from typing import Type

import torch
from nemo.collections.asr.models import (
    EncDecHybridRNNTCTCBPEModel,
    EncDecCTCModelBPE,
    EncDecRNNTBPEModel,
)


# Supported model classes
MODEL_CLASSES = {
    'EncDecHybridRNNTCTCBPEModel': EncDecHybridRNNTCTCBPEModel,
    'EncDecCTCModelBPE': EncDecCTCModelBPE,
    'EncDecRNNTBPEModel': EncDecRNNTBPEModel,
}


def get_model_class_from_ckpt(ckpt_path: str) -> Type:
    """
    Attempt to infer model class from checkpoint metadata.
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Returns:
        Inferred model class or default to EncDecHybridRNNTCTCBPEModel
    """
    print(f"Attempting to infer model class from checkpoint...")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Check for model class hints in checkpoint
        if 'pytorch-lightning_version' in ckpt:
            print(f"  PyTorch Lightning checkpoint detected")
        
        if 'state_dict' in ckpt:
            state_keys = list(ckpt['state_dict'].keys())
            print(f"  Found {len(state_keys)} state dict keys")
            
            # Look for RNNT decoder presence to distinguish hybrid vs pure CTC
            has_rnnt = any('joint' in k or 'rnnt' in k.lower() for k in state_keys)
            if has_rnnt:
                print(f"  Detected RNNT components (hybrid model)")
                return EncDecHybridRNNTCTCBPEModel
            else:
                print(f"  No RNNT detected, assuming CTC-only")
                return EncDecCTCModelBPE
    except Exception as e:
        print(f"  Warning: Could not inspect checkpoint: {e}")
    
    print(f"  Defaulting to EncDecHybridRNNTCTCBPEModel")
    return EncDecHybridRNNTCTCBPEModel


def convert_ckpt_to_nemo(
    ckpt_path: str,
    output_path: str,
    model_class: Type = None,
    map_location: str = 'cpu'
) -> str:
    """
    Convert a PyTorch Lightning checkpoint to a NeMo package.
    
    Args:
        ckpt_path: Path to input .ckpt file
        output_path: Path for output .nemo file
        model_class: Model class to use (auto-detect if None)
        map_location: Device to load checkpoint on ('cpu' or 'cuda')
        
    Returns:
        Path to saved .nemo file
        
    Raises:
        FileNotFoundError: If checkpoint does not exist
        ValueError: If conversion fails
    """
    ckpt_path = Path(ckpt_path)
    output_path = Path(output_path)
    
    # Validate input
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT TO NEMO CONVERSION")
    print(f"{'='*80}")
    print(f"Input:  {ckpt_path}")
    print(f"Output: {output_path}")
    
    # Auto-detect model class if not provided
    if model_class is None:
        model_class = get_model_class_from_ckpt(str(ckpt_path))
    else:
        print(f"Using specified model class: {model_class.__name__}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading checkpoint...")
    print(f"  Device: {map_location}")
    
    try:
        # Load model from checkpoint
        model = model_class.load_from_checkpoint(
            str(ckpt_path),
            map_location=map_location
        )
        print(f"✓ Model loaded successfully")
        
        # Display model info
        if hasattr(model, 'cfg'):
            cfg = model.cfg
            print(f"\nModel Configuration:")
            if hasattr(cfg, 'encoder'):
                print(f"  Encoder:")
                print(f"    - d_model: {cfg.encoder.d_model}")
                print(f"    - num_layers: {cfg.encoder.num_layers}")
                print(f"    - num_heads: {cfg.encoder.num_heads}")
                if hasattr(cfg.encoder, 'att_context_size'):
                    print(f"    - att_context_size: {cfg.encoder.att_context_size}")
            
            if hasattr(cfg, 'tokenizer'):
                print(f"  Tokenizer: {cfg.tokenizer.dir}")
        
        # Get model size
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel Parameters: {total_params:,}")
        
        # Save to NeMo format
        print(f"\nSaving to NeMo format...")
        model.save_to(str(output_path))
        
        # Verify output
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Successfully saved: {output_path}")
            print(f"  File size: {size_mb:.1f} MB")
            
            print(f"\n{'='*80}")
            print(f"✅ CONVERSION COMPLETE")
            print(f"{'='*80}")
            print(f"You can now load the model with:")
            print(f"  from nemo.collections.asr.models import {model_class.__name__}")
            print(f"  model = {model_class.__name__}.restore_from('{output_path}')")
            
            return str(output_path)
        else:
            raise ValueError(f"Output file was not created: {output_path}")
            
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        raise


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Convert NeMo checkpoint (.ckpt) to NeMo package (.nemo)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--ckpt',
        required=True,
        help='Path to input .ckpt checkpoint file'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Path for output .nemo file'
    )
    
    parser.add_argument(
        '--model-class',
        choices=list(MODEL_CLASSES.keys()),
        default=None,
        help='Model class to use (auto-detect if not specified)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to load checkpoint on'
    )
    
    args = parser.parse_args()
    
    try:
        # Get model class
        model_class = None
        if args.model_class:
            model_class = MODEL_CLASSES[args.model_class]
        
        # Perform conversion
        output = convert_ckpt_to_nemo(
            ckpt_path=args.ckpt,
            output_path=args.output,
            model_class=model_class,
            map_location=args.device
        )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
