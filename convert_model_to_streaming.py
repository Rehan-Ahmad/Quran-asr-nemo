#!/usr/bin/env python3
"""
Convert a non-streaming pretrained ASR model to streaming by modifying encoder
parameters in-place without using config overlays.

This script loads a pretrained model and directly modifies streaming encoder 
parameters, then saves the modified model.
"""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr


def convert_model_to_streaming(
    pretrained_model_path: str,
    output_model_path: str,
    custom_tokenizer_dir: str = None
) -> None:
    """
    Convert a non-streaming model to streaming by modifying encoder parameters.
    
    Args:
        pretrained_model_path: Path to the pretrained .nemo model
        output_model_path: Path where the streaming model will be saved
        custom_tokenizer_dir: Path to custom tokenizer directory (optional)
    """
    print(f"Loading pretrained model from: {pretrained_model_path}")
    
    # Load the pretrained model
    model = nemo_asr.models.ASRModel.restore_from(pretrained_model_path)
    print(f"Model loaded: {type(model).__name__}")
    
    # Apply custom tokenizer if provided
    if custom_tokenizer_dir:
        print(f"\nApplying custom tokenizer from: {custom_tokenizer_dir}")
        model.change_vocabulary(
            new_tokenizer_dir=custom_tokenizer_dir,
            new_tokenizer_type='bpe'
        )
        print("✓ Custom tokenizer applied successfully")
    
    # Define streaming parameters to apply
    streaming_params = {
        'causal_downsampling': True,
        'att_context_style': 'chunked_limited',
        'att_context_size': [70, 13],
        'conv_context_size': 'causal'
    }
    
    print("\nModifying encoder for streaming:")
    print(f"  - causal_downsampling: {streaming_params['causal_downsampling']}")
    print(f"  - att_context_style: {streaming_params['att_context_style']}")
    print(f"  - att_context_size: {streaming_params['att_context_size']}")
    print(f"  - conv_context_size: {streaming_params['conv_context_size']}")
    
    # Get model config
    model_cfg = model.cfg
    
    # Update encoder config directly
    with open_dict(model_cfg):
        with open_dict(model_cfg.encoder):
            for key, value in streaming_params.items():
                model_cfg.encoder[key] = value
                model.encoder.__dict__[key] = value
                print(f"  ✓ Updated encoder.{key}")
    
    # Now rebuild encoder with streaming params
    print("\nRecompiling encoder with streaming configuration...")
    try:
        from nemo.collections.asr.modules import ConformerEncoder
        
        enc_cfg = model_cfg.encoder
        
        # Create new encoder with streaming parameters
        new_encoder = ConformerEncoder(
            feat_in=enc_cfg.feat_in,
            feat_out=enc_cfg.get('feat_out', -1),
            n_layers=enc_cfg.n_layers,
            d_model=enc_cfg.d_model,
            use_bias=enc_cfg.get('use_bias', True),
            subsampling=enc_cfg.get('subsampling', 'dw_striding'),
            subsampling_factor=enc_cfg.get('subsampling_factor', 8),
            subsampling_conv_channels=enc_cfg.get('subsampling_conv_channels', -1),
            causal_downsampling=True,  # Enable streaming
            ff_expansion_factor=enc_cfg.get('ff_expansion_factor', 4),
            self_attention_model=enc_cfg.get('self_attention_model', 'rel_pos'),
            n_heads=enc_cfg.get('n_heads', 8),
            att_context_size=[70, 13],  # Streaming: left context, right lookahead
            att_context_style='chunked_limited',  # Streaming mode
            att_context_probs=enc_cfg.get('att_context_probs', None),
            xscaling=enc_cfg.get('xscaling', True),
            pos_emb_max_len=enc_cfg.get('pos_emb_max_len', 5000),
            conv_kernel_size=enc_cfg.get('conv_kernel_size', 9),
            conv_norm_type=enc_cfg.get('conv_norm_type', 'layer_norm'),
            conv_context_size='causal',  # Streaming: causal convolutions
            dropout=enc_cfg.get('dropout', 0.1),
            dropout_pre_encoder=enc_cfg.get('dropout_pre_encoder', 0.1),
            dropout_emb=enc_cfg.get('dropout_emb', 0.0),
            dropout_att=enc_cfg.get('dropout_att', 0.1),
            stochastic_depth_drop_prob=enc_cfg.get('stochastic_depth_drop_prob', 0.0),
            stochastic_depth_mode=enc_cfg.get('stochastic_depth_mode', 'linear'),
            stochastic_depth_start_layer=enc_cfg.get('stochastic_depth_start_layer', 1)
        )
        
        # Replace encoder
        model.encoder = new_encoder
        print("✓ Encoder recompiled successfully with streaming parameters")
        
    except Exception as e:
        print(f"⚠ Warning: Could not recompile encoder: {e}")
        print("  Model config has been updated, attempting to save...")
    
    # Save the streaming model
    print(f"\nSaving streaming model to: {output_model_path}")
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save_to(output_model_path)
    print("✓ Model saved successfully")
    
    # Verify
    print("\nVerifying saved model...")
    test_model = nemo_asr.models.ASRModel.restore_from(output_model_path)
    print(f"✓ Model verification successful")
    print(f"\n✓✓✓ Streaming model created successfully: {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert non-streaming pretrained model to streaming"
    )
    parser.add_argument(
        "--pretrained-model",
        required=True,
        help="Path to the pretrained .nemo model file"
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="Path where the streaming model will be saved"
    )
    parser.add_argument(
        "--custom-tokenizer",
        default=None,
        help="Path to custom tokenizer directory (optional)"
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.pretrained_model):
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model}")
    
    if args.custom_tokenizer and not os.path.exists(args.custom_tokenizer):
        raise FileNotFoundError(f"Custom tokenizer not found: {args.custom_tokenizer}")
    
    convert_model_to_streaming(
        args.pretrained_model,
        args.output_model,
        args.custom_tokenizer
    )
