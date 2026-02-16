#!/usr/bin/env python3
"""
Evaluate a trained NeMo ASR model on test data.
This script loads a .nemo checkpoint and computes WER/CER metrics on the test set.
"""

import os
import json
import re
from pathlib import Path
import nemo.collections.asr as nemo_asr


def find_best_checkpoint(experiment_dir: str) -> str:
    """Find the best checkpoint file by parsing val_wer from filenames."""
    experiment_path = Path(experiment_dir)
    
    # Find all checkpoint files
    nemo_files = list(experiment_path.rglob("checkpoints/*.nemo"))
    ckpt_files = list(experiment_path.rglob("checkpoints/*.ckpt"))
    
    if not nemo_files and not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {experiment_dir}")
    
    # Prefer .nemo files (these are the packaged models for inference)
    if nemo_files:
        # Sort by modification time and get the most recent .nemo
        # (NeMo saves the best model as .nemo at the end of training)
        nemo_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        best_checkpoint = nemo_files[0]
        print(f"\nUsing most recent .nemo checkpoint (best model)")
    else:
        # If no .nemo files, parse .ckpt files for lowest val_wer
        print(f"\nNo .nemo files found, searching .ckpt files for best val_wer")
        
        best_checkpoint = None
        best_wer = float('inf')
        
        for ckpt_file in ckpt_files:
            # Skip epoch=0 checkpoints (validation not run yet)
            if 'epoch=0' in ckpt_file.name or 'epoch=00' in ckpt_file.name:
                continue
                
            # Look for pattern: val_wer=0.1234
            match = re.search(r'val_wer=([\d.]+)', ckpt_file.name)
            if match:
                wer_value = float(match.group(1))
                if wer_value < best_wer and wer_value > 0:  # Exclude 0.0000 which is invalid
                    best_wer = wer_value
                    best_checkpoint = ckpt_file
        
        if best_checkpoint is None:
            # Fall back to most recent .ckpt
            ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            best_checkpoint = ckpt_files[0]
            print(f"Note: Using most recent .ckpt file")
        else:
            print(f"Best .ckpt checkpoint: val_wer={best_wer:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Selected checkpoint: {best_checkpoint.relative_to(experiment_path.parent.parent)}")
    print(f"{'='*70}\n")
    
    return str(best_checkpoint)


def evaluate_model(
    checkpoint_path: str, 
    test_manifest: str, 
    output_file: str = None,
    decoder_type: str = None,
    att_context_size: list = None,
    auto_detect: bool = True
):
    """
    Load model from checkpoint and evaluate on test data.
    
    Args:
        checkpoint_path: Path to .nemo checkpoint file
        test_manifest: Path to test manifest JSON file
        output_file: Optional path to save predictions JSON
        decoder_type: Decoder type for hybrid models ('rnnt' or 'ctc'). If None and auto_detect=True, will be auto-detected
        att_context_size: Attention context size for streaming models, e.g., [140, 27]. If None and auto_detect=True, will be auto-detected
        auto_detect: If True, automatically detect streaming vs non-streaming from model config (default: True)
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Restore model from checkpoint
    asr_model = nemo_asr.models.ASRModel.restore_from(checkpoint_path)
    asr_model.eval()
    
    # Auto-detect streaming configuration from model.cfg if not explicitly specified
    if auto_detect:
        print(f"\n{'='*70}")
        print(f"Auto-detecting model configuration from checkpoint...")
        
        # Check for streaming encoder configuration
        has_encoder_cfg = hasattr(asr_model.cfg, 'encoder')
        is_streaming = False
        detected_att_context_size = None
        
        if has_encoder_cfg:
            encoder_cfg = asr_model.cfg.encoder
            # Check if encoder has att_context_size configured
            if hasattr(encoder_cfg, 'att_context_size') and encoder_cfg.att_context_size:
                detected_att_context_size = encoder_cfg.att_context_size
                is_streaming = True
        
        # Auto-detect decoder type from model class
        detected_decoder_type = None
        model_class_name = type(asr_model).__name__
        
        if 'EncDecHybridRNNTCTC' in model_class_name:
            # For hybrid models, check config for preferred decoder
            if hasattr(asr_model.cfg, 'decoder') and hasattr(asr_model.cfg.decoder, 'pred_type'):
                detected_decoder_type = asr_model.cfg.decoder.pred_type
            else:
                # Default to RNNT for hybrid models if not specified
                detected_decoder_type = 'rnnt'
        elif 'EncDecRNNT' in model_class_name:
            detected_decoder_type = 'rnnt'
        elif 'EncDecCTC' in model_class_name:
            detected_decoder_type = 'ctc'
        
        # Use detected values if not explicitly provided
        if decoder_type is None and detected_decoder_type:
            decoder_type = detected_decoder_type
        if att_context_size is None and detected_att_context_size:
            att_context_size = detected_att_context_size
        
        # Print detection results
        print(f"  Model class: {model_class_name}")
        print(f"  Streaming: {'Yes' if is_streaming else 'No'}")
        if detected_decoder_type:
            print(f"  Decoder type detected: {detected_decoder_type}")
        if detected_att_context_size:
            print(f"  Attention context size detected: {detected_att_context_size}")
        print(f"{'='*70}\n")
    
    # Apply decoder type and streaming configuration (CRITICAL FOR HYBRID/STREAMING MODELS)
    if decoder_type and hasattr(asr_model, 'change_decoding_strategy'):
        try:
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
            from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
            
            print(f"\n{'='*70}")
            print(f"Applying decoding configuration:")
            print(f"  decoder_type: {decoder_type}")
            
            if decoder_type == 'rnnt':
                decoding_cfg = RNNTDecodingConfig()
            elif decoder_type == 'ctc':
                decoding_cfg = CTCDecodingConfig()
            else:
                raise ValueError(f"Unknown decoder_type: {decoder_type}. Must be 'rnnt' or 'ctc'.")
            
            asr_model.change_decoding_strategy(decoding_cfg, decoder_type=decoder_type)
            print(f"  ✓ Decoding strategy changed to {decoder_type.upper()}")
        except Exception as e:
            print(f"  Warning: Could not change decoding strategy: {e}")
    
    # Apply streaming attention context size if specified
    if att_context_size and hasattr(asr_model.encoder, 'set_default_att_context_size'):
        try:
            print(f"  att_context_size: {att_context_size}")
            asr_model.encoder.set_default_att_context_size(att_context_size)
            print(f"  ✓ Applied attention context size for streaming")
        except Exception as e:
            print(f"  Warning: Could not set attention context size: {e}")
    
    print(f"{'='*70}\n")
    
    print(f"Model architecture: {type(asr_model).__name__}")
    print(f"Vocabulary size: {asr_model.decoder.vocab_size if hasattr(asr_model.decoder, 'vocab_size') else 'N/A'}")
    
    # Verify test manifest exists
    if not os.path.exists(test_manifest):
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")
    
    print(f"\nRunning evaluation on: {test_manifest}")
    print(f"{'='*70}\n")
    
    # Load ground truth from manifest and extract audio file paths
    with open(test_manifest, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    audio_files = [sample['audio_filepath'] for sample in test_data]
    
    print(f"Transcribing {len(audio_files)} audio files...")
    
    # Transcribe test data
    # Note: num_workers=0 to avoid multiprocessing issues during evaluation
    hypotheses = asr_model.transcribe(
        audio_files,
        batch_size=32,
        return_hypotheses=True,
        num_workers=0,
        verbose=True
    )
    
    # Calculate metrics
    if isinstance(hypotheses, tuple):
        predictions, alignments = hypotheses
    else:
        predictions = hypotheses
    
    # Compute WER and CER using NeMo's built-in metrics
    from nemo.collections.asr.metrics.wer import word_error_rate
    
    # Prepare references and hypotheses lists
    references = [sample['text'] for sample in test_data]
    
    # Extract text from Hypothesis objects (use .text attribute)
    hypothesis_texts = []
    for pred in predictions:
        if hasattr(pred, 'text'):
            hypothesis_texts.append(pred.text)
        elif isinstance(pred, dict):
            hypothesis_texts.append(pred['text'])
        elif isinstance(pred, str):
            hypothesis_texts.append(pred)
        else:
            hypothesis_texts.append(str(pred))
    
    print(f"Computing metrics for {len(test_data)} samples...")
    
    # Calculate overall WER
    overall_wer = word_error_rate(hypothesis_texts, references)
    
    # Calculate overall CER (character-level)
    # Remove spaces and treat as character sequence
    references_chars = [ref.replace(' ', '') for ref in references]
    hypothesis_chars = [hyp.replace(' ', '') for hyp in hypothesis_texts]
    overall_cer = word_error_rate(hypothesis_chars, references_chars, use_cer=True)
    
    # Calculate per-sample metrics for detailed results
    results = []
    for sample, ref, hyp in zip(test_data, references, hypothesis_texts):
        sample_wer = word_error_rate([hyp], [ref])
        sample_cer = word_error_rate([hyp.replace(' ', '')], [ref.replace(' ', '')], use_cer=True)
        
        results.append({
            'audio_filepath': sample['audio_filepath'],
            'reference': ref,
            'prediction': hyp,
            'wer': sample_wer,
            'cer': sample_cer
        })
    
    # Use overall metrics
    avg_wer = overall_wer
    avg_cer = overall_cer
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Test samples: {len(predictions)}")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # Save predictions if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'checkpoint': checkpoint_path,
                'test_manifest': test_manifest,
                'avg_wer': avg_wer,
                'avg_cer': avg_cer,
                'num_samples': len(predictions),
                'predictions': results
            }, f, ensure_ascii=False, indent=2)
        print(f"Detailed results saved to: {output_file}\n")
    
    # Show some example predictions
    print(f"Sample predictions (first 5):")
    print(f"{'-'*70}")
    for i, result in enumerate(results[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Reference:  {result['reference']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  WER: {result['wer']:.4f}, CER: {result['cer']:.4f}")
    print(f"\n{'-'*70}\n")


def main():
    # Configuration
    EXPERIMENTS_ROOT = "nemo_experiments"
    TEST_MANIFEST = "data/manifests/test.json"
    OUTPUT_FILE = "evaluation_results.json"
    
    # ===== OPTIONAL: Override auto-detection =====
    # If you want to manually specify decoder type or streaming context,
    # uncomment the lines below. Otherwise, they will be auto-detected.
    # DECODER_TYPE = 'rnnt'           # e.g., 'rnnt' or 'ctc'
    # ATT_CONTEXT_SIZE = [140, 27]    # e.g., [140, 27] or None
    DECODER_TYPE = None              # None = auto-detect
    ATT_CONTEXT_SIZE = None          # None = auto-detect
    # =============================================
    
    # List available models
    if not os.path.exists(EXPERIMENTS_ROOT):
        raise FileNotFoundError(f"Experiments directory not found: {EXPERIMENTS_ROOT}")
    
    models = [d for d in os.listdir(EXPERIMENTS_ROOT)
              if os.path.isdir(os.path.join(EXPERIMENTS_ROOT, d))]
    models.sort()
    
    if not models:
        raise FileNotFoundError(f"No models found in {EXPERIMENTS_ROOT}")
    
    print(f"\nAvailable models in {EXPERIMENTS_ROOT}:")
    for i, model in enumerate(models):
        print(f"  {i}: {model}")
    
    # Auto-select via environment variables (for automation)
    env_model_name = os.getenv("EVAL_MODEL_NAME")
    env_model_index = os.getenv("EVAL_MODEL_INDEX")
    
    if env_model_name:
        if env_model_name not in models:
            raise ValueError(f"EVAL_MODEL_NAME not found: {env_model_name}")
        selected_model = env_model_name
    elif env_model_index is not None:
        try:
            selection = int(env_model_index)
        except ValueError as exc:
            raise ValueError("EVAL_MODEL_INDEX must be an integer") from exc
        if selection < 0 or selection >= len(models):
            raise ValueError(f"EVAL_MODEL_INDEX out of range: {selection}")
        selected_model = models[selection]
    else:
        # Get user selection
        while True:
            try:
                selection = int(input(f"\nSelect model (0-{len(models)-1}): "))
                if 0 <= selection < len(models):
                    break
                print(f"Invalid selection. Please enter 0-{len(models)-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")
        selected_model = models[selection]
    EXPERIMENT_DIR = os.path.join(EXPERIMENTS_ROOT, selected_model)
    
    print(f"\nSelected: {selected_model}")
    
    # Find best checkpoint
    checkpoint_path = find_best_checkpoint(EXPERIMENT_DIR)
    
    # Evaluate with auto-detection enabled
    print(f"\n[INFO] Auto-detection enabled: streaming & decoder type will be detected from checkpoint")
    print()
    
    evaluate_model(
        checkpoint_path, 
        TEST_MANIFEST, 
        OUTPUT_FILE,
        decoder_type=DECODER_TYPE,
        att_context_size=ATT_CONTEXT_SIZE,
        auto_detect=True  # Enable auto-detection
    )


if __name__ == "__main__":
    main()
