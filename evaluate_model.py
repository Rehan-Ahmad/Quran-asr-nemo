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


def evaluate_model(checkpoint_path: str, test_manifest: str, output_file: str = None):
    """
    Load model from checkpoint and evaluate on test data.
    
    Args:
        checkpoint_path: Path to .nemo checkpoint file
        test_manifest: Path to test manifest JSON file
        output_file: Optional path to save predictions JSON
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Restore model from checkpoint
    asr_model = nemo_asr.models.ASRModel.restore_from(checkpoint_path)
    asr_model.eval()
    
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
    EXPERIMENT_DIR = "nemo_experiments/FastConformer-Hybrid-Transducer-CTC-BPE-Streaming"
    TEST_MANIFEST = "data/manifests/test.json"
    OUTPUT_FILE = "evaluation_results.json"
    
    # Find best checkpoint
    checkpoint_path = find_best_checkpoint(EXPERIMENT_DIR)
    
    # Evaluate
    evaluate_model(checkpoint_path, TEST_MANIFEST, OUTPUT_FILE)


if __name__ == "__main__":
    main()
