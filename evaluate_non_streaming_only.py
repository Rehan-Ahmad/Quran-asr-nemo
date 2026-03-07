"""
Quick script to evaluate non-streaming model and combine with existing streaming results
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, List
from jiwer import wer, cer
from tqdm import tqdm

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel


# Configuration
NON_STREAMING_MODEL = './nemo_experiments/FastConformer-Custom-Tokenizer/2026-02-14_08-36-37/checkpoints/FastConformer-Custom-Tokenizer.nemo'
TEST_MANIFEST = './data/manifests/test.json'
OUTPUT_DIR = './streaming_context_evaluation'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_test_data(manifest_path: str) -> List[Dict]:
    """Load test data from manifest"""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def evaluate_non_streaming(model, test_samples: List[Dict]):
    """Evaluate non-streaming model"""
    print(f"\n{'='*60}")
    print(f"Evaluating Non-Streaming (Full Context) Model")
    print(f"{'='*60}")
    
    results = []
    total_wer = 0
    total_cer = 0
    exact_matches = 0
    total_time = 0
    
    # Evaluate
    for sample in tqdm(test_samples, desc="Non-Streaming"):
        audio_path = sample['audio_filepath']
        reference = sample['text']
        
        # Transcribe
        start_time = time.time()
        with torch.no_grad():
            hypothesis = model.transcribe([audio_path], batch_size=1)[0]
        inference_time = time.time() - start_time
        
        # Extract text
        pred_text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
        
        # Calculate metrics
        sample_wer = wer(reference, pred_text)
        sample_cer = cer(reference, pred_text)
        exact_match = reference == pred_text
        
        total_wer += sample_wer
        total_cer += sample_cer
        if exact_match:
            exact_matches += 1
        total_time += inference_time
        
        results.append({
            'audio': audio_path,
            'reference': reference,
            'prediction': pred_text,
            'wer': sample_wer * 100,
            'cer': sample_cer * 100,
            'exact_match': exact_match,
            'inference_time': inference_time
        })
    
    # Calculate averages
    num_samples = len(test_samples)
    avg_wer = (total_wer / num_samples) * 100
    avg_cer = (total_cer / num_samples) * 100
    exact_match_rate = (exact_matches / num_samples) * 100
    avg_time = total_time / num_samples
    throughput = num_samples / total_time
    
    summary = {
        'context': 'Full Context (Non-Streaming)',
        'wer': avg_wer,
        'cer': avg_cer,
        'exact_match_count': exact_matches,
        'exact_match_rate': exact_match_rate,
        'avg_inference_time': avg_time,
        'throughput': throughput,
        'total_samples': num_samples
    }
    
    # Print results
    print(f"\n📊 Results for Non-Streaming (Full Context):")
    print(f"   WER: {avg_wer:.2f}%")
    print(f"   CER: {avg_cer:.2f}%")
    print(f"   Exact Match: {exact_matches}/{num_samples} ({exact_match_rate:.2f}%)")
    print(f"   Avg Time: {avg_time:.3f}s")
    print(f"   Throughput: {throughput:.2f} samples/s")
    
    # Save results
    output_path = Path(OUTPUT_DIR) / 'non_streaming_full_context.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"   ✓ Saved: {output_path}")
    
    return summary, results


def combine_all_results():
    """Load all streaming results and combine with non-streaming"""
    output_dir = Path(OUTPUT_DIR)
    
    # Load all existing results
    all_summaries = []
    
    # Load streaming results
    streaming_files = [
        ('streaming_70_13.jsonl', 'Streaming [70, 13] (Best Accuracy)'),
        ('streaming_70_6.jsonl', 'Streaming [70, 6] (Balanced)'),
        ('streaming_70_1.jsonl', 'Streaming [70, 1] (Low Latency)'),
        ('streaming_70_0.jsonl', 'Streaming [70, 0] (Causal Only)')
    ]
    
    for filename, label in streaming_files:
        filepath = output_dir / filename
        if filepath.exists():
            # Calculate summary from detailed results
            with open(filepath, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f]
            
            total_samples = len(results)
            avg_wer = sum(r['wer'] for r in results) / total_samples
            avg_cer = sum(r['cer'] for r in results) / total_samples
            exact_matches = sum(1 for r in results if r['exact_match'])
            exact_match_rate = (exact_matches / total_samples) * 100
            avg_time = sum(r['inference_time'] for r in results) / total_samples
            throughput = total_samples / sum(r['inference_time'] for r in results)
            
            all_summaries.append({
                'context': label,
                'wer': avg_wer,
                'cer': avg_cer,
                'exact_match_count': exact_matches,
                'exact_match_rate': exact_match_rate,
                'avg_inference_time': avg_time,
                'throughput': throughput,
                'total_samples': total_samples
            })
    
    # Load non-streaming result
    non_streaming_file = output_dir / 'non_streaming_full_context.jsonl'
    if non_streaming_file.exists():
        with open(non_streaming_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        
        total_samples = len(results)
        avg_wer = sum(r['wer'] for r in results) / total_samples
        avg_cer = sum(r['cer'] for r in results) / total_samples
        exact_matches = sum(1 for r in results if r['exact_match'])
        exact_match_rate = (exact_matches / total_samples) * 100
        avg_time = sum(r['inference_time'] for r in results) / total_samples
        throughput = total_samples / sum(r['inference_time'] for r in results)
        
        all_summaries.append({
            'context': 'Non-Streaming (Full Context)',
            'wer': avg_wer,
            'cer': avg_cer,
            'exact_match_count': exact_matches,
            'exact_match_rate': exact_match_rate,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'total_samples': total_samples
        })
    
    # Save combined summary
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    
    # Print comparison table
    print_comparison_table(all_summaries)
    
    return all_summaries


def print_comparison_table(summaries: List[Dict]):
    """Print comprehensive comparison table"""
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*100}\n")
    
    # Header
    print(f"{'Configuration':<40} {'WER %':<10} {'CER %':<10} {'Exact Match':<15} {'Avg Time':<12} {'Throughput':<12}")
    print(f"{'-'*40} {'-'*10} {'-'*10} {'-'*15} {'-'*12} {'-'*12}")
    
    # Data rows
    for summary in summaries:
        context = summary['context']
        wer_val = summary['wer']
        cer_val = summary['cer']
        exact_rate = summary['exact_match_rate']
        avg_time = summary['avg_inference_time']
        throughput = summary['throughput']
        
        print(f"{context:<40} {wer_val:>8.2f}% {cer_val:>9.2f}% {exact_rate:>12.2f}% {avg_time:>10.3f}s {throughput:>11.2f}/s")
    
    # Find best performers
    print(f"\n{'='*100}")
    print(f"BEST PERFORMERS")
    print(f"{'='*100}")
    
    best_wer = min(summaries, key=lambda x: x['wer'])
    best_cer = min(summaries, key=lambda x: x['cer'])
    best_exact = max(summaries, key=lambda x: x['exact_match_rate'])
    fastest = min(summaries, key=lambda x: x['avg_inference_time'])
    highest_throughput = max(summaries, key=lambda x: x['throughput'])
    
    print(f"🏆 Best WER: {best_wer['context']} ({best_wer['wer']:.2f}%)")
    print(f"🏆 Best CER: {best_cer['context']} ({best_cer['cer']:.2f}%)")
    print(f"🏆 Best Exact Match: {best_exact['context']} ({best_exact['exact_match_rate']:.2f}%)")
    print(f"⚡ Fastest: {fastest['context']} ({fastest['avg_inference_time']:.3f}s per sample)")
    print(f"⚡ Highest Throughput: {highest_throughput['context']} ({highest_throughput['throughput']:.2f} samples/s)")
    print(f"{'='*100}\n")


def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print("="*60)
    print("Non-Streaming Model Evaluation")
    print("="*60)
    
    # Load test data
    print("\n[1/2] Loading test data...")
    test_samples = load_test_data(TEST_MANIFEST)
    print(f"   ✓ Loaded {len(test_samples)} test samples")
    
    # Load non-streaming model
    print("\n[2/2] Loading non-streaming model (Hybrid with Full Context)...")
    print(f"   Path: {NON_STREAMING_MODEL.split('/')[-1]}")
    non_streaming_model = EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=NON_STREAMING_MODEL)
    non_streaming_model = non_streaming_model.to(DEVICE)
    non_streaming_model.eval()
    print(f"   ✓ Model loaded")
    
    # Evaluate
    evaluate_non_streaming(non_streaming_model, test_samples)
    
    # Combine all results and print comparison table
    print("\n" + "="*60)
    print("Combining All Results")
    print("="*60)
    combine_all_results()


if __name__ == '__main__':
    main()
