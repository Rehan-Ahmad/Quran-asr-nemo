"""
Comprehensive Streaming vs Non-Streaming Evaluation
Tests all 4 streaming context configurations against non-streaming model

Usage:
    python evaluate_streaming_contexts.py
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from jiwer import wer, cer
from tqdm import tqdm

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
import nemo.collections.asr as nemo_asr


# Configuration
STREAMING_CHECKPOINT = './nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/epoch=49-step=45200.ckpt'
# Note: This is also a Hybrid model, but uses full context (non-streaming)
NON_STREAMING_MODEL = './nemo_experiments/FastConformer-Custom-Tokenizer/2026-02-14_08-36-37/checkpoints/FastConformer-Custom-Tokenizer.nemo'
TEST_MANIFEST = './data/manifests/test.json'
OUTPUT_DIR = './streaming_context_evaluation'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# All streaming context configurations to test
STREAMING_CONTEXTS = [
    ([70, 13], "Best Accuracy"),
    ([70, 6], "Balanced"),
    ([70, 1], "Low Latency"),
    ([70, 0], "Causal Only")
]


def load_test_data(manifest_path: str) -> List[Dict]:
    """Load test data from manifest"""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def evaluate_streaming_context(
    model: EncDecHybridRNNTCTCBPEModel,
    context: List[int],
    context_name: str,
    test_samples: List[Dict]
) -> Dict:
    """Evaluate streaming model with specific context configuration"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating Streaming: {context} - {context_name}")
    print(f"{'='*60}")
    
    # Set context
    model.encoder.att_context_size = context
    print(f"✓ Context set: {model.encoder.att_context_size}")
    
    results = []
    total_wer = 0
    total_cer = 0
    exact_matches = 0
    total_time = 0
    
    # Evaluate
    for sample in tqdm(test_samples, desc=f"Streaming {context}"):
        audio_path = sample['audio_filepath']
        reference = sample['text']
        
        # Transcribe
        start_time = time.time()
        with torch.no_grad():
            prediction = model.transcribe(audio=[audio_path], batch_size=1)[0]
        inference_time = time.time() - start_time
        
        # Extract text
        pred_text = prediction.text if hasattr(prediction, 'text') else str(prediction)
        
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
    
    num_samples = len(test_samples)
    avg_wer = (total_wer / num_samples) * 100
    avg_cer = (total_cer / num_samples) * 100
    exact_match_rate = (exact_matches / num_samples) * 100
    avg_time = total_time / num_samples
    throughput = num_samples / total_time
    
    summary = {
        'context': context,
        'context_name': context_name,
        'model_type': 'Streaming (Hybrid RNNT+CTC)',
        'num_samples': num_samples,
        'wer': avg_wer,
        'cer': avg_cer,
        'exact_matches': exact_matches,
        'exact_match_rate': exact_match_rate,
        'total_time': total_time,
        'avg_time_per_sample': avg_time,
        'throughput': throughput
    }
    
    print(f"\n📊 Results for {context_name} {context}:")
    print(f"   WER: {avg_wer:.2f}%")
    print(f"   CER: {avg_cer:.2f}%")
    print(f"   Exact Match: {exact_matches}/{num_samples} ({exact_match_rate:.2f}%)")
    print(f"   Avg Time: {avg_time:.3f}s")
    print(f"   Throughput: {throughput:.2f} samples/s")
    
    return summary, results


def evaluate_non_streaming(
    model,
    test_samples: List[Dict]
) -> Dict:
    """Evaluate non-streaming CTC model"""
    
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
            # For Hybrid model, use same API as streaming
            dataloader_cfg = {
                'manifest_filepath': sample,  # Pass dict or path
                'batch_size': 1,
                'sample_rate': 16000
            }
            hypothesis = model.transcribe([audio_path], batch_size=1)[0]
        inference_time = time.time() - start_time
        
        # Extract text (CTC model returns string directly)
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
    
    num_samples = len(test_samples)
    avg_wer = (total_wer / num_samples) * 100
    avg_cer = (total_cer / num_samples) * 100
    exact_match_rate = (exact_matches / num_samples) * 100
    avg_time = total_time / num_samples
    throughput = num_samples / total_time
    
    summary = {
        'context': 'Full Context',
        'context_name': 'Full Context',
        'model_type': 'Non-Streaming (CTC)',
        'num_samples': num_samples,
        'wer': avg_wer,
        'cer': avg_cer,
        'exact_matches': exact_matches,
        'exact_match_rate': exact_match_rate,
        'total_time': total_time,
        'avg_time_per_sample': avg_time,
        'throughput': throughput
    }
    
    print(f"\n📊 Results for Non-Streaming CTC:")
    print(f"   WER: {avg_wer:.2f}%")
    print(f"   CER: {avg_cer:.2f}%")
    print(f"   Exact Match: {exact_matches}/{num_samples} ({exact_match_rate:.2f}%)")
    print(f"   Avg Time: {avg_time:.3f}s")
    print(f"   Throughput: {throughput:.2f} samples/s")
    
    return summary, results


def print_comparison_table(all_summaries: List[Dict]):
    """Print comprehensive comparison table"""
    
    print("\n" + "="*100)
    print(" " * 30 + "COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*100)
    
    # Header
    print(f"\n{'Configuration':<30} {'WER %':<10} {'CER %':<10} {'Exact %':<10} {'Avg Time':<12} {'Throughput':<12}")
    print("-" * 100)
    
    # Streaming contexts
    for summary in all_summaries:
        if 'Streaming' in summary['model_type']:
            context_str = f"{summary['context']} - {summary['context_name']}"
            print(f"{context_str:<30} {summary['wer']:<10.2f} {summary['cer']:<10.2f} "
                  f"{summary['exact_match_rate']:<10.2f} {summary['avg_time_per_sample']:<12.3f} "
                  f"{summary['throughput']:<12.2f}")
    
    print("-" * 100)
    
    # Non-streaming
    for summary in all_summaries:
        if 'Non-Streaming' in summary['model_type']:
            print(f"{'Non-Streaming (CTC)':<30} {summary['wer']:<10.2f} {summary['cer']:<10.2f} "
                  f"{summary['exact_match_rate']:<10.2f} {summary['avg_time_per_sample']:<12.3f} "
                  f"{summary['throughput']:<12.2f}")
    
    print("="*100)
    
    # Find best performers
    print("\n🏆 BEST PERFORMERS:")
    
    best_wer = min(all_summaries, key=lambda x: x['wer'])
    print(f"   Lowest WER: {best_wer['context_name']} ({best_wer['wer']:.2f}%)")
    
    best_cer = min(all_summaries, key=lambda x: x['cer'])
    print(f"   Lowest CER: {best_cer['context_name']} ({best_cer['cer']:.2f}%)")
    
    best_exact = max(all_summaries, key=lambda x: x['exact_match_rate'])
    print(f"   Best Exact Match: {best_exact['context_name']} ({best_exact['exact_match_rate']:.2f}%)")
    
    fastest = min(all_summaries, key=lambda x: x['avg_time_per_sample'])
    print(f"   Fastest: {fastest['context_name']} ({fastest['avg_time_per_sample']:.3f}s/sample)")
    
    print("\n📈 STREAMING CONTEXT TRENDS:")
    streaming_summaries = [s for s in all_summaries if 'Streaming' in s['model_type']]
    if len(streaming_summaries) > 1:
        print(f"   As right context decreases [13→6→1→0]:")
        wer_trend = "increases" if streaming_summaries[-1]['wer'] > streaming_summaries[0]['wer'] else "decreases"
        speed_trend = "increases" if streaming_summaries[-1]['avg_time_per_sample'] < streaming_summaries[0]['avg_time_per_sample'] else "decreases"
        print(f"     - WER {wer_trend}")
        print(f"     - Speed {speed_trend}")


def main():
    """Main evaluation pipeline"""
    
    print("="*80)
    print(" " * 20 + "STREAMING CONTEXT EVALUATION")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Test Manifest: {TEST_MANIFEST}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    
    # Load test data
    print("\n[1/3] Loading test data...")
    test_samples = load_test_data(TEST_MANIFEST)
    print(f"   ✓ Loaded {len(test_samples)} test samples")
    
    # Load streaming model
    print("\n[2/3] Loading streaming model (Hybrid RNNT+CTC)...")
    print(f"   Path: {STREAMING_CHECKPOINT.split('/')[-1]}")
    streaming_model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(STREAMING_CHECKPOINT)
    streaming_model = streaming_model.to(DEVICE)
    streaming_model.eval()
    print(f"   ✓ Model loaded")
    print(f"   ✓ Available contexts: {streaming_model.cfg.encoder.att_context_size}")
    
    # Load non-streaming model
    print("\n[3/3] Loading non-streaming model (Hybrid with Full Context)...")
    print(f"   Path: {NON_STREAMING_MODEL.split('/')[-1]}")
    non_streaming_model = EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=NON_STREAMING_MODEL)
    non_streaming_model = non_streaming_model.to(DEVICE)
    non_streaming_model.eval()
    print(f"   ✓ Model loaded")
    
    # Evaluate all configurations
    all_summaries = []
    all_detailed_results = {}
    
    # Test all streaming contexts
    for context, context_name in STREAMING_CONTEXTS:
        summary, results = evaluate_streaming_context(
            streaming_model,
            context,
            context_name,
            test_samples
        )
        all_summaries.append(summary)
        
        # Save detailed results
        output_file = Path(OUTPUT_DIR) / f"streaming_{context[0]}_{context[1]}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"   ✓ Saved: {output_file}")
        
        all_detailed_results[f"Streaming {context}"] = results
    
    # Test non-streaming
    summary, results = evaluate_non_streaming(non_streaming_model, test_samples)
    all_summaries.append(summary)
    
    # Save non-streaming results
    output_file = Path(OUTPUT_DIR) / "non_streaming_ctc.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"   ✓ Saved: {output_file}")
    
    all_detailed_results["Non-Streaming CTC"] = results
    
    # Print comparison table
    print_comparison_table(all_summaries)
    
    # Save summary
    summary_file = Path(OUTPUT_DIR) / "evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summaries': all_summaries,
            'test_samples': len(test_samples),
            'device': DEVICE,
            'streaming_model': STREAMING_CHECKPOINT,
            'non_streaming_model': NON_STREAMING_MODEL
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Summary saved: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - Detailed results: streaming_*.jsonl, non_streaming_ctc.jsonl")
    print(f"  - Summary: evaluation_summary.json")


if __name__ == "__main__":
    main()
