#!/usr/bin/env python3
"""
Evaluate the impact of streaming conversion by measuring key metrics:
- Latency (algorithm latency + E2E latency)
- Real-time factor (RTF)
- Memory consumption
- Accuracy (WER, CER)
- Throughput
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse

import nemo.collections.asr as nemo_asr
from nemo.utils import logging
from jiwer import wer, cer


class StreamingImpactEvaluator:
    """Evaluate streaming model metrics vs non-streaming baseline."""
    
    def __init__(self, streaming_model_path, baseline_model_path=None, test_manifest_path=None):
        """
        Args:
            streaming_model_path: Path to streaming-converted model
            baseline_model_path: Path to original non-streaming model (optional)
            test_manifest_path: Path to test manifest JSON
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
        
        # Load models
        logging.info(f"Loading streaming model from: {streaming_model_path}")
        self.streaming_model = nemo_asr.models.ASRModel.restore_from(streaming_model_path)
        self.streaming_model = self.streaming_model.eval()
        self.streaming_model = self.streaming_model.to(self.device)
        
        self.baseline_model = None
        if baseline_model_path and os.path.exists(baseline_model_path):
            logging.info(f"Loading baseline model from: {baseline_model_path}")
            self.baseline_model = nemo_asr.models.ASRModel.restore_from(baseline_model_path)
            self.baseline_model = self.baseline_model.eval()
            self.baseline_model = self.baseline_model.to(self.device)
        
        # Load test data
        self.test_samples = []
        if test_manifest_path and os.path.exists(test_manifest_path):
            self._load_test_manifest(test_manifest_path)
        
        logging.info(f"Device: {self.device}")
        logging.info(f"Streaming model: {type(self.streaming_model).__name__}")
        if self.baseline_model:
            logging.info(f"Baseline model: {type(self.baseline_model).__name__}")
    
    def _load_test_manifest(self, manifest_path, max_samples=10):
        """Load test samples from manifest."""
        import json
        count = 0
        with open(manifest_path) as f:
            for line in f:
                if count >= max_samples:
                    break
                data = json.loads(line)
                audio_path = data.get('audio_filepath')
                text = data.get('text', '')
                
                if audio_path and text and os.path.exists(audio_path):
                    self.test_samples.append({'audio': audio_path, 'text': text})
                    count += 1
        
        logging.info(f"Loaded {len(self.test_samples)} test samples")
    
    def measure_latency(self, audio_duration_sec=5.0):
        """
        Measure algorithm latency and E2E latency.
        
        Algorithm latency: Time before first output is available
        E2E latency: Total processing time
        """
        logging.info("\n" + "="*70)
        logging.info("LATENCY MEASUREMENT")
        logging.info("="*70)
        
        # Create synthetic audio
        sample_rate = self.streaming_model.cfg.sample_rate
        audio = torch.randn(int(audio_duration_sec * sample_rate))
        
        # Warm up
        with torch.no_grad():
            _ = self.streaming_model.transcribe([audio.cpu().numpy()])
        
        # Measure streaming model
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start = time.perf_counter()
        
        with torch.no_grad():
            streaming_pred = self.streaming_model.transcribe([audio.cpu().numpy()])
        
        torch.cuda.synchronize() if self.device == 'cuda' else None
        streaming_time = time.perf_counter() - start
        
        # Algorithm latency for streaming = chunk_processing_time
        chunk_duration = 26e-3  # Based on your att_context_size and subsampling
        
        self.results['latency'] = {
            'audio_duration_sec': audio_duration_sec,
            'streaming_e2e_latency_sec': streaming_time,
            'estimated_algorithm_latency_sec': chunk_duration,
            'streaming_speed': f"{audio_duration_sec / streaming_time:.1f}x real-time"
        }
        
        logging.info(f"\n✓ Streaming Model:")
        logging.info(f"  Audio duration: {audio_duration_sec:.2f}s")
        logging.info(f"  Total E2E latency: {streaming_time:.3f}s")
        logging.info(f"  Algorithm latency (chunk): ~{chunk_duration*1000:.1f}ms")
        logging.info(f"  Speed: {audio_duration_sec/streaming_time:.1f}x real-time")
        
        # Compare with baseline if available
        if self.baseline_model:
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.perf_counter()
            
            with torch.no_grad():
                baseline_pred = self.baseline_model.transcribe([audio.cpu().numpy()])
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            baseline_time = time.perf_counter() - start
            
            self.results['latency']['baseline_e2e_latency_sec'] = baseline_time
            self.results['latency']['baseline_speed'] = f"{audio_duration_sec / baseline_time:.1f}x"
            self.results['latency']['latency_improvement_factor'] = baseline_time / streaming_time
            
            logging.info(f"\n✓ Baseline Model:")
            logging.info(f"  Total E2E latency: {baseline_time:.3f}s")
            logging.info(f"  Speed: {audio_duration_sec/baseline_time:.1f}x real-time")
            logging.info(f"\n✓ Improvement: {baseline_time/streaming_time:.2f}x faster!")
    
    def measure_rtf(self):
        """
        Measure Real-Time Factor (RTF).
        RTF = Model processing time / Audio duration
        """
        logging.info("\n" + "="*70)
        logging.info("REAL-TIME FACTOR (RTF)")
        logging.info("="*70)
        
        if len(self.test_samples) == 0:
            logging.warning("No test samples loaded, skipping RTF measurement")
            return
        
        import librosa
        
        rtf_values = []
        
        for sample_idx, sample in enumerate(tqdm(self.test_samples, desc="Measuring RTF")):
            audio_path = sample['audio']
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.streaming_model.cfg.sample_rate)
            audio_duration = len(audio) / sr
            
            # Measure processing time
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.streaming_model.transcribe([audio])
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            processing_time = time.perf_counter() - start
            
            rtf = processing_time / audio_duration
            rtf_values.append(rtf)
        
        avg_rtf = np.mean(rtf_values)
        std_rtf = np.std(rtf_values)
        
        self.results['rtf'] = {
            'mean_rtf': float(avg_rtf),
            'std_rtf': float(std_rtf),
            'min_rtf': float(np.min(rtf_values)),
            'max_rtf': float(np.max(rtf_values)),
            'interpretation': 'RTF < 0.1 = 10x faster than real-time' if avg_rtf < 0.1 else \
                             'RTF 0.1-0.5 = Usable for streaming' if avg_rtf < 0.5 else \
                             'RTF > 1.0 = Cannot keep up with real-time (offline only)'
        }
        
        logging.info(f"\n✓ Real-Time Factor (RTF):")
        logging.info(f"  Mean RTF: {avg_rtf:.4f}")
        logging.info(f"  Std RTF: {std_rtf:.4f}")
        logging.info(f"  Min RTF: {np.min(rtf_values):.4f}")
        logging.info(f"  Max RTF: {np.max(rtf_values):.4f}")
        logging.info(f"  Interpretation: {self.results['rtf']['interpretation']}")
    
    def measure_memory(self):
        """Measure peak GPU memory consumption."""
        logging.info("\n" + "="*70)
        logging.info("MEMORY CONSUMPTION")
        logging.info("="*70)
        
        if len(self.test_samples) == 0:
            logging.warning("No test samples loaded, skipping memory measurement")
            return
        
        import librosa
        
        # Clear cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        
        # Measure streaming model
        sample = self.test_samples[0]
        audio_path = sample['audio']
        audio, sr = librosa.load(audio_path, sr=self.streaming_model.cfg.sample_rate)
        
        if self.device == 'cuda':
            torch.cuda.reset_max_memory_allocated()
            start_mem = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            _ = self.streaming_model.transcribe([audio])
        
        if self.device == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated()
            streaming_mem_mb = (peak_mem - start_mem) / 1024 / 1024
        else:
            streaming_mem_mb = "N/A (CPU)"
        
        self.results['memory'] = {
            'streaming_model_peak_mb': streaming_mem_mb,
            'device': self.device
        }
        
        logging.info(f"\n✓ Streaming Model:")
        logging.info(f"  Peak memory: {streaming_mem_mb} MB" if isinstance(streaming_mem_mb, str) else f"  Peak memory: {streaming_mem_mb:.1f} MB")
        
        # Compare with baseline
        if self.baseline_model:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
                start_mem = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                _ = self.baseline_model.transcribe([audio])
            
            if self.device == 'cuda':
                peak_mem = torch.cuda.max_memory_allocated()
                baseline_mem_mb = (peak_mem - start_mem) / 1024 / 1024
                
                self.results['memory']['baseline_model_peak_mb'] = baseline_mem_mb
                self.results['memory']['memory_reduction_pct'] = (1 - streaming_mem_mb/baseline_mem_mb) * 100 if baseline_mem_mb > 0 else 0
                
                logging.info(f"\n✓ Baseline Model:")
                logging.info(f"  Peak memory: {baseline_mem_mb:.1f} MB")
                logging.info(f"  Memory reduction: {self.results['memory']['memory_reduction_pct']:.1f}%")
    
    def measure_accuracy(self):
        """Measure WER and CER on test set."""
        logging.info("\n" + "="*70)
        logging.info("ACCURACY METRICS (WER/CER)")
        logging.info("="*70)
        
        if len(self.test_samples) == 0:
            logging.warning("No test samples loaded, skipping accuracy measurement")
            return
        
        import librosa
        
        streaming_wer_scores = []
        streaming_cer_scores = []
        baseline_wer_scores = []
        baseline_cer_scores = []
        
        for sample in tqdm(self.test_samples, desc="Measuring accuracy"):
            audio_path = sample['audio']
            reference_text = sample['text']
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.streaming_model.cfg.sample_rate)
            
            # Streaming prediction
            with torch.no_grad():
                streaming_pred = self.streaming_model.transcribe([audio])
            streaming_text = streaming_pred[0] if isinstance(streaming_pred, list) else str(streaming_pred)
            
            # Calculate metrics
            s_wer = wer(reference_text, streaming_text)
            s_cer = cer(reference_text, streaming_text)
            
            streaming_wer_scores.append(s_wer)
            streaming_cer_scores.append(s_cer)
            
            # Baseline prediction
            if self.baseline_model:
                with torch.no_grad():
                    baseline_pred = self.baseline_model.transcribe([audio])
                baseline_text = baseline_pred[0] if isinstance(baseline_pred, list) else str(baseline_pred)
                
                b_wer = wer(reference_text, baseline_text)
                b_cer = cer(reference_text, baseline_text)
                
                baseline_wer_scores.append(b_wer)
                baseline_cer_scores.append(b_cer)
        
        avg_streaming_wer = np.mean(streaming_wer_scores)
        avg_streaming_cer = np.mean(streaming_cer_scores)
        
        self.results['accuracy'] = {
            'streaming_wer': float(avg_streaming_wer),
            'streaming_cer': float(avg_streaming_cer),
        }
        
        logging.info(f"\n✓ Streaming Model:")
        logging.info(f"  Average WER: {avg_streaming_wer:.2f}%")
        logging.info(f"  Average CER: {avg_streaming_cer:.2f}%")
        
        if self.baseline_model:
            avg_baseline_wer = np.mean(baseline_wer_scores)
            avg_baseline_cer = np.mean(baseline_cer_scores)
            
            self.results['accuracy']['baseline_wer'] = float(avg_baseline_wer)
            self.results['accuracy']['baseline_cer'] = float(avg_baseline_cer)
            self.results['accuracy']['wer_degradation_pct'] = (avg_streaming_wer - avg_baseline_wer) / avg_baseline_wer * 100
            self.results['accuracy']['cer_degradation_pct'] = (avg_streaming_cer - avg_baseline_cer) / avg_baseline_cer * 100
            
            logging.info(f"\n✓ Baseline Model:")
            logging.info(f"  Average WER: {avg_baseline_wer:.2f}%")
            logging.info(f"  Average CER: {avg_baseline_cer:.2f}%")
            logging.info(f"\n✓ Accuracy Comparison:")
            logging.info(f"  WER degradation: {self.results['accuracy']['wer_degradation_pct']:.2f}%")
            logging.info(f"  CER degradation: {self.results['accuracy']['cer_degradation_pct']:.2f}%")
    
    def print_summary(self):
        """Print comprehensive summary."""
        logging.info("\n" + "="*70)
        logging.info("STREAMING CONVERSION IMPACT SUMMARY")
        logging.info("="*70)
        
        print("\n" + json.dumps(self.results, indent=2))
        
        logging.info("\n" + "="*70)
        logging.info("KEY TAKEAWAYS")
        logging.info("="*70)
        
        if 'latency' in self.results:
            latency = self.results['latency']
            logging.info(f"✓ Algorithm latency: ~{latency['estimated_algorithm_latency_sec']*1000:.1f}ms per chunk")
            logging.info(f"✓ E2E latency: {latency['streaming_e2e_latency_sec']:.3f}s for {latency['audio_duration_sec']:.1f}s audio")
            if 'latency_improvement_factor' in latency:
                logging.info(f"✓ Speed improvement: {latency['latency_improvement_factor']:.2f}x faster than baseline")
        
        if 'rtf' in self.results:
            rtf = self.results['rtf']
            logging.info(f"✓ Real-Time Factor: {rtf['mean_rtf']:.4f} ({rtf['interpretation']})")
        
        if 'memory' in self.results:
            mem = self.results['memory']
            logging.info(f"✓ Memory efficiency: {mem['streaming_model_peak_mb']}")
        
        if 'accuracy' in self.results:
            acc = self.results['accuracy']
            logging.info(f"✓ WER: {acc['streaming_wer']:.2f}%")
            if 'wer_degradation_pct' in acc:
                logging.info(f"  (Degradation: {acc['wer_degradation_pct']:.2f}% vs baseline)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate streaming conversion impact")
    parser.add_argument('--streaming-model', required=True, help='Path to streaming model')
    parser.add_argument('--baseline-model', help='Path to baseline non-streaming model')
    parser.add_argument('--test-manifest', help='Path to test manifest')
    
    args = parser.parse_args()
    
    evaluator = StreamingImpactEvaluator(
        streaming_model_path=args.streaming_model,
        baseline_model_path=args.baseline_model,
        test_manifest_path=args.test_manifest
    )
    
    evaluator.measure_latency()
    evaluator.measure_rtf()
    evaluator.measure_memory()
    evaluator.measure_accuracy()
    evaluator.print_summary()
    
    # Save results to JSON
    results_path = '/data/SAB_PhD/quranNemoASR/streaming_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    logging.info(f"\n✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
