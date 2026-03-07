#!/usr/bin/env python3
"""
Evaluate Quran Arabic Hybrid Model with Both RNNT and CTC Decoders
Compare performance between the two decoding strategies
"""

import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from jiwer import wer, cer
import json
from tqdm import tqdm

def evaluate_with_decoder(model, manifest, decoder_type='rnnt'):
    """Evaluate model with specified decoder"""
    print(f"\n{'='*80}")
    print(f"EVALUATING WITH {decoder_type.upper()} DECODER")
    print(f"{'='*80}")
    
    # Switch decoder if needed
    if decoder_type == 'ctc':
        ctc_cfg = CTCDecodingConfig()
        model.change_decoding_strategy(ctc_cfg, decoder_type='ctc')
        print(f"✓ Switched to CTC decoder")
    else:
        print(f"✓ Using RNNT decoder (default)")
    
    print(f"Current decoder: {model.cur_decoder}")
    
    # Transcribe
    print(f"\nTranscribing test set...")
    transcriptions = model.transcribe(
        audio=manifest,
        batch_size=16
    )
    
    # Load references and compute metrics
    print(f"Computing metrics...")
    references = []
    predictions = []
    results = []
    
    with open(manifest, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            ref_text = sample['text'].strip()
            # Extract text from Hypothesis object
            pred_text = transcriptions[i].text if hasattr(transcriptions[i], 'text') else str(transcriptions[i]).strip()
            
            references.append(ref_text)
            predictions.append(pred_text)
            
            # Store result
            results.append({
                'audio_filepath': sample['audio_filepath'],
                'duration': sample.get('duration', 0),
                'text': ref_text,
                'pred_text': pred_text
            })
    
    # Compute WER and CER
    dataset_wer = wer(references, predictions) * 100
    dataset_cer = cer(references, predictions) * 100
    
    # Count exact matches
    exact_matches = sum(1 for r, p in zip(references, predictions) if r == p)
    
    # Calculate metrics
    metrics = {
        'decoder': decoder_type.upper(),
        'samples': len(references),
        'wer': dataset_wer,
        'cer': dataset_cer,
        'exact_matches': exact_matches,
        'exact_match_rate': (exact_matches / len(references)) * 100
    }
    
    return metrics, results

def main():
    # Configuration
    CHECKPOINT_PATH = './nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/epoch=49-step=45200.ckpt'
    TEST_MANIFEST = './data/manifests/test.json'
    OUTPUT_DIR = './nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints'
    
    print("=" * 80)
    print("HYBRID MODEL DECODER COMPARISON")
    print("Comparing RNNT vs CTC Decoders")
    print("=" * 80)
    
    # Load model
    print(f"\n📂 Loading checkpoint...")
    model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location='cpu'
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    model = model.to(device)
    model.eval()
    
    # Model info
    print(f"\n🔍 Model Configuration:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Vocab size: {model.tokenizer.vocab_size}")
    print(f"   Encoder att_context_size: {model.encoder.att_context_size}")
    print(f"   RNNT Decoder: {hasattr(model, 'decoder')}")
    print(f"   CTC Decoder: {hasattr(model, 'ctc_decoder')}")
    
    # Evaluate with both decoders
    all_results = {}
    
    # 1. RNNT Decoder
    rnnt_metrics, rnnt_results = evaluate_with_decoder(model, TEST_MANIFEST, 'rnnt')
    all_results['rnnt'] = rnnt_results
    
    # Save RNNT results
    rnnt_output = f'{OUTPUT_DIR}/hybrid_eval_rnnt.jsonl'
    with open(rnnt_output, 'w', encoding='utf-8') as f:
        for result in rnnt_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"\n💾 RNNT results saved to: {rnnt_output}")
    
    # 2. CTC Decoder
    ctc_metrics, ctc_results = evaluate_with_decoder(model, TEST_MANIFEST, 'ctc')
    all_results['ctc'] = ctc_results
    
    # Save CTC results
    ctc_output = f'{OUTPUT_DIR}/hybrid_eval_ctc.jsonl'
    with open(ctc_output, 'w', encoding='utf-8') as f:
        for result in ctc_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"\n💾 CTC results saved to: {ctc_output}")
    
    # Compare results
    print(f"\n{'='*80}")
    print("📊 DECODER COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<25} {'RNNT':<15} {'CTC':<15} {'Winner':<10}")
    print("-" * 80)
    print(f"{'Samples':<25} {rnnt_metrics['samples']:<15} {ctc_metrics['samples']:<15} {'-':<10}")
    print(f"{'Word Error Rate (WER)':<25} {rnnt_metrics['wer']:<15.2f} {ctc_metrics['cer']:<15.2f} {'RNNT' if rnnt_metrics['wer'] < ctc_metrics['wer'] else 'CTC':<10}")
    print(f"{'Char Error Rate (CER)':<25} {rnnt_metrics['cer']:<15.2f} {ctc_metrics['cer']:<15.2f} {'RNNT' if rnnt_metrics['cer'] < ctc_metrics['cer'] else 'CTC':<10}")
    print(f"{'Exact Matches':<25} {rnnt_metrics['exact_matches']:<15} {ctc_metrics['exact_matches']:<15} {'RNNT' if rnnt_metrics['exact_matches'] > ctc_metrics['exact_matches'] else 'CTC':<10}")
    print(f"{'Exact Match Rate %':<25} {rnnt_metrics['exact_match_rate']:<15.2f} {ctc_metrics['exact_match_rate']:<15.2f} {'RNNT' if rnnt_metrics['exact_match_rate'] > ctc_metrics['exact_match_rate'] else 'CTC':<10}")
    
    # Determine overall winner
    rnnt_score = (rnnt_metrics['wer'] < ctc_metrics['wer']) + (rnnt_metrics['cer'] < ctc_metrics['cer']) + (rnnt_metrics['exact_match_rate'] > ctc_metrics['exact_match_rate'])
    ctc_score = (ctc_metrics['wer'] < rnnt_metrics['wer']) + (ctc_metrics['cer'] < rnnt_metrics['cer']) + (ctc_metrics['exact_match_rate'] > rnnt_metrics['exact_match_rate'])
    
    print(f"\n{'='*80}")
    if rnnt_score > ctc_score:
        print(f"🏆 OVERALL WINNER: RNNT Decoder ({rnnt_score}/3 metrics)")
        print(f"   Recommendation: Use RNNT decoder for this model")
    elif ctc_score > rnnt_score:
        print(f"🏆 OVERALL WINNER: CTC Decoder ({ctc_score}/3 metrics)")
        print(f"   Recommendation: Use CTC decoder for this model")
    else:
        print(f"🤝 TIE: Both decoders perform similarly")
    print(f"{'='*80}")
    
    # Show comparison samples
    print(f"\n📝 SAMPLE COMPARISON (First 5):")
    for i in range(min(5, len(rnnt_results))):
        print(f"\n[Sample {i+1}] {rnnt_results[i]['audio_filepath'].split('/')[-1]}")
        print(f"   REF:       {rnnt_results[i]['text']}")
        print(f"   RNNT PRED: {rnnt_results[i]['pred_text']}")
        print(f"   CTC PRED:  {ctc_results[i]['pred_text']}")
        
        rnnt_match = "✓" if rnnt_results[i]['text'] == rnnt_results[i]['pred_text'] else "✗"
        ctc_match = "✓" if ctc_results[i]['text'] == ctc_results[i]['pred_text'] else "✗"
        print(f"   RNNT: {rnnt_match}  |  CTC: {ctc_match}")
        
        if rnnt_match != ctc_match:
            print(f"   ⚠️  Decoders disagree!")

if __name__ == '__main__':
    main()
