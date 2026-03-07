#!/usr/bin/env python3
"""
Evaluate Quran Arabic Streaming ASR Model
Following the Hindi finetuning approach from salesken/Hindi-FastConformer-Streaming-ASR
"""

import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.utils.transcribe_utils import write_transcription
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from omegaconf import OmegaConf, open_dict
from jiwer import wer, cer
import json
from tqdm import tqdm

def main():
    # Configuration
    CHECKPOINT_PATH = './nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/epoch=49-step=45200.ckpt'
    TEST_MANIFEST = './data/manifests/test.json'
    OUTPUT_DIR = './nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints'
    OUTPUT_FILE = f'{OUTPUT_DIR}/quran_streaming_eval_results.jsonl'
    
    print("=" * 80)
    print("QURAN ARABIC STREAMING ASR EVALUATION")
    print("Based on Reference: salesken/Hindi-FastConformer-Streaming-ASR")
    print("=" * 80)
    
    # Load model
    print(f"\n📂 Loading checkpoint...")
    print(f"   Path: {CHECKPOINT_PATH.split('/')[-1]}")
    model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location='cpu'
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    model = model.to(device)
    model.eval()
    
    # Verify configuration
    print(f"\n🔍 Model Configuration:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Decoder: {model.cur_decoder}")
    print(f"   Vocab size: {model.tokenizer.vocab_size}")
    print(f"   Encoder att_context_size (active): {model.encoder.att_context_size}")
    print(f"   Config att_context_size: {model.cfg.encoder.att_context_size}")
    
    # Use RNNT decoder (default, as trained)
    print(f"\n⚙️  Using RNNT decoder (default from training)")
    
    # Transcribe test set
    print(f"\n🎙️  Transcribing test set...")
    print(f"   Manifest: {TEST_MANIFEST}")
    
    # Transcribe
    print(f"\n   Processing audio files...")
    transcriptions = model.transcribe(
        audio=TEST_MANIFEST,
        batch_size=16
    )
    
    # Load references and compute metrics
    print(f"\n📊 Computing metrics...")
    references = []
    predictions = []
    results = []
    
    with open(TEST_MANIFEST, 'r', encoding='utf-8') as f:
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
    
    # Write results
    print(f"\n💾 Saving results...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"   Saved to: {OUTPUT_FILE}")
    
    # Print metrics
    print(f"\n" + "=" * 80)
    print("📈 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Samples:        {len(references)}")
    print(f"Word Error Rate (WER): {dataset_wer:.2f}%")
    print(f"Char Error Rate (CER): {dataset_cer:.2f}%")
    print(f"Exact matches:  {exact_matches}/{len(references)} ({exact_matches/len(references)*100:.2f}%)")
    print("=" * 80)
    
    # Show sample predictions
    print(f"\n📝 Sample Predictions (first 5):")
    for i in range(min(5, len(results))):
        print(f"\n[{i+1}] {results[i]['audio_filepath'].split('/')[-1]}")
        print(f"   REF:  {results[i]['text']}")
        print(f"   PRED: {results[i]['pred_text']}")
        match = "✓" if results[i]['text'] == results[i]['pred_text'] else "✗"
        print(f"   {match}")

if __name__ == '__main__':
    main()
