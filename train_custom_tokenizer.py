#!/usr/bin/env python3
"""
Train a custom SentencePiece BPE tokenizer on Quran dataset.
This tokenizer will include all Quranic diacritics and special characters.
"""

import json
import os
from pathlib import Path
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model


def extract_text_from_manifests(manifest_files, output_file):
    """Extract all text from manifest files into a single corpus file."""
    print(f"\nExtracting text from manifests...")
    
    texts = []
    total_chars = 0
    
    for manifest_file in manifest_files:
        if not os.path.exists(manifest_file):
            print(f"  Warning: {manifest_file} not found, skipping")
            continue
            
        print(f"  Reading: {manifest_file}")
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get('text', '').strip()
                if text:
                    texts.append(text)
                    total_chars += len(text)
    
    # Write to corpus file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))
    
    print(f"\n  Extracted {len(texts)} utterances")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Saved to: {output_file}")
    
    return len(texts), total_chars


def train_tokenizer(corpus_file, output_dir, vocab_size=1024):
    """Train SentencePiece tokenizer."""
    print("\nTraining BPE tokenizer...")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train tokenizer using NeMo's function
    create_spt_model(
        data_file=corpus_file,
        vocab_size=vocab_size,
        sample_size=-1,  # Use all data
        do_lower_case=False,  # Keep original case and diacritics
        output_dir=output_dir,
        character_coverage=1.0,  # Include all characters (important for diacritics!)
    )
    
    print(f"\n✓ Tokenizer trained successfully!")
    print(f"  Model files saved to: {output_dir}/")


def test_tokenizer(tokenizer_dir, test_texts):
    """Test the trained tokenizer on sample texts."""
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
    
    print(f"\nTesting tokenizer...")
    
    # Load tokenizer
    tokenizer = SentencePieceTokenizer(model_path=f"{tokenizer_dir}/tokenizer.model")
    
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"\nSample tokenization:")
    
    for i, text in enumerate(test_texts, 1):
        tokens = tokenizer.text_to_ids(text)
        decoded = tokenizer.ids_to_text(tokens)
        
        print(f"\n  {i}. Original:  {text}")
        print(f"     Tokens:    {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"     Decoded:   {decoded}")
        print(f"     Match:     {'✓' if text == decoded else '✗'}")


def main():
    """Main execution."""
    print("="*70)
    print("Training Custom SentencePiece Tokenizer for Quran ASR")
    print("="*70)
    
    # Configuration
    MANIFEST_DIR = "data/manifests"
    CORPUS_FILE = "tokenizer/quran_text_corpus.txt"
    TOKENIZER_DIR = "tokenizer/quran_tokenizer_bpe_v1024"
    VOCAB_SIZE = 1024
    # NeMo's create_spt_model uses BPE by default.
    
    # Manifest files to use for training
    manifest_files = [
        f"{MANIFEST_DIR}/train.json",
        f"{MANIFEST_DIR}/val.json",
        # Optionally include test.json (usually keep separate for evaluation)
        # f"{MANIFEST_DIR}/test.json",
    ]
    
    # Step 1: Extract text corpus
    print(f"\nStep 1: Extract text corpus")
    print("-"*70)
    num_texts, num_chars = extract_text_from_manifests(manifest_files, CORPUS_FILE)
    
    # Step 2: Train tokenizer
    print(f"\nStep 2: Train tokenizer")
    print("-"*70)
    train_tokenizer(CORPUS_FILE, TOKENIZER_DIR, VOCAB_SIZE)
    
    # Step 3: Test tokenizer
    print(f"\nStep 3: Test tokenizer")
    print("-"*70)
    
    # Use texts with Quranic diacritics to verify they're handled correctly
    test_texts = [
        "وَلَقَدْ فَتَنَّا ٱلَّذِينَ مِن قَبْلِهِمْ فَلَيَعْلَمَنَّ ٱللَّهُ ٱلَّذِينَ صَدَقُوا۟ وَلَيَعْلَمَنَّ ٱلْكَٰذِبِينَ",
        "فِيهِمَا عَيْنَانِ نَضَّاخَتَانِ",
        "وَإِنَّهُ لَتَنزِيلُ رَبِّ الْعَالَمِينَ",
    ]
    test_tokenizer(TOKENIZER_DIR, test_texts)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Tokenizer Training Complete!")
    print(f"{'='*70}")
    print(f"\nTokenizer location: {TOKENIZER_DIR}/")
    print(f"Files created:")
    print(f"  - tokenizer.model (SentencePiece model)")
    print(f"  - tokenizer.vocab (vocabulary file)")
    print(f"  - vocab.txt (NeMo format vocabulary)")
    print(f"\nNext steps:")
    print(f"  1. Update finetune config to use: {TOKENIZER_DIR}")
    print(f"  2. Run fine-tuning: bash train_nemo_finetune.sh")
    print(f"  3. Evaluate and compare WER/CER with custom tokenizer")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
