#!/usr/bin/env python3
"""Diagnostic script to identify WER > 1 root causes."""

import json
from pathlib import Path
from typing import Optional
import re

def check_manifest_integrity(manifest_path: Path, sample_size: int = 10) -> dict:
    """Check if manifest entries have valid paths and text."""
    print(f"\n{'='*80}")
    print(f"CHECKING MANIFEST: {manifest_path}")
    print(f"{'='*80}")
    
    issues = {
        "missing_audio": [],
        "empty_duration": [],
        "empty_text": [],
        "text_issues": [],
    }
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return issues
    
    with manifest_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total entries: {len(lines)}")
    
    for i, entry in enumerate(lines[:sample_size]):
        audio_path = entry.get("audio_filepath", "")
        text = entry.get("text", "")
        duration = entry.get("duration", 0)
        
        # Check audio path
        full_path = Path(audio_path)
        if not full_path.exists() and not Path("data/audio") / audio_path.lstrip("data/audio/"):
            issues["missing_audio"].append(f"Line {i}: {audio_path}")
        
        # Check duration
        if not duration or duration <= 0:
            issues["empty_duration"].append(f"Line {i}: duration={duration}")
        
        # Check text
        if not text or len(text.strip()) == 0:
            issues["empty_text"].append(f"Line {i}: empty text")
        
        # Check for text issues (diacritics, normalization)
        if text:
            has_diacritics = bool(re.search(r'[\u064B-\u0652]', text))  # Arabic diacritics
            has_extra_spaces = "  " in text  # Double spaces
            has_punctuation = bool(re.search(r'[،ؚ؛؟!.*\-_]', text))  # Arabic/special punctuation
            
            if has_diacritics or has_extra_spaces or has_punctuation:
                issues["text_issues"].append({
                    "line": i,
                    "text": text[:100],
                    "has_diacritics": has_diacritics,
                    "has_extra_spaces": has_extra_spaces,
                    "has_punctuation": has_punctuation,
                })
        
        # Print first few samples
        if i < min(3, sample_size):
            print(f"\n[Entry {i}]")
            print(f"  Audio: {audio_path}")
            print(f"  Duration: {duration}")
            print(f"  Text: {text}")
    
    # Print issues summary
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f"\n⚠️  {issue_type.upper()}: {len(issue_list)} instances")
            for issue in issue_list[:3]:
                print(f"   - {issue}")
            if len(issue_list) > 3:
                print(f"   ... and {len(issue_list) - 3} more")
    
    return issues


def check_tokenizer(tokenizer_dir: Path) -> dict:
    """Check tokenizer configuration."""
    print(f"\n{'='*80}")
    print(f"CHECKING TOKENIZER: {tokenizer_dir}")
    print(f"{'='*80}")
    
    info = {}
    
    # Check for model and vocab files
    model_file = tokenizer_dir / "tokenizer.model"
    vocab_file = tokenizer_dir / "tokenizer.vocab"
    
    if model_file.exists():
        info["model_file"] = f"✓ Found ({model_file.stat().st_size / 1024 / 1024:.2f} MB)"
    else:
        info["model_file"] = "✗ Missing"
    
    if vocab_file.exists():
        with vocab_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        info["vocab_size"] = len(lines)
        info["vocab_sample"] = "\n".join([f"  Token {i}: {line.strip()}" for i, line in enumerate(lines[:10])])
    else:
        info["vocab_file"] = "✗ Missing"
    
    # Check for BOS/EOS tokens
    if vocab_file.exists():
        with vocab_file.open("r", encoding="utf-8") as f:
            vocab_lines = f.readlines()
        
        bos_found = any("<bos>" in line or "▁<bos>" in line for line in vocab_lines)
        eos_found = any("<eos>" in line or "▁<eos>" in line for line in vocab_lines)
        
        info["has_bos"] = "✓" if bos_found else "✗"
        info["has_eos"] = "✓" if eos_found else "✗"
    
    for key, value in info.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    return info


def check_pretrained_compatibility() -> None:
    """Check if pretrained model is compatible."""
    print(f"\n{'='*80}")
    print(f"CHECKING PRETRAINED MODEL COMPATIBILITY")
    print(f"{'='*80}")
    
    print("\nKNOWN ISSUES:")
    print("1. Pretrained 'stt_en_fastconformer_transducer_large' is ENGLISH model")
    print("   - Uses English SentencePiece tokenizer (English vocab ~512-1024 tokens)")
    print("   - Fine-tuning on Arabic WITHOUT setting 'init_from_pretrained_model=null'")
    print("   - This means: Encoder weights are from English, decoder/tokenizer are Arabic")
    print("   - MISMATCH ALERT: The model learns but produces garbage because tokenizer differs")
    print("\n2. TEXT NORMALIZATION:")
    print("   - Arabic text with diacritics (شَدّة, فتحة, etc.) causes tokenizer mismatches")
    print("   - Pretrained model may not have been trained on diacritic-heavy text")
    print("\n3. SOLUTION:")
    print("   a) Option A: Train from scratch (init_from_pretrained_model=null)")
    print("   b) Option B: Use Arabic-specific pretrained model (if available)")
    print("   c) Option C: Remove diacritics from text (normalize to undiacritized form)")


def main() -> None:
    base_path = Path("/data/SAB_PhD/quranNemoASR")
    
    # Check manifests
    for manifest_file in ["train.json", "val.json", "test.json"]:
        manifest_path = base_path / "data" / "manifests" / manifest_file
        check_manifest_integrity(manifest_path, sample_size=5)
    
    # Check tokenizer
    tokenizer_path = base_path / "tokenizer" / "tokenizer_spe_bpe_v1024_bos_eos"
    check_tokenizer(tokenizer_path)
    
    # Check compatibility
    check_pretrained_compatibility()
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION:")
    print(f"{'='*80}")
    print("""
The high WER (>0.99) is likely caused by:

1. TOKENIZER MISMATCH (Most Likely):
   - Pretrained model (English) decoder -> Arabic tokenizer
   - Solution: Either train from scratch OR use fine-tuning approach that resets decoder

2. TEXT NORMALIZATION:
   - Arabic diacritics in training data
   - Solution: Remove diacritics before training or ensure preprocessor normalizes them

3. DATA ISSUES:
   - Manifest corruptions, empty fields, mismatched audio/text
   - Solution: Verify manifest integrity

IMMEDIATE ACTIONS:
1. Try setting init_from_pretrained_model=null to train from scratch
2. Or: Remove Arabic diacritics from text (normalize to undiacritized form)
3. Check if first validation epoch shows any learning signal
4. Monitor loss curves (rnnt_loss, ctc_loss, combined) to see if they decrease
    """)


if __name__ == "__main__":
    main()
