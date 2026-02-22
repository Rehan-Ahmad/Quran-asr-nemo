#!/usr/bin/env python3
"""Offline evaluation helper for streaming FastConformer models.

Usage example:
  . .venv/bin/activate
  python scripts/offline_streaming_eval.py \
    --nemo pretrained_models/stt_en_fastconformer_hybrid_large_streaming_multi.nemo \
    --ckpt nemo_experiments/.../checkpoints/epoch=24-step=22600.ckpt \
    --tokenizer tokenizer/quran_tokenizer_bpe_v1024 \
    --manifest data/manifests/test.json \
    --out_dir nemo_experiments/.../finetune/2026-02-18_15-04-25 \
    --batch_size 8

This script will:
 - load a base .nemo
 - (optionally) load a Lightning .ckpt state_dict into the model
 - (optionally) change the tokenizer
 - run offline transcription over the manifest
 - write predictions and per-sample WER/CER JSONL files to --out_dir
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List, Dict

import torch
from nemo.collections.asr.models import ASRModel


def load_manifest(manifest_path: str) -> List[Dict]:
    # Support both JSON array manifests and JSONL (one JSON object per line).
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    text = text.strip()
    if not text:
        return []
    # Try parsing as a JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # fallthrough to JSONL parsing
        pass

    # Fallback: parse as JSONL (one JSON object per line)
    items: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def edit_distance(a: List[str], b: List[str]) -> int:
    # classic DP edit distance
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # deletion
                           dp[i][j - 1] + 1,      # insertion
                           dp[i - 1][j - 1] + cost)  # substitution
    return dp[n][m]


def wer_of_pair(ref: str, hyp: str) -> float:
    r_words = ref.strip().split()
    h_words = hyp.strip().split()
    if len(r_words) == 0:
        return 1.0 if len(h_words) > 0 else 0.0
    ed = edit_distance(r_words, h_words)
    return ed / len(r_words)


def cer_of_pair(ref: str, hyp: str) -> float:
    r_chars = list(ref.strip())
    h_chars = list(hyp.strip())
    if len(r_chars) == 0:
        return 1.0 if len(h_chars) > 0 else 0.0
    ed = edit_distance(r_chars, h_chars)
    return ed / len(r_chars)


def apply_checkpoint_if_present(model: ASRModel, ckpt_path: str) -> None:
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.endswith('.nemo'):
        # If user provided a full .nemo, restore anew
        print(f"Restoring model from .nemo: {ckpt_path}")
        # Re-create model from .nemo and return
        new_model = ASRModel.restore_from(ckpt_path)
        model.__dict__.update(new_model.__dict__)
        return
    # assume Lightning .ckpt or torch save
    print(f"Loading checkpoint state_dict from: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = None
    if isinstance(ck, dict):
        # common key in Lightning: 'state_dict'
        state = ck.get('state_dict', None)
        if state is None:
            # some checkpoints store nested under 'model' or similar
            for k, v in ck.items():
                if isinstance(v, dict) and 'state_dict' in v:
                    state = v['state_dict']
                    break
    if state is None:
        raise RuntimeError('state_dict not found in checkpoint (expected Lightning .ckpt)')
    # make keys compatible if needed (no transformation here; user tested strict=False works)
    model.load_state_dict(state, strict=False)


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument('--nemo', required=True, help='Path to base .nemo file')
    p.add_argument('--ckpt', default=None, help='Optional Lightning .ckpt to load (state_dict)')
    p.add_argument('--tokenizer', default=None, help='Optional tokenizer dir to apply via change_vocabulary')
    p.add_argument('--manifest', required=True, help='Test manifest (json list)')
    p.add_argument('--out_dir', required=True, help='Directory to write predictions and scores')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    args = p.parse_args(argv)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.out_dir, exist_ok=True)

    print('Restoring base model from .nemo:', args.nemo)
    model = ASRModel.restore_from(args.nemo)

    if args.ckpt:
        apply_checkpoint_if_present(model, args.ckpt)

    if args.tokenizer:
        print('Applying tokenizer from:', args.tokenizer)
        model.change_vocabulary(new_tokenizer_dir=args.tokenizer, new_tokenizer_type='bpe')

    # move model to device
    if device == 'cuda':
        model = model.cuda()
        print('Moved model to CUDA')
    else:
        print('Using CPU for inference')

    # load manifest
    samples = load_manifest(args.manifest)
    audio_paths = [s['audio_filepath'] for s in samples]
    refs = [s.get('text','') for s in samples]
    print(f'Loaded {len(audio_paths)} samples from {args.manifest}')

    # transcribe using ASRModel.transcribe
    print(f'Transcribing offline (batch_size={args.batch_size})...')
    hyps = model.transcribe(paths2audio_files=audio_paths, batch_size=args.batch_size)

    # compute per-sample metrics and overall
    sample_scores = []
    wer_sum = 0.0
    cer_sum = 0.0
    for ref, hyp, s in zip(refs, hyps, samples):
        w = wer_of_pair(ref, hyp)
        c = cer_of_pair(ref, hyp)
        wer_sum += w
        cer_sum += c
        sample_scores.append({'audio_filepath': s['audio_filepath'], 'wer': w, 'cer': c})

    overall_wer = wer_sum / max(1, len(sample_scores))
    overall_cer = cer_sum / max(1, len(sample_scores))

    # write predictions
    preds_path = os.path.join(args.out_dir, 'predictions_offline_streaming.jsonl')
    scores_path = os.path.join(args.out_dir, 'predictions_offline_streaming_scores.jsonl')
    with open(preds_path, 'w', encoding='utf-8') as pf:
        for s, hyp in zip(samples, hyps):
            out = dict(s)
            out['pred_text'] = hyp
            pf.write(json.dumps(out, ensure_ascii=False) + '\n')
    with open(scores_path, 'w', encoding='utf-8') as sf:
        for row in sample_scores:
            sf.write(json.dumps(row, ensure_ascii=False) + '\n')

    print('Wrote outputs:')
    print(' -', preds_path)
    print(' -', scores_path)
    print(f'Overall WER: {overall_wer:.4f}, CER: {overall_cer:.4f}')


if __name__ == '__main__':
    main()
