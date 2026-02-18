#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quran/Arabic ASR Evaluation (Non-Streaming + Streaming) with Auto-Discovery

Features:
- Auto-discover models under nemo_experiments/**/checkpoints/*.nemo
- Interactive model selection when model_path is not provided
- Auto-detect streaming vs non-streaming from run config/name
- Offline eval matches old-script behavior (WER + CER without spaces)
- Optional Arabic/Quran normalization (OFF by default)
- Streaming eval:
   * Native API if available (transcribe_streaming or transcribe_vad_streaming)
   * Simulated chunked streaming fallback (no lookahead)
- Hydra-safe path handling (absolute paths)

Usage examples and notes are in the file header.
"""

import os
import re
import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from hydra.utils import to_absolute_path
from nemo.utils import logging
from nemo.core.config import hydra_runner

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.wer import word_error_rate
import yaml

# ------------------------------------------------------------------------------------
# Configs
# ------------------------------------------------------------------------------------

@dataclass
class TextProcConfig:
    # Keep these False to match old behavior; use arabic_norm/rm_punct for Quran/Arabic
    separate_punctuation: bool = False
    do_lowercase: bool = False
    rm_punctuation: bool = False
    punctuation_marks: str = ".,?،؛؟"  # include Arabic punctuation if ever used


@dataclass
class EvaluationConfig:
    # Model selection
    model_path: Optional[str] = None               # .nemo; if None → interactive selector
    decoder_type: Optional[str] = None             # For hybrid RNNT; otherwise None

    # Data
    dataset_manifest: str = "data/manifests/test.json"  # JSONL with audio_filepath + text
    gt_text_attr_name: str = "text"

    # Output
    output_pred_manifest: str = "predictions.jsonl"     # JSONL: original fields + pred_text
    output_scores_manifest: Optional[str] = None        # JSONL per-sample metrics; auto if None
    save_examples: int = 5

    # Eval options
    batch_size: int = 32
    use_cer_as_primary: bool = False
    compute_per_sample: bool = True

    # Arabic/Quran normalization (OFF by default to match old script)
    arabic_norm: bool = False
    rm_punct: bool = False
    text_processing: TextProcConfig = field(default_factory=TextProcConfig)

    # Device
    cuda: bool = True

    # Streaming
    streaming_eval: bool = False          # OFF by default (old-script behavior)
    chunk_len_ms: int = 320
    sample_rate: Optional[int] = None     # if None, infer from model cfg
    left_context: int = 40                # for logging only (simulated path)
    right_context: int = 0                # streaming strict (no lookahead)

    # Tolerance (optional gate on primary metric)
    tolerance: Optional[float] = None


# ------------------------------------------------------------------------------------
# Arabic / Quran Normalization
# ------------------------------------------------------------------------------------

_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")  # harakat + Quranic marks
_TATWEEL = "\u0640"
_ALEF_MAP = {
    "\u0622": "ا",  # آ
    "\u0623": "ا",  # أ
    "\u0625": "ا",  # إ
    "\u0671": "ا",  # ٱ
}
_YA_MAP = {"\u0649": "ي"}  # ى -> ي
_TA_MARBUTA_MAP = {"\u0629": "ه"}  # ة -> ه (pick mapping consistent with your training)
_AR_PUNCT = "،؛؟"

def normalize_arabic(text: str, remove_punct: bool = True) -> str:
    """Light Arabic/Quran normalization (opt-in)."""
    if not text:
        return text
    text = _ARABIC_DIACRITICS.sub("", text).replace(_TATWEEL, "")
    for src, dst in _ALEF_MAP.items():
        text = text.replace(src, dst)
    for src, dst in _YA_MAP.items():
        text = text.replace(src, dst)
    for src, dst in _TA_MARBUTA_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_punct:
        text = re.sub(rf"[{_AR_PUNCT}\.,;:\?\!\-\"'()\{{\}}\[\]\\\/]", "", text)
    return text


# ------------------------------------------------------------------------------------
# Discovery + Streaming Detection
# ------------------------------------------------------------------------------------

def _parse_timestamp(name: str) -> datetime:
    try:
        return datetime.strptime(name, '%Y-%m-%d_%H-%M-%S')
    except Exception:
        return datetime.min

def _find_latest_run_dir(arch_path: Path) -> Optional[Path]:
    runs = []
    # Direct children with date-named dirs
    for p in arch_path.iterdir():
        if p.is_dir() and p.name[:4].isdigit():
            runs.append(p)
    # Nested (e.g., finetune/2026-*)
    for p in arch_path.iterdir():
        if p.is_dir() and not p.name[:4].isdigit():
            for sp in p.iterdir():
                if sp.is_dir() and sp.name[:4].isdigit():
                    runs.append(sp)
    if not runs:
        return None
    return max(runs, key=lambda x: _parse_timestamp(x.name))

def _find_best_nemo(run_dir: Path) -> Optional[Path]:
    ckpts = run_dir / "checkpoints"
    if not ckpts.exists():
        return None
    nemo_files = list(ckpts.glob("*.nemo"))
    if not nemo_files:
        return None
    nemo_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return nemo_files[0]

def _check_is_streaming(run_dir: Path, arch_name: str) -> bool:
    """Detect streaming flag via hparams.yaml (att_context_size / 'streaming') or name."""
    hparams = run_dir / "hparams.yaml"
    if not hparams.exists():
        hparams = run_dir.parent / "hparams.yaml"
    try:
        if hparams.exists():
            with open(hparams, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            # Find att_context_size or 'streaming' in config tree
            def _find_key(d, key):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k == key:
                            return v
                        res = _find_key(v, key)
                        if res is not None:
                            return res
                return None
            att = _find_key(cfg, "att_context_size")
            if att is not None:
                flat = []
                def _flatten(x):
                    if isinstance(x, list):
                        for y in x:
                            _flatten(y)
                    else:
                        try:
                            flat.append(int(x))
                        except Exception:
                            pass
                _flatten(att)
                for v in flat:
                    if v >= 0:
                        return True
            # fallback: string search
            cfg_str = json.dumps(cfg, default=str).lower()
            if "streaming" in cfg_str:
                return True
    except Exception:
        pass
    return "streaming" in arch_name.lower()

def _discover_models(root: str = "nemo_experiments") -> Dict[str, Dict]:
    base = Path(root)
    if not base.exists():
        return {}
    archs = {}
    for arch_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        latest_run = _find_latest_run_dir(arch_dir)
        if latest_run is None:
            continue
        nemo = _find_best_nemo(latest_run)
        if nemo is None:
            continue
        is_streaming = _check_is_streaming(latest_run, arch_dir.name)
        archs[arch_dir.name] = {
            "latest_run": latest_run.name,
            "latest_run_path": str(latest_run),
            "best_ckpt": str(nemo),
            "best_ckpt_name": nemo.name,
            "is_streaming": is_streaming,
            "size_mb": nemo.stat().st_size / (1024**2),
        }
    return archs

def _select_model_interactive() -> Tuple[str, bool]:
    archs = _discover_models()
    if not archs:
        raise ValueError("No .nemo models found under nemo_experiments/**/checkpoints/")
    items = list(archs.items())
    print("\n" + "="*100)
    print("📦 AVAILABLE MODEL ARCHITECTURES")
    print("="*100)
    for i, (name, info) in enumerate(items, 1):
        t = "🌊 STREAMING" if info["is_streaming"] else "📶 STANDARD"
        print(f"\n{i:2d}. [{t}] {name}")
        print(f"    Latest Run: {info['latest_run']}")
        print(f"    Best Model: {info['best_ckpt_name']}")
        print(f"    Size:       {info['size_mb']:.1f} MB")
    print("\n" + "="*100)
    while True:
        try:
            choice = int(input(f"Select architecture (1-{len(items)}): ").strip())
            if 1 <= choice <= len(items):
                name, info = items[choice - 1]
                print(f"\n✓ Selected: {name}")
                print(f"✓ Type:     {'STREAMING' if info['is_streaming'] else 'STANDARD'}")
                print(f"✓ Model:    {info['best_ckpt']}")
                return info['best_ckpt'], bool(info['is_streaming'])
            else:
                print(f"❌ Enter a number between 1 and {len(items)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


# ------------------------------------------------------------------------------------
# IO + Metrics
# ------------------------------------------------------------------------------------

def _to_abs(path: Optional[str]) -> Optional[str]:
    return None if path is None else to_absolute_path(path)

def load_manifest(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def _hyp_text(obj) -> str:
    if obj is None:
        return ""
    if hasattr(obj, "text"):
        return obj.text
    if isinstance(obj, dict) and "text" in obj:
        return obj["text"]
    if isinstance(obj, str):
        return obj
    return str(obj)

def _strip_spaces(xs: List[str]) -> List[str]:
    return [x.replace(" ", "") for x in xs]

def compute_wer_cer(hypotheses: List[str], references: List[str]) -> Tuple[float, float]:
    """WER (fraction), CER without spaces (fraction)."""
    wer = word_error_rate(hypotheses=hypotheses, references=references, use_cer=False)
    cer = word_error_rate(hypotheses=_strip_spaces(hypotheses), references=_strip_spaces(references), use_cer=True)
    # Guard older versions that return percentages:
    wer = wer / 100.0 if wer > 1.0 else wer
    cer = cer / 100.0 if cer > 1.0 else cer
    return float(wer), float(cer)

def maybe_apply_arabic_norm(texts: List[str], enable: bool, remove_punct: bool) -> List[str]:
    if not enable:
        return texts
    return [normalize_arabic(t, remove_punct=remove_punct) for t in texts]


# ------------------------------------------------------------------------------------
# Transcription (Offline + Streaming)
# ------------------------------------------------------------------------------------

def transcribe_offline(model: ASRModel, audio_paths: List[str], batch_size: int, decoder_type: Optional[str] = None) -> List[str]:
    kwargs = {"return_hypotheses": True}
    if decoder_type:
        kwargs["decoder_type"] = decoder_type  # Hybrid models only
    preds = model.transcribe(audio_paths, batch_size=batch_size, **kwargs)
    return [_hyp_text(p) for p in preds]

def _robust_load_audio(wav_path: str, target_sr: int) -> Tuple["list", int]:
    """Load audio; try torchaudio → soundfile → librosa; resample if needed."""
    import numpy as np
    # torchaudio
    try:
        import torchaudio
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0).numpy().astype("float32")  # mono
    except Exception:
        # soundfile
        try:
            import soundfile as sf
            wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if hasattr(wav, "ndim") and wav.ndim == 2:
                wav = wav.mean(axis=1)
        except Exception:
            # librosa
            try:
                import librosa
                wav, sr = librosa.load(wav_path, sr=None, mono=True)
                wav = wav.astype("float32")
            except Exception as e:
                raise RuntimeError(f"Failed to load audio: {wav_path}; {e}")

    # Resample if needed
    if sr != target_sr:
        try:
            import torchaudio.functional as AF
            t = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
            wav = AF.resample(t, orig_freq=sr, new_freq=target_sr).squeeze(0).numpy()
            sr = target_sr
        except Exception:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception:
                # naive linear interpolation
                duration = len(wav) / sr
                t_old = np.linspace(0, duration, num=len(wav), endpoint=False)
                t_new = np.linspace(0, duration, num=int(duration * target_sr), endpoint=False)
                wav = np.interp(t_new, t_old, wav).astype("float32")
                sr = target_sr
    return wav, sr


def transcribe_streaming_simulated(
    model: ASRModel,
    paths: List[str],
    chunk_len_ms: int,
    target_sr: int,
    decoder_type: Optional[str] = None,
) -> List[str]:
    """
    Simulated streaming: decode successive audio chunks without future context,
    concatenate partial hypotheses. This approximates streaming constraints when
    no native streaming API is available.

    NOTE: This is stricter than offline and may over/under-segment words.
    """
    preds = []
    for wav_path in paths:
        wav, sr = _robust_load_audio(wav_path, target_sr=target_sr)
        chunk_len = int(target_sr * (chunk_len_ms / 1000.0))
        texts: List[str] = []
        for start in range(0, len(wav), chunk_len):
            end = min(start + chunk_len, len(wav))
            if end - start < max(1, int(0.2 * chunk_len)):
                break
            import tempfile, soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                sf.write(tmp_wav.name, wav[start:end], target_sr, subtype="PCM_16")
                chunk_text = transcribe_offline(model, [tmp_wav.name], batch_size=1, decoder_type=decoder_type)[0]
                if chunk_text:
                    texts.append(chunk_text.strip())
        final_text = " ".join(texts).strip()
        preds.append(final_text)
    return preds


def transcribe_streaming_native(
    model: ASRModel,
    paths: List[str],
    chunk_len_ms: int,
    target_sr: int,
    decoder_type: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Try to use a native streaming method if the model exposes it.
    Returns list of texts or None if not supported.
    """
    for attr in ("transcribe_streaming", "transcribe_vad_streaming"):
        if hasattr(model, attr):
            fn = getattr(model, attr)
            try:
                preds = fn(
                    paths,
                    chunk_len_ms=chunk_len_ms,
                    batch_size=1,
                    return_hypotheses=True,
                    **({"decoder_type": decoder_type} if decoder_type else {})
                )
                return [_hyp_text(p) for p in preds]
            except Exception as e:
                logging.warning(f"Native streaming method {attr} failed: {e}")
                return None
    return None


def _log_examples(refs: List[str], hyps: List[str], n: int = 5):
    print("\n" + "-" * 80)
    print(f"Sample predictions (first {min(n, len(refs))}):")
    print("-" * 80)
    for i, (r, h) in enumerate(list(zip(refs, hyps))[:n], 1):
        print(f"[{i}]")
        print(f"REF: {r}")
        print(f"HYP: {h}")
        print()


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig) -> EvaluationConfig:
    # Make paths absolute (Hydra-safe)
    cfg.dataset_manifest = _to_abs(cfg.dataset_manifest)
    cfg.output_pred_manifest = _to_abs(cfg.output_pred_manifest)
    if cfg.output_scores_manifest is None:
        base, _ = os.path.splitext(cfg.output_pred_manifest)
        cfg.output_scores_manifest = base + "_scores.jsonl"
    else:
        cfg.output_scores_manifest = _to_abs(cfg.output_scores_manifest)
    if cfg.model_path:
        cfg.model_path = _to_abs(cfg.model_path)

    logging.info("=" * 100)
    logging.info("QURAN / ARABIC ASR EVALUATION")
    logging.info("=" * 100)

    # If model_path not supplied, offer interactive selection from nemo_experiments
    inferred_streaming: Optional[bool] = None
    if cfg.model_path is None:
        logging.info("No model_path provided — launching interactive selector from nemo_experiments/")
        cfg.model_path, inferred_streaming = _select_model_interactive()
        cfg.model_path = _to_abs(cfg.model_path)
    else:
        # If user supplied a .nemo under nemo_experiments, try to infer streaming flag
        try:
            p = Path(cfg.model_path)
            if "nemo_experiments" in str(p):
                if p.parent.name == "checkpoints":
                    run_dir = p.parent.parent
                    inferred_streaming = _check_is_streaming(run_dir, run_dir.parent.name)
        except Exception:
            inferred_streaming = None

    # Load model
    logging.info(f"Loading model from: {cfg.model_path}")
    model = ASRModel.restore_from(str(cfg.model_path))
    model.eval()

    # Move model to GPU if requested
    if cfg.cuda and torch.cuda.is_available():
        model = model.cuda()
        logging.info("✓ Using GPU")
    else:
        logging.info("✓ Using CPU")

    # Determine sample rate
    if cfg.sample_rate is None:
        try:
            cfg.sample_rate = int(getattr(model, "_cfg", {}).get("sample_rate", 16000))
        except Exception:
            cfg.sample_rate = 16000
    logging.info(f"Sample rate assumed: {cfg.sample_rate} Hz")

    # Load manifest
    samples = load_manifest(cfg.dataset_manifest)
    audio_paths = [s["audio_filepath"] for s in samples]
    refs_raw = [s.get(cfg.gt_text_attr_name, "") for s in samples]

    # Decide streaming mode
    streaming_mode = bool(cfg.streaming_eval)
    if not streaming_mode and inferred_streaming is True:
        logging.info("Model appears to be streaming-capable; switching to streaming_eval")
        streaming_mode = True

    # --- Transcribe ---
    start_time = time.time()
    if not streaming_mode:
        logging.info("Mode: OFFLINE (non-streaming) — matching old-script behavior")
        hyps_raw = transcribe_offline(model, audio_paths, batch_size=cfg.batch_size, decoder_type=cfg.decoder_type)
    else:
        logging.info("Mode: STREAMING")
        hyps_native = transcribe_streaming_native(model, audio_paths, cfg.chunk_len_ms, cfg.sample_rate, decoder_type=cfg.decoder_type)
        if hyps_native is not None:
            hyps_raw = hyps_native
            logging.info("✓ Used native streaming API")
        else:
            logging.info("Native streaming not available; falling back to simulated chunked streaming")
            hyps_raw = transcribe_streaming_simulated(model, audio_paths, cfg.chunk_len_ms, cfg.sample_rate, decoder_type=cfg.decoder_type)
    decode_time = time.time() - start_time
    logging.info(f"Decoding completed in {decode_time:.2f} sec; ~{len(audio_paths)/max(1e-6, decode_time):.2f} utt/s")

    # Save predictions JSONL (refs + hyps)
    with open(cfg.output_pred_manifest, "w", encoding="utf-8") as f:
        for s, hyp in zip(samples, hyps_raw):
            out = dict(s)
            out["pred_text"] = hyp
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    logging.info(f"Predictions saved to: {cfg.output_pred_manifest}")

    # --- Text processing: default OFF to match old behavior ---
    refs = list(refs_raw)
    hyps = list(hyps_raw)

    # (Optional) Apply Quran/Arabic normalization symmetrically
    refs = maybe_apply_arabic_norm(refs, enable=cfg.arabic_norm, remove_punct=cfg.rm_punct)
    hyps = maybe_apply_arabic_norm(hyps, enable=cfg.arabic_norm, remove_punct=cfg.rm_punct)

    # --- Metrics ---
    wer, cer = compute_wer_cer(hyps, refs)

    # Print results
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)
    print(f"Total samples : {len(refs)}")
    print(f"WER           : {wer*100:.2f}% ({wer:.4f})")
    print(f"CER (no space): {cer*100:.2f}% ({cer:.4f})")
    print("=" * 100 + "\n")

    _log_examples(refs_raw, hyps_raw, n=cfg.save_examples)

    # Per-sample metrics (optional)
    if cfg.compute_per_sample:
        with open(cfg.output_scores_manifest, "w", encoding="utf-8") as f:
            for s, r, h in zip(samples, refs, hyps):
                w_i, c_i = compute_wer_cer([h], [r])
                f.write(json.dumps({
                    "audio_filepath": s["audio_filepath"],
                    "reference": r,
                    "prediction": h,
                    "wer": float(w_i),
                    "cer": float(c_i),
                }, ensure_ascii=False) + "\n")
        logging.info(f"Per-sample metrics saved to: {cfg.output_scores_manifest}")

    # Tolerance check (optional)
    primary = cer if cfg.use_cer_as_primary else wer
    metric_name = "CER" if cfg.use_cer_as_primary else "WER"
    if cfg.tolerance is not None:
        if primary > cfg.tolerance:
            raise ValueError(f"{metric_name}={primary:.4f} exceeded tolerance={cfg.tolerance}")
        else:
            logging.info(f"✓ {metric_name}={primary:.4f} within tolerance={cfg.tolerance}")

    # Return snapshot of results in config (handy for pipelines)
    ocfg = OmegaConf.structured(cfg)
    with open_dict(ocfg):
        ocfg.wer = float(wer)
        ocfg.cer = float(cer)
        ocfg.metric_name = metric_name
        ocfg.metric_value = float(primary)
    return ocfg


if __name__ == "__main__":
    main()
