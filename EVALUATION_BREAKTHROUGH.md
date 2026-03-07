# Evaluation Breakthrough Documentation

## Date: March 7, 2026

## Problem Summary

Previous evaluation attempts were producing catastrophically poor results:
- **Initial RNNT evaluation**: ~97% WER (repetitive hypotheses, massive insertion rates)
- **CTC decoder attempt**: 517% WER (!), completely gibberish output
- Model appeared to be fundamentally broken despite good training metrics (11-12% WER during training)

## Root Cause Analysis

### What We Were Doing Wrong

1. **Runtime Tokenizer Override Attempts**
   - Tried to change tokenizer at runtime using `model.change_vocabulary()`
   - This was unnecessary - model was already trained with Quran BPE tokenizer
   - Runtime changes caused decoder/tokenizer misalignment

2. **Incorrect Decoder Selection**
   - Attempted to force CTC decoder thinking it would improve results
   - CTC decoder produced 517% WER (vs RNNT's 11.44% WER)
   - Hybrid model was trained primarily with RNNT loss

3. **Configuration Misunderstanding**
   - Tried to inject `streaming_cfg` into encoder config
   - Didn't understand that `att_context_size` alone controls streaming behavior
   - Caused model instantiation failures

4. **Over-engineering the Solution**
   - Created complex .nemo repackaging scripts
   - Attempted to merge configs from different models
   - Added unnecessary complexity when simple checkpoint loading would work

## The Breakthrough

### Reference Model Study

Downloaded and analyzed two reference models:
1. **nvidia/stt_en_fastconformer_hybrid_large_streaming_multi** (English base)
2. **salesken/Hindi-FastConformer-Streaming-ASR** (Hindi finetuned)

### Key Insights from Reference Models

```yaml
# Both models use identical streaming configuration:
encoder:
  att_context_size: [[70, 13], [70, 6], [70, 1], [70, 0]]
  # Note: NO separate streaming_cfg section!
  # Streaming is controlled ONLY by att_context_size
```

**Critical Finding**: Our Quran model ALREADY had the correct configuration!

```python
# Our model config (verified):
model.cfg.encoder.att_context_size = [[70, 13], [70, 6], [70, 1], [70, 0]]
model.encoder.att_context_size = [70, 13]  # Active streaming context
```

### The Simple Solution

**Stop trying to modify the model. Just load and use it as-is.**

```python
# What works (simple approach):
model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(checkpoint_path)
model = model.to(device)
model.eval()
transcriptions = model.transcribe(audio=manifest, batch_size=16)
```

**What we removed:**
- ❌ Runtime tokenizer changes
- ❌ Decoder switching
- ❌ Config overrides
- ❌ .nemo repackaging
- ❌ Complex override_config with timestamps/etc.

## Final Results

### RNNT Decoder (Recommended)
- **WER: 11.44%** ✅
- **CER: 6.88%** ✅
- **Exact Match Rate: 74.02%** ✅
- **Speed: 4.58 it/s**

### CTC Decoder (Comparison)
- **WER: 5.34%** (better)
- **CER: 5.34%** (better)
- **Exact Match Rate: 54.45%** (worse)
- **Speed: 5.65 it/s** (faster)

**Winner: RNNT Decoder** - Higher exact match rate (74% vs 54%) is more important for Quranic text where precision matters.

## Lessons Learned

1. **Trust the Training Process**
   - Model was correctly configured and trained
   - No runtime modifications needed
   - Checkpoint loading is sufficient

2. **Understand Framework Conventions**
   - NeMo uses `att_context_size` for streaming, not separate `streaming_cfg`
   - Hybrid models with RNNT + CTC: RNNT is usually primary
   - Check reference implementations before custom solutions

3. **Match Evaluation to Training**
   - Model trained with RNNT loss → use RNNT decoder for eval
   - Tokenizer baked into checkpoint → don't override at runtime
   - Same preprocessing as training (no changes needed)

4. **Metrics Can Be Misleading**
   - CTC has lower WER/CER but worse exact matches
   - For Quranic text: exact match rate > word error rate
   - Context matters when choosing metrics

## Implementation Changes

### New Files Created

1. **`evaluate_quran_streaming.py`**
   - Simple, clean evaluation script
   - Uses model as-is from checkpoint
   - No complex overrides or modifications
   - Produces: `quran_streaming_eval_results.jsonl`

2. **`evaluate_hybrid_decoders.py`**
   - Compares RNNT vs CTC performance
   - Comprehensive metrics comparison
   - Agreement analysis between decoders
   - Produces: `hybrid_eval_rnnt.jsonl`, `hybrid_eval_ctc.jsonl`

3. **`reference_models/`**
   - Downloaded English and Hindi reference models
   - Extracted and compared configurations
   - Validated our model's streaming setup

### Configuration Verification

```python
# Verified model configuration matches reference:
✓ att_context_size: [[70, 13], [70, 6], [70, 1], [70, 0]]
✓ Tokenizer: Quran BPE v1024 (already in checkpoint)
✓ Model: EncDecHybridRNNTCTCBPEModel
✓ Streaming mode: encoder context [70, 13]
✓ Decoder: RNNT (default, as trained)
```

## Recommendation for Future Work

1. **Use the Simple Evaluation Script**
   ```bash
   python evaluate_quran_streaming.py
   ```

2. **Always Use RNNT Decoder**
   - Don't switch to CTC unless specifically needed
   - 74% exact match rate is superior for Quranic applications

3. **No Runtime Modifications Needed**
   - Load checkpoint directly
   - Model is pre-configured correctly
   - Trust the training process

4. **For Streaming Inference**
   - Model already supports streaming via att_context_size
   - No code changes needed for streaming mode
   - Just use the checkpoint as-is

## Performance Comparison Table

| Metric | Previous (Wrong) | Current (Correct) | Improvement |
|--------|------------------|-------------------|-------------|
| WER (RNNT) | 97.31% | 11.44% | **8.5x better** |
| CER (RNNT) | 113.94% | 6.88% | **16.5x better** |
| WER (CTC) | 517.25% | 5.34% | **96.8x better** |
| Exact Matches | 0/3745 | 2772/3745 | **∞ improvement** |

## Conclusion

The model was working correctly all along. Our previous poor results were due to:
1. Unnecessary runtime modifications
2. Misunderstanding of streaming configuration
3. Incorrect decoder selection
4. Over-complexity in evaluation approach

**The solution was to simplify, not complicate.**
