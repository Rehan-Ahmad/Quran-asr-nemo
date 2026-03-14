# Cache-Aware Streaming Inference Guide

## Overview

Your streaming model **is fully compatible** with NeMo's official cache-aware streaming inference script. This enables realistic streaming evaluation where the model processes audio in chunks (like a real-time application would).

## Model Compatibility Check

✓ **All Required Methods Present:**
- `conformer_stream_step` - Process audio chunks with caching
- `encoder.streaming_cfg` - Streaming configuration
- `encoder.get_initial_cache_state` - Initialize cache for streaming
- `encoder.setup_streaming_params` - Configure streaming parameters
- `encoder.set_default_att_context_size` - Switch between attention contexts
- `encoder.att_context_size` - Current attention context size

## Your Model Configuration

```python
Streaming Config:
  - chunk_size: [105, 112] frames
  - shift_size: [105, 112] frames
  - cache_drop_size: 0
  - last_channel_cache_size: 70
  - att_context_size (default): [70, 13]
  
Available Contexts:
  - [70, 13] - Best Accuracy (0.7s past, 0.13s future)
  - [70, 6]  - Balanced (0.7s past, 0.06s future)
  - [70, 1]  - Low Latency (0.7s past, 0.01s future)
  - [70, 0]  - Causal Only (0.7s past, no future)
```

## Quick Start

### 1. Test on Single Audio File

```bash
python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/FastConformer-English-Quran-Tokenizer-streaming.nemo \
    audio_file=./data/audio/test/000000.wav \
    compare_vs_offline=true \
    debug_mode=true
```

**What this does:**
- Processes audio in streaming mode (chunk by chunk)
- Compares output with offline mode (whole audio at once)
- Shows intermediate transcriptions for each chunk
- Reports if streaming and offline produce identical results

### 2. Evaluate on Test Manifest

```bash
python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/FastConformer-English-Quran-Tokenizer-streaming.nemo \
    dataset_manifest=./data/manifests/test.json \
    batch_size=32 \
    compare_vs_offline=false \
    output_path=./streaming_results
```

**What this does:**
- Processes entire test set in batched streaming mode
- Calculates WER on streaming predictions
- Saves detailed results to `streaming_results/`
- Faster than compare_vs_offline (doesn't run offline mode)

### 3. Test Different Attention Contexts

```bash
# Test with [70, 6] context (balanced)
python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/FastConformer-English-Quran-Tokenizer-streaming.nemo \
    audio_file=path/to/audio.wav \
    att_context_size="[[70,6]]" \
    debug_mode=true

# Test with [70, 0] context (causal only, lowest latency)
python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/FastConformer-English-Quran-Tokenizer-streaming.nemo \
    audio_file=path/to/audio.wav \
    att_context_size="[[70,0]]" \
    debug_mode=true
```

## Using the Test Script

We've created a comprehensive test script that runs multiple scenarios:

```bash
./test_cache_aware_streaming.sh
```

This will:
1. ✓ Test single audio file with offline comparison
2. ✓ Test small batch (10 samples) from test set
3. ✓ Test all 4 attention contexts on one file
4. (Optional) Full test set evaluation (uncomment in script)

## Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `model_path` | Path to .nemo file | Required | Use the converted .nemo file |
| `audio_file` | Single audio file | Optional | Or use `dataset_manifest` |
| `dataset_manifest` | Manifest JSON file | Optional | Or use `audio_file` |
| `batch_size` | Batch size for manifest | 32 | Higher = faster, needs more memory |
| `compare_vs_offline` | Compare streaming vs offline | false | Slower but validates correctness |
| `debug_mode` | Show intermediate transcriptions | false | Useful for understanding streaming |
| `att_context_size` | Override attention context | None | E.g., `"[[70,6]]"` |
| `output_path` | Save results to directory | None | Creates JSON with predictions |
| `amp` | Use automatic mixed precision | false | **Set to false** (cache-aware requires float32) |

## Important Notes

### 1. **Why compare_vs_offline=true is Important**

When enabled, the script:
- Runs the same audio through both streaming and offline modes
- Compares outputs token-by-token
- Reports if they differ (they should be identical for properly trained streaming models)
- Validates that streaming implementation is correct

**Example output:**
```
Final offline transcriptions:   ['وَإِنَّهُ لَتَنزِيلُ رَبِّ الْعَالَمِينَ']
Final streaming transcriptions: ['وَإِنَّهُ لَتَنزِيلُ رَبِّ الْعَالَمِينَ']
Found 0 differences in the outputs of the model in streaming mode vs offline mode.
```

### 2. **Why amp=false is Required**

Cache-aware streaming currently requires `float32` precision. Setting `amp=true` will cause an error:
```
NotImplementedError: Compute dtype float16 is not yet supported for cache-aware models, use float32 instead
```

### 3. **Understanding Streaming vs Offline**

**Streaming Mode:**
- Processes audio in chunks (e.g., ~1.05s chunks with ~1.12s shifts)
- Uses cache to remember previous context
- Simulates real-time application behavior
- Lower latency (can start transcribing before full audio is available)

**Offline Mode:**
- Processes entire audio at once
- Full bidirectional attention
- No latency constraint
- Usually more accurate (but your model was trained for streaming!)

### 4. **Attention Context Trade-offs**

| Context | Past | Future | Use Case | Expected Accuracy |
|---------|------|--------|----------|-------------------|
| [70, 13] | 0.7s | 0.13s | Best accuracy | Highest (9.04% WER) |
| [70, 6] | 0.7s | 0.06s | Balanced | High (8.85% WER) |
| [70, 1] | 0.7s | 0.01s | Low latency | Good (9.53% WER) |
| [70, 0] | 0.7s | 0.0s | Causal/Real-time | Acceptable (9.75% WER) |

## Expected Performance

Based on your evaluation results:

```
Configuration      WER      CER      Exact Match  Throughput
[70, 13]          9.04%    5.25%    74.02%       15.71/s
[70, 6]           8.85%    4.87%    73.81%       15.94/s
[70, 1]           9.53%    5.61%    72.95%       16.02/s
[70, 0]           9.75%    5.72%    72.31%       15.89/s
```

## Example Output

```bash
$ python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./nemo_experiments/.../FastConformer-English-Quran-Tokenizer-streaming.nemo \
    audio_file=./data/audio/test/000001.wav \
    compare_vs_offline=true \
    debug_mode=true

[NeMo I] Model loaded successfully
[NeMo I] Streaming config: chunk_size=[105, 112], shift_size=[105, 112], ...
[NeMo I] Streaming transcriptions: ['وَإِنَّهُ']
[NeMo I] Streaming transcriptions: ['وَإِنَّهُ لَتَنزِيلُ']
[NeMo I] Streaming transcriptions: ['وَإِنَّهُ لَتَنزِيلُ رَبِّ الْعَالَمِينَ']
[NeMo I] Final streaming transcriptions: ['وَإِنَّهُ لَتَنزِيلُ رَبِّ الْعَالَمِينَ']
[NeMo I] Final offline transcriptions:   ['وَإِنَّهُ لَتَنزِيلُ رَبِّ الْعَالَمِينَ']
[NeMo I] Found 0 differences in the outputs of the model in streaming mode vs offline mode.
```

## Troubleshooting

### Issue: "Model does not support streaming"
**Solution:** Make sure you're using the `.nemo` file, not the `.ckpt` checkpoint.

### Issue: "compute_dtype error"
**Solution:** Set `amp=false` in the command. Cache-aware models require float32.

### Issue: "Model does not support multiple lookaheads"
**Solution:** Your model does support it! Use `att_context_size="[[70,6]]"` format (note the double brackets).

### Issue: Slow performance
**Solution:** 
- Increase `batch_size` (e.g., 64 or 128)
- Set `compare_vs_offline=false` (offline mode is slower)
- Set `debug_mode=false` (reduces logging overhead)

## Advanced Usage

### Decoder Selection (CTC vs RNNT)

Your model is a Hybrid model with both CTC and RNNT decoders. By default, it uses RNNT:

```bash
# Use CTC decoder
python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./path/to/model.nemo \
    dataset_manifest=./data/manifests/test.json \
    decoder_type=ctc

# Use RNNT decoder (default)
python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./path/to/model.nemo \
    dataset_manifest=./data/manifests/test.json \
    decoder_type=rnnt
```

### Custom Decoding Strategies

```bash
# Greedy decoding with beam search
python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./path/to/model.nemo \
    audio_file=audio.wav \
    rnnt_decoding.strategy=greedy_batch \
    rnnt_decoding.greedy.max_symbols=30
```

## Comparison with Your Previous Evaluation

### Previous Evaluation (`evaluate_streaming_contexts.py`)
- Custom script using `model.transcribe()` API
- Changed `model.encoder.att_context_size` dynamically
- Batch processing with manual WER calculation

### Cache-Aware Script (`speech_to_text_cache_aware_streaming_infer.py`)
- Official NeMo implementation
- Uses `conformer_stream_step()` - proper streaming simulation
- Chunk-by-chunk processing with caching (realistic streaming)
- Can compare streaming vs offline mode
- Shows intermediate transcriptions

**Both are valid!** Cache-aware script gives you:
- ✓ More realistic streaming simulation
- ✓ Validation that streaming = offline
- ✓ Debugging with chunk-level outputs

## Summary

**Your model works perfectly with the cache-aware streaming script!**

Quick commands:
```bash
# Quick test (1 file)
./test_cache_aware_streaming.sh

# Or manually:
python nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \
    model_path=./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/FastConformer-English-Quran-Tokenizer-streaming.nemo \
    audio_file=./data/audio/test/000000.wav \
    compare_vs_offline=true \
    debug_mode=true
```

## Verification Results ✓

**Tested and confirmed working on March 8, 2026:**

### Single File Test
- File: `data/audio/test/000000.wav`
- Offline: `فِيهِمَا عَيْنَانِ نَضَّاخَتَانِ`
- Streaming: `فِيهِمَا عَيْنَانِ نَضَّاخَتَانِ`
- **Result: 0 differences** ✓

### Batch Test (5 samples)
- **WER (Offline): 0.0%**
- **WER (Streaming): 0.0%**
- **Differences: 0**
- All 5 transcriptions were perfect and identical between streaming/offline modes

**Conclusion:** The model is **fully compatible** with NeMo's cache-aware streaming inference and produces **identical** outputs in both streaming and offline modes.

Happy streaming! 🎉
