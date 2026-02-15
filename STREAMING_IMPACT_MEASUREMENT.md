# Streaming Model Conversion: Impact Measurement Guide

## Overview
Converting a non-streaming ASR model to streaming enables real-time inference at the cost of potentially small accuracy degradation. This guide explains how to measure and interpret the impact.

## Key Metrics

### 1. **Algorithm Latency** ⏱️
The time before the first output becomes available (crucial for streaming).

**Non-streaming model:**
- Must process entire audio sequence
- Latency = full audio duration
- Example: 10s audio = 10s latency before first character

**Streaming model:**
- Processes in chunks with limited left context
- Latency = chunk size / sample rate
- Example: With 70-frame left context at 8x subsampling, window_stride=0.01s:
  - Latency ≈ (70 / 8) × 0.01s = **~87ms per chunk**
  - Much faster! Can output characters after ~87ms of audio

**Formula:**
```
Algorithm Latency = (left_context / subsampling_factor) × window_stride
                  = (70 / 8) × 0.01s = 0.0875s ≈ 87ms
```

---

### 2. **Real-Time Factor (RTF)** 🚀
How fast the model processes audio compared to audio duration.

**Formula:**
```
RTF = Model Processing Time / Audio Duration
```

**Interpretation:**
- **RTF < 0.1**: Model is 10× faster than real-time → Excellent for streaming
- **RTF 0.1-0.5**: Model is 2-10× real-time → Good for streaming
- **RTF 0.5-1.0**: Model runs at audio speed → Marginal for streaming
- **RTF > 1.0**: Model is slower than audio → Cannot do real-time

**Example:**
```
Audio duration: 5 seconds
Processing time: 0.5 seconds
RTF = 0.5 / 5 = 0.1 → 10× real-time ✓ Excellent
```

---

### 3. **End-to-End Latency** ⏱️ 
Total time from audio input to final transcript.

**Non-streaming (batch):**
- E2E Latency = Algorithm Latency + Processing Time
- Example: 10s audio + 0.5s processing = 10.5s total

**Streaming (continuous):**
- E2E Latency ≈ Algorithm Latency (once streaming starts)
- Example: 87ms algorithm latency + 0.5s processing starts immediately
- First characters appear after ~87ms, rest appears continuously

**Key benefit:** User sees results appearing in real-time rather than waiting for entire audio.

---

### 4. **Memory Consumption** 💾
GPU/CPU memory used during inference.

**Why streaming uses less memory:**
- Limited context = smaller internal buffers
- No need to store full audio attention matrices
- Reduced cache requirements

**Typical savings:**
- Non-streaming: Full attention matrix = O(n²) memory
- Streaming: Limited context = O(context_size) memory
- Expected reduction: **20-40%** for moderate audio lengths

---

### 5. **Accuracy Metrics** 📊

#### Word Error Rate (WER)
```
WER = (Insertions + Deletions + Substitutions) / Total Words × 100%
```

#### Character Error Rate (CER)
```
CER = (Char Insertions + Deletions + Substitutions) / Total Chars × 100%
```

**Streaming accuracy loss sources:**
1. **Limited lookahead**: Cannot see future context (right_context = 13 frames only)
2. **Boundary effects**: Words near chunk boundaries may have lower confidence
3. **No backtracking**: Cannot revise earlier decisions when future context arrives

**Typical WER degradation:** **1-3%** for well-tuned streaming models

**Example:**
```
Baseline (non-streaming): WER = 15%
Streaming model: WER = 16.5%
Degradation: +1.5% (acceptable trade-off for real-time capability)
```

---

### 6. **Throughput** 📈
How many audio hours can be processed per GPU hour.

**Calculation:**
```
Throughput = (Audio Duration) / (Total Processing Time) × 3600
Example: 100 hours / 20 hours processing = 18 audio-hours/GPU-hour
```

**Streaming advantage:**
- Can pipeline multiple streams
- Lower memory per stream = more concurrent streams
- Example: Process 5 simultaneous streams vs 2 for non-streaming

---

## How Conversion Affects Each Metric

### ✅ Improvements with Streaming Conversion

| Metric | Before | After | Benefit |
|--------|--------|-------|---------|
| **Algorithm Latency** | 5-10s | 50-100ms | 50-100× faster |
| **RTF** | N/A | 0.05-0.1 | Real-time capable |
| **Memory/Streams** | 1 stream | 3-5 streams | Higher throughput |
| **Concurrency** | Sequential | Parallel | Handle many users |

### ⚠️ Trade-offs with Streaming Conversion

| Metric | Impact | Magnitude |
|--------|--------|-----------|
| **WER** | Slight increase | +1-3% |
| **CER** | Slight increase | +1-3% |
| **Complexity** | Code is more complex | Moderate |

---

## Measuring the Impact: Step-by-Step

### Step 1: Run Evaluation Script
```bash
python evaluate_streaming_impact.py \
  --streaming-model=pretrained_models/stt_ar_fastconformer_hybrid_large_streaming.nemo \
  --baseline-model=pretrained_models/stt_ar_fastconformer_hybrid_large_pcd.nemo \
  --test-manifest=data/manifests/test.json
```

### Step 2: Key Outputs to Look For

```json
{
  "latency": {
    "audio_duration_sec": 5.0,
    "streaming_e2e_latency_sec": 0.751,
    "estimated_algorithm_latency_sec": 0.026,
    "streaming_speed": "6.7x real-time",
    "baseline_e2e_latency_sec": 3.245,
    "latency_improvement_factor": 4.32
  },
  "rtf": {
    "mean_rtf": 0.0423,
    "interpretation": "RTF < 0.1 = 10x faster than real-time"
  },
  "memory": {
    "streaming_model_peak_mb": 1250.5,
    "baseline_model_peak_mb": 1680.3,
    "memory_reduction_pct": 25.6
  },
  "accuracy": {
    "streaming_wer": 16.5,
    "baseline_wer": 15.0,
    "wer_degradation_pct": 10.0
  }
}
```

### Step 3: Interpret Results

**Excellent streaming conversion if:**
- ✅ RTF < 0.1 (10× real-time)
- ✅ Algorithm latency < 100ms
- ✅ WER degradation < 2%
- ✅ Memory reduction > 20%

**Good streaming conversion if:**
- ✅ RTF < 0.5 (2× real-time)
- ✅ Algorithm latency < 200ms
- ✅ WER degradation < 3%
- ✅ Memory reduction > 15%

---

## Your Streaming Model Configuration

### Your Settings:
```yaml
encoder:
  causal_downsampling: true           # Enable streaming
  att_context_size: [70, 13]          # 70 frames left, 13 right
  att_context_style: chunked_limited  # Limited attention
  conv_context_size: causal           # Streaming convolutions
  subsampling_factor: 8               # 8× subsampling
  window_stride: 0.01                 # 10ms frames
```

### Expected Metrics:
```
Algorithm Latency: ~87ms per chunk
  = (70 / 8) × 0.01s = 0.0875s

Right Lookahead: ~130ms
  = (13 / 8) × 0.01s = 0.0163s × 8 = 0.13s

Full Algorithm Latency: ~87ms (before first output)
  + 130ms lookahead = ~217ms total before final decision

RTF: Expected 0.05-0.10 (10-20× real-time)
Memory: Expected 20-30% reduction
WER: Expected ~1-2% degradation
```

---

## How to Measure During Inference

### In Production:

**Measure latency during actual streaming:**
```python
import time
start = time.time()
for audio_chunk in stream:
    output = model.transcribe_chunk(audio_chunk)
    latency = time.time() - start
    print(f"Latency: {latency*1000:.1f}ms")
```

**Monitor RTF continuously:**
```python
processing_time = time.time() - start
audio_duration = num_chunks * chunk_duration
rtf = processing_time / audio_duration
print(f"RTF: {rtf:.4f}")

if rtf > 1.0:
    print("WARNING: Cannot keep up with real-time!")
```

**Track accuracy on production data:**
```python
predicted = model.transcribe(audio)
reference = get_ground_truth(audio_id)
wer = calculate_wer(reference, predicted)
log_metric('streaming_wer', wer)
```

---

## Decision Flow

```
START
  ↓
Is RTF < 0.1? 
  YES → Excellent for streaming ✓
  NO → Is RTF < 0.5?
    YES → Good for streaming ✓
    NO → Is RTF < 1.0?
      YES → Marginal, may struggle ⚠️
      NO → Not suitable for real-time ✗

Is WER degradation < 2%?
  YES → Acceptable trade-off ✓
  NO → May need model tuning ⚠️

Is memory < baseline?
  YES → Good efficiency ✓
  NO → Not streaming (check config)

CONCLUSION: Ready for production ✓
```

---

## Summary

The streaming conversion validates as successful if:

1. **Latency:** Algorithm latency < 100ms, RTF < 0.1
2. **Accuracy:** WER degradation < 2%
3. **Efficiency:** Memory reduction > 20%
4. **Functionality:** Model configured correctly with streaming parameters

Your Quranic ASR streaming model should meet all these criteria and be ready for real-time deployment!
