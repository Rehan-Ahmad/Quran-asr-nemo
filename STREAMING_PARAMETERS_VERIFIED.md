# Streaming Conversion Parameters (NeMo-Verified)

Based on official NeMo cache-aware streaming configs:
- [FastConformer-CTC Streaming Config](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_ctc_bpe_streaming.yaml)
- [Conformer-RNNT Streaming Config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/conformer/cache_aware_streaming/conformer_transducer_bpe_streaming.yaml)

---

## ✅ Core Encoder Settings

```yaml
model:
  encoder:
    att_context_style: chunked_limited     # Enables limited-context attention
    att_context_size: [70, 13]             # [left, right]; -1 = unlimited (offline)
    causal_downsampling: true              # Prevents future peeking in subsampling
    conv_context_size: causal              # Causal convolutions (no right look-ahead)
```

**Important:** Left context should be divisible by (right_context + 1).  
Example: 70 % (13+1) = 0 ✓

---

## ✅ Preprocessor & Validation

```yaml
model:
  preprocessor:
    normalize: "NA"        # No global normalization (streaming-friendly)

  compute_eval_loss: false # Disable to avoid OOM on long validation utterances
```

---

## ⏱️ Latency Calculation (Corrected)

**Look-ahead latency** (algorithmic latency before first output):

```
Latency (seconds) = right_context × subsampling_factor × window_stride
```

**Example with current config:**
```
att_context_size: [70, 13]
subsampling_factor: 8 (FastConformer default)
window_stride: 0.01 (10ms)

Look-ahead latency = 13 × 8 × 0.01 = 1.04 seconds (1040ms)
```

**Note:** 
- **Left context (70)** is cached history → no added latency
- **Right context (13)** is look-ahead → adds 1040ms latency

---

## 🎯 Latency vs Accuracy Trade-off

| `att_context_size` | Look-ahead Latency | Use Case |
|-------------------|-------------------|----------|
| `[70, 4]` | ~320ms | Low-latency voice assistants |
| `[70, 8]` | ~640ms | Balanced (recommended for real-time) |
| `[70, 13]` | ~1040ms | Quality-focused (current setting) |
| `[140, 27]` | ~2160ms | Near-offline quality |
| `[-1, -1]` | Full audio | Non-streaming (offline) |

**Recommendation:** For real-time Quranic ASR, consider evaluating `[70, 8]` (640ms) for better responsiveness.

---

## 🔍 Verification After Conversion

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from("streaming_model.nemo")
enc_cfg = model.cfg.encoder

# Verify streaming parameters
assert enc_cfg.att_context_style == 'chunked_limited', "Not streaming!"
assert enc_cfg.att_context_size == [70, 13], "Wrong context!"
assert enc_cfg.causal_downsampling == True, "Not causal!"
assert enc_cfg.conv_context_size == 'causal', "Convs not causal!"

# Verify preprocessor
assert model.cfg.preprocessor.normalize == 'NA', "Wrong normalization!"
assert model.cfg.compute_eval_loss == False, "Eval loss not disabled!"

# Calculate actual latency
right_ctx = enc_cfg.att_context_size[1]
latency_ms = right_ctx * 8 * 10  # subsampling=8, stride=10ms
print(f"✓ Streaming enabled with {latency_ms}ms look-ahead")
```

---

## 📝 Key Corrections from NeMo Documentation

### 1. **Latency Formula**
- **Correct:** Look-ahead latency = `right_context × subsampling × window_stride`
- **Why:** Left context is cached history (no added latency), only right-context adds algorithmic look-ahead

### 2. **Preprocessor Normalization**
- **Correct:** Use `"NA"` (as per official NeMo streaming YAMLs)
- **Not recommended:** `"online"` is not in official streaming configs

### 3. **Eval Loss**
- **Why disable:** To avoid OOM on long validation utterances, especially for RNNT/Hybrid models
- **Not:** "Can't compute loss" but rather "shouldn't for memory efficiency"

### 4. **Subsampling Type**
- **Both valid:** Both `striding` and `dw_striding` appear in official streaming configs
- **Causality controlled by:** `causal_downsampling=true`, not subsampling type

---

## 📚 References

- NeMo FastConformer Cache-Aware Streaming: [GitHub](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_ctc_bpe_streaming.yaml)
- NeMo Conformer Cache-Aware Streaming: [GitHub](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/conformer/cache_aware_streaming/conformer_transducer_bpe_streaming.yaml)
- Cache-Aware Streaming ASR Paper: [arXiv](https://arxiv.org/html/2312.17279v2)
- NVIDIA NeMo ASR Models: [Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- Streaming ASR Tutorial: [Notebook](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Streaming_ASR.ipynb)
