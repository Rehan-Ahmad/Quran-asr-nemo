# 🚀 Quick Start: Streaming vs Non-Streaming Gradio App

## ✅ Test Results

Both models loaded successfully and tested:
- **Streaming model**: att_context_size [[70, 13], [70, 6], [70, 1], [70, 0]]
- **Non-streaming model**: Full context [10000, 10000]
- **Test transcription**: ✅ IDENTICAL outputs
- **Sample performance** (8.09s audio): 
  - Streaming: 0.921s (limited context att_context_size)
  - Non-streaming: 0.495s ← **Faster for short audio!**
  - **Note**: Non-streaming is faster for short clips; streaming benefits appear with longer audio or real-time requirements

## 🎯 Launch the App

```bash
# Activate environment
source .venv/bin/activate

# Launch Gradio interface
python app_streaming_comparison.py
```

**App will be available at**: http://localhost:7860

## 📋 Features

### Dual Inference Modes
- ✅ **Streaming**: Limited context window (suitable for real-time)
- ✅ **Non-streaming**: Full context (maximum accuracy)
- ✅ **Both simultaneously**: Direct performance comparison

### Performance Metrics
- ⏱️ **Inference Time**: Latency comparison
- 📊 **WER/CER**: Accuracy metrics (if reference provided)
- 🎯 **Exact Match**: Binary accuracy check
- 📈 **Output Comparison**: Visual diff between modes

### Audio Input
- 🎙️ **Microphone**: Record directly in browser
- 📁 **Upload**: Any audio format (auto-converted to 16kHz)
- 🔊 **Batch Tests**: Test multiple samples from test set

## 📊 Expected Performance

Based on our breakthrough evaluation (11.44% WER):

| Metric | Streaming (RNNT) | Non-Streaming | Notes |
|--------|------------------|---------------|-------|
| **WER** | ~11.44% | Similar or better | Depends on audio duration |
| **CER** | ~6.88% | Similar or better | Context helps longer audio |
| **Exact Match** | ~74% | Expected similar | For Quranic text |
| **Speed** | Varies | Usually faster for short audio | Streaming better for long/real-time |

### Important: "Streaming" ≠ "Faster"

**Streaming is about latency, not throughput!**

### When Streaming is Better
- ✅ **Real-time applications** (process as audio arrives)
- ✅ **Very long audio** (>30s, memory benefits from limited context)
- ✅ **Online inference** (start transcribing before audio ends)
- ✅ **Lower first-token latency** (begins output sooner)

### When Non-Streaming is Faster
- ✅ **Short audio clips** (<10s) ← **Most test samples are 3-10s**
- ✅ **Batch processing** (all audio available upfront)
- ✅ **GPU optimization** (better parallelization with full context)
- ✅ **Throughput** (more audio processed per second)

## 🎯 Use Cases

### 1. Quality Assurance
Compare both modes to verify streaming doesn't degrade quality significantly on your specific audio data.

### 2. Real-Time Testing
Record live audio via microphone and see immediate transcription in both modes.

### 3. Performance Benchmarking
Upload various audio lengths to find the crossover point where streaming becomes beneficial.

### 4. Reference Validation
Provide ground truth text to calculate WER/CER and validate accuracy.

## 📁 Related Files

| File | Purpose |
|------|---------|
| [app_streaming_comparison.py](app_streaming_comparison.py) | Main Gradio application |
| [test_streaming_app.py](test_streaming_app.py) | Quick validation test |
| [STREAMING_APP_GUIDE.md](STREAMING_APP_GUIDE.md) | Comprehensive documentation |
| [EVALUATION_BREAKTHROUGH.md](EVALUATION_BREAKTHROUGH.md) | Context on evaluation approach |

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Kill existing Gradio processes
lsof -ti:7860 | xargs kill -9
```

### Memory Issues
- Reduce batch size in code if OOM
- Use CPU mode: set `DEVICE = 'cpu'` in line 17

### Audio Format Issues
```bash
# Convert audio to 16kHz mono WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 📝 Example Workflow

1. **Launch app**: `python app_streaming_comparison.py`
2. **Open browser**: Navigate to http://localhost:7860
3. **Upload audio**: From `data/audio/test/` directory
4. **Add reference**: Copy from `data/manifests/test.json`
5. **Enable both modes**: Check both checkboxes
6. **Transcribe**: Click "🎯 Transcribe" button
7. **Compare results**: View side-by-side comparison

## 🎨 Interface Preview

```
┌─────────────────────────────────────────────────┐
│ 🎙️ Quran Arabic ASR Comparison                 │
├─────────────────────────────────────────────────┤
│                                                 │
│ [Upload Audio] or [🎤 Record]                  │
│                                                 │
│ Reference Text (Optional):                      │
│ ┌─────────────────────────────────────────────┐ │
│ │ Enter Arabic text here...                   │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ☑ Enable Streaming    ☑ Enable Non-Streaming  │
│                                                 │
│           [🎯 Transcribe]                       │
│                                                 │
├─────────────────────┬───────────────────────────┤
│ Streaming Results   │ Non-Streaming Results     │
│ ─────────────────── │ ──────────────────────    │
│ Transcription: ...  │ Transcription: ...        │
│ Time: 0.921s        │ Time: 0.495s              │
│ WER: 11.44%         │ WER: 11.20%               │
├─────────────────────┴───────────────────────────┤
│          ⚡ Performance Comparison               │
│ ───────────────────────────────────────────     │
│ Output: ✅ IDENTICAL                            │
│ Speed: Non-Streaming 1.86x faster              │
│ Accuracy: Non-Streaming (lower WER)            │
└─────────────────────────────────────────────────┘
```

## 💡 Tips

1. **First run slower**: Model warmup takes extra time
2. **GPU recommended**: ~10x faster than CPU
3. **Test set samples**: Use `data/audio/test/*.wav` for validated results
4. **Reference accuracy**: Copy exact text from `data/manifests/test.json`
5. **Batch testing**: Keep app running for multiple tests

## 🎉 Ready to Launch!

All tests passed. You can now:

```bash
python app_streaming_comparison.py
```

---

**Built with ❤️ using NVIDIA NeMo + Gradio**
