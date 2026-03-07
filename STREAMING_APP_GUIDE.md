# Streaming vs Non-Streaming ASR Comparison App

## Overview

Interactive Gradio application for comparing streaming and non-streaming inference modes of the Quran Arabic ASR model.

## Features

### 🎯 Dual Inference Modes
- **Streaming Mode**: Limited context window `[[70, 13], [70, 6], [70, 1], [70, 0]]`
  - Suitable for **real-time/online** applications
  - **Lower first-token latency** (starts output sooner)
  - Memory efficient for long audio
  - **Note**: May be slower for short audio due to chunking overhead
  
- **Non-Streaming Mode**: Full audio context
  - Better accuracy (full context available)
  - **Faster for short audio** (<10s) due to parallel processing
  - Higher memory usage for very long audio
  - Requires complete audio before starting

### 📊 Performance Metrics
- **Inference Time**: Latency comparison between modes
- **WER/CER**: Word and Character Error Rates (if reference provided)
- **Exact Match**: Binary accuracy check
- **Output Comparison**: Visual diff between transcriptions

### 🎙️ Audio Input Options
- Upload pre-recorded audio files
- Record directly via microphone
- Supports various audio formats (converted to 16kHz internally)

## Usage

### Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch app
python app_streaming_comparison.py
```

The app will be available at: **http://localhost:7860**

### Step-by-Step Guide

1. **Upload/Record Audio**
   - Click "Upload Audio" to select a file
   - OR click "Microphone" to record live
   - Recommended: 16kHz mono WAV for best results

2. **Add Reference Text (Optional)**
   - Enter correct Arabic transcription
   - Enables WER/CER/Exact Match calculations
   - Use examples provided below audio input

3. **Select Inference Modes**
   - ✅ Streaming Mode - for limited context inference
   - ✅ Non-Streaming Mode - for full context inference
   - Enable both for comparison

4. **Transcribe**
   - Click "🎯 Transcribe" button
   - Wait for inference (typically 1-5 seconds)
   - View results in three sections:
     - Left: Streaming results
     - Right: Non-streaming results
     - Bottom: Performance comparison

## Results Interpretation

### Streaming Results Panel
- **Transcription**: Arabic text output
- **Inference Time**: Processing latency
- **Context Mode**: Attention context size used
- **Metrics**: WER/CER if reference provided

### Non-Streaming Results Panel
- Same format as streaming panel
- Shows full-context inference results

### Comparison Panel
- **Output Comparison**: Whether both modes agree
- **Speed Comparison**: Latency difference and speedup
- **Accuracy Comparison**: WER/CER winner (if reference provided)

## Expected Performance

Based on evaluation results (from EVALUATION_BREAKTHROUGH.md):

| Mode | WER | CER | Exact Match | Speed |
|------|-----|-----|-------------|-------|
| Streaming (RNNT) | ~11.44% | ~6.88% | ~74% | 4.58 it/s |
| Non-Streaming | Expected similar or better | - | - | Varies |

## Technical Details

### Model Configuration

```python
# Streaming Mode
encoder.att_context_size = [[70, 13], [70, 6], [70, 1], [70, 0]]
# Limited attention context per layer

# Non-Streaming Mode  
encoder.att_context_size = [10000, 10000]
# Effectively unlimited context
```

### Dependencies
- `gradio` - Web interface
- `nemo` - ASR model framework
- `torch` - Deep learning backend
- `jiwer` - WER/CER calculation
- `numpy` - Numerical operations

### Checkpoint Location
```
./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/epoch=49-step=45200.ckpt
```

## Use Cases

### 1. Real-Time Applications
**Use Streaming Mode:**
- Live transcription systems (audio arriving in chunks)
- Interactive voice applications (respond while speaking)
- Resource-constrained environments (memory limits)
- **Low first-token latency** (start output quickly)
- Very long recordings (>30s, memory benefits)

### 2. Offline/Batch Processing
**Use Non-Streaming Mode:**
- Archival transcription (all audio available)
- Maximum accuracy requirements (full context helps)
- Post-processing pipelines (throughput over latency)
- **Short audio clips** (<10s, faster processing)
- No real-time constraints

### 3. Quality Assurance
**Use Both Modes:**
- Verify streaming doesn't degrade quality significantly
- Identify audio segments where context matters
- Benchmark inference speed vs accuracy tradeoffs

## Troubleshooting

### Model Loading Errors
```bash
# Verify checkpoint exists
ls -lh ./nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/epoch=49-step=45200.ckpt

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Audio Format Issues
- Ensure audio is 16kHz sample rate
- Convert using: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`
- App auto-converts but may introduce latency

### Memory Issues
- Models loaded once at startup (saves memory)
- Reduce batch_size in code if OOM errors
- Use CPU mode if GPU memory insufficient

### Port Already in Use
```bash
# Kill existing process
lsof -ti:7860 | xargs kill -9

# Or use different port in code:
# app.launch(server_port=7861)
```

## Advanced Configuration

### Custom Checkpoint Path
Edit line 18 in `app_streaming_comparison.py`:
```python
CHECKPOINT_PATH = 'path/to/your/checkpoint.ckpt'
```

### Adjust Context Windows
Modify non-streaming context (line 47):
```python
non_streaming_model.encoder.att_context_size = [10000, 10000]
# Increase for longer context, decrease for memory savings
```

### Change Port/Host
Edit `app.launch()` call (line 391):
```python
app.launch(
    server_name="0.0.0.0",  # Change to "127.0.0.1" for localhost only
    server_port=7860,        # Change port number
    share=True               # Enable public URL via Gradio share
)
```

## Example Workflow

```bash
# 1. Start app
python app_streaming_comparison.py

# 2. Open browser to http://localhost:7860

# 3. Upload test audio from data/audio/test/

# 4. Add reference from data/manifests/test.json

# 5. Enable both modes and transcribe

# 6. Compare results in comparison panel
```

## Performance Tips

1. **First Inference**: Takes longer (model warmup)
2. **Batch Multiple**: If testing many files, keep app running
3. **GPU Recommended**: ~10x faster than CPU
4. **Audio Quality**: Clean recordings yield better results
5. **Reference Text**: Copy from test manifest for accurate metrics

## Related Files

- `evaluate_quran_streaming.py` - Batch evaluation script
- `evaluate_hybrid_decoders.py` - RNNT vs CTC comparison
- `EVALUATION_BREAKTHROUGH.md` - Detailed evaluation docs
- `app_gradio_asr.py` - Original Gradio app (single model)

## Citation

If using this app for research/publication:

```bibtex
@software{quran_asr_streaming_comparison,
  title={Quran Arabic ASR Streaming vs Non-Streaming Comparison App},
  author={Sharjeel Abid Butt},
  year={2026},
  note={Built with NVIDIA NeMo ASR framework}
}
```

## License

Same as parent project.

---

**Built with ❤️ using NVIDIA NeMo ASR + Gradio**
