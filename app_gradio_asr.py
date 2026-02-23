
import gradio as gr
from pathlib import Path
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
import numpy as np

# Path to your .nemo model
MODEL_PATH = "nemo_experiments/FastConformer-Custom-Tokenizer/2026-02-14_08-36-37/checkpoints/FastConformer-Custom-Tokenizer.nemo"

# Load model once
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=MODEL_PATH)

# Convert audio to 16kHz wav

def convert_wav_to_16k(input_wav_path, output_file_path, sr=16000):
    y, s = librosa.load(input_wav_path, sr=sr)
    sf.write(output_file_path, y, s)
    return output_file_path


# Buffered/chunked inference helper
def chunk_audio(samples, sample_rate, chunk_len_sec, left_ctx_sec=0, right_ctx_sec=0):
    chunk_len = int(chunk_len_sec * sample_rate)
    left_ctx = int(left_ctx_sec * sample_rate)
    right_ctx = int(right_ctx_sec * sample_rate)
    total_len = len(samples)
    chunks = []
    for start in range(0, total_len, chunk_len):
        l = max(0, start - left_ctx)
        r = min(total_len, start + chunk_len + right_ctx)
        chunks.append(samples[l:r])
    return chunks

def predict(uploaded_wav, buffered_inference=False, chunk_len_sec=5.0, left_ctx_sec=0.0, right_ctx_sec=0.0):
    out_path = "converted.wav"
    audio_conversion = convert_wav_to_16k(uploaded_wav, out_path)
    if not buffered_inference:
        prediction = asr_model.transcribe([audio_conversion])
        if hasattr(prediction[0], 'text'):
            return prediction[0].text
        return prediction[0]
    # Buffered/chunked inference
    samples, sr = sf.read(audio_conversion)
    chunks = chunk_audio(samples, sr, chunk_len_sec, left_ctx_sec, right_ctx_sec)
    results = []
    for chunk in chunks:
        # Save chunk to temp wav
        sf.write("temp_chunk.wav", chunk, sr)
        pred = asr_model.transcribe(["temp_chunk.wav"])
        if hasattr(pred[0], 'text'):
            results.append(pred[0].text)
        else:
            results.append(pred[0])
    return " ".join(results)


# Gradio interface with buffered/chunked inference option
with gr.Blocks() as demo:
    gr.Markdown("# NeMo ASR Demo with Buffered/Chunked Inference Option")
    audio_input = gr.Audio(value=None, label="Audio file", type="filepath")
    buffered_checkbox = gr.Checkbox(label="Enable Buffered/Chunked Inference", value=False)
    chunk_len = gr.Number(label="Chunk Length (seconds)", value=5.0, visible=True)
    left_ctx = gr.Number(label="Left Context (seconds)", value=0.0, visible=True)
    right_ctx = gr.Number(label="Right Context (seconds)", value=0.0, visible=True)
    output_text = gr.Text(label="Transcription")

    def update_visibility(buffered_inference):
        return {
            chunk_len: gr.update(visible=buffered_inference),
            left_ctx: gr.update(visible=buffered_inference),
            right_ctx: gr.update(visible=buffered_inference),
        }

    buffered_checkbox.change(update_visibility, inputs=[buffered_checkbox], outputs=[chunk_len, left_ctx, right_ctx])

    transcribe_btn = gr.Button("Transcribe")
    transcribe_btn.click(
        predict,
        inputs=[audio_input, buffered_checkbox, chunk_len, left_ctx, right_ctx],
        outputs=output_text
    )

demo.launch(debug=True, share=True)
