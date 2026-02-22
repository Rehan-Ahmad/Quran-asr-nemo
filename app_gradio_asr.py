import gradio as gr
from pathlib import Path
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf

# Path to your .nemo model
MODEL_PATH = "nemo_experiments/FastConformer-Custom-Tokenizer/2026-02-14_08-36-37/checkpoints/FastConformer-Custom-Tokenizer.nemo"

# Load model once
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=MODEL_PATH)

# Convert audio to 16kHz wav

def convert_wav_to_16k(input_wav_path, output_file_path, sr=16000):
    y, s = librosa.load(input_wav_path, sr=sr)
    sf.write(output_file_path, y, s)
    return output_file_path

# Prediction function

def predict(uploaded_wav):
    out_path = "converted.wav"
    audio_conversion = convert_wav_to_16k(uploaded_wav, out_path)
    # Use correct argument for hybrid RNNT/CTC model
    prediction = asr_model.transcribe([audio_conversion])
    # Extract text from hypothesis
    if hasattr(prediction[0], 'text'):
        return prediction[0].text
    return prediction[0]

# Gradio interface

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(value=None, label="Audio file", type="filepath"),
    outputs=gr.Text(label="Transcription")
)

demo.launch(debug=True, share=True)
