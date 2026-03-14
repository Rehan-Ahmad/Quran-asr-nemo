import nemo.collections.asr as nemo_asr
import numpy as np
import torch
import torchaudio.functional as AF
import gradio as gr
from silero_vad import get_speech_timestamps, load_silero_vad

# ======================
# CONFIG
# ======================
TARGET_SAMPLE_RATE = 16000
BUFFER_DURATION = 5  # seconds

# Silero VAD tuning
SILERO_THRESHOLD = 0.5
SILERO_MIN_SPEECH_MS = 120
SILERO_MIN_SILENCE_MS = 80
SILERO_SPEECH_PAD_MS = 40
SILENCE_RESET_SEC = 1.5  # clear rolling ASR context after sustained silence

BUFFER_SIZE = int(TARGET_SAMPLE_RATE * BUFFER_DURATION)

# ======================
# LOAD MODELS
# ======================
MODEL_PATH = "FastConformer-Custom-Tokenizer.nemo"
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=MODEL_PATH)
asr_model.eval()

vad_model = load_silero_vad()


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if audio.size == 0:
        return audio
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    resampled = AF.resample(audio_t, orig_freq=src_sr, new_freq=dst_sr)
    return resampled.squeeze(0).cpu().numpy().astype(np.float32)


def has_speech(audio_chunk_f32: np.ndarray) -> bool:
    timestamps = get_speech_timestamps(
        audio_chunk_f32,
        vad_model,
        sampling_rate=TARGET_SAMPLE_RATE,
        threshold=SILERO_THRESHOLD,
        min_speech_duration_ms=SILERO_MIN_SPEECH_MS,
        min_silence_duration_ms=SILERO_MIN_SILENCE_MS,
        speech_pad_ms=SILERO_SPEECH_PAD_MS,
    )
    return len(timestamps) > 0


def init_state():
    return {
        "audio_buffer": np.zeros(0, dtype=np.float32),
        "prev_text": "",
        "full_text": "",
        "silence_duration": 0.0,
    }


def process_stream(mic_audio, state):
    if state is None:
        state = init_state()

    if mic_audio is None:
        return state["full_text"], state

    sr, audio = mic_audio
    if audio is None:
        return state["full_text"], state

    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0

    audio_chunk = _resample_audio(audio, src_sr=int(sr), dst_sr=TARGET_SAMPLE_RATE)
    if audio_chunk.size == 0:
        return state["full_text"], state

    speech_present = has_speech(audio_chunk)
    chunk_duration = float(audio_chunk.shape[0]) / TARGET_SAMPLE_RATE

    if not speech_present:
        state["silence_duration"] += chunk_duration
        if state["silence_duration"] >= SILENCE_RESET_SEC:
            state["audio_buffer"] = np.zeros(0, dtype=np.float32)
            state["prev_text"] = ""
        return state["full_text"], state

    state["silence_duration"] = 0.0

    state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_chunk])
    if len(state["audio_buffer"]) > BUFFER_SIZE:
        state["audio_buffer"] = state["audio_buffer"][-BUFFER_SIZE:]

    with torch.no_grad():
        transcription = (asr_model.transcribe([state["audio_buffer"]], verbose=False)[0]).text

    if transcription != state["prev_text"]:
        new_part = transcription[len(state["prev_text"]) :]
        state["full_text"] += new_part
        state["prev_text"] = transcription

    return state["full_text"], state


def clear_transcript():
    return "", init_state()


with gr.Blocks() as demo:
    gr.Markdown("## Streaming ASR")
    gr.Markdown("Click record on the microphone and speak. Transcript updates live.")

    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="Microphone",
    )
    transcript = gr.Textbox(label="Transcript", lines=10)
    state = gr.State(init_state())
    clear_btn = gr.Button("Clear")

    mic.stream(
        fn=process_stream,
        inputs=[mic, state],
        outputs=[transcript, state],
    )
    clear_btn.click(fn=clear_transcript, outputs=[transcript, state])


if __name__ == "__main__":
    demo.queue().launch(debug=True, share=True)
