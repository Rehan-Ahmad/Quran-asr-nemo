"""
Gradio App for Streaming vs Non-Streaming ASR Comparison
Allows runtime testing and performance evaluation of streaming model

Usage:
    python app_streaming_comparison.py
"""

import gradio as gr
import torch
import time
import json
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
from jiwer import wer, cer

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
import nemo.collections.asr as nemo_asr

# Configuration
STREAMING_CHECKPOINT_PATH = './nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/epoch=49-step=45200.ckpt'
NON_STREAMING_MODEL_PATH = './nemo_experiments/FastConformer-Custom-Tokenizer/2026-02-14_08-36-37/checkpoints/FastConformer-Custom-Tokenizer.nemo'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global models (loaded once at startup)
streaming_model = None
non_streaming_model = None


def load_models():
    """Load both streaming and non-streaming versions of the model"""
    global streaming_model, non_streaming_model
    
    print("Loading models...")
    print(f"Device: {DEVICE}")
    
    # Load streaming model from checkpoint (with context restrictions)
    print("\n[1/2] Loading streaming model from checkpoint...")
    print(f"   Path: {STREAMING_CHECKPOINT_PATH.split('/')[-1]}")
    streaming_model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(STREAMING_CHECKPOINT_PATH)
    streaming_model = streaming_model.to(DEVICE)
    streaming_model.eval()
    
    # Verify streaming configuration
    att_context = streaming_model.cfg.encoder.att_context_size
    print(f"   ✓ Loaded - att_context_size: {att_context}")
    print(f"   ✓ Default active context: {streaming_model.encoder.att_context_size}")
    
    # Load non-streaming model from .nemo file (CTC model, full context)
    print("\n[2/2] Loading non-streaming model from .nemo...")
    print(f"   Path: {NON_STREAMING_MODEL_PATH.split('/')[-1]}")
    non_streaming_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=NON_STREAMING_MODEL_PATH)
    non_streaming_model = non_streaming_model.to(DEVICE)
    non_streaming_model.eval()
    
    # Check non-streaming model context
    print(f"   ✓ Loaded CTC model (non-streaming, full context)")
    if hasattr(non_streaming_model, 'encoder') and hasattr(non_streaming_model.encoder, 'att_context_size'):
        print(f"   ✓ att_context_size: {non_streaming_model.encoder.att_context_size}")
    
    print("\n✓ Both models loaded successfully")
    print(f"\nModel Info:")
    print(f"  Streaming (Hybrid RNNT+CTC): {STREAMING_CHECKPOINT_PATH.split('/')[-1]}")
    print(f"  Non-Streaming (CTC):         {NON_STREAMING_MODEL_PATH.split('/')[-1]}")



def set_streaming_context(context_choice: str):
    """
    Set the streaming context window based on user selection
    
    Args:
        context_choice: String representation of context choice
    """
    global streaming_model
    
    context_map = {
        "[70, 13] - Best Accuracy (0.7s past, 0.13s future)": [70, 13],
        "[70, 6] - Balanced (0.7s past, 0.06s future)": [70, 6],
        "[70, 1] - Low Latency (0.7s past, 0.01s future)": [70, 1],
        "[70, 0] - Causal Only (0.7s past, no future)": [70, 0]
    }
    
    context = context_map.get(context_choice, [70, 13])
    
    if streaming_model is not None and hasattr(streaming_model.encoder, 'att_context_size'):
        streaming_model.encoder.att_context_size = context
        print(f"✓ Streaming context updated to: {context}")
    
    return context


def transcribe_audio(
    audio_path: str,
    reference_text: str = "",
    streaming_context: str = "[70, 13] - Best Accuracy (0.7s past, 0.13s future)",
    use_streaming: bool = True,
    use_non_streaming: bool = True
) -> Tuple[str, str, str]:
    """
    Transcribe audio with both streaming and non-streaming models
    
    Args:
        audio_path: Path to audio file
        reference_text: Optional reference text for metrics
        streaming_context: Selected streaming context window
        use_streaming: Enable streaming mode
        use_non_streaming: Enable non-streaming mode
    
    Returns:
        Tuple of (streaming_result_html, non_streaming_result_html, comparison_html)
    """
    if audio_path is None:
        return "❌ No audio provided", "", ""
    
    # Set streaming context based on selection
    selected_context = set_streaming_context(streaming_context)
    
    # Get audio duration for context
    import soundfile as sf
    audio_data, sample_rate = sf.read(audio_path)
    audio_duration = len(audio_data) / sample_rate
    
    results = {'audio_duration': audio_duration, 'streaming_context': selected_context}
    
    # Streaming inference
    if use_streaming:
        print("Running streaming inference...")
        start_time = time.time()
        
        with torch.no_grad():
            streaming_result = streaming_model.transcribe(
                audio=[audio_path],
                batch_size=1
            )[0]
        
        streaming_time = time.time() - start_time
        
        # Extract text from Hypothesis object
        streaming_text = streaming_result.text if hasattr(streaming_result, 'text') else str(streaming_result)
        
        results['streaming'] = {
            'text': streaming_text,
            'time': streaming_time,
            'mode': 'Streaming (Hybrid RNNT+CTC, Limited Context)',
            'context': str(streaming_model.encoder.att_context_size),
            'context_description': f"Left: {streaming_model.encoder.att_context_size[0]} frames (~{streaming_model.encoder.att_context_size[0]*0.01:.2f}s), Right: {streaming_model.encoder.att_context_size[1]} frames (~{streaming_model.encoder.att_context_size[1]*0.01:.2f}s)"
        }
    
    # Non-streaming inference
    if use_non_streaming:
        print("Running non-streaming inference...")
        start_time = time.time()
        
        with torch.no_grad():
            non_streaming_result = non_streaming_model.transcribe(
                audio=[audio_path],
                batch_size=1
            )[0]
        
        non_streaming_time = time.time() - start_time
        
        # Extract text from Hypothesis object
        non_streaming_text = non_streaming_result.text if hasattr(non_streaming_result, 'text') else str(non_streaming_result)
        
        # Get context info if available
        context_info = "Full Context (CTC Model)"
        if hasattr(non_streaming_model, 'encoder') and hasattr(non_streaming_model.encoder, 'att_context_size'):
            context_info = f"Full Context: {non_streaming_model.encoder.att_context_size}"
        
        results['non_streaming'] = {
            'text': non_streaming_text,
            'time': non_streaming_time,
            'mode': 'Non-Streaming (CTC, Full Context)',
            'context': context_info
        }
    
    # Format results
    streaming_html = format_result(results.get('streaming'), reference_text, "Streaming")
    non_streaming_html = format_result(results.get('non_streaming'), reference_text, "Non-Streaming")
    comparison_html = format_comparison(results, reference_text)
    
    return streaming_html, non_streaming_html, comparison_html


def format_result(result: Dict, reference: str, mode: str) -> str:
    """Format individual result as HTML"""
    if result is None:
        return f"<p><i>{mode} mode not selected</i></p>"
    
    html = f"""
    <div style="padding: 15px; background-color: #f0f0f0; border-radius: 8px; margin: 10px 0;">
        <h3 style="margin-top: 0; color: #2c3e50;">🎯 {mode} Result</h3>
        
        <div style="margin: 10px 0;">
            <strong>📝 Transcription:</strong>
            <div style="padding: 10px; background-color: white; border-left: 4px solid #3498db; margin-top: 5px; font-family: 'Amiri', 'Traditional Arabic', serif; font-size: 18px; direction: rtl; color: #2c3e50;">
                {result['text']}
            </div>
        </div>
        
        <div style="margin: 10px 0;">
            <strong>⏱️ Inference Time:</strong> <code>{result['time']:.3f}s</code>
        </div>
        
        <div style="margin: 10px 0;">
            <strong>🔧 Context Window:</strong> <code>{result['context']}</code>
            <br/>
            <small style="color: #7f8c8d;">{result.get('context_description', '')}</small>
        </div>
    """
    
    # Calculate metrics if reference provided
    if reference and reference.strip():
        pred_text = result['text']
        wer_score = wer(reference, pred_text) * 100
        cer_score = cer(reference, pred_text) * 100
        exact_match = reference == pred_text
        
        html += f"""
        <div style="margin: 15px 0; padding: 10px; background-color: #ecf0f1; border-radius: 5px;">
            <strong>📊 Accuracy Metrics:</strong>
            <ul style="margin: 5px 0;">
                <li>WER: <code>{wer_score:.2f}%</code></li>
                <li>CER: <code>{cer_score:.2f}%</code></li>
                <li>Exact Match: <code>{'✅ Yes' if exact_match else '❌ No'}</code></li>
            </ul>
        </div>
        """
    
    html += "</div>"
    return html


def format_comparison(results: Dict, reference: str) -> str:
    """Format comparison between streaming and non-streaming"""
    if not results:
        return "<p><i>No results to compare</i></p>"
    
    streaming = results.get('streaming')
    non_streaming = results.get('non_streaming')
    audio_duration = results.get('audio_duration', 0)
    streaming_context = results.get('streaming_context', [70, 13])
    
    if not streaming or not non_streaming:
        return "<p><i>Both modes must be selected for comparison</i></p>"
    
    # Text comparison
    same_output = streaming['text'] == non_streaming['text']
    
    # Speed comparison
    speedup = non_streaming['time'] / streaming['time'] if streaming['time'] > 0 else 0
    faster = "Streaming" if streaming['time'] < non_streaming['time'] else "Non-Streaming"
    time_diff = abs(streaming['time'] - non_streaming['time'])
    
    # Context-aware explanation
    if audio_duration < 10:
        speed_context = f"✓ <strong>Expected</strong>: Non-streaming is faster for short audio (<10s). Streaming benefits appear with longer audio or real-time scenarios."
    elif audio_duration < 30:
        speed_context = f"⚠️ <strong>Transition zone</strong>: Results vary. Streaming provides lower first-token latency but similar total time."
    else:
        speed_context = f"✓ <strong>Expected</strong>: For long audio (>30s), streaming provides memory benefits and lower first-token latency."
    
    html = f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 10px 0;">
        <h3 style="margin-top: 0;">⚡ Performance Comparison</h3>
        
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4 style="margin-top: 0;">🎵 Audio & Configuration</h4>
            <p style="font-size: 16px; margin: 5px 0;">
                <strong>Audio Duration:</strong> {audio_duration:.2f}s
            </p>
            <p style="font-size: 16px; margin: 5px 0;">
                <strong>Streaming Context:</strong> <code style="background-color: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px;">{streaming_context}</code>
            </p>
            <p style="font-size: 13px; margin: 5px 0; opacity: 0.9;">
                Left: {streaming_context[0]} frames (~{streaming_context[0]*0.01:.2f}s) | Right: {streaming_context[1]} frames (~{streaming_context[1]*0.01:.2f}s)
            </p>
        </div>
        
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4 style="margin-top: 0;">🎯 Output Comparison</h4>
            <p style="font-size: 16px;">
                {'✅ <strong>IDENTICAL</strong> - Both models produced the same transcription' if same_output else '❌ <strong>DIFFERENT</strong> - Models produced different outputs'}
            </p>
        </div>
        
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4 style="margin-top: 0;">⏱️ Speed Comparison</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 5px;"><strong>Streaming:</strong></td>
                    <td style="padding: 5px;"><code>{streaming['time']:.3f}s</code></td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><strong>Non-Streaming:</strong></td>
                    <td style="padding: 5px;"><code>{non_streaming['time']:.3f}s</code></td>
                </tr>
                <tr style="background-color: rgba(255, 255, 255, 0.2);">
                    <td style="padding: 5px;"><strong>Difference:</strong></td>
                    <td style="padding: 5px;"><code>{time_diff:.3f}s</code></td>
                </tr>
                <tr style="background-color: rgba(255, 255, 255, 0.2);">
                    <td style="padding: 5px;"><strong>Faster:</strong></td>
                    <td style="padding: 5px;"><strong>{faster}</strong> ({speedup:.2f}x)</td>
                </tr>
            </table>
            <p style="margin-top: 10px; font-size: 14px; background-color: rgba(255, 255, 255, 0.15); padding: 10px; border-radius: 5px;">
                💡 {speed_context}
            </p>
        </div>
    """
    
    # Accuracy comparison if reference provided
    if reference and reference.strip():
        streaming_wer = wer(reference, streaming['text']) * 100
        non_streaming_wer = wer(reference, non_streaming['text']) * 100
        streaming_cer = cer(reference, streaming['text']) * 100
        non_streaming_cer = cer(reference, non_streaming['text']) * 100
        
        streaming_exact = reference == streaming['text']
        non_streaming_exact = reference == non_streaming['text']
        
        better_wer = "Streaming" if streaming_wer < non_streaming_wer else "Non-Streaming" if non_streaming_wer < streaming_wer else "Tie"
        better_cer = "Streaming" if streaming_cer < non_streaming_cer else "Non-Streaming" if non_streaming_cer < streaming_cer else "Tie"
        
        html += f"""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4 style="margin-top: 0;">📊 Accuracy Comparison</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th style="padding: 5px; text-align: left;">Metric</th>
                    <th style="padding: 5px; text-align: center;">Streaming</th>
                    <th style="padding: 5px; text-align: center;">Non-Streaming</th>
                    <th style="padding: 5px; text-align: center;">Winner</th>
                </tr>
                <tr style="background-color: rgba(255, 255, 255, 0.1);">
                    <td style="padding: 5px;">WER</td>
                    <td style="padding: 5px; text-align: center;">{streaming_wer:.2f}%</td>
                    <td style="padding: 5px; text-align: center;">{non_streaming_wer:.2f}%</td>
                    <td style="padding: 5px; text-align: center;"><strong>{better_wer}</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">CER</td>
                    <td style="padding: 5px; text-align: center;">{streaming_cer:.2f}%</td>
                    <td style="padding: 5px; text-align: center;">{non_streaming_cer:.2f}%</td>
                    <td style="padding: 5px; text-align: center;"><strong>{better_cer}</strong></td>
                </tr>
                <tr style="background-color: rgba(255, 255, 255, 0.1);">
                    <td style="padding: 5px;">Exact Match</td>
                    <td style="padding: 5px; text-align: center;">{'✅' if streaming_exact else '❌'}</td>
                    <td style="padding: 5px; text-align: center;">{'✅' if non_streaming_exact else '❌'}</td>
                    <td style="padding: 5px; text-align: center;"><strong>{'Streaming' if streaming_exact and not non_streaming_exact else 'Non-Streaming' if non_streaming_exact and not streaming_exact else 'Both' if streaming_exact and non_streaming_exact else 'Neither'}</strong></td>
                </tr>
            </table>
        </div>
        """
    
    html += "</div>"
    return html


def create_gradio_interface():
    """Create Gradio interface"""
    
    # Custom CSS for better Arabic text rendering
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');
    
    .arabic-text {
        font-family: 'Amiri', 'Traditional Arabic', serif;
        font-size: 18px;
        direction: rtl;
        text-align: right;
        color: #2c3e50 !important;
    }
    
    .results-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 20px;
    }
    
    /* Ensure good contrast for all text */
    div[style*="background-color: white"] {
        color: #2c3e50 !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Streaming vs Non-Streaming ASR Comparison") as app:
        gr.Markdown("""
        # 🎙️ Quran Arabic ASR: Streaming vs Non-Streaming Comparison
        
        Compare performance between **streaming** (Hybrid RNNT+CTC, configurable context) and **non-streaming** (CTC, full context) models.
        
        ### ⚠️ Important Notes
        
        - **Streaming Model**: Hybrid RNNT+CTC with configurable attention context
        - **Non-Streaming Model**: CTC-only with full attention context
        - **Different architectures**: Results may vary due to model differences, not just streaming vs non-streaming
        
        ### How to Use:
        1. **Upload or record audio** (Quranic Arabic recitation recommended)
        2. **Select streaming context window** (for streaming model only)
        3. **Optional**: Provide reference text for accuracy metrics
        4. **Select inference modes** (streaming, non-streaming, or both)
        5. Click **Transcribe** and compare results!
        
        ### Model Info:
        - **Streaming**: FastConformer Hybrid RNNT+CTC (epoch=49-step=45200.ckpt)
        - **Non-Streaming**: FastConformer CTC (FastConformer-Custom-Tokenizer.nemo)
        - **Streaming Contexts**: [[70, 13], [70, 6], [70, 1], [70, 0]]
        - **Expected WER**: ~11.44% (streaming RNNT), varies for non-streaming CTC
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio Input (16kHz recommended)"
                )
                
                reference_input = gr.Textbox(
                    label="Reference Text (Optional - for accuracy metrics)",
                    placeholder="Enter the correct Arabic transcription here...",
                    lines=3,
                    elem_classes="arabic-text"
                )
                
                streaming_context_selector = gr.Dropdown(
                    choices=[
                        "[70, 13] - Best Accuracy (0.7s past, 0.13s future)",
                        "[70, 6] - Balanced (0.7s past, 0.06s future)",
                        "[70, 1] - Low Latency (0.7s past, 0.01s future)",
                        "[70, 0] - Causal Only (0.7s past, no future)"
                    ],
                    value="[70, 13] - Best Accuracy (0.7s past, 0.13s future)",
                    label="🎚️ Streaming Context Window",
                    info="Select attention context for streaming mode. Larger right context = better accuracy but higher latency"
                )
                
                with gr.Row():
                    use_streaming = gr.Checkbox(
                        label="Enable Streaming Mode",
                        value=True,
                        info="Limited context window"
                    )
                    use_non_streaming = gr.Checkbox(
                        label="Enable Non-Streaming Mode",
                        value=True,
                        info="Full context"
                    )
                
                transcribe_btn = gr.Button("🎯 Transcribe", variant="primary", size="lg")
        
        with gr.Row():
            streaming_output = gr.HTML(label="Streaming Results")
            non_streaming_output = gr.HTML(label="Non-Streaming Results")
        
        with gr.Row():
            comparison_output = gr.HTML(label="Performance Comparison")
        
        # Examples
        gr.Markdown("### 📚 Example Quranic Verses (for reference text)")
        gr.Examples(
            examples=[
                ["بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"],
                ["الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"],
                ["الرَّحْمَٰنِ الرَّحِيمِ"],
                ["مَالِكِ يَوْمِ الدِّينِ"],
                ["إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ"],
            ],
            inputs=reference_input,
            label="Common Quranic Text Examples"
        )
        
        # Connect transcribe button
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, reference_input, streaming_context_selector, use_streaming, use_non_streaming],
            outputs=[streaming_output, non_streaming_output, comparison_output]
        )
        
        gr.Markdown("""
        ---
        ### 💡 Understanding the Results:
        
        **Speed Comparison Context:**
        - **Short audio (<10s)**: Non-streaming typically faster (better GPU parallelization)
        - **Long audio (>30s)**: Streaming benefits from lower memory usage
        - **Real-time use**: Streaming provides lower first-token latency (starts transcribing sooner)
        
        **Accuracy:**
        - **Streaming mode** uses limited context windows - suitable for real-time applications
        - **Non-streaming mode** uses full audio context - better accuracy but requires complete audio
        - For **best results**, provide 16kHz mono WAV audio
        - Reference text enables **WER/CER calculation** and exact match verification
        
        ### 📊 Expected Performance:
        - **Streaming**: ~11.44% WER, 74% exact match rate
        - **Non-streaming**: Typically slightly better accuracy (full context helps)
        - **Output agreement**: Usually identical for clear Quranic recitations
        
        Built using NVIDIA NeMo ASR framework 🚀
        """)
    
    return app


def main():
    """Main entry point"""
    print("=" * 60)
    print("Quran Arabic ASR: Streaming vs Non-Streaming Comparison")
    print("=" * 60)
    
    # Load models at startup
    load_models()
    
    print("\n" + "=" * 60)
    print("Launching Gradio interface...")
    print("=" * 60)
    
    # Create and launch app
    app = create_gradio_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
