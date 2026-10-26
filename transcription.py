import streamlit as st
from faster_whisper import WhisperModel

@st.cache_resource
def load_whisper_model():
    """Ładuje zoptymalizowany model Whisper z cache Streamlita."""
    return WhisperModel(
        "medium", 
        device="cpu", 
        compute_type="int8", 
        cpu_threads=8,
        num_workers=2
    )

def transcribe_full_audio(audio_path, model, custom_vocabulary=""):
    """
    Przyjmuje cały plik audio na raz i wykonuje ciągłą transkrypcję (Late Fusion).
    Zwraca listę słowników: [{"start": float, "end": float, "text": str}, ...].
    """
    initial_prompt = custom_vocabulary.strip() if custom_vocabulary and custom_vocabulary.strip() else None

    segments, _ = model.transcribe(
        audio_path,
        language="pl",
        beam_size=1,
        temperature=0.0,
        best_of=1,
        vad_filter=True,
        initial_prompt=initial_prompt
    )

    transcript_segments = []
    for s in segments:
        text = s.text.strip()
        if text:
            try:
                start = float(s.start)
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(s.end)
            except (TypeError, ValueError):
                end = 0.0

            transcript_segments.append({
                "start": start,
                "end": end,
                "text": text
            })

    return transcript_segments

def transcribe_chunk(audio_chunk_path, model, custom_vocabulary=""):
    """Zachowane dla kompatybilności wstecznej."""
    segments = transcribe_full_audio(audio_chunk_path, model, custom_vocabulary)
    return " ".join([s["text"] for s in segments])
