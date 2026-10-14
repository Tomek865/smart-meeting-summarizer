from faster_whisper import WhisperModel
import streamlit as st

def format_time(seconds):
    """Pomocnicza funkcja do formatowania czasu (MM:SS)"""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

@st.cache_resource
def load_whisper_model():
    """Ładuje model Whisper z cache Streamlita."""
    return WhisperModel("medium", device="cpu", compute_type="int8")

def transcribe_audio(audio_path, model):
    """Wykonuje transkrypcję z polskim językiem i zwraca listę segmentów."""
    segments, _ = model.transcribe(
        audio_path, 
        beam_size=5, 
        language="pl",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    transcription_data = []
    for segment in segments:
        transcription_data.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "start_str": format_time(segment.start),
            "end_str": format_time(segment.end)
        })
        
    return transcription_data