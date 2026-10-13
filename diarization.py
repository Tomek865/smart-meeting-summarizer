import os
from dotenv import load_dotenv
import streamlit as st
from pyannote.audio import Pipeline
import soundfile as sf
import torch

load_dotenv()

def format_time(seconds):
    """Pomocnicza funkcja do formatowania czasu w sekundach na MM:SS"""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

@st.cache_resource
def load_diarization_model():
    """
    Ładuje pipeline pyannote.audio przy użyciu tokenu z .env.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        st.error("Brak tokenu HF_TOKEN! Upewnij się, że masz plik .env")
        return None

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    return pipeline

def diarize_audio(audio_path, pipeline):
    """
    Analizuje plik audio i zwraca listę kto i kiedy mówił.
    """
    data, sample_rate = sf.read(audio_path, dtype='float32')
    
    if data.ndim == 1:
        data = data.reshape(1, -1) 
    else:
        data = data.T 
        
    waveform = torch.from_numpy(data)
    
    diarization = pipeline({
        "uri": "processed_audio", 
        "waveform": waveform, 
        "sample_rate": sample_rate
    })
    
    diarization_data = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        diarization_data.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "start_str": format_time(turn.start),
            "end_str": format_time(turn.end)
        })
        
    return diarization_data