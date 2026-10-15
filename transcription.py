import streamlit as st
from faster_whisper import WhisperModel

@st.cache_resource
def load_whisper_model():
    """Ładuje model Whisper z cache Streamlita."""
    return WhisperModel("medium", device="cpu", compute_type="int8")

def transcribe_chunk(audio_chunk_path, model, custom_vocabulary=""):
    """
    Przyjmuje krótki wycinek audio i zwraca sam tekst. 
    Nie wie nic o mówcach ani czasach - robi tylko transkrypcję.
    """
    initial_prompt = custom_vocabulary if custom_vocabulary.strip() else None
    
    segments, _ = model.transcribe(
        audio_chunk_path, 
        beam_size=5, 
        language="pl",
        initial_prompt=initial_prompt
    )
    
    chunk_text = " ".join([s.text.strip() for s in segments])
    return chunk_text