import streamlit as st
import soundfile as sf
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

def transcribe_chunk(audio_chunk_path, model, custom_vocabulary=""):
    """
    Przyjmuje krótki wycinek audio i zwraca sam tekst w najkrótszym możliwym czasie.
    Nie wie nic o mówcach ani czasach - robi tylko transkrypcję.
    """
    initial_prompt = custom_vocabulary if custom_vocabulary.strip() else None
    
    audio_data, _ = sf.read(audio_chunk_path, dtype="float32")
    
    segments, _ = model.transcribe(
        audio_data, 
        beam_size=1,
        best_of=1,
        temperature=0.0,                   
        condition_on_previous_text=False,  
        without_timestamps=True,           
        vad_filter=True,                   
        language="pl",
        initial_prompt=initial_prompt
    )
    
    chunk_text = " ".join([s.text.strip() for s in segments])
    return chunk_text