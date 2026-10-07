import streamlit as st
from pydub import AudioSegment
from faster_whisper import WhisperModel
import os

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def normalize_audio(uploaded_file):
    input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    output_path = os.path.join(TEMP_DIR, "processed_audio.wav")
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    
    return output_path

def format_time(seconds):
    """Pomocnicza funkcja do formatowania czasu w sekundach na MM:SS"""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

@st.cache_resource
def load_whisper_model():
    """
    Ładuje model tylko raz i trzyma go w pamięci cache Streamlita.
    Dzięki temu kolejne transkrypcje będą startować od razu.
    """
    return WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_audio(audio_path, model):
    """
    Wykonuje transkrypcję i zwraca listę segmentów.
    """
    segments, info = model.transcribe(audio_path, beam_size=5, language="pl")
    
    transcription_data = []
    for segment in segments:
        transcription_data.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
        
    return transcription_data

def main():
    st.set_page_config(page_title="App", layout="centered")
    st.title("System Transkrypcji i Diaryzacji")

    whisper_model = load_whisper_model()

    uploaded_file = st.file_uploader("Wybierz plik", type=['mp3', 'wav', 'm4a'])

    if uploaded_file:
        st.info(f"Przetwarzanie pliku: {uploaded_file.name}")
        
        processed_path = normalize_audio(uploaded_file)
        st.audio(processed_path)
        
        if st.button("Uruchom transkrypcję (Whisper)"):
            with st.spinner("Model analizuje dźwięk... To może potrwać od kilku sekund do paru minut."):
                results = transcribe_audio(processed_path, whisper_model)
                
                st.success("Transkrypcja zakończona!")
                
                st.markdown("### Wynik transkrypcji:")
                for res in results:
                    start_str = format_time(res['start'])
                    end_str = format_time(res['end'])
                    st.markdown(f"**[{start_str} - {end_str}]** {res['text']}")

if __name__ == "__main__":
    main()