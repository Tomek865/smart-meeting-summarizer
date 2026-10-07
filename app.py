import streamlit as st
from pydub import AudioSegment
import os

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def normalize_audio(uploaded_file):
    """
    Zapisuje wgrany plik i konwertuje go do formatu optymalnego 
    dla Whisper i PyAnnote (WAV, 16kHz, Mono).
    """
    input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    output_path = os.path.join(TEMP_DIR, "processed_audio.wav")
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    
    return output_path

def main():
    st.set_page_config(page_title="App", layout="centered")
    st.title("System Transkrypcji i Diaryzacji")
    st.write("Wgraj plik audio, aby rozpocząć proces.")

    uploaded_file = st.file_uploader("Wybierz plik", type=['mp3', 'wav', 'm4a'])

    if uploaded_file:
        st.info(f"Przetwarzanie pliku: {uploaded_file.name}")
        
        try:
            processed_path = normalize_audio(uploaded_file)
            st.success("Plik znormalizowany pomyślnie (WAV, 16kHz, Mono).")
            st.audio(processed_path)
            
        except Exception as e:
            st.error(f"Błąd przetwarzania: {e}")
            st.warning("Upewnij się, że masz zainstalowane ffmpeg w systemie.")

if __name__ == "__main__":
    main()