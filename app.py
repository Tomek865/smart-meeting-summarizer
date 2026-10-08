import streamlit as st
from audio_processing import normalize_audio
from transcription import load_whisper_model, transcribe_audio
from diarization import load_diarization_model, diarize_audio

def main():
    st.set_page_config(page_title="Audio app", layout="centered")
    st.title("System Transkrypcji i Diaryzacji")

    whisper_model = load_whisper_model()
    diarization_pipeline = load_diarization_model()

    uploaded_file = st.file_uploader("Wybierz plik audio", type=['mp3', 'wav', 'm4a'])

    if uploaded_file:
        st.info(f"Przetwarzanie pliku: {uploaded_file.name}")
        
        processed_path = normalize_audio(uploaded_file)
        st.audio(processed_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Uruchom transkrypcję"):
                with st.spinner("Whisper analizuje tekst..."):
                    results = transcribe_audio(processed_path, whisper_model)
                    st.success("Transkrypcja zakończona!")
                    for res in results:
                        st.markdown(f"**[{res['start_str']} - {res['end_str']}]** {res['text']}")

        with col2:
            if st.button("Uruchom diaryzację"):
                if diarization_pipeline is None:
                    st.error("Model diaryzacji nie został poprawnie załadowany.")
                else:
                    with st.spinner("PyAnnote analizuje mówców (to może chwilę potrwać)..."):
                        diarization_results = diarize_audio(processed_path, diarization_pipeline)
                        st.success("Diaryzacja zakończona!")
                        for res in diarization_results:
                            st.markdown(f"**[{res['start_str']} - {res['end_str']}]** {res['speaker']}")

if __name__ == "__main__":
    main()