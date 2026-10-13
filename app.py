import streamlit as st
from audio_processing import normalize_audio
from transcription import load_whisper_model, transcribe_audio
from diarization import load_diarization_model, diarize_audio
from visualization import plot_diarization_waveform
from synchronization import merge_transcription_and_diarization

def main():
    st.set_page_config(page_title="Audio AI Pipeline", layout="wide")
    st.title("System Transkrypcji i Diaryzacji")

    if 'transcription_results' not in st.session_state:
        st.session_state.transcription_results = None
    if 'diarization_results' not in st.session_state:
        st.session_state.diarization_results = None

    whisper_model = load_whisper_model()
    diarization_pipeline = load_diarization_model()

    uploaded_file = st.file_uploader("Wybierz plik audio", type=['mp3', 'wav', 'm4a'])

    if uploaded_file:
        st.info(f"Przetwarzanie pliku: {uploaded_file.name}")
        
        processed_path = normalize_audio(uploaded_file)
        st.audio(processed_path)
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Uruchom transkrypcję", use_container_width=True):
                with st.spinner("Whisper analizuje tekst..."):
                    st.session_state.transcription_results = transcribe_audio(processed_path, whisper_model)
                    st.success("Transkrypcja zakończona!")
            
            if st.session_state.transcription_results:
                st.markdown("### Wynik Transkrypcji")
                for res in st.session_state.transcription_results:
                    st.markdown(f"**[{res['start_str']} - {res['end_str']}]** {res['text']}")

        with col2:
            if st.button("Uruchom diaryzację", use_container_width=True):
                if diarization_pipeline is None:
                    st.error("Model diaryzacji nie został poprawnie załadowany.")
                else:
                    with st.spinner("PyAnnote analizuje mówców..."):
                        st.session_state.diarization_results = diarize_audio(processed_path, diarization_pipeline)
                        st.success("Diaryzacja zakończona!")
                        
            if st.session_state.diarization_results:
                st.markdown("### Wykres Mówców")
                fig = plot_diarization_waveform(processed_path, st.session_state.diarization_results)
                st.pyplot(fig)
                
                st.markdown("### Rozpoznani Mówcy (Czas)")
                for res in st.session_state.diarization_results:
                    st.markdown(f"**[{res['start_str']} - {res['end_str']}]** {res['speaker']}")
        st.divider()
        if st.session_state.transcription_results and st.session_state.diarization_results:
            st.markdown("## Połączony Scenariusz Spotkania")
            
            combined_script = merge_transcription_and_diarization(
                st.session_state.transcription_results, 
                st.session_state.diarization_results
            )
            
            for line in combined_script:
                st.markdown(f"**[{line['start_str']} - {line['end_str']}] {line['speaker']}:** {line['text']}")

if __name__ == "__main__":
    main()