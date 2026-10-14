import streamlit as st
from audio_processing import normalize_audio
from transcription import load_whisper_model, transcribe_audio
from diarization import load_diarization_model, diarize_audio
from visualization import plot_diarization_waveform
from synchronization import merge_transcription_and_diarization
from summarization import generate_summary

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
            st.markdown("### Ustawienia Transkrypcji")
            if st.button("Uruchom transkrypcję", use_container_width=True):
                with st.spinner("Whisper analizuje tekst..."):
                    st.session_state.transcription_results = transcribe_audio(processed_path, whisper_model)
                    st.success("Transkrypcja zakończona!")
            
            if st.session_state.transcription_results:
                with st.expander("Pokaż surowy tekst (tylko z Whispera)", expanded=False):
                    for res in st.session_state.transcription_results:
                        st.markdown(f"**[{res['start_str']} - {res['end_str']}]** {res['text']}")

        with col2:
            st.markdown("### Ustawienia Diaryzacji")
            known_speakers = st.checkbox("Znam dokładną liczbę mówców", value=False)
            
            num_speakers = None
            if known_speakers:
                num_speakers = st.number_input("Podaj liczbę mówców:", min_value=1, max_value=10, value=2, step=1)
            else:
                st.info("Automatyczne szacowanie liczby mówców (1-5).")

            if st.button("Uruchom diaryzację", use_container_width=True):
                if diarization_pipeline is None:
                    st.error("Model diaryzacji nie został poprawnie załadowany.")
                else:
                    with st.spinner("PyAnnote analizuje mówców..."):
                        st.session_state.diarization_results = diarize_audio(
                            processed_path, 
                            diarization_pipeline, 
                            num_speakers=num_speakers,
                            min_speakers=1, 
                            max_speakers=5
                        )
                        st.success("Diaryzacja zakończona!")
                        
            if st.session_state.diarization_results:
                with st.expander("Pokaż podział na mówców i wykres (PyAnnote)", expanded=False):
                    fig = plot_diarization_waveform(processed_path, st.session_state.diarization_results)
                    st.pyplot(fig)
                    for res in st.session_state.diarization_results:
                        st.markdown(f"**[{res['start_str']} - {res['end_str']}]** {res['speaker']}")

        st.divider()
        if st.session_state.transcription_results and st.session_state.diarization_results:
            
            combined_script = merge_transcription_and_diarization(
                st.session_state.transcription_results, 
                st.session_state.diarization_results
            )
            
            st.markdown("## Połączony Scenariusz Spotkania")
            with st.expander("Rozwiń pełny dialog ze spotkania", expanded=False):
                for line in combined_script:
                    st.markdown(f"**[{line['start_str']} - {line['end_str']}] {line['speaker']}:** {line['text']}")
            
            st.markdown("## Inteligentne Podsumowanie (LLM)")
            if st.button("Wygeneruj notatkę ze spotkania", type="primary"):
                with st.spinner("Model czyta transkrypt i tworzy podsumowanie..."):
                    summary = generate_summary(combined_script)
                    if summary:
                        st.success("Gotowe!")
                        st.info(summary)

if __name__ == "__main__":
    main()