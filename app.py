import streamlit as st
from audio_processing import prepare_dual_audio
from transcription import load_whisper_model
from diarization import load_diarization_model, diarize_audio
from visualization import plot_diarization_waveform
from synchronization import build_synchronized_transcript
from summarization import generate_summary

def main():
    st.set_page_config(page_title="Audio AI Pipeline", layout="wide")
    st.title("System Transkrypcji i Diaryzacji")

    if 'last_file_id' not in st.session_state:
        st.session_state.last_file_id = None
    if 'diar_path' not in st.session_state:
        st.session_state.diar_path = None
    if 'trans_path' not in st.session_state:
        st.session_state.trans_path = None
    if 'diarization_results' not in st.session_state:
        st.session_state.diarization_results = None
    if 'combined_script' not in st.session_state:
        st.session_state.combined_script = None

    whisper_model = load_whisper_model()
    diarization_pipeline = load_diarization_model()

    uploaded_file = st.file_uploader("Wybierz plik audio", type=['mp3', 'wav', 'm4a'])

    if uploaded_file:
        st.info(f"Aktywny plik: {uploaded_file.name}")
        
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}_v5"
        
        if st.session_state.last_file_id != current_file_id:
            with st.spinner("Trwa dwutorowa optymalizacja audio (Architektura Dual-Track)..."):
                diar_path, trans_path = prepare_dual_audio(uploaded_file)
                st.session_state.diar_path = diar_path
                st.session_state.trans_path = trans_path
                st.session_state.last_file_id = current_file_id
                st.session_state.diarization_results = None
                st.session_state.combined_script = None
                
        if st.session_state.trans_path:
            st.audio(st.session_state.trans_path)
        
        st.divider()
        st.markdown("### Krok 1: Analiza Mówców (PyAnnote)")
        known_speakers = st.checkbox("Znam dokładną liczbę mówców", value=False)
        
        num_speakers = None
        if known_speakers:
            num_speakers = st.number_input("Podaj liczbę mówców:", min_value=1, max_value=10, value=2, step=1)
        else:
            st.info("System działa w trybie pełnej automatyzacji (samodzielne szacowanie mówców).")

        if st.button("Uruchom diaryzację", type="secondary", use_container_width=True):
            if diarization_pipeline is None:
                st.error("Model diaryzacji nie został poprawnie załadowany.")
            else:
                with st.spinner("PyAnnote analizuje mówców..."):
                    st.session_state.diarization_results = diarize_audio(
                        st.session_state.diar_path,
                        diarization_pipeline, 
                        num_speakers=num_speakers,
                        min_speakers=1, 
                        max_speakers=5
                    )
                    st.success("Diaryzacja zakończona pomyślnie!")
                    
        if st.session_state.diarization_results:
            with st.expander("Pokaż podział na mówców i wykres (PyAnnote)", expanded=True):
                fig = plot_diarization_waveform(st.session_state.diar_path, st.session_state.diarization_results)
                st.pyplot(fig)
                for res in st.session_state.diarization_results:
                    st.markdown(f"**[{res['start_str']} - {res['end_str']}]** {res['speaker']}")

        st.divider()
        st.markdown("### Krok 2: Inteligentna Transkrypcja (Whisper)")
        
        custom_vocab = st.text_input("Słownik branżowy (opcjonalnie)", placeholder="np. API, Streamlit, Scrum, backend")

        if st.button("Uruchom transkrypcję", type="primary", use_container_width=True):
            if not st.session_state.diarization_results:
                st.warning("Najpierw musisz wykonać Krok 1 (Diaryzację)!")
            else:
                st.session_state.combined_script = build_synchronized_transcript(
                    st.session_state.trans_path,
                    st.session_state.diarization_results, 
                    whisper_model, 
                    custom_vocabulary=custom_vocab
                )
                st.success("Transkrypcja i synchronizacja zakończone!")

        st.divider()
        if st.session_state.combined_script:
            
            st.markdown("## Połączony Scenariusz Spotkania")
            with st.expander("Rozwiń pełny dialog ze spotkania", expanded=False):
                for line in st.session_state.combined_script:
                    st.markdown(f"**[{line['start_str']} - {line['end_str']}] {line['speaker']}:** {line['text']}")
            
            st.markdown("## Inteligentne Podsumowanie (LLM)")
            if st.button("Wygeneruj notatkę ze spotkania", type="primary"):
                with st.spinner("Model czyta transkrypt i tworzy podsumowanie..."):
                    summary = generate_summary(st.session_state.combined_script)
                    if summary:
                        st.success("Gotowe!")
                        st.info(summary)

if __name__ == "__main__":
    main()