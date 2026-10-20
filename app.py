import os
import streamlit as st
from audio_processing import prepare_dual_audio
from transcription import load_whisper_model
from diarization import load_diarization_model, diarize_audio
from visualization import plot_diarization_waveform, plot_speaker_share_pie_chart
from synchronization import build_synchronized_transcript
from summarization import generate_summary
from export_utils import generate_txt_report, generate_pdf_report

def main():
    st.set_page_config(page_title="Audio AI Pipeline", layout="wide")
    st.title("System Transkrypcji i Diaryzacji")

    default_states = {
        'last_file_id': None,
        'diar_path': None,
        'trans_path': None,
        'diarization_results': None,
        'combined_script': None,
        'summary_text': None
    }
    for key, val in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = val

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
                st.session_state.summary_text = None
                
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
            with st.expander("📊 Analityka Mówców i Wykresy (PyAnnote)", expanded=True):
                col_chart1, col_chart2 = st.columns([2, 1])
                
                with col_chart1:
                    fig_wave = plot_diarization_waveform(st.session_state.diar_path, st.session_state.diarization_results)
                    st.pyplot(fig_wave)
                    
                with col_chart2:
                    fig_pie = plot_speaker_share_pie_chart(st.session_state.diarization_results)
                    st.pyplot(fig_pie)
                    
                st.markdown("#### Chronologiczny podział na mówców:")
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
            
            unique_speakers = sorted(list(set(item["speaker"] for item in st.session_state.combined_script)))
            speaker_names = {}
            with st.expander(" Personalizacja mówców (opcjonalnie nazwij rozmówców)", expanded=False):
                cols = st.columns(len(unique_speakers) if unique_speakers else 1)
                for idx, spk in enumerate(unique_speakers):
                    with cols[idx % len(cols)]:
                        speaker_names[spk] = st.text_input(f"Nazwa dla {spk}:", value=spk, key=f"rename_{spk}")
            
            st.markdown("## Połączony Scenariusz Spotkania")
            with st.expander("Rozwiń pełny dialog ze spotkania", expanded=False):
                for line in st.session_state.combined_script:
                    disp_spk = speaker_names.get(line['speaker'], line['speaker'])
                    st.markdown(f"**[{line['start_str']} - {line['end_str']}] {disp_spk}:** {line['text']}")
            
            st.markdown("## Inteligentne Podsumowanie (LLM)")
            
            col_llm1, col_llm2, col_llm3 = st.columns([1, 1, 2])
            with col_llm1:
                summary_style = st.selectbox(
                    "Wybierz styl podsumowania:",
                    options=["Biznesowy", "Naukowy / Akademicki", "Luźna notatka (Casual)"],
                    index=0
                )
            with col_llm2:
                summary_language = st.selectbox(
                    "Język notatki:",
                    options=["Polski", "Angielski"],
                    index=0
                )
            with col_llm3:
                custom_instructions = st.text_area(
                    "Dodatkowe instrukcje (opcjonalnie):",
                    placeholder="np. Skup się tylko na wypowiedziach Jana. Wypunktuj wszystkie wzmianki o budżecie projektu."
                )

            if st.button("Wygeneruj notatkę ze spotkania", type="primary"):
                with st.spinner(f"Model ({summary_style}, {summary_language}) czyta transkrypt i tworzy podsumowanie..."):
                    st.session_state["summary_text"] = generate_summary(
                        st.session_state.combined_script,
                        summary_style=summary_style,
                        custom_instructions=custom_instructions,
                        language=summary_language
                    )
            
            
            current_summary = st.session_state.get("summary_text")
            if current_summary:
                st.success("Podsumowanie gotowe!")
                st.info(current_summary)

            st.divider()
            st.markdown("## Eksport wyników (PDF / TXT)")
            
            col_opt1, col_opt2 = st.columns([2, 1])
            with col_opt1:
                export_mode = st.radio(
                    "Wybierz co chcesz wyeksportować:",
                    options=[
                        "Pełny raport (Podsumowanie + Transkrypcja)",
                        "Tylko podsumowanie",
                        "Tylko transkrypcja"
                    ],
                    index=0
                )
            with col_opt2:
                include_timestamps = st.checkbox("Dołącz czasy [MM:SS] w transkrypcji", value=True)

            if export_mode in ["Tylko podsumowanie", "Pełny raport (Podsumowanie + Transkrypcja)"] and not current_summary:
                st.warning("Wybrałeś opcję z podsumowaniem, ale jeszcze go nie wygenerowałeś powyżej. Kliknij 'Wygeneruj notatkę ze spotkania' lub wybierz 'Tylko transkrypcja'.")

            base_filename = os.path.splitext(uploaded_file.name)[0]
            
            txt_data = generate_txt_report(
                transcript=st.session_state.combined_script,
                summary=current_summary,
                export_mode=export_mode,
                include_timestamps=include_timestamps,
                speaker_names=speaker_names
            )
            
            pdf_data = generate_pdf_report(
                transcript=st.session_state.combined_script,
                summary=current_summary,
                export_mode=export_mode,
                include_timestamps=include_timestamps,
                speaker_names=speaker_names
            )

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="Pobierz jako plik TXT",
                    data=txt_data.encode("utf-8"),
                    file_name=f"raport_{base_filename}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_dl2:
                st.download_button(
                    label="Pobierz jako plik PDF",
                    data=pdf_data,
                    file_name=f"raport_{base_filename}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()