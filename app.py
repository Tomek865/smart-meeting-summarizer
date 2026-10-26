import os
import time
import streamlit as st
from audio_processing import prepare_dual_audio
from transcription import load_whisper_model
from diarization import load_diarization_model, diarize_audio
from visualization import plot_diarization_waveform, plot_speaker_share_pie_chart
from synchronization import build_synchronized_transcript
from summarization import generate_summary
from export_utils import generate_txt_report, generate_pdf_report

def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_stepper(current_step):
    steps = [
        (1, "Plik audio"),
        (2, "Analiza mówców"),
        (3, "Transkrypcja"),
        (4, "Podsumowanie i eksport")
    ]
    html = '<div class="stepper-wrapper">'
    for num, label in steps:
        if num == current_step:
            cls = "step-item step-active"
        elif num < current_step:
            cls = "step-item step-completed"
        else:
            cls = "step-item"
        html += f'<div class="{cls}">{label}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Meeting Assistant", layout="wide", initial_sidebar_state="collapsed")
    load_css("style.css")

    default_states = {
        'current_step': 1,
        'last_file_id': None,
        'uploaded_filename': None,
        'raw_audio_bytes': None,
        'diar_path': None,
        'trans_path': None,
        'diarization_results': None,
        'combined_script': None,
        'summary_text': None,
        'summary_language': "Polski",
        'custom_vocab': "",
        'speaker_names': {}
    }
    for key, val in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.markdown('<h1 class="main-header">Meeting Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automatyczna transkrypcja, rozpoznawanie mówców i generowanie notatek AI</p>', unsafe_allow_html=True)

    if "models_loaded" not in st.session_state:
        loading_screen = st.empty()
        
        loading_screen.markdown("""
        <div class="loading-container">
            <div class="custom-loader"></div>
            <div class="loading-title">Ładowanie</div>
            <div class="loading-subtitle">Wczytywanie modeli Whisper i PyAnnote. Proszę czekać.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.whisper_model = load_whisper_model()
        st.session_state.diarization_pipeline = load_diarization_model()
        
        loading_screen.empty()
        st.session_state.models_loaded = True

    whisper_model = st.session_state.whisper_model
    diarization_pipeline = st.session_state.diarization_pipeline

    render_stepper(st.session_state.current_step)

    if st.session_state.current_step == 1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Wgraj nagranie ze spotkania</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">Wybierz plik audio z dysku. System automatycznie przygotuje ścieżki sygnału w tle, zachowując oryginał do odsłuchu.</div>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Wybierz plik audio (MP3, WAV, M4A)", type=['mp3', 'wav', 'm4a'], label_visibility="collapsed")

            if uploaded_file:
                current_file_id = f"{uploaded_file.name}_{uploaded_file.size}_v6"
                st.session_state.uploaded_filename = uploaded_file.name
                
                if st.session_state.last_file_id != current_file_id:
                    st.session_state.raw_audio_bytes = uploaded_file.getvalue()
                    with st.spinner("Optymalizacja dźwięku w tle (Dual-Track Audio DSP)..."):
                        print("Rozpoczęto przygotowanie audio...")
                        start_prep = time.time()
                        diar_path, trans_path = prepare_dual_audio(uploaded_file)
                        prep_time = time.time() - start_prep
                        print(f"Przygotowanie audio zajęło: {prep_time:.2f} sekundy")
                        
                        st.session_state.diar_path = diar_path
                        st.session_state.trans_path = trans_path
                        st.session_state.last_file_id = current_file_id
                        st.session_state.diarization_results = None
                        st.session_state.combined_script = None
                        st.session_state.summary_text = None
                        st.session_state.speaker_names = {}

            if st.session_state.raw_audio_bytes:
                with st.container(border=True):
                    col_info, col_player = st.columns([1, 2], vertical_alignment="center")
                    with col_info:
                        st.markdown(f"**Oryginalny plik:**\n\n`{st.session_state.uploaded_filename}`")
                    with col_player:
                        st.audio(st.session_state.raw_audio_bytes)
                
                col_left, col_center, col_right = st.columns([1, 2, 1])
                with col_right:
                    if st.button("Analiza mówców", type="primary", use_container_width=True):
                        st.session_state.current_step = 2
                        st.rerun()

    elif st.session_state.current_step == 2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Analiza mówców (Diaryzacja)</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">Identyfikacja uczestników rozmowy oraz wyznaczenie dokładnych przedziałów czasowych ich wypowiedzi.</div>', unsafe_allow_html=True)
            
            with st.container(border=True):
                col_cfg1, col_cfg2, col_btn = st.columns([1.2, 1.3, 1.2], vertical_alignment="center")
                with col_cfg1:
                    known_speakers = st.checkbox("Znam dokładną liczbę mówców", value=False)
                with col_cfg2:
                    if known_speakers:
                        num_speakers = st.number_input("Liczba uczestników:", min_value=1, max_value=10, value=2, step=1)
                    else:
                        num_speakers = None
                        st.caption("Tryb automatyczny: detekcja od 1 do 5 mówców.")
                with col_btn:
                    run_diar = st.button("Uruchom diaryzację", type="primary", use_container_width=True)

            if run_diar:
                with st.spinner("Analiza cech biometrycznych głosów..."):
                    print("Rozpoczęto diaryzację...")
                    start_diar = time.time()
                    st.session_state.diarization_results = diarize_audio(
                        st.session_state.diar_path, diarization_pipeline, 
                        num_speakers=num_speakers, min_speakers=1, max_speakers=5
                    )
                    diar_time = time.time() - start_diar
                    print(f"Diaryzacja zajęła: {diar_time:.2f} sekundy")

            if st.session_state.diarization_results:
                st.success("Analiza mówców została zakończona.")
                
                with st.expander("Pokaż wykresy analityczne (Oś czasu i udział procentowy)", expanded=False):
                    col_chart1, col_chart2 = st.columns([1.8, 1], vertical_alignment="center")
                    with col_chart1:
                        fig_wave = plot_diarization_waveform(st.session_state.diar_path, st.session_state.diarization_results)
                        st.pyplot(fig_wave)
                    with col_chart2:
                        fig_pie = plot_speaker_share_pie_chart(st.session_state.diarization_results)
                        st.pyplot(fig_pie)

                with st.expander("Pokaż szczegółowe przedziały czasowe", expanded=False):
                    for res in st.session_state.diarization_results:
                        st.markdown(f"**[{res['start_str']} - {res['end_str']}]** {res['speaker']}")

            col_back, col_space, col_next = st.columns([1, 2, 1])
            with col_back:
                if st.button("Wstecz", type="secondary", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()
            with col_next:
                if st.session_state.diarization_results:
                    if st.button("Transkrypcja", type="primary", use_container_width=True):
                        st.session_state.current_step = 3
                        st.rerun()

    elif st.session_state.current_step == 3:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Transkrypcja i personalizacja rozmówców</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">Zamiana mowy na tekst z wykorzystaniem modelu Whisper oraz opcjonalne przypisanie imion.</div>', unsafe_allow_html=True)
            
            with st.container(border=True):
                col_v1, col_v2 = st.columns([2.5, 1], vertical_alignment="bottom")
                with col_v1:
                    st.session_state.custom_vocab = st.text_input(
                        "Słownik branżowy (opcjonalnie):",
                        value=st.session_state.custom_vocab,
                        placeholder="np. API, Scrum, React, architektura"
                    )
                with col_v2:
                    run_trans = st.button("Uruchom transkrypcję", type="primary", use_container_width=True)

            if run_trans:
                with st.spinner("Zamiana mowy na tekst i synchronizacja ze ścieżką mówców..."):
                    print("Rozpoczęto transkrypcję...")
                    start_trans = time.time()
                    st.session_state.combined_script = build_synchronized_transcript(
                        st.session_state.trans_path,
                        st.session_state.diarization_results, 
                        whisper_model, 
                        custom_vocabulary=st.session_state.custom_vocab
                    )
                    trans_time = time.time() - start_trans
                    print(f"Transkrypcja zajęła: {trans_time:.2f} sekundy")

            if st.session_state.combined_script:
                st.success("Transkrypcja i synchronizacja tekstu zakończona.")
                
                with st.expander("Opcjonalnie: Przypisz imiona do wykrytych mówców", expanded=False):
                    st.caption("Wpisane imiona od razu zaktualizują dialog oraz eksportowany raport PDF/TXT.")
                    unique_speakers = sorted(list(set(item["speaker"] for item in st.session_state.combined_script)))
                    cols = st.columns(len(unique_speakers) if unique_speakers else 1)
                    for idx, spk in enumerate(unique_speakers):
                        with cols[idx % len(cols)]:
                            default_name = st.session_state.speaker_names.get(spk, spk)
                            st.session_state.speaker_names[spk] = st.text_input(
                                f"Etykieta dla {spk}", value=default_name, key=f"rename_{spk}"
                            )

                with st.expander("Pokaż pełny zapis dialogu ze spotkania", expanded=False):
                    for line in st.session_state.combined_script:
                        disp_spk = st.session_state.speaker_names.get(line['speaker'], line['speaker'])
                        st.markdown(f"**[{line['start_str']} - {line['end_str']}] {disp_spk}:** {line['text']}")

            col_back, col_space, col_next = st.columns([1, 2, 1])
            with col_back:
                if st.button("Wstecz", type="secondary", use_container_width=True):
                    st.session_state.current_step = 2
                    st.rerun()
            with col_next:
                if st.session_state.combined_script:
                    if st.button("Podsumowanie i eksport", type="primary", use_container_width=True):
                        st.session_state.current_step = 4
                        st.rerun()

    elif st.session_state.current_step == 4:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Podsumowanie AI i eksport raportu</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">Wygeneruj ustrukturyzowaną notatkę ze spotkania i pobierz gotową dokumentację w formacie PDF lub TXT.</div>', unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown('<div class="section-label">Parametry podsumowania</div>', unsafe_allow_html=True)
                col_llm1, col_llm2 = st.columns(2)
                with col_llm1:
                    summary_style = st.selectbox("Styl notatki:", ["Biznesowy", "Naukowy / Akademicki", "Luźna notatka (Casual)"])
                with col_llm2:
                    summary_language = st.selectbox("Język wyjściowy:", ["Polski", "Angielski"])
                
                custom_instructions = st.text_input(
                    "Dodatkowe wytyczne dla modelu (opcjonalnie):",
                    placeholder="np. Wypunktuj wszystkie ustalenia dotyczące budżetu i terminów realizacji"
                )

                col_btn_space1, col_btn_gen, col_btn_space2 = st.columns([1, 1.5, 1])
                with col_btn_gen:
                    run_sum = st.button("Wygeneruj podsumowanie AI", type="primary", key="generate_summary_btn", use_container_width=True)

            if run_sum:
                with st.spinner(f"Generowanie podsumowania ({summary_style}, {summary_language})..."):
                    print("Rozpoczęto generowanie podsumowania...")
                    start_sum = time.time()
                    st.session_state["summary_language"] = summary_language
                    st.session_state["summary_text"] = generate_summary(
                        st.session_state.combined_script,
                        summary_style=summary_style,
                        custom_instructions=custom_instructions,
                        language=summary_language
                    )
                    sum_time = time.time() - start_sum
                    print(f"Generowanie podsumowania (LLM) zajęło: {sum_time:.2f} sekundy")
            
            current_summary = st.session_state.get("summary_text")
            active_lang = st.session_state.get("summary_language", summary_language)

            if current_summary:
                st.success("Notatka została wygenerowana i jest gotowa do eksportu.")
                with st.expander("Pokaż treść wygenerowanej notatki AI", expanded=True):
                    st.markdown(current_summary)

            with st.container(border=True):
                st.markdown('<div class="section-label">Pobieranie raportu</div>', unsafe_allow_html=True)
                col_opt1, col_opt2 = st.columns([2.2, 1], vertical_alignment="center")
                with col_opt1:
                    export_mode = st.radio(
                        "Zawartość dokumentu:",
                        ["Pełny raport (Podsumowanie + Transkrypcja)", "Tylko podsumowanie", "Tylko transkrypcja"],
                        horizontal=True
                    )
                with col_opt2:
                    include_timestamps = st.checkbox("Dołącz znaczniki czasu [MM:SS]", value=True)

                if "podsumowanie" in export_mode.lower() and not current_summary:
                    st.warning("Wybrany tryb eksportu wymaga wygenerowania podsumowania powyżej.")
                else:
                    base_filename = os.path.splitext(st.session_state.uploaded_filename)[0] if st.session_state.uploaded_filename else "spotkanie"
                    
                    txt_data = generate_txt_report(
                        transcript=st.session_state.combined_script,
                        summary=current_summary, 
                        export_mode=export_mode,
                        include_timestamps=include_timestamps, 
                        speaker_names=st.session_state.speaker_names,
                        language=active_lang
                    )
                    pdf_data = generate_pdf_report(
                        transcript=st.session_state.combined_script,
                        summary=current_summary, 
                        export_mode=export_mode,
                        include_timestamps=include_timestamps, 
                        speaker_names=st.session_state.speaker_names,
                        language=active_lang
                    )

                    is_en = "ang" in str(active_lang).lower() or "en" in str(active_lang).lower()
                    prefix = "Report" if is_en else "Raport"

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button("Pobierz plik TXT", data=txt_data.encode("utf-8"), file_name=f"{prefix}_{base_filename}.txt", mime="text/plain", type="secondary", use_container_width=True)
                    with col_dl2:
                        st.download_button("Pobierz plik PDF", data=pdf_data, file_name=f"{prefix}_{base_filename}.pdf", mime="application/pdf", type="primary", use_container_width=True)

            col_back, col_space, col_reset = st.columns([1, 2, 1])
            with col_back:
                if st.button("Wstecz", type="secondary", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
            with col_reset:
                if st.button("Nowa analiza", type="secondary", use_container_width=True):
                    st.session_state.current_step = 1
                    st.session_state.last_file_id = None
                    st.session_state.raw_audio_bytes = None
                    st.session_state.trans_path = None
                    st.session_state.diarization_results = None
                    st.session_state.combined_script = None
                    st.session_state.summary_text = None
                    st.session_state.summary_language = "Polski"
                    st.rerun()

    st.markdown('<p class="footer-text">© 2026 Projekt Inżynierski</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
