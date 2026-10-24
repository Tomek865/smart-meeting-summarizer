#Meeting Summarizer

Aplikacja internetowa stworzona w ramach projektu inżynierskiego. Służy do automatycznej transkrypcji spotkań, rozpoznawania mówców (diaryzacji) oraz generowania inteligentnych podsumowań i notatek z wykorzystaniem dużych modeli językowych (LLM).

#Główne funkcje

Dual-Track Audio DSP: Wstępne przetwarzanie i optymalizacja dźwięku.

Diaryzacja : Rozpoznawanie nakładających się głosów i przypisywanie ich do konkretnych przedziałów czasowych (PyAnnote 3.1).

Transkrypcja mowy : Zamiana mowy na tekst z pomijaniem ciszy (VAD) i wsparciem dla słowników branżowych (Faster-Whisper).

Generowanie podsumowań AI: Automatyczne tworzenie ustrukturyzowanych notatek w różnych stylach przy użyciu Groq API.

Eksport raportów: Pobieranie gotowej dokumentacji ze spotkania w formacie PDF oraz TXT.

#Wykorzystane technologie

Język: Python 3.11

Interfejs użytkownika: Streamlit

Przetwarzanie audio: PyDub, librosa, noisereduce

Modele AI: pyannote.audio, faster-whisper, Groq API

Optymalizacja wydajności: ONNX Runtime, kwantyzacja int8

Testowanie: pytest, unittest.mock, jiwer (WER), pyannote.metrics (DER)

#Wymagania wstępne

Zainstalowany Python 3.11.x.

Zainstalowany FFmpeg dodany do zmiennych środowiskowych PATH.

Aktywne tokeny dostępowe do:

HuggingFace (dla modelu PyAnnote)

Groq Cloud (dla modelu językowego)

⚙️ Instalacja i konfiguracja

Klonowanie repozytorium:

git clone <adres-repozytorium>
cd smart-meeting-summarizer


#Utworzenie i aktywacja środowiska wirtualnego:

python -m venv venv
Na systemie Windows:
venv\Scripts\activate
Na systemie Linux/Mac:
source venv/bin/activate


#Instalacja zależności:

pip install -r requirements.txt


#Konfiguracja zmiennych środowiskowych:
Utwórz plik .env w głównym katalogu projektu i dodaj swoje klucze API (lub umieść je bezpiecznie w kodzie/Streamlit secrets, zgodnie z implementacją):

HUGGINGFACE_TOKEN=twoj_token_hf
GROQ_API_KEY=twoj_klucz_groq


#Uruchomienie aplikacji

Aby uruchomić aplikację w przeglądarce, wpisz w terminalu z aktywnym środowiskiem:

streamlit run app.py
