import sys
from unittest.mock import MagicMock, patch, mock_open
import matplotlib.figure
import numpy as np
import pytest

class DummyTensor:
    pass

mock_torch = MagicMock()
mock_torch.Tensor = DummyTensor 

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.types'] = MagicMock()
sys.modules['torch.fft'] = MagicMock()

sys.modules['torchvision'] = MagicMock()
sys.modules['pyannote'] = MagicMock()
sys.modules['pyannote.audio'] = MagicMock()
sys.modules['faster_whisper'] = MagicMock()

from export_utils import generate_txt_report, generate_pdf_report, clean_txt_summary, format_pdf_line
from summarization import generate_summary
from visualization import plot_diarization_waveform, plot_speaker_share_pie_chart
from diarization import diarize_audio
from audio_processing import prepare_dual_audio
from transcription import transcribe_full_audio, transcribe_chunk
from synchronization import build_synchronized_transcript

# --- DANE TESTOWE (FIXTURES) ---

@pytest.fixture
def sample_transcript():
    return [
        {"start_str": "00:00", "end_str": "00:05", "speaker": "SPEAKER_00", "text": "Cześć, zaczynamy."},
        {"start_str": "00:05", "end_str": "00:10", "speaker": "SPEAKER_01", "text": "Jasne, przejdźmy do konkretów."}
    ]

@pytest.fixture
def sample_diarization():
    return [
        {"start": 0.0, "end": 5.0, "start_str": "00:00", "end_str": "00:05", "speaker": "SPEAKER_00"}
    ]

# --- 1. TESTY EKSPORTU ---

def test_txt_report_full(sample_transcript):
    result = generate_txt_report(
        transcript=sample_transcript, 
        summary="Notatka", 
        export_mode="Pełny raport (Podsumowanie + Transkrypcja)", 
        include_timestamps=True, 
        speaker_names={"SPEAKER_00": "Jan"}
    )
    assert "Notatka" in result
    assert "Jan: Cześć" in result
    assert "--- PODSUMOWANIE SPOTKANIA ---" in result

def test_txt_report_english_and_removes_markdown_asterisks(sample_transcript):
    summary_with_bold = "- **Zadanie 1:** Wykonać projekt.\n* **Całe zdanie w gwiazdkach.**"
    result = generate_txt_report(
        transcript=sample_transcript, 
        summary=summary_with_bold, 
        export_mode="Pełny raport (Podsumowanie + Transkrypcja)", 
        include_timestamps=True, 
        speaker_names={"SPEAKER_00": "John"},
        language="Angielski"
    )
    assert "MEETING REPORT (AI AUDIO)" in result
    assert "--- MEETING SUMMARY ---" in result
    assert "--- MEETING TRANSCRIPT ---" in result
    assert "**" not in result
    assert "- Zadanie 1: Wykonać projekt." in result
    assert "* Całe zdanie w gwiazdkach." in result

def test_clean_txt_summary_and_pdf_line():
    raw = "* **Jan Kowalski:** Przedstawił raport.\n- **Ustalono nowy termin.**"
    cleaned = clean_txt_summary(raw)
    assert "**" not in cleaned
    assert "* Jan Kowalski: Przedstawił raport." in cleaned
    assert "- Ustalono nowy termin." in cleaned
    
    line1 = format_pdf_line("* **Jan Kowalski:** Przedstawił raport.")
    assert "**" not in line1
    assert "<b>Jan Kowalski:</b>" in line1
    
    line2 = format_pdf_line("- **Ustalono nowy termin.**")
    assert "**" not in line2
    assert "Ustalono nowy termin." in line2

@patch('export_utils.SimpleDocTemplate')
def test_pdf_report_generation(mock_doc_template, sample_transcript):
    with patch('export_utils.io.BytesIO') as mock_bytes_io:
        mock_bytes_instance = MagicMock()
        mock_bytes_instance.getvalue.return_value = b"%PDF-1.4 mock bytes"
        mock_bytes_io.return_value = mock_bytes_instance
        
        result = generate_pdf_report(
            transcript=sample_transcript, 
            summary="Raport testowy", 
            export_mode="Pełny raport (Podsumowanie + Transkrypcja)", 
            include_timestamps=True, 
            speaker_names={},
            language="Polski"
        )
        assert result == b"%PDF-1.4 mock bytes"

@patch('export_utils.SimpleDocTemplate')
def test_pdf_report_generation_english(mock_doc_template, sample_transcript):
    with patch('export_utils.io.BytesIO') as mock_bytes_io:
        mock_bytes_instance = MagicMock()
        mock_bytes_instance.getvalue.return_value = b"%PDF-1.4 mock bytes"
        mock_bytes_io.return_value = mock_bytes_instance
        
        result = generate_pdf_report(
            transcript=sample_transcript, 
            summary="* **Task 1:** Complete project.\n- **All done.**", 
            export_mode="Pełny raport (Podsumowanie + Transkrypcja)", 
            include_timestamps=True, 
            speaker_names={},
            language="Angielski"
        )
        assert result == b"%PDF-1.4 mock bytes"

# --- 2. TESTY NOWEJ TRANSKRYPCJI (FASTER WHISPER) ---

def test_transcribe_full_audio_handles_faster_whisper():
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.5
    mock_segment.text = " Test transkrypcji."
    
    mock_model.transcribe.return_value = ([mock_segment], None)
    
    result = transcribe_full_audio("dummy.wav", mock_model, custom_vocabulary="Słownik")
    
    assert len(result) == 1
    assert result[0] == {"start": 0.0, "end": 2.5, "text": "Test transkrypcji."}
    mock_model.transcribe.assert_called_once_with(
        "dummy.wav",
        language="pl",
        beam_size=1,
        temperature=0.0,
        best_of=1,
        vad_filter=True,
        initial_prompt="Słownik"
    )

def test_transcribe_chunk_backward_compatibility():
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = " Test transkrypcji."
    
    mock_model.transcribe.return_value = ([mock_segment], None)
    
    result = transcribe_chunk("dummy_chunk.wav", mock_model, custom_vocabulary="Słownik")
    assert result == "Test transkrypcji."

# --- 3. TESTY LLM ---

@patch('summarization.Groq') 
def test_generate_summary_api_call(mock_groq_class, sample_transcript):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Sukces."
    mock_groq_class.return_value = mock_client
    
    result = generate_summary(sample_transcript, "Biznesowy", "", "Polski")
    assert result == "Sukces."

@patch('summarization.Groq') 
def test_generate_summary_english_and_long_audio(mock_groq_class):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Success in English."
    mock_groq_class.return_value = mock_client
    
    long_script = [
        {"start_str": "00:00", "end_str": "00:10", "speaker": "SPEAKER_00", "text": "Detailed discussion " * 30}
        for _ in range(20)
    ]
    result = generate_summary(long_script, "Biznesowy", "Focus on ROI", "Angielski")
    assert result == "Success in English."
    call_args = mock_client.chat.completions.create.call_args[1]
    system_content = call_args["messages"][0]["content"]
    assert "EXTENSIVE & HIGHLY DETAILED SUMMARY REQUIRED" in system_content
    assert "SPECIAL USER INSTRUCTIONS" in system_content

# --- 4. TESTY WIZUALIZACJI ---

@patch('visualization.sf.read', return_value=(np.zeros((100, 2)), 16000))
def test_plot_waveform_returns_figure(mock_sf_read, sample_diarization):
    fig = plot_diarization_waveform("dummy.wav", sample_diarization)
    assert isinstance(fig, matplotlib.figure.Figure)

def test_plot_pie_chart_returns_figure(sample_diarization):
    fig = plot_speaker_share_pie_chart(sample_diarization)
    assert isinstance(fig, matplotlib.figure.Figure)

# --- 5. TESTY DIARYZACJI ---

@patch('diarization.sf.read', return_value=(np.random.uniform(-0.5, 0.5, 16000 * 70).astype(np.float32), 16000))
def test_diarize_audio_logic(mock_sf_read):
    mock_pipeline = MagicMock()
    class MockSegment:
        def __init__(self, start, end):
            self.start = start
            self.end = end
    
    mock_pipeline.return_value.speaker_diarization.itertracks.return_value = [
        (MockSegment(0.0, 65.5), None, "SPEAKER_00")
    ]
    
    result = diarize_audio("dummy.wav", mock_pipeline, num_speakers=1)
    
    assert len(result) == 1
    assert result[0]["end_str"] == "01:05"

# --- 6. TESTY PRZETWARZANIA AUDIO ---

@patch('audio_processing.compress_dynamic_range') 
@patch('audio_processing.nr.reduce_noise', return_value=np.zeros(100))
@patch('audio_processing.sf.write')
@patch('builtins.open', new_callable=mock_open)
@patch('audio_processing.AudioSegment')
def test_prepare_dual_audio_valid_file(mock_audio_segment, mock_file, mock_sf_write, mock_nr, mock_compress):
    mock_audio_instance = MagicMock()
    mock_audio_segment.from_file.return_value = mock_audio_instance
    mock_compress.return_value = mock_audio_instance
    
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "test.mp3"
    mock_uploaded_file.getbuffer.return_value = b"bytes"
    
    diar_path, trans_path = prepare_dual_audio(mock_uploaded_file)
    assert mock_audio_segment.from_file.called
    assert mock_sf_write.called

# --- 7. TESTY SYNCHRONIZACJI (LATE FUSION) ---

def test_build_synchronized_transcript(sample_diarization):
    mock_whisper = MagicMock()
    mock_segment = MagicMock()
    mock_segment.start = 0.5
    mock_segment.end = 4.0
    mock_segment.text = " Cześć, zaczynamy."
    mock_whisper.transcribe.return_value = ([mock_segment], None)

    result = build_synchronized_transcript("dummy.wav", sample_diarization, mock_whisper, "")
    
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"
    assert "Cześć" in result[0]["text"]
    assert result[0]["start_str"] == "00:00"
    assert result[0]["end_str"] == "00:04"

def test_build_synchronized_transcript_merges_same_speaker():
    diarization_data = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}
    ]
    mock_whisper = MagicMock()
    seg1 = MagicMock()
    seg1.start = 0.0
    seg1.end = 3.0
    seg1.text = " Pierwsze zdanie."
    
    seg2 = MagicMock()
    seg2.start = 3.5
    seg2.end = 7.0
    seg2.text = " Drugie zdanie."
    
    mock_whisper.transcribe.return_value = ([seg1, seg2], None)
    
    result = build_synchronized_transcript("dummy.wav", diarization_data, mock_whisper, "")
    
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "Pierwsze zdanie. Drugie zdanie."
    assert result[0]["start_str"] == "00:00"
    assert result[0]["end_str"] == "00:07"

def test_build_synchronized_transcript_multiple_speakers():
    diarization_data = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.1, "end": 10.0, "speaker": "SPEAKER_01"}
    ]
    mock_whisper = MagicMock()
    seg1 = MagicMock()
    seg1.start = 0.0
    seg1.end = 4.5
    seg1.text = " Dzień dobry wszystkim."
    
    seg2 = MagicMock()
    seg2.start = 5.5
    seg2.end = 8.0
    seg2.text = " Witam serdecznie."
    
    mock_whisper.transcribe.return_value = ([seg1, seg2], None)
    
    result = build_synchronized_transcript("dummy.wav", diarization_data, mock_whisper, "")
    
    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "Dzień dobry wszystkim."
    assert result[1]["speaker"] == "SPEAKER_01"
    assert result[1]["text"] == "Witam serdecznie."
