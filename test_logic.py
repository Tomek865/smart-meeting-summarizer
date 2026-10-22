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

from export_utils import generate_txt_report, generate_pdf_report
from summarization import generate_summary
from visualization import plot_diarization_waveform, plot_speaker_share_pie_chart
from diarization import diarize_audio
from audio_processing import prepare_dual_audio
from transcription import transcribe_chunk
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
            speaker_names={}
        )
        assert result == b"%PDF-1.4 mock bytes"

# --- 2. TESTY NOWEJ TRANSKRYPCJI (FASTER WHISPER) ---

def test_transcribe_chunk_handles_faster_whisper():
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " Test transkrypcji."
    
    mock_model.transcribe.return_value = ([mock_segment], None)
    
    result = transcribe_chunk("dummy_chunk.wav", mock_model, custom_vocabulary="Słownik")
    
    assert result == "Test transkrypcji."
    mock_model.transcribe.assert_called_once_with(
        "dummy_chunk.wav", beam_size=5, language="pl", initial_prompt="Słownik"
    )

# --- 3. TESTY LLM ---

@patch('summarization.Groq') 
def test_generate_summary_api_call(mock_groq_class, sample_transcript):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Sukces."
    mock_groq_class.return_value = mock_client
    
    result = generate_summary(sample_transcript, "Biznesowy", "", "Polski")
    assert result == "Sukces."

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
@patch('audio_processing.sf.read', return_value=(np.zeros(100), 16000))
@patch('builtins.open', new_callable=mock_open)
@patch('audio_processing.AudioSegment')
def test_prepare_dual_audio_valid_file(mock_audio_segment, mock_file, mock_sf_read, mock_sf_write, mock_nr, mock_compress):
    mock_audio_instance = MagicMock()
    mock_audio_segment.from_file.return_value = mock_audio_instance
    mock_compress.return_value = mock_audio_instance
    
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "test.mp3"
    mock_uploaded_file.getbuffer.return_value = b"bytes"
    
    diar_path, trans_path = prepare_dual_audio(mock_uploaded_file)
    assert mock_audio_segment.from_file.called
    assert mock_sf_read.called

@patch('synchronization.AudioSegment')
def test_build_synchronized_transcript(mock_audio_segment, sample_diarization):
    mock_whisper = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " Cześć, zaczynamy."
    mock_whisper.transcribe.return_value = ([mock_segment], None)

    mock_audio_instance = MagicMock()
    mock_audio_segment.from_file.return_value = mock_audio_instance
    
    result = build_synchronized_transcript("dummy.wav", sample_diarization, mock_whisper, "")
    
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"
    assert "Cześć" in result[0]["text"]