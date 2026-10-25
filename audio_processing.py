import os
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from scipy.signal import butter, lfilter

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return lfilter(b, a, data)

def fast_normalize(data):
    """Błyskawiczna normalizacja na macierzach NumPy."""
    max_val = np.max(np.abs(data))
    return data / max_val if max_val > 0 else data

def fast_compress_dynamic_range(data, ratio=2.0):
    """Wektorowa kompresja dynamiki (AGC) w NumPy zamiast powolnego pydub."""
    compressed = np.tanh(data * ratio)
    return fast_normalize(compressed)

def prepare_dual_audio(uploaded_file):
    """
    Tworzy dwie wersje pliku całkowicie w pamięci RAM:
    - Tor 1 (Diaryzacja): naturalny głos z łagodnym odszumianiem.
    - Tor 2 (Transkrypcja): szybkie AGC w NumPy + filtr pasmowy + odszumianie.
    """
    input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    diarization_path = os.path.join(TEMP_DIR, "for_diarization.wav")
    transcription_path = os.path.join(TEMP_DIR, "for_transcription.wav")
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    audio = AudioSegment.from_file(input_path).set_channels(1).set_frame_rate(16000)
    rate = 16000
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    
    data_d = fast_normalize(samples)
    reduced_d = nr.reduce_noise(y=data_d, sr=rate, prop_decrease=0.6, stationary=True, n_jobs=1)
    sf.write(diarization_path, reduced_d, rate)
    
    data_t = fast_compress_dynamic_range(samples)
    
    lowcut_t = 80.0
    highcut_t = min(7500.0, (rate / 2.0) - 100.0)
    
    if highcut_t > lowcut_t:
        filtered_trans = bandpass_filter(data_t, lowcut_t, highcut_t, rate)
    else:
        filtered_trans = data_t
        
    reduced_t = nr.reduce_noise(y=filtered_trans, sr=rate, prop_decrease=0.8, stationary=True, n_jobs=1)
    sf.write(transcription_path, fast_normalize(reduced_t), rate)
    
    return diarization_path, transcription_path