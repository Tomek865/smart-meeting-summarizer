import os
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from scipy.signal import butter, lfilter

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    y = lfilter(b, a, data)
    return y

def prepare_dual_audio(uploaded_file):
    """
    Tworzy dwie wersje pliku zoptymalizowane pod różne zadania:
    - Tor 1 (Diaryzacja): naturalny głos z łagodnym odszumianiem (nie gubi krótkich słów).
    - Tor 2 (Transkrypcja): kompresja dynamiki AGC + filtr pasmowy dla Whispera.
    """
    input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    diarization_path = os.path.join(TEMP_DIR, "for_diarization.wav")
    transcription_path = os.path.join(TEMP_DIR, "for_transcription.wav")
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    audio = AudioSegment.from_file(input_path).set_channels(1).set_frame_rate(16000)
    
    audio_diar = normalize(audio)
    temp_diar = os.path.join(TEMP_DIR, "temp_diar.wav")
    audio_diar.export(temp_diar, format="wav")
    
    data_d, rate_d = sf.read(temp_diar)
    reduced_d = nr.reduce_noise(y=data_d, sr=rate_d, prop_decrease=0.6)
    sf.write(diarization_path, reduced_d, rate_d)
    
    if os.path.exists(temp_diar):
        os.remove(temp_diar)
    
    audio_trans = compress_dynamic_range(audio)
    audio_trans = normalize(audio_trans)
    
    temp_trans = os.path.join(TEMP_DIR, "temp_dsp.wav")
    audio_trans.export(temp_trans, format="wav")
    
    data_t, rate_t = sf.read(temp_trans)
    lowcut_t = 80.0
    highcut_t = min(7500.0, (rate_t / 2.0) - 100.0)
    
    if highcut_t > lowcut_t:
        filtered_trans = bandpass_filter(data_t, lowcut_t, highcut_t, rate_t)
    else:
        filtered_trans = data_t
        
    reduced_t = nr.reduce_noise(y=filtered_trans, sr=rate_t, prop_decrease=0.8)
    sf.write(transcription_path, reduced_t, rate_t)
    
    if os.path.exists(temp_trans):
        os.remove(temp_trans)
        
    return diarization_path, transcription_path