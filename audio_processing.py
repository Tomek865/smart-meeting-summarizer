import os
from pydub import AudioSegment
import soundfile as sf
import noisereduce as nr

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def normalize_audio(uploaded_file):
    """
    Zapisuje wgrany plik, konwertuje go do formatu WAV, 16kHz, Mono
    oraz nakłada zaawansowany algorytm redukcji szumów.
    """
    input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    output_path = os.path.join(TEMP_DIR, "processed_audio.wav")
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    temp_wav_path = os.path.join(TEMP_DIR, "temp_for_nr.wav")
    audio.export(temp_wav_path, format="wav")
    
    data, rate = sf.read(temp_wav_path)
    
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
    
    sf.write(output_path, reduced_noise, rate)
    
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)
        
    return output_path