import os
from pydub import AudioSegment

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def normalize_audio(uploaded_file):
    """
    Zapisuje wgrany plik i konwertuje go do formatu WAV, 16kHz, Mono.
    """
    input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    output_path = os.path.join(TEMP_DIR, "processed_audio.wav")
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    
    return output_path