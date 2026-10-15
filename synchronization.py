import os
from pydub import AudioSegment
import streamlit as st
from transcription import transcribe_chunk

def build_synchronized_transcript(audio_path, diarization_data, whisper_model, custom_vocabulary=""):
    """
    Zarządza osią czasu: tnie audio na podstawie diaryzacji, 
    zleca transkrypcję i łączy wyniki w spójny dialog.
    """
    audio = AudioSegment.from_file(audio_path)
    combined_script = []
    
    sorted_diarization = sorted(diarization_data, key=lambda x: x['start'])
    total_segments = len(sorted_diarization)
    
    progress_bar = st.progress(0, text="Synchronizacja: transkrybowanie wypowiedzi krok po kroku...")

    for i, segment in enumerate(sorted_diarization):
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        
        chunk = audio[start_ms:end_ms]
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        
        chunk_text = transcribe_chunk(chunk_path, whisper_model, custom_vocabulary)
        
        if chunk_text:
            combined_script.append({
                "start_str": segment['start_str'],
                "end_str": segment['end_str'],
                "speaker": segment['speaker'],
                "text": chunk_text
            })
            
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
            
        progress_bar.progress((i + 1) / total_segments, text=f"Zsynchronizowano {i+1}/{total_segments} fragmentów")
        
    progress_bar.empty()
    
    merged_script = []
    for current in combined_script:
        if not merged_script:
            merged_script.append(current)
            continue
            
        previous = merged_script[-1]
        if previous['speaker'] == current['speaker']:
            previous['end_str'] = current['end_str']
            previous['text'] += " " + current['text']
        else:
            merged_script.append(current)

    return merged_script