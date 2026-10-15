import os
from dotenv import load_dotenv
import streamlit as st
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np
import torch

load_dotenv()

def format_time(seconds):
    """Pomocnicza funkcja do formatowania czasu w sekundach na MM:SS"""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

@st.cache_resource
def load_diarization_model():
    """
    Ładuje pipeline pyannote.audio przy użyciu tokenu z .env.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        st.error("Brak tokenu HF_TOKEN! Upewnij się, że masz plik .env")
        return None

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    return pipeline

def diarize_audio(audio_path, pipeline, num_speakers=None, min_speakers=1, max_speakers=5):
    """
    Analizuje plik audio, wykrywa mówców i domyka pominięte przez AI krótkie słowa na podstawie energii fali.
    """
    try:
        hparams = pipeline.parameters(instantiated=True)
        if "clustering" in hparams:
            hparams["clustering"]["threshold"] = 0.74
            if "min_cluster_size" in hparams["clustering"]:
                hparams["clustering"]["min_cluster_size"] = 15
        if "segmentation" in hparams and "min_duration_off" in hparams["segmentation"]:
            hparams["segmentation"]["min_duration_off"] = 0.1
        pipeline.instantiate(hparams)
    except Exception as e:
        print(f"Informacja: Nie nadpisano hiperparametrów: {e}")

    data, sample_rate = sf.read(audio_path, dtype='float32')
    
    mono_data = data if data.ndim == 1 else data[:, 0]
    
    if data.ndim == 1:
        tensor_data = data.reshape(1, -1)
    else:
        tensor_data = data.T 
        
    waveform = torch.from_numpy(tensor_data)
    audio_file = {"waveform": waveform, "sample_rate": sample_rate}

    if num_speakers is not None:
        diarization = pipeline(audio_file, num_speakers=num_speakers)
    else:
        diarization = pipeline(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
        
    raw_data = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        raw_data.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    if not raw_data:
        return []

    merged_data = []
    for current in raw_data:
        if not merged_data:
            merged_data.append(current)
            continue
            
        previous = merged_data[-1]
        
        if previous["speaker"] == current["speaker"] and (current["start"] - previous["end"]) < 2.0:
            previous["end"] = max(previous["end"], current["end"])
        else:
            merged_data.append(current)

    for i in range(len(merged_data) - 1):
        prev_seg = merged_data[i]
        next_seg = merged_data[i + 1]
        gap_start = prev_seg["end"]
        gap_end = next_seg["start"]
        gap_duration = gap_end - gap_start
        
        if 0.1 < gap_duration <= 2.0:
            start_sample = int(gap_start * sample_rate)
            end_sample = int(gap_end * sample_rate)
            gap_audio = np.abs(mono_data[start_sample:end_sample])
            
            if len(gap_audio) > 0 and np.max(gap_audio) > 0.08:
                active_indices = np.where(gap_audio > 0.05)[0]
                if len(active_indices) > 0:
                    peak_start_time = gap_start + (active_indices[0] / sample_rate)
                    peak_end_time = gap_start + (active_indices[-1] / sample_rate)
                    
                    dist_to_prev = peak_start_time - prev_seg["end"]
                    dist_to_next = next_seg["start"] - peak_end_time
                    
                    if dist_to_prev <= dist_to_next:
                        prev_seg["end"] = min(next_seg["start"], peak_end_time + 0.1)
                    else:
                        next_seg["start"] = max(prev_seg["end"], peak_start_time - 0.1)

    speaker_mapping = {}
    next_idx = 0
    final_data = []
    
    for seg in merged_data:
        orig_spk = seg["speaker"]
        if orig_spk not in speaker_mapping:
            speaker_mapping[orig_spk] = f"SPEAKER_{next_idx:02d}"
            next_idx += 1
            
        final_data.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker_mapping[orig_spk],
            "start_str": format_time(seg["start"]),
            "end_str": format_time(seg["end"])
        })
            
    return final_data