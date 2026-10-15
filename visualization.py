import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

def plot_diarization_waveform(audio_path, diarization_results):
    data, rate = sf.read(audio_path)
    if data.ndim > 1:
        data = data[:, 0]
    
    times = np.arange(len(data)) / float(rate)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, data, color='gray', alpha=0.5)
    
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    speaker_colors = {}
    
    for res in diarization_results:
        spk = res['speaker']
        if spk not in speaker_colors:
            speaker_colors[spk] = color_palette[len(speaker_colors) % len(color_palette)]
        
        ax.axvspan(res['start'], res['end'], color=speaker_colors[spk], alpha=0.4, 
                   label=spk if spk not in ax.get_legend_handles_labels()[1] else "")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.set_xlabel('Czas [s]')
    ax.set_ylabel('Amplituda')
    ax.set_title('Wizualizacja podziału audio na mówców')
    plt.tight_layout()
    
    return fig