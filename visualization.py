import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

def get_speaker_colors(diarization_results):
    """
    Zwraca spójny słownik kolorów dla mówców, aby na wszystkich 
    wykresach dana osoba miała dokładnie ten sam kolor.
    """
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    speaker_colors = {}
    for res in diarization_results:
        spk = res['speaker']
        if spk not in speaker_colors:
            speaker_colors[spk] = color_palette[len(speaker_colors) % len(color_palette)]
    return speaker_colors

def plot_diarization_waveform(audio_path, diarization_results):
    """Rysuje liniowy wykres fali dźwiękowej (timeline)."""
    data, rate = sf.read(audio_path)
    if data.ndim > 1:
        data = data[:, 0]
    
    times = np.arange(len(data)) / float(rate)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, data, color='gray', alpha=0.5)
    
    speaker_colors = get_speaker_colors(diarization_results)
    
    for res in diarization_results:
        spk = res['speaker']
        ax.axvspan(res['start'], res['end'], color=speaker_colors[spk], alpha=0.4, 
                   label=spk if spk not in ax.get_legend_handles_labels()[1] else "")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.set_xlabel('Czas [s]')
    ax.set_ylabel('Amplituda')
    ax.set_title('Wizualizacja osi czasu (Timeline)')
    plt.tight_layout()
    
    return fig

def plot_speaker_share_pie_chart(diarization_results):
    """Rysuje wykres kołowy obrazujący procentowy udział w rozmowie."""
    speaker_times = {}
    for res in diarization_results:
        dur = res['end'] - res['start']
        spk = res['speaker']
        speaker_times[spk] = speaker_times.get(spk, 0.0) + dur
        
    labels = list(speaker_times.keys())
    sizes = list(speaker_times.values())
    
    speaker_colors = get_speaker_colors(diarization_results)
    colors = [speaker_colors[spk] for spk in labels]
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=140, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'alpha': 0.85}
    )
    
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=11)
    
    ax.set_title('Udział procentowy mówców', fontsize=13, weight='bold', pad=15)
    plt.tight_layout()
    
    return fig