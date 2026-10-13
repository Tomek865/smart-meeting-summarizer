import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

def plot_diarization_waveform(audio_path, diarization_results):
    """
    Rysuje falę dźwiękową i nakłada na nią kolorowe bloki oznaczające mówców.
    """
    # Wczytanie audio
    data, sample_rate = sf.read(audio_path, dtype='float32')
    
    # Jeśli plik jest stereo (2 kanały), uśredniamy go do mono, żeby wykres był czytelniejszy
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    # Wygenerowanie osi czasu w sekundach
    duration = len(data) / sample_rate
    time_axis = np.linspace(0, duration, len(data))

    # Inicjalizacja wykresu
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Rysowanie szarej fali dźwiękowej (waveform)
    ax.plot(time_axis, data, color='gray', alpha=0.5, linewidth=0.5)

    # Kolory przypisane do mówców (obsłuży do 4 różnych głosów, dla więcej użyje domyślnego)
    colors = {
        'SPEAKER_00': 'blue', 
        'SPEAKER_01': 'orange', 
        'SPEAKER_02': 'green',
        'SPEAKER_03': 'purple'
    }

    # Rysowanie bloków dla każdego segmentu diaryzacji
    for res in diarization_results:
        speaker = res['speaker']
        color = colors.get(speaker, 'red')
        
        # axvspan tworzy pionowy blok od czasu 'start' do 'end'
        ax.axvspan(res['start'], res['end'], color=color, alpha=0.3, label=speaker)

    # Filtrowanie legendy, aby każdy mówca (kolor) wyświetlił się w niej tylko raz
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")
    ax.set_title("Wizualizacja podziału audio na mówców")
    
    # Ciasne dopasowanie układu, żeby dobrze wyglądało w Streamlit
    plt.tight_layout()
    
    return fig