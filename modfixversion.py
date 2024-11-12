import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Définition des paramètres pour 5 filtres passe-bande
filtres_passe_bande = [
    {"freq_basse": 20, "freq_haute": 200, "Q": 30.5, "gain": 1.0},
    {"freq_basse": 200, "freq_haute": 900, "Q": 0.5, "gain": 1.2},
    {"freq_basse": 1000, "freq_haute": 3000, "Q": 10.5, "gain": 0.8},
    {"freq_basse": 4000, "freq_haute": 9000, "Q": 1.0, "gain": 1.9},
    {"freq_basse": 10000, "freq_haute": 16000, "Q": 20.5, "gain": 2.0}
]

def filtre_passe_bande(signal, freq_ech, freq_basse, freq_haute, Q, gain=1.0):
    order = 2
    nyquist = 0.5 * freq_ech
    normal_cutoff_basse = freq_basse / nyquist
    normal_cutoff_haute = freq_haute / nyquist
    
    b, a = butter(order, [normal_cutoff_basse, normal_cutoff_haute], btype='band')
    signal_filtre = lfilter(b, a, signal)
    return signal_filtre * gain

def appliquer_filtres(signal, freq_ech, filtres):
    return [filtre_passe_bande(signal, freq_ech, 
                               filtre["freq_basse"], 
                               filtre["freq_haute"], 
                               filtre["Q"], 
                               filtre["gain"]) for filtre in filtres]

def filtre_passe_bas1(signal, freq_ech, freq_coupure, gain):
    nyquist = 0.5 * freq_ech
    normal_cutoff = freq_coupure / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal) * gain

def filtreNumTemp(data):
    a0, a1 = 0.1153, -0.1153
    b1, b2 = 1.8777, -0.8809
    filtered_data = np.zeros_like(data)
    filtered_data[0] = a0 * data[0]
    filtered_data[1] = a0 * data[1] + a1 * data[0] + b1 * filtered_data[0]
    for i in range(2, len(data)):
        filtered_data[i] = (a0 * data[i] + a1 * data[i-1] + 
                            b1 * filtered_data[i-1] + b2 * filtered_data[i-2])
    return filtered_data

def generer_signal(type, freq_ech):
    if type == 0:
        # Signal sinusoïdal
        f_sin, duree = 2000, 1
        t = np.linspace(0, duree, int(freq_ech * duree), endpoint=False)
        return np.sin(2 * np.pi * f_sin * t), t, False
    elif type == 1:
        # Fichier wav
        freq_ech, data = wavfile.read('LW_20M_amis.wav')
        t = np.linspace(0, len(data) / freq_ech, len(data), endpoint=False)
        return data, t, True
    else:
        # Impulsion
        duree = 3
        data = np.zeros(freq_ech * duree)
        data[0] = 1
        t = np.linspace(0, duree, int(freq_ech * duree), endpoint=False)
        return data, t, False

def main():
    type = 0
    freq_ech = 44100
    data, t, creationWavFiltre = generer_signal(type, freq_ech)

    # Vérifier si le fichier est stéréo ou mono
    if len(data.shape) > 1:
        data = data[:, 0]

    fft_result = np.fft.fft(data)
    frequences = np.fft.fftfreq(len(fft_result), d=1/freq_ech)

    # Application des filtres passe-bande
    signaux_filtres = appliquer_filtres(data, freq_ech, filtres_passe_bande)
    fft_results_filtres = [np.fft.fft(signal) for signal in signaux_filtres]

    # Application du filtre numérique temporel
    signal_filtre_temp = filtreNumTemp(data)
    fft_result_filtre_temp = np.fft.fft(signal_filtre_temp)

    # Écrire le fichier WAV
    if creationWavFiltre:
        signal_combine = np.sum(signaux_filtres, axis=0)
        signal_combine_normalise = np.clip(signal_combine, -32768, 32767).astype(np.int16)
        wavfile.write('wav_filtre_combine.wav', freq_ech, signal_combine_normalise)

        signal_filtre_temp_normalise = np.clip(signal_filtre_temp, -32768, 32767).astype(np.int16)
        wavfile.write('wav_filtre_temp.wav', freq_ech, signal_filtre_temp_normalise)

    # Création des graphiques
    plt.figure(figsize=(15, 15))

    # Signal temporel
    plt.subplot(3, 1, 1)
    plt.plot(t, data, label='Signal Origine', alpha=0.7)
    for i, signal_filtre in enumerate(signaux_filtres):
        plt.plot(t, signal_filtre, label=f'Filtre {i+1}', alpha=0.7)
    plt.plot(t, signal_filtre_temp, label='Filtre Numérique Temporel', linestyle='--')
    plt.xlabel('Temps [s]')
    plt.ylabel('Amplitude')
    plt.title('Signaux temporels')
    plt.legend()

    # FFT des signaux filtrés passe-bande
    plt.subplot(3, 1, 2)
    plt.plot(frequences[:len(frequences)//2], np.abs(fft_result)[:len(fft_result)//2], label='Original', alpha=0.7)
    for i, fft_result_filtre in enumerate(fft_results_filtres):
        plt.plot(frequences[:len(frequences)//2], np.abs(fft_result_filtre)[:len(fft_result_filtre)//2], 
                 label=f'Filtre {i+1}', alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Spectres de Fourier - Filtres Passe-Bande')
    plt.legend()

    # FFT du signal filtré numériquement
    plt.subplot(3, 1, 3)
    plt.plot(frequences[:len(frequences)//2], np.abs(fft_result)[:len(fft_result)//2], label='Original', alpha=0.7)
    plt.plot(frequences[:len(frequences)//2], np.abs(fft_result_filtre_temp)[:len(fft_result_filtre_temp)//2], 
             label='Filtre Numérique Temporel', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Spectre de Fourier - Filtre Numérique Temporel')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()