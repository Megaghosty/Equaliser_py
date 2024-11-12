import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Définition des paramètres pour 5 filtres numériques
filtres_numeriques = [
    {"a0": 0.2183, "a1": -0.2183, "b1": 1.7505, "b2": -0.7661, "gain": 2.0},
    {"a0": 0.194, "a1": -0.194, "b1": 1.5568, "b2": -0.6813, "gain": 1.2},
    {"a0": 0.1989, "a1": -0.1989, "b1": -1.2747, "b2": -0.5578, "gain": 0.8},
    {"a0": 0.1249, "a1": -0.1249, "b1": 1.0023, "b2": -0.4386, "gain": 1.9},
    {"a0": 0.0972, "a1": -0.00972, "b1": 0.7800, "b2": -0.3414, "gain": 2.0}
]

def filtre_numerique(signal, a0, a1, b1, b2, gain=1.0):
    filtered_data = np.zeros_like(signal)
    filtered_data[0] = a0 * signal[0]
    filtered_data[1] = a0 * signal[1] + a1 * signal[0] + b1 * filtered_data[0]
    filtered_data[2:] = (a0 * signal[2:] + a1 * signal[1:-1] + 
                         b1 * filtered_data[1:-1] + b2 * filtered_data[:-2])
    return filtered_data * gain

def appliquer_filtres(signal, filtres):
    return [filtre_numerique(signal, 
                             filtre["a0"], 
                             filtre["a1"], 
                             filtre["b1"], 
                             filtre["b2"], 
                             filtre["gain"]) for filtre in filtres]

def reponse_impulsionnelle(a0, a1, b1, b2, gain, n_samples=1000):
    h = np.zeros(n_samples)
    h[0] = 1  # Impulsion unitaire
    y = np.zeros(n_samples)
    y[0] = a0 * h[0]
    y[1] = a0 * h[1] + a1 * h[0] + b1 * y[0]
    for n in range(2, n_samples):
        y[n] = a0 * h[n] + a1 * h[n-1] + b1 * y[n-1] + b2 * y[n-2]
    return y * gain

# Chargement du signal
freq_ech, data = wavfile.read('LW_20M_amis.wav')
if len(data.shape) > 1:
    data = data[:, 0]  # Prendre le premier canal si stéréo

# Création du vecteur temps
t = np.linspace(0, len(data) / freq_ech, len(data), endpoint=False)

# Sous-échantillonnage pour l'affichage
display_step = max(1, len(data) // 10000)
t_display = t[::display_step]
data_display = data[::display_step]

# Calculer la FFT
fft_result = np.fft.fft(data)
frequences = np.fft.fftfreq(len(fft_result), d=1/freq_ech)
fft_display = np.abs(fft_result[:len(fft_result)//2])
freq_display = frequences[:len(frequences)//2]

# Appliquer les filtres
signaux_filtres = appliquer_filtres(data, filtres_numeriques)
fft_results_filtres = [np.fft.fft(signal) for signal in signaux_filtres]

# Créer un signal combiné en additionnant tous les signaux filtrés
signal_combine = np.sum(signaux_filtres, axis=0)

# Normaliser et convertir en int16 pour le fichier WAV
signal_combine_normalise = np.int16(signal_combine / np.max(np.abs(signal_combine)) * 32767)

# Écrire le fichier WAV filtré
wavfile.write('wav_filtre_combine.wav', freq_ech, signal_combine_normalise)

# Créer la figure et les axes
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

# Tracer les signaux temporels
ax1.plot(t_display, data_display, label='Signal Origine', alpha=0.7)
for i, signal_filtre in enumerate(signaux_filtres):
    ax1.plot(t_display, signal_filtre[::display_step], label=f'Filtre {i+1}', alpha=0.7)
ax1.set_xlabel('Temps [s]')
ax1.set_ylabel('Amplitude')
ax1.set_title('Signaux temporels')
ax1.legend()

# Tracer les spectres de Fourier
ax2.plot(freq_display, fft_display, label='Original', alpha=0.7)
for i, fft_result_filtre in enumerate(fft_results_filtres):
    fft_display_filtre = np.abs(fft_result_filtre[:len(fft_result_filtre)//2])
    ax2.plot(freq_display, fft_display_filtre, label=f'Filtre {i+1}', alpha=0.7)
ax2.set_xscale('log')
ax2.set_xlabel('Fréquence (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Spectres de Fourier')
ax2.legend()

# Tracer les réponses impulsionnelles
for i, filtre in enumerate(filtres_numeriques):
    h = reponse_impulsionnelle(filtre["a0"], filtre["a1"], filtre["b1"], filtre["b2"], filtre["gain"])
    ax3.plot(h[:200], label=f'Filtre {i+1}')  # Afficher seulement les 200 premiers échantillons
ax3.set_xlabel('Échantillons')
ax3.set_ylabel('Amplitude')
ax3.set_title('Réponses Impulsionnelles')
ax3.legend()

plt.tight_layout()
plt.show()