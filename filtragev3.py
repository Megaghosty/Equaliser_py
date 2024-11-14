import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

# Définition des paramètres pour 5 filtres numériques
filtres_numeriques = [
    {"a0": 0.2183, "a1": -0.2183, "b1": 1.7505, "b2": -0.7661, "gain_statique": 1.0},
    {"a0": 0.194, "a1": -0.194, "b1": 1.5568, "b2": -0.6813, "gain_statique": 1.0},
    {"a0": 0.1989, "a1": -0.1989, "b1": -1.2747, "b2": -0.5578, "gain_statique": 1.0},
    {"a0": 0.1249, "a1": -0.1249, "b1": 1.0023, "b2": -0.4386, "gain_statique": 1.0},
    {"a0": 0.0972, "a1": -0.00972, "b1": 0.7800, "b2": -0.3414, "gain_statique": 1.0}
]

# Variables globales
freq_ech = None
data = None
filtered_data = None

# Créer la fenêtre principale
root = tk.Tk()
root.title("Égaliseur Audio")
root.geometry("600x500")

def filtre_numerique(signal, a0, a1, b1, b2, gain_statique=1.0):
    filtered_data = np.zeros_like(signal)
    filtered_data[0] = a0 * signal[0]
    filtered_data[1] = a0 * signal[1] + a1 * signal[0] + b1 * filtered_data[0]
    filtered_data[2:] = (a0 * signal[2:] + a1 * signal[1:-1] + 
                         b1 * filtered_data[1:-1] + b2 * filtered_data[:-2])
    return filtered_data * gain_statique

def appliquer_filtres(signal, filtres):
    return [filtre_numerique(signal, 
                             filtre["a0"], 
                             filtre["a1"], 
                             filtre["b1"], 
                             filtre["b2"], 
                             filtre["gain_statique"]) for filtre in filtres]

def reponse_impulsionnelle(a0, a1, b1, b2, gain_statique, n_samples=1000):
    h = np.zeros(n_samples)
    h[0] = 1  # Impulsion unitaire
    y = np.zeros(n_samples)
    y[0] = a0 * h[0]
    y[1] = a0 * h[1] + a1 * h[0] + b1 * y[0]
    for n in range(2, n_samples):
        y[n] = a0 * h[n] + a1 * h[n-1] + b1 * y[n-1] + b2 * y[n-2]
    return y * gain_statique

# Fonction pour charger un fichier audio
def load_audio():
    global freq_ech, data
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        freq_ech, data = wavfile.read(file_path)
        if len(data.shape) > 1:
            data = data[:, 0]  # Prendre le premier canal si stéréo
        status_label.config(text="Fichier audio chargé")
        update_equalizer()

# Fonction pour mettre à jour l'égaliseur
def update_equalizer(*args):
    global filtered_data
    for i, slider in enumerate(sliders):
        filtres_numeriques[i]["gain_statique"] = slider.get() / 50  # Normaliser à [0, 2]
        labels[i].config(text=f"Filtre {i+1}: {slider.get():.1f}")
    
    if data is not None:
        filtered_signals = appliquer_filtres(data, filtres_numeriques)
        filtered_data = np.sum(filtered_signals, axis=0)
        status_label.config(text="Audio filtré")

# Fonction pour enregistrer l'audio filtré
def save_filtered_audio():
    if filtered_data is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if file_path:
            normalized_data = np.int16(filtered_data / np.max(np.abs(filtered_data)) * 32767)
            wavfile.write(file_path, freq_ech, normalized_data)
            status_label.config(text=f"Fichier audio filtré enregistré")
            generate_and_save_plots()

def generate_and_save_plots():
    global freq_ech
    if data is None or filtered_data is None:
        return

    if freq_ech is None:
        freq_ech = 44100  # Valeur par défaut si aucun fichier audio n'a été chargé

    # Créer une figure avec 5 sous-graphiques
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    axs = axs.ravel()  # Rendre le tableau d'axes unidimensionnel pour un accès simplifié

    # 1. Signal original vs Signal filtré
    time = np.arange(len(data)) / freq_ech
    axs[0].plot(time, data, label='Original')
    axs[0].plot(time, filtered_data, label='Filtré')
    axs[0].set_title('Signal Original vs Filtré')
    axs[0].set_xlabel('Temps (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()

    # 2. Transformée de Fourier
    fft_original = np.fft.fft(data)
    fft_filtered = np.fft.fft(filtered_data)
    freqs = np.fft.fftfreq(len(data), 1/freq_ech)
    axs[1].plot(freqs[:len(freqs)//2], np.abs(fft_original)[:len(freqs)//2], label='Original')
    axs[1].plot(freqs[:len(freqs)//2], np.abs(fft_filtered)[:len(freqs)//2], label='Filtré')
    axs[1].set_title('Transformée de Fourier')
    axs[1].set_xlabel('Fréquence (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_xscale('log')
    axs[1].legend()

    # 3. Réponses impulsionnelles des filtres individuels
    for i, filtre in enumerate(filtres_numeriques):
        h = reponse_impulsionnelle(filtre["a0"], filtre["a1"], filtre["b1"], filtre["b2"], filtre["gain_statique"])
        axs[2].plot(h, label=f'Filtre {i+1}')
    axs[2].set_title('Réponses Impulsionnelles des Filtres')
    axs[2].set_xlabel('Échantillons')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()

    # 4. Réponses en fréquence des filtres individuels
    for i, filtre in enumerate(filtres_numeriques):
        h = reponse_impulsionnelle(filtre["a0"], filtre["a1"], filtre["b1"], filtre["b2"], filtre["gain_statique"])
        w, H = np.fft.fft(h), np.abs(np.fft.fft(h))
        axs[3].plot(w[:len(w)//2], H[:len(H)//2], label=f'Filtre {i+1}')
    axs[3].set_title('Réponses en Fréquence des Filtres')
    axs[3].set_xlabel('Fréquence normalisée')
    axs[3].set_ylabel('Magnitude')
    axs[3].legend()

    # 5. Réponse en fréquence globale
    n_samples = 1000
    impulse_global = np.zeros(n_samples)
    for filtre in filtres_numeriques:
        impulse_global += reponse_impulsionnelle(filtre["a0"], filtre["a1"], filtre["b1"], filtre["b2"], filtre["gain_statique"], n_samples)
    
    # Calculer la transformée de Fourier pour obtenir la réponse en fréquence globale
    freq_response_global = np.abs(np.fft.fft(impulse_global))[:n_samples // 2]
    freqs_global = np.fft.fftfreq(n_samples, d=1/freq_ech)[:n_samples // 2]

    # Limiter l'affichage à la plage audible (20 Hz - 20 kHz)
    mask = (freqs_global >= 20) & (freqs_global <= 20000)
    freqs_global_plot = freqs_global[mask]
    freq_response_global_plot = freq_response_global[mask]

    axs[4].semilogx(freqs_global_plot, freq_response_global_plot, label='Réponse en Fréquence Globale', color='black')
    axs[4].set_title('Réponse en Fréquence Globale')
    axs[4].set_xlabel('Fréquence (Hz)')
    axs[4].set_ylabel('Magnitude')
    axs[4].set_xlim(20, 20000)  # Limiter l'axe x à la plage audible
    axs[4].legend()

    plt.tight_layout()

    # Obtenir le chemin du dossier du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Créer le nom du fichier
    file_name = "graphiques_audio.jpg"
    
    # Joindre le chemin du dossier et le nom du fichier
    file_path = os.path.join(script_dir, file_name)
    
    # Sauvegarder la figure
    plt.savefig(file_path, format='jpg', dpi=300)
    plt.close(fig)
    status_label.config(text="Graphiques générés et enregistrés")

# Créer un cadre pour les sliders
sliders_frame = ttk.Frame(root)
sliders_frame.pack(pady=20)

# Créer les sliders et les labels
sliders = []
labels = []
for i in range(5):
    frame = ttk.Frame(sliders_frame)
    frame.pack(side=tk.LEFT, padx=10)

    label = ttk.Label(frame, text=f"Filtre {i+1}: 50.0")
    label.pack()

    slider = ttk.Scale(
        frame,
        from_=100,
        to=0,
        orient='vertical',
        length=300,
        command=update_equalizer
    )
    slider.set(50)  
    slider.pack()

    sliders.append(slider)
    labels.append(label)

# Créer les boutons
button_frame = ttk.Frame(root)
button_frame.pack(pady=20)

load_button = ttk.Button(button_frame, text="Charger Audio", command=load_audio)
load_button.pack(side=tk.LEFT, padx=5)

save_button = ttk.Button(button_frame, text="Enregistrer Audio Filtré", command=save_filtered_audio)
save_button.pack(side=tk.LEFT, padx=5)

# Créer un label pour le statut
status_label = ttk.Label(root, text="Prêt")
status_label.pack(pady=10)

# Lancement
root.mainloop()