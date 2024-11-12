import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

lowcut = 500.0 
highcut = 1500.0  
fs = 44000.0  

order = 4  #
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(order, [low, high], btype='band')

# Affichage de la réponse en fréquence
w, h = signal.freqz(b, a, worN=2000)
plt.plot((fs * 0.5 / np.pi) * w, abs(h), 'b')
plt.plot(lowcut, 0.5 * np.sqrt(2), 'ko')
plt.plot(highcut, 0.5 * np.sqrt(2), 'ko')
plt.axvline(lowcut, color='k')
plt.axvline(highcut, color='k')
plt.xlim(0, 0.5 * fs)
plt.title("Réponse en fréquence du filtre passe-bande")
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Gain')
plt.grid()
plt.show()
