#Importation bibliothèques
from numpy import *
from matplotlib.pyplot import *
from pylab import *

# définition fonction de tranfert
#PV : fréquence plus parlant (effectivement ce n'est qu'a des facteur pres
ft=8000 # fréquence de mon signal
dureeT = 10/ft   #Observation temporelle sur 10 période


H0=1 # gain statique
fc=500# fréquence de coupure

fe=441000 # fréquence ech

######################## A SUPP
#fcnt=wt/we
#wcnt=2*pi*fcnt
##################################


#Signal temporel : 
t = np.linspace(0, dureeT, int(dureeT * fe), endpoint=False)  # Vecteur temps
x = np.sin(2 * np.pi * ft * t)  # Signal sinusoïdal

wc=2*np.pi*fc #Pulsation de coupure

def H(f):
    return H0 / (1 + 1j * f * 2 * np.pi / wc)


#Test à mofidier...
def filtreNumTemp(data, alpha):
    filtered_data = np.zeros_like(data) #Creation donnee sortie meme type que data
    filtered_data[0] = data[0]
    filtered_data[1] = 0.25*(data[1] + data[0])
    filtered_data[2] = 0.25*(data[2] + data[1] + data[0])



    for i in range(3, len(data)):
        filtered_data[i] = (1 - alpha) * filtered_data[i-1] + alpha * data[i]
    return filtered_data




#PV Ce n'est pas les puissance
#Découpage des puissances de 1  à 100000
puissance=arange(0,6,0.01)

#définition des pulsations
f=10**puissance
fn=f/(fe)




#définition du module en dB
module=20*log10(absolute(H(f)))


################################## Filtrage de mon signal
# Calculer la transformée de Fourier du signal
X = np.fft.fft(x)
fSignal = np.fft.fftfreq(len(x), 1/fe)

# Appliquer le filtre dans le domaine fréquentiel
H = H(fSignal)
Y = X * H

# Revenir dans le domaine temporel en utilisant la transformée de Fourier inverse
y = np.fft.ifft(Y).real



plt.figure()
#tracé du diagramme de bode
subplot(211) # nb graphes : 2 colonne :1 ligne 1
semilogx(f,module)
grid(True)


# Affichage des signaux
plt.figure()
plt.plot(t, x, label='Signal original')
plt.plot(t, y, label='Signal filtré', linestyle='--')
plt.xlabel('Temps [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Signal Sinusoïdal avec Filtrage Passe-Bas')
plt.grid()


# Appliquer le filtre passe-bas
alpha = 0.1  # Paramètre de lissage
y = filtreNumTemp(x, fc/fe)
# Affichage des signaux
plt.figure()
plt.plot(t, x, label='Signal original')
plt.plot(t, y, label='Signal filtré', linestyle='--')
plt.xlabel('Temps [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Filtre Passe-Bas de Premier Ordre')
plt.grid()
plt.show()
###Autre méthode




