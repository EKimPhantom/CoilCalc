from scipy import integrate as integrate
from scipy import fft as fft
import numpy as np

from matplotlib import pyplot as plt


class Field():
    def __init__(self, multipoles) -> None:
        self.multipoles = multipoles    # multipole = [[n, C(n), a(n)], [n1, C(n1), a(n1)], ...]

class Signal(Field):
    def __init__(self, multipoles, phi, signal, w) -> None:
        super().__init__(multipoles)
        self.phi = phi
        self.signal = signal
        self.w = w
    
    def __add__(self, other):
        #first_arg = set(self.multipoles)
        #second_arg = set(other.multipoles)
        #    print("WARNING: Adding signals from different fields!!! (Operation valid for first field multipoles)")

        if self.w!= other.w:
            print("WARNING: Adding signals with different rotational velocities!!! (Operation valid for first field)")

        new_phi = {}
        for n_pole in self.multipoles:
            n, c, a = n_pole[0], n_pole[1], n_pole[2]
            new_phi[n] = self.phi[n] + other.phi[n]
        new_signal = self.signal + other.signal
        return Signal(self.multipoles, new_phi, new_signal, self.w)
    
class Coil():
    def __init__(self, r1, r2, aperture, L, w) -> None:
        self.r1, self.r2, self.L = r1, r2, L
        self.ref = 0.7 * aperture
        self.w = w
    
    def straight_coil_signal(self, field: Field, t, turn=0, delta=0, mult=1):
        length = self.L - 2 * turn * delta
        rad1 = np.sign(self.r1) * abs(self.r1 - turn * delta)
        rad2 = np.sign(self.r2) * abs(self.r2 - turn * delta)
        phi = {}
        signal = np.zeros_like(t)
        for n_pole in field.multipoles:

            n, c, a = n_pole[0], n_pole[1], n_pole[2]
            phi[n] = (length / (n * self.ref ** (n - 1))) * (rad1 ** n - rad2 ** n)
            signal += mult*phi[n] * c * np.cos(n*self.w*t - n*a*np.ones_like(t)) * n * self.w

        return Signal(field.multipoles, phi, signal, self.w)
    
    def curved_coil_signal(self, field: Field, t, curve, turn=0, delta=0, mult=1):

        phi = {}
        length = self.L - turn * delta
        rad1 = np.sign(self.r1) * abs(self.r1 - turn * delta)
        rad2 = np.sign(self.r2) * abs(self.r2 - turn * delta)
        signal = np.zeros_like(t)
        width = rad1 - rad2

        for n_pole in field.multipoles:

            n, c, a = n_pole[0], n_pole[1], n_pole[2]
            phi_k = 0

            for k in range(0,int((n-1)/2+1), 1):
                phi_k +=  2*(width/2) ** (2*k + 1) * combinations(n, 2*k + 1) * integrate.quad(lambda x: curve(x) ** (n - 2*k - 1), turn * delta, length)[0] * (1 / (n * self.ref ** (n - 1)))

            phi[n]= phi_k
            signal += mult*phi_k * c * n * self.w *np.sin(n*self.w*t - n*a*np.ones_like(t))
        return Signal(field.multipoles, phi, signal, self.w)
    
    def sine_deformation(self, x, curvature=0.02):
        return curvature * (self.r1 - self.r2) * np.sin(np.pi*x/self.L)
    
    @staticmethod
    def null_f(x):
        return 0




class Integration_coil(Coil):
    def __init__(self, r1, r2, aperture, L, w, num_turns, delta) -> None:
        super().__init__(r1, r2, aperture, L, w)
        self.num_turns = num_turns
        self.delta = delta

    def small_coils_signal(self, field: Field, t, curve=Coil.null_f):
        
        phi = {}
        signal = np.zeros_like(t)
        width = self.r1 - self.r2
        for n_pole in field.multipoles:

            n, c, a = n_pole[0], n_pole[1], n_pole[2]
            phi_k = 0
            for i in range(self.num_turns - 1):
                phi_k -= (self.delta * self.ref/n) * (((width - self.delta * i + curve(self.delta * i))/(2 * self.ref)) ** n - ((width - self.delta * (i + 1) + curve(self.delta * i))/(2 * self.ref)) ** n)
            phi[n]= phi_k
            signal += phi_k * c * n * self.w *np.sin(n*self.w*t - n*a*np.ones_like(t))

        return Signal(field.multipoles, phi, signal, self.w)
    
    def get_signal(self, field: Field, t):

        result = self.small_coils_signal(field, t)
        for n in range(self.num_turns):
            if n < self.num_turns -1:
                result+= self.straight_coil_signal(field, t, turn=n, delta=self.delta, mult=2)
            else:
                result+= self.straight_coil_signal(field, t, turn=n, delta=self.delta)

        return result

    def get_signal_curved(self, field: Field, t, curve):

        result = self.small_coils_signal(field, t, curve=curve)
        for n in range(self.num_turns):
            if n < self.num_turns -1:
                result+= self.curved_coil_signal(field, t, curve, turn=n, delta=self.delta, mult=2)
            else:
                result+= self.curved_coil_signal(field, t, curve, turn=n, delta=self.delta)

        return result

    def get_signal_simpified(self, field: Field, t):
        phi = {}
        signal = np.zeros_like(t)
        width = abs(self.r1 - self.r2)
        for n_pole in field.multipoles:

            n, c, a = n_pole[0], n_pole[1], n_pole[2]
            phi_k = 0

            if n%2 == 1:
                phi_k += 2 * ((self.L - 2 * self.num_turns * self.delta) * self.ref / n * ((width - 2 * self.num_turns * self.delta)/self.ref)**n)
                for i in range(0,n-1,1):
                    phi_k += 4 * ((self.L - 2 * i * self.delta) * self.ref / n * ((width - 2 * i * self.delta)/self.ref)**n)

            for i in range(0,n - 1,1):
                phi_k -= (self.delta ** 2) / n * ((width - 2 * i * self.delta)/(2 * self.ref)) ** (n - 1)
            
            phi[n] = phi_k

            signal += phi_k * c * n * self.w *np.sin(n*self.w*t - n*a*np.ones_like(t))
        return Signal(field.multipoles, phi, signal, self.w)
    
def combinations(n, k): 
    if 0 <= k <= n:
        nn = 1
        kk = 1
        for t in range(1, min(k, n - k) + 1):
            nn *= n
            kk *= t
            n -= 1
        return nn // kk
    else:
        return 0

def analyze_signal(signal: Signal, axis):

    fs = 1/(axis[1] - axis[0])
    fourrier = fft.fftshift(fft.fft(signal.signal))
    #freq_axis = np.linspace(-fs/(2 * signal.w), fs/(2 *signal.w), len(fourrier))
    freq_axis = fft.fftshift(fft.fftfreq(len(axis), 1/fs)) / (signal.w/ (2 * np.pi))

    bins = []
    bin_axis = []
    freq_fs = 1/(freq_axis[-1] - freq_axis[-2])
    for i in range(int(len(freq_axis)/freq_fs)):
       bins.append(np.sum(fourrier[int(i*freq_fs - freq_fs/2):int(i*freq_fs + freq_fs/2)+1]))
       bin_axis.append(freq_axis[int(i*freq_fs)])
    bins = np.array(bins)
    bin_axis = np.array(bin_axis)           

    axis_to_harmonic = int(len(bin_axis)/2)
    gradients = np.abs(bins)
    for phi_n in signal.phi:
        if signal.phi[phi_n] != 0:
            gradients[axis_to_harmonic + phi_n] *= 1/signal.phi[phi_n]*np.sqrt(2/np.pi) / phi_n /100
            gradients[axis_to_harmonic - phi_n] *= 1/signal.phi[phi_n]*np.sqrt(2/np.pi) / phi_n /100
    alpha = np.zeros_like(fourrier)

    return gradients, bin_axis

