import sys
import numpy as np
sys.path.insert(0, '..')

from profiles import iterprofiles

from scipy import signal
import matplotlib.pyplot as plt
rng = np.random.default_rng()

#Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
#0.001 V**2/Hz of white noise sampled at 10 kHz.

import scipy.signal as ss


def fourierTransform(x, npersegment, noverlap):
    N = len(x)
    X=0
    stride = npersegment - noverlap
    i=0
    while True:
        i0 = i *stride
        i1 = i0 + npersegment
        if i1>=N:
            break
        _x = x[i0:i1]
        ss.detrend(x, type='linear', overwrite_data=True)
        window = ss.hann(npersegment, sym=False)
        _x *= window
        _X = np.fft.fft(_x)
        if len(_X)%2==0:
            X += _X[:npersegment//2+1]
        else:
            raise NotImplementedError()
        i+=1
    X/=i
    return X

    
fs = 1
N = 1e4
amp = 2*np.sqrt(2)
freq = 0.1
noise_power = 0.001 * fs / 2 * 0.1
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

y = amp*np.sin(2*np.pi*freq*time+0.1/freq)
y += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

t = np.linspace(0,1, len(y))/fs

N=32
X = fourierTransform(x, N, N//2)
Y = fourierTransform(y, N, N//2)
XY=Y/X
real_part = XY.real
imag_part = XY.imag
phi = np.arctan2(imag_part, real_part)
fn = 0.5* fs
f = np.linspace(0, fn, len(X))*2*np.pi

plt.plot(f, phi)
plt.plot(f, f*10)
Q                
#Compute and plot the power spectral density.

f, Pxx_den = signal.welch(x, fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

#If we average the last half of the spectral density, to exclude the
#peak, we can recover the noise power on the signal.

np.mean(Pxx_den[256:])


#Now compute and plot the power spectrum.

f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.show()

#The peak height in the power spectrum is an estimate of the RMS
#amplitude.

np.sqrt(Pxx_spec.max())

    
