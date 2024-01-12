import numpy as np

from scipy.signal import detrend
from scipy.signal._spectral_py import _fft_helper

def gain_and_delay(t, x, y, nperseg):
    ''' computes a coherence spectrum of H, assuming
    
        Y(s) = H(s) * X(s)
  
        returns omega, gain and phase delay.

    '''
    _n = len(t)
    mantis = int(np.log2(_n))
    n = 2**mantis
    t=t[:n]
    x = x[:n]
    y = y[:n]
    dt = np.diff(t).mean()
    fs = 1/dt
    if nperseg%2:
        num_freqs = (nperseg+1)//2
    else:
        num_freqs = nperseg//2 + 1
    X=_fft_helper(x,np.hanning(nperseg),lambda x: detrend(x,type='constant'),nperseg, nperseg//2, nperseg,'oneside')[...,:num_freqs]*2
    Y=_fft_helper(y,np.hanning(nperseg),lambda x: detrend(x,type='constant'),nperseg, nperseg//2, nperseg,'oneside')[...,:num_freqs]*2

    YX = (Y/X).mean(axis=0)
    a=YX.real
    b=YX.imag
    fn=0.5*fs
    omega=np.arange(num_freqs)*fn/float(num_freqs)*2.*np.pi
    return omega,a**2+b**2,np.arctan(b/a)

class CrossSpectral(ProfileSplitter):
    def __init__(self,data={},window_size=9,threshold_bar_per_second=1e-3):
        ProfileSplitter.__init__(self,data,window_size,threshold_bar_per_second)
    
    def ideal_length(self,n):
        return 2**int(np.log2(n))

    def series_length(self, cast):
        if cast is None:
            series_length = [p.i_cast.shape[0] for p in self]
        elif cast =='up':
            series_length = [p.i_up.shape[0] for p in self]
        else:
            series_length = [p.i_down.shape[0] for p in self]
        min_series_length = min(series_length)
        ideal_series_length = self.ideal_length(min_series_length)
        return ideal_series_length

    def do_ffts(self,parameter,fft_length, cast):
        fftC = []
        for p in self:
            if cast is None:
                n = p.i_cast.shape[0]//2
            elif cast == 'up':
                n = p.i_up.shape[0]//2
            else:
                n = p.i_down.shape[0]//2
            j0 = n - fft_length//2
            j1 = n + fft_length//2
            if cast is None:
                _, C = p.get_cast(parameter)
            elif cast == 'up':
                _, C = p.get_upcast(parameter)
            else:
                _, C = p.get_downcast(parameter)
            Cw = C[j0:j1]
            fftC.append(np.fft.fft(Cw))
        fftC = np.array(fftC)
        return fftC.mean(axis = 0)

    def Hs(self,param0,param1, cast=None):
        ''' Computes coherence based on time series
            per cast. Consider the .coherence() method for the same
            thing, but then on all data.

            if cast is None: then use up and down casts
            other options: cast='up' cast = 'down'

        '''
        print("Consider using the .coherence() method")

        sl=self.series_length(cast)

        FC = self.do_ffts(param0, sl, cast)
        FT = self.do_ffts(param1, sl, cast)
        FCT=(FC/FT)[:sl//2]
        a=FCT.real
        b=FCT.imag
        p = self[0]
        if cast is None:
            t = self.data[self.T_str][p.i_cast]
        elif cast == 'up':
            t = self.data[self.T_str][p.i_up]
        else:
            t = self.data[self.T_str][p.i_down]
        dT=np.diff(t).mean()
        fn=0.5*1./dT
        omega=np.arange(sl/2)*fn/float(sl/2)*2.*np.pi
        print("sample length:",sl)
        return omega,a**2+b**2,np.arctan(b/a)

    def coherence(self, param0, param1, nperseg):
        ''' computes a coherence spectrum of H, assuming

        Y(s) = H(s) * X(s)
  
        returns omega, gain and phase delay.

        '''
        t = self.data["time"]
        x = self.data[param0]
        y = self.data[param1]
        return gain_and_delay(t, x, y, nperseg)
