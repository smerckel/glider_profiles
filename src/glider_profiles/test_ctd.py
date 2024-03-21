import sys
sys.path.insert(0,'.')

import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

import profiles, ctd
import dbdreader

dbd_path = "/home/lucas/gliderdata/nsb3_201907/hd/comet-2019-203-05-000.?bd"
dbd_path = "/home/lucas/gliderdata/nsb3_201907/hd/comet-2019-209-00-000.?bd"

dbd = dbdreader.MultiDBD(dbd_path)

t, C, T, D = dbd.get_CTD_sync()

data = dict(time=t, pressure=D, C=C*10, T=T, D=D*10)

tl = ctd.ThermalLag(data)
tl.interpolate_data(dt=1.)
tl.split_profiles()

tau = 0.032
lag_filter_temperature = ctd.filters.LagFilter(1., tau)
#tl.data["C"] = lag_filter_temperature.filter(tl.data["time"], tl.data["C"])
alpha = 0.85
tl.data["C"][1:] = alpha * tl.data["C"][1:] + (1-alpha) * tl.data["C"][:-1]
tl.add_salinity_and_density()



H = []
for i, cast in enumerate(tl.get_casts()):
    s = cast.T.shape[0]
    i0 = (s - 2**8) // 2
    i1 = i0 + 2**8
    _F = np.fft.fft(ss.detrend(cast.T[i0:i1]))
    _G = np.fft.fft(ss.detrend(cast.C[i0:i1]))
    _H = _G / _F
    if _H.shape[0] != 2**8:
        continue
    H.append(_G / _F)
avg = lambda x : np.convolve(x, np.ones(25)/25, 'valid')    
H = np.mean(H, axis=0)
H = H[2**7:]
dt = np.mean(np.diff(tl.data["time"]))
omega = np.linspace(0, 1, 2**7) * 2*np.pi/dt/2
phi = np.arctan(H.imag, H.real)
#plt.plot(avg(omega), avg(phi),'o')

def f(x):
    xm = np.mean(x)
    xstd = np.std(x)
    return (x - xm)/xstd

def f(x):
    return x

# Subclassed method, later include methods into ctd.ThermalLag
class ThermoclineFinder(ctd.ThermalLag):

    '''
    Parameters
    ----------
    minimum_gradient_dTdz : float {1}
        minimum value of largest gradient of T

    threshold_gradient_dTdz : float {0.3}
        the edge of the thermocline is marked where the gradient drops below this value

    thermocline_window : tuple of float {(0, 1e9)}
        the thermocline's edges should be found within specified window in order to be considered.

    '''
    
    def __init__(self, data, *, 
                 minimum_gradient_dTdz = 1,
                 threshold_gradient_dTdz = 0.3,
                 thermocline_window = (0, 1e9)):
        super().__init__(data)
        self.minimum_gradient_dTdz = minimum_gradient_dTdz
        self.threshold_gradient_dTdz = threshold_gradient_dTdz
        self.thermocline_window = thermocline_window
        
    def mask_thermocline(self):
        ''' Mask thermocline regions of all profiles
        '''
        if not self.has_mask:
            self.clear_mask()
        for j, (down_cast, up_cast) in  enumerate(self.get_down_up_casts()):
            i0 = self._get_index_thermocline_edge(down_cast)
            i1 = self._get_index_thermocline_edge(up_cast)
            if i0 and i1:
                z0 = 10 * down_cast.pressure[i0]
                z1 = 10 * up_cast.pressure[i1]
                if z1 > z0:
                    condition = np.logical_and(down_cast.pressure*10>=z0,
                                               down_cast.pressure*10<=z1)
                    idx = np.where(condition)[0]
                    down_cast.mask[idx] = True
                    condition = np.logical_and(up_cast.pressure*10>=z0,
                                               up_cast.pressure*10<=z1)
                    idx = np.where(condition)[0]
                    up_cast.mask[idx] = True
                    print(up_cast.mask)
                    
    def _get_index_thermocline_edge(self, cast):
        # Workhorse algorithm:
        # z and T are extracted, then the maximum gradient is looked up. If not large enough, return None
        # signalling nothing found. Then we look up the index where the gradient is max, and then go back to
        # find the location where the gradient drops below the threshold. This works for both down casts and
        #
        increment = -1 
        z = cast.pressure*10
        T = cast.T
        dTdz = - np.gradient(T)/np.gradient(z)
        if np.max(dTdz) < self.minimum_gradient_dTdz:
            return None
        idx = np.argmax(dTdz)
        max_idx = T.shape[0]
        while True:
            if dTdz[idx] < self.threshold_gradient_dTdz:
                found = True
                break
            if idx == 0 or idx == max_idx or z[idx]<self.thermocline_window[0] or z[idx]>self.thermocline_window[1]:
                found = False
                break
            idx += increment
            
        if not found:
            return None
        return idx
    
tl = ThermoclineFinder(data,
                       minimum_gradient_dTdz = 1,
                       threshold_gradient_dTdz=0.2,
                       thermocline_window=(10, 20))
tl.interpolate_data(dt=1.)
tl.mask_thermocline()
    


scale=10

if 0:
    for i, (down_cast, up_cast) in enumerate(tl.get_down_up_casts()):
        plt.plot(np.gradient(down_cast.T)/np.gradient(down_cast.D)+i*scale, down_cast.D, 'C0')
        plt.plot(np.gradient(up_cast.T)/np.gradient(up_cast.D)+i*scale, up_cast.D, 'C3')


for i, (down_cast, up_cast) in enumerate(tl.get_down_up_casts()):
    plt.plot(down_cast.T+i*scale, down_cast.D, 'C0')
    plt.plot(up_cast.T+i*scale, up_cast.D, 'C3')
    plt.plot(down_cast.T.compress(down_cast.mask)+i*scale, down_cast.D.compress(down_cast.mask), 'k')
    plt.plot(up_cast.T.compress(up_cast.mask)+i*scale, up_cast.D.compress(up_cast.mask), 'k')
    
    
for i, (down_cast, up_cast) in enumerate(tl.get_down_up_casts()):
    plt.plot(f(down_cast.S)+i*scale, down_cast.D, 'C1')
    plt.plot(f(up_cast.S)+i*scale, up_cast.D, 'C2')
    plt.plot(down_cast.S.compress(down_cast.mask)+i*scale, down_cast.D.compress(down_cast.mask), 'k')
    plt.plot(up_cast.S.compress(up_cast.mask)+i*scale, up_cast.D.compress(up_cast.mask), 'k')


    

plt.ylim(40,0)

# #tl.apply_thermal_lag_correction(0.2, 0.05)


# tl.select_profiles(3)
# c = tl.get_casts()[3]
# plt.plot(c.S, -c.pressure)

# coefs = tl.calibrate(initial_values=[0.001, 0.02], method="S-profile")

# plt.plot(c.S, -c.pressure)
