# Example of how to correct for the thermal lag
#
# Required is data with a temperature gradient 
# select a number of profiles (not too many, here one ebd file is used
# find the edges of the thermocline, and mask this depth range as well as depth
# ranges that are outside the scope of the measurements (apply_mask).
# use the calibrate method to find alpha and beta.
# use the apply_thermal_lag_correction for correction of data with known alpha and beta.

import numpy as np
import pylab as pl

from profiles import iterprofiles, ctd
import dbdreader
import fast_gsw


dbd = dbdreader.DBD("/home/lucas/gliderdata/helgoland201407/hd/amadeus-2014-215-00-019.ebd")
_, tctd, C, T, D = dbd.get_sync("sci_ctd41cp_timestamp",["sci_water_cond","sci_water_temp",
                                                         "sci_water_pressure"])
    
# remove zero times.
tctd, C, T, D = np.compress(tctd>0, [tctd, C, T, D], axis = 1)
data = dict(time = tctd,
            pressure = D,
            C = C*10,
            T = T,
            D = D*10)

#ctd_tl = ctd.ThermalLagFreeFlush(data)
ctd_tl = ctd.ThermalLag(data)

ctd_tl.interpolate_data(dt = 2)
#ctd_tl.lag_filter_pressure_data(1.07, other_pressure_parameters = ["D"])
#ctd_tl.lag_filter_pressure_data(5, other_pressure_parameters = ["D"])
ctd_tl.split_profiles()

ctd_tl.data['w'] = np.gradient(ctd_tl.data['D'])/np.gradient(ctd_tl.data['time'])


ctd_tl.clear_mask()
ctd_tl.apply_mask(lambda z: z<2)
ctd_tl.apply_mask(lambda z: z>11)
#ctd_tl.apply_mask(lambda z: np.logical_and(z>11, z<15.7))


c0, c1 = ctd_tl.calibrate(54, 8, initial_values=[0, 0.15])

beta=0.055
print("c0: {0:5.3f} (-), c1: {1:5.3f} (-), beta: {2:5.3f} (s^-1), tau: {3:5.3f} (s).".format(c0, c1, beta, 1/beta))


ctd_tl.apply_thermal_lag_correction(0,0.1, beta=1/16)

ctd_tl.data['rho'] = fast_gsw.rho(ctd_tl.data['Ccor'],
                                  ctd_tl.data['T'],
                                  ctd_tl.data['D'],
                                  8, 54)

ctd_tl.data['S'] = fast_gsw.SA(ctd_tl.data['Ccor'],
                                  ctd_tl.data['T'],
                                  ctd_tl.data['D'],
                                  8, 54)


# show that it really works.
s=1
fc=1
f, ax = pl.subplots(3,1, sharex=True, sharey=True)
for i, p in enumerate(ctd_tl[::1]):
    TDd = p.get_downcast("S", "D")
    TDu = p.get_upcast("S", "D")
    ax[0].plot(TDd[0]-TDd[0].mean()+i*s, TDd[1],'r')
    ax[0].plot(TDu[0]-TDd[0].mean()+i*s, TDu[1],'b')
    TDd = p.get_downcast("rho", "D")
    TDu = p.get_upcast("rho", "D")
    ax[1].plot(TDd[0]-TDd[0].mean()+i*s, TDd[1],'r')
    ax[1].plot(TDu[0]-TDd[0].mean()+i*s, TDu[1],'b')
    TDd = p.get_downcast("w", "D")
    TDu = p.get_upcast("w", "D")
    ax[2].plot(fc*TDd[0]+i*s, TDd[1],'r')
    ax[2].plot(fc*TDu[0]+i*s, TDu[1],'b')
[_ax.set_ylim(40,0) for _ax in ax]
pl.show()


