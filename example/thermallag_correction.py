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

dbd = dbdreader.DBD("/home/lucas/gliderdata/helgoland201407/hd/sebastian-2014-215-00-093.ebd")
_, tctd, C, T, D = dbd.get_sync("sci_ctd41cp_timestamp",["sci_water_cond","sci_water_temp",
                                                         "sci_water_pressure"])
    
# remove zero times.
tctd, C, T, D = np.compress(tctd>0, [tctd, C, T, D], axis = 1)
data = dict(time = tctd,
            pressure = D,
            C = C*10,
            T = T,
            D = D*10)

ctd_tl = ctd.ThermalLag(data)

ctd_tl.interpolate_data(dt = 2)
ctd_tl.lag_filter_pressure_data(1.07, other_pressure_parameters = ["D"])
ctd_tl.split_profiles()

# exclude all above 3 m and above 36 and in between 15 and 25 m:
ctd_tl.apply_mask(lambda z: z<3)
ctd_tl.apply_mask(lambda z: z>36)
ctd_tl.apply_mask(lambda z: np.logical_and(z<25, z>15))

alpha, beta = ctd_tl.calibrate(54, 8)
print("alpha: {0:5.3f} (-), beta: {1:5.3f} (s^-1), tau: {2:5.3f} (s).".format(alpha, beta, 1/beta))

ctd_tl.data['rho'] = fast_gsw.rho(ctd_tl.data['C'],
                                  ctd_tl.data['T'],
                                  ctd_tl.data['D'],
                                  8, 54)

# show that it really works.
s=1
f, ax = pl.subplots(2,1, sharex=True)
for i, p in enumerate(ctd_tl[::1]):
    TDd = p.get_downcast("S", "D")
    TDu = p.get_upcast("S", "D")
    ax[0].plot(TDd[0]+i*s, TDd[1],'r')
    ax[0].plot(TDu[0]+i*s, TDu[1],'b')
    TDd = p.get_downcast("rho", "D")
    TDu = p.get_upcast("rho", "D")
    ax[1].plot(TDd[0]-1000+i*s, TDd[1],'r')
    ax[1].plot(TDu[0]-1000+i*s, TDu[1],'b')

pl.show()


