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
_, tctd, C, T, D = dbd.get_sync("sci_ctd41cp_timestamp","sci_water_cond","sci_water_temp",
                                "sci_water_pressure")
    
# remove zero times.
tctd, C, T, D = np.compress(tctd>0, [tctd, C, T, D], axis = 1)
data = dict(time = tctd,
            pressure = D,
            C = C*10,
            T = T,
            D = D*10)


ctd_tl = ctd.ThermalLag(data, lat=54, lon=7)

ctd_tl.interpolate_data(dt = 1)
ctd_tl.lag_filter_pressure_data(1.07, other_pressure_parameters = ["D"])
ctd_tl.split_profiles()

ctd_tl.data['rho0'] = fast_gsw.pot_rho(ctd_tl.data['C'],
                                       ctd_tl.data['T'],
                                       ctd_tl.data['D'],
                                       8, 54).copy()
ctd_tl.apply_short_time_mismatch(tau=0.3)


# exclude all above 3 m and above 36 and in between 15 and 25 m:
z = ctd_tl.data['D'] # note that interpolation may lead to a different size compared to D when read from DBD files.
ctd_tl.apply_mask(z<3)
ctd_tl.apply_mask(z>36)
ctd_tl.apply_mask(np.logical_and(z<25, z>15))

alpha, beta = ctd_tl.calibrate(method='S-profile')
print("alpha: {0:5.3f} (-), beta: {1:5.3f} (s^-1), tau: {2:5.3f} (s).".format(alpha, beta, 1/beta))

C1 = ctd_tl.data['C'].copy()
T1 = ctd_tl.data['T'].copy()
D1= ctd_tl.data['D'].copy()

ctd_tl.data['rho'] = fast_gsw.pot_rho(ctd_tl.data['Ccor'],
                                      ctd_tl.data['T'],
                                      ctd_tl.data['D'],
                                      8, 54)

# show that it really works.
s=1
f, ax = pl.subplots(2,1, sharex=True)
for i, p in enumerate(ctd_tl[::1]):
    if i==0:
        labeld = 'Down cast'
        labelu = 'Up cast'
        label0 = 'Raw'
    else:
        labeld = labelu = label0 = None
    TDd = p.get_downcast("S", "D")
    TDu = p.get_upcast("S", "D")
    ax[0].plot(TDd[0]+i*s, TDd[1],'C0', label=labeld)
    ax[0].plot(TDu[0]+i*s, TDu[1],'C1', label=labelu)
    TDd = p.get_downcast("rho0", "D")
    TDu = p.get_upcast("rho0", "D")
    ax[1].plot(TDd[0]-1000+i*s, TDd[1],'C2', label=label0)
    ax[1].plot(TDu[0]-1000+i*s, TDu[1],'C3', label=label0)
    TDd = p.get_downcast("rho", "D")
    TDu = p.get_upcast("rho", "D")
    ax[1].plot(TDd[0]-1000+i*s, TDd[1],'C0', label=labeld)
    ax[1].plot(TDu[0]-1000+i*s, TDu[1],'C1', label=labelu)
ax[1].set_xlabel('Profile number')
ax[0].set_ylabel('Absolute Salinity')
ax[1].set_ylabel(r'Potential density (kg m^{-3})')

ax[0].legend()
ax[1].legend()
pl.show()


