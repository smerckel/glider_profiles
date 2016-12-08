import numpy as np
from scipy.optimize import fmin

from profiles import iterprofiles, filters
import fast_gsw

class ThermalLag(iterprofiles.ProfileSplitter):
    ''' Class to correct for the thermal lag issue.

    The steps to take are:

    * create data in a similar way as ProfileSplitter()
    
    * interpolate_data() to for better filtering

    * optional: lag the pressure parameter

    * apply_thermal_lag_correction()

    * to find the thermal lag parameters:
        o select some data with a thermocline
        o apply a mask so that the boundaries and thermocline are excluded
        o run calibrate()

    '''
    def __init__(self,data, **kwds):
        iterprofiles.ProfileSplitter.__init__(self, data, **kwds)
        self.make_empty_mask()
        self.thermal_lag_filter = filters.ThermalLagFilter(1,1,1)
        self.gamma = 1.0 # about correct for C in mS/cm
        
    def interpolate_data(self,dt = 1):
        tctd = self.data["time"]
        ti = np.arange(tctd.min(), tctd.max()+dt, dt)
        for k, v in self.data.items():
            if k=="time":
                continue
            self.data[k] = np.interp(ti, tctd, v)
        self.data["time"] = ti

    def lag_filter_pressure_data(self, delay, other_pressure_parameters = []):
        ti = self.data["time"]
        p = ["pressure"] + other_pressure_parameters
        LF = filters.LagFilter(1,delay)
        for k in p:
            self.data[k] = LF.filter(ti, self.data[k])

    def make_empty_mask(self, dz = 0.1):
        P = self.data['pressure']*10
        zi = np.arange(P.min(), P.max()+dz, dz)
        mask = np.zeros_like(zi, int)
        self.z_mask = np.ma.masked_array(zi, mask)

    def apply_mask(self, mask_fun, operator = "|"):
        z = self.z_mask.data
        mask = mask_fun(z)
        if operator == "|":
            self.z_mask.mask |= mask
        elif operator == "&":
            self.z_mask.mask &= mask
        else:
            raise NotImplementedError
        
    def calibrate(self, lat = 54, lon =8):
        x = self.get_thermal_lag_coefs(lat = lat, lon = lon)
        self.apply_thermal_lag_correction(*x, lon = lon, lat = lat)
        return x
    
    def get_thermal_lag_coefs(self, lat = 54, lon = 8):
        alpha, beta = fmin(self.cost_function, [0.02, 0.03], args=("C", "T", lon, lat))
        return alpha, beta


    def apply_thermal_lag_correction(self, alpha, beta,
                                     Cparameter="C", Tparameter="T", lon=8, lat=54):
        self.thermal_lag_filter.set_parameters(self.gamma ,alpha, beta)
        t = self.data['time']
        C = self.data[Cparameter]
        T = self.data[Tparameter]
        Delta_C = self.thermal_lag_filter.filter(t, C)
        Cp = C + Delta_C
        self.data['Ccor'] = Cp
        self.data['S'] = fast_gsw.SA(Cp, T, self.data["pressure"]*10, lon, lat)

    def cost_function(self, x, Cparameter, Tparameter, lon ,lat):
        self.apply_thermal_lag_correction(*x, Cparameter, Tparameter, lon, lat)
        return np.sum([self.get_profile_score(p, "S") for p in self])
    
    def get_profile_score(self, p, parameter = "S"):
        S_d, z_d = p.get_downcast(parameter, "pressure")
        S_u, z_u = p.get_upcast(parameter, "pressure")
        z_d*=10
        z_u*=10
        S_d_i = np.interp(self.z_mask, z_d, S_d)
        S_u_i = np.interp(self.z_mask, z_u[::-1], S_u[::-1])
        dx = (S_d_i - S_u_i)**2
        dx = np.ma.masked_array(dx, self.z_mask.mask)
        return np.sum(dx)
