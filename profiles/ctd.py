import numpy as np
from scipy.optimize import fmin
from scipy.interpolate import pchip
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
        self.clear_mask()
        self.thermal_lag_filter = filters.ThermalLagFilter(1,1,1)
        self.gamma = 1.0 # about correct for C in mS/cm
        

    def clear_mask(self, dz = 0.1):
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
        
    def calibrate(self, lat = 54, lon =8, initial_values = [0.02, 0.03]):
        x = self.get_thermal_lag_coefs(lat = lat, lon = lon,
                                       initial_values=initial_values)
        self.apply_thermal_lag_correction(*x, lon = lon, lat = lat)
        return x
    
    def get_thermal_lag_coefs(self, lat = 54, lon = 8, initial_values = [0.02, 0.03]):
        x = fmin(self.cost_function, initial_values, args=("C", "T", lon, lat))
        return x


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
        self.apply_thermal_lag_correction(*x, Cparameter=Cparameter, Tparameter=Tparameter,
                                          lon=lon, lat=lat)
        return np.sum([self.get_profile_score(p, "S") for p in self[1:]])
    
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





class ThermalLagFreeFlush(ThermalLag):
    '''Class to correct for the thermal lag issue for free flush CTD.

    The difference with the pumped ThermalLag is that depending on the speed of
    the glider, the coefficients may vary.

    Still we can use the methods from the pumped CTD to estimate the
    coeficients, but we need to mask the profiles in accordance to the
    constant (enough) vertical speeds. That means that if, for example, the upcast
    is faster than the down cast, then we estimate the parameters for the upcast first
    by masking the the thermocline AND the layer below. For the down cast, we mask
    the thermocline AND the layer above.

    Then we have to provide some relationship between alpha, beta and the vertical speed.


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
        ThermalLag.__init__(self, data, **kwds)

    def calibrate(self, beta, lat = 54, lon =8, initial_value = 0.02):
        x = self.get_thermal_lag_coefs(lat = lat, lon = lon,
                                       initial_values=[initial_value, beta])
        self.apply_thermal_lag_correction(x, beta, lon = lon, lat = lat)
        return float(x)
    
    def get_thermal_lag_coefs(self, lat = 54, lon = 8, initial_values = [0.02, 0.03]):
        x = fmin(self.cost_function, initial_values[0], args=(initial_values[1],
                                                              "C", "T", lon, lat))
        return x


    def cost_function(self, x, beta, Cparameter, Tparameter, lon ,lat):
        self.apply_thermal_lag_correction(x, beta, Cparameter=Cparameter, Tparameter=Tparameter,
                                          lon=lon, lat=lat)
        return np.sum([self.get_profile_score(p, "S") for p in self[1:]])

        
    def apply_thermal_lag_correction2(self, c0, c1, beta=0.05,
                                     Cparameter="C", Tparameter="T", Dparameter="D",
                                     lon=8, lat=54, U = None):
        t = self.data['time']
        C = self.data[Cparameter]
        T = self.data[Tparameter]
        D = self.data[Dparameter]
        dt = np.gradient(t)
        if U is None:
            U = np.abs(np.gradient(D)/dt)
        idx = np.where(U<0.04)
        if len(idx[0]):
            U[idx]=0.04
        Delta_C = np.empty_like(C)
        for i, (_U, _dt, _T) in enumerate(zip(U, dt, T)):
            if i==0:
                __T = _T
                continue
            _alpha = c0/_U + c1
            a0 = self.gamma * _alpha/beta*2/_dt
            a1 = -a0
            b0 = 1 + 2/_dt/beta
            b1 = 1 - 2/_dt/beta
            Delta_C[i] = a0/b0*_T + a1/b0*__T - b1/b0*Delta_C[i-1]
            __T = _T
            
        Cp = C + Delta_C
        self.data['Ccor'] = Cp
        self.data['S'] = fast_gsw.SA(Cp, T, self.data["pressure"]*10, lon, lat)
