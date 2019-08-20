import numpy as np
from scipy.optimize import fmin
from scipy.interpolate import pchip
from profiles import iterprofiles, filters
try:
    import fast_gsw
except ImportError:
    import gsw as fast_gsw

    def __SA(C, t, p, lon, lat):
        SP = fast_gsw.SP_from_C(C,t, p)
        SA = fast_gsw.SA_from_SP(SP, p, lon, lat)
        return SA
    fast_gsw.SA = __SA
    

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
        self.thermal_lag_filter = filters.ThermalLagFilter(1,1,1)
        self.gamma = 1.0 # about correct for C in mS/cm
        self.mask=None

    def clear_mask(self):
        self.mask = np.zeros_like(self.data['time'], bool)

    def apply_mask(self, mask, operator = "|"):
        if self.mask is None:
            self.clear_mask()
        if operator == "|":
            self.mask |= mask
        elif operator == "&":
            self.mask &= mask
        else:
            raise NotImplementedError
        
    def calibrate(self, lat = 54, lon =8, initial_values = [0.02, 0.03], Cparameter="C", Tparameter="T"):
        x = self.get_thermal_lag_coefs(lat = lat, lon = lon,
                                       initial_values=initial_values,
                                       Cparameter=Cparameter, Tparameter=Tparameter)
        self.apply_thermal_lag_correction(*x, lon = lon, lat = lat)
        return x
    
    def get_thermal_lag_coefs(self, lat = 54, lon = 8, initial_values = [0.02, 0.03],
                              Cparameter="C", Tparameter="T"):
        x = fmin(self.cost_function, initial_values, args=(Cparameter, Tparameter, lon, lat))
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

    def apply_short_time_mismatch(self, tau):
        if not 'Craw' in self.data.keys():
            self.data['Craw'] = self.data['C'].copy()
        lf = filters.LagFilter(1, tau)
        self.data['C'] = lf.filter(self.data['time'], self.data['Craw'])

        

    def cost_function(self, x, Cparameter, Tparameter, lon ,lat):
        self.apply_thermal_lag_correction(*x, Cparameter=Cparameter, Tparameter=Tparameter,
                                          lon=lon, lat=lat)
        return np.sum([self.get_profile_score(p, "S") for p in self[1:]])
    
    def get_profile_score(self, p, parameter = "S", dz=0.01):
        if self.mask is None:
            self.clear_mask()
        i_down = p.i_down
        i_up = p.i_up
        z_d = self.data['pressure'][i_down]
        z_u = self.data['pressure'][i_up]
        zi = np.arange(min(z_d.min(), z_u.min()), max(z_d.max(), z_u.max())+dz, dz)
        mask_i = (np.interp(zi, z_d, self.mask[i_down])+0.5).astype(int)
        mask_i |= (np.interp(zi, z_u[::-1], self.mask[i_up[::-1]])+0.5).astype(int)
        mask_i = mask_i.astype(bool)
        v_d = np.interp(zi, z_d, self.data[parameter][i_down])
        v_u = np.interp(zi, z_u[::-1], self.data[parameter][i_up[::-1]])
        dx = (v_d.compress(~mask_i)-v_u.compress(~mask_i))**2
        return sum(dx)






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
