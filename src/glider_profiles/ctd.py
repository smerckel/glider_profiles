from functools import partial
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.interpolate import pchip

import dbdreader
from glider_profiles import profiles, filters
try:
    import fast_gsw
except ImportError:
    import gsw as fast_gsw

    def __SA(C, t, p, lon, lat):
        SP = fast_gsw.SP_from_C(C,t, p)
        SA = fast_gsw.SA_from_SP(SP, p, lon, lat)
        return SA
    
    def __CT(C, t, p, lon, lat):
        SP = fast_gsw.SP_from_C(C,t, p)
        SA = fast_gsw.SA_from_SP(SP, p, lon, lat)
        CT = fast_gsw.CT_from_t(SA, t, p)
        return CT

    def __pot_rho(C, t, p, lon, lat):
        SP = fast_gsw.SP_from_C(C,t, p)
        SA = fast_gsw.SA_from_SP(SP, p, lon, lat)
        pot_rho = fast_gsw.pot_rho_t_exact(SA, t, p, 0)
        return pot_rho

    def __rho(C, t, p, lon, lat):
        SP = fast_gsw.SP_from_C(C,t, p)
        SA = fast_gsw.SA_from_SP(SP, p, lon, lat)
        rho = fast_gsw.rho_t_exact(SA, t, p)
        return rho

    fast_gsw.SA = __SA
    fast_gsw.CT = __CT
    fast_gsw.pot_rho = __pot_rho
    fast_gsw.rho = __rho

CalibrationResult = namedtuple("CalibrationResult", "alpha beta tau number_of_profiles")


class ThermalLagFilter(object):

    def __init__(self, data, dt=1.0):
        self.data = self.interpolate_data(dt)

    def interpolate_data(self, data, dt, time_str="time"):
        self.original_time_base = data[time_str]
        tm = data[time_str]
        t0 = tm[0]
        t1 = tm[-1]
        ti = np.arange(t0, t1+dt, dt)
        for k,v in data.items():
            if k==time_str:
                continue
            data[k] = np.interp(ti, tm, v)
        self.split_profiles()

class ThermalLag(profiles.ProfileSplitter):
    '''Class to correct for the thermal lag issue.

    The steps to take are:

    * create data in a similar way as ProfileSplitter()
    
    * interpolate_data() for better filtering

    * optional: lag the pressure parameter

    * apply_thermal_lag_correction()

    * to find the thermal lag parameters:
        o select some data with a thermocline
        o apply a mask so that the boundaries and thermocline are excluded
        o run calibrate()


    Parameters
    ----------
    data : dictionary of np.arrays
        a data dictionary as usual for ProfileSplitter

    method : string {"S-profile", "ST-diagram"} ("S-profile")
        sets the method for computing the error
    
    kwds : keywords passed on to :obj:`ProfileSplitter`

    Note
    ----
    
    The method can be either "S-profile" or "ST-diagram". 

    In case of "S-profile" a down and upcast pair of salinity data is
    interpolated on a common z coordinate and the cost_function
    returns the area between the two profiles.  This works well in
    coastal regions (shallow water) where the temperature gradient can
    be substantial. Additionally it makes sense to apply a mask to
    mask very strong gradients where short time mismatches are the
    dominant source of error.

    The "ST-diagram" minimises a down and upcast pair in absolute
    salinity and conservative temperature space. This works better in
    situations where there is no clear or strong temperature gradient.

    '''
    def __init__(self,data, method='S-profile', lat=None, lon=None, **kwds):
        super().__init__(data, **kwds)
        self.thermal_lag_filter = filters.ThermalLagFilter(1,1,1)
        self.gamma = 1.0 # about correct for C in mS/cm
        self.method = method
        self.has_mask = False
        self.lat = lat or 54.
        self.lon = lon or 8.
        self._time_original = None
        self.add_salinity_and_density()
        
    def interpolate_data(self, dt = 1.):
        t = self.data['time']
        self._time_original = t
        ti = np.arange(t.min(), t.max(), dt)
        keys = list(self.data.keys())
        for k in keys:
            if k!='time':
                self.data[k] = np.interp(ti, t, self.data[k])
        self.data['time'] = ti
    
    def clear_mask(self):
        ''' Clears the mask and sets the size of the mask equal to the time vector.'''
        self.data['mask'] = np.zeros_like(self.data['time'], bool)
        self.has_mask=True
        
    def apply_mask(self, mask, operator = "|"):
        ''' Applies a mask subject to a given operator

        Parameters
        ----------
        mask : np.array of bool or int (1/0)
            data mask
        operator : str {"|", "&"}
            specifies the operator. The OR operator adds the mask to the current mask (union), 
            the AND operator sets the mask to the intersection of both masks.
        '''
        if not self.has_mask:
            self.clear_mask()
        if operator == "|":
            self.data['mask'] |= mask
        elif operator == "&":
            self.data['mask'] &= mask
        else:
            raise NotImplementedError

    def select_profiles(self, start, end=None):
        c = self.get_casts()
        t_start = c[start].time.min()
        if end is None:
            end = start
        t_end = c[end].time.max()
        t = self.data["time"]
        self.apply_mask(t<t_start)
        self.apply_mask(t>t_end)
        
    def calibrate(self, initial_values = [0.02, 0.03], Cparameter="C", Tparameter="T", method=None, lat=None, lon=None, **kwds):
        '''calibrate

        Method that calibrates the coefficients of the thermal lag model.

        Parameters
        ----------
        initial_values : tuple of floats
            initial values for alpha and beta
        Cparameter : str 
            key of conductivity values in self.data dictionary
        Tparameter : str 
            key of temperature values in self.data dictionary
        method : str {"S-profile", "ST-diagram", None}
            sets method of minimisation method, see also constructor help.
            If None, the constructor set value is used.
        lat, lon : float or None
            sets the latitude and longitude for evaluation of SA.
        **kwds : dictionary of options passed to the method function.

        For ST-diagram: 
        N : int
            number of nodes to represent the ST-curve.
        
        Notes
        -----

        Two methods are provided to minimise a cost function. Because
        most of the action happens at temperature interfaces, this
        affects mostly the S-T diagram in a small region, and a
        minimisation method might not work as well as expected.  For
        specific sites, such as a stratified North Sea, it may be
        beneficial to use S-profile method on selected profiles.

        '''
        lat = lat or self.lat
        lon = lon or self.lon
        if (lat is None) or (lon is None):
            raise ValueError("No latitude/longitude coordinates specified.")
        method = method or self.method
        methods = {"S-profile":self.get_profile_score_S_profile,
                   "ST-diagram":self.get_profile_score_ST_diagram}
        
        method_fun = partial(methods[method], **kwds)
        
        x = self.get_thermal_lag_coefs(method_fun, lat = lat, lon = lon,
                                       initial_values=initial_values,
                                       Cparameter=Cparameter, Tparameter=Tparameter)
        self.apply_thermal_lag_correction(*x, lon = lon, lat = lat)
        return x
    
    def get_thermal_lag_coefs(self, method_fun, lat = 54, lon = 8, initial_values = [0.02, 0.03],
                              Cparameter="C", Tparameter="T"):
        x = fmin(self.cost_function, initial_values, args=(method_fun, Cparameter, Tparameter, lon, lat))
        return x


    def apply_thermal_lag_correction(self, alpha, beta,
                                     Cparameter="C", Tparameter="T", lon=None, lat=None):
        self.thermal_lag_filter.set_parameters(self.gamma ,alpha, beta)
        t = self.data['time']
        C = self.data[Cparameter]
        T = self.data[Tparameter]
        Delta_C = self.thermal_lag_filter.filter(t, C)
        Cp = C + Delta_C
        self.data['Ccor'] = Cp
        lat = lat or self.lat
        lon = lon or self.lon
        self.add_salinity_and_density(Cp, T, self.data['pressure']*10, lat, lon)
        
    def add_salinity_and_density(self, C=None, T=None, P=None, lat=None, lon=None):
        C = C or self.data['C']
        T = T or self.data['T']
        P = P or self.data['pressure']*10
        lat = lat or self.lat
        lon = lon or self.lon
        
        self.data['S'] = fast_gsw.SA(C, T, P, lon, lat)
        self.data['CT'] = fast_gsw.CT(C, T, P, lon, lat)
        self.data['pot_rho'] = fast_gsw.pot_rho(C, T, P, lon, lat)
        self.data['rho'] = fast_gsw.rho(C, T, P, lon, lat)
        
    def apply_short_time_mismatch(self, tau):
        if not 'Craw' in self.data.keys():
            self.data['Craw'] = self.data['C'].copy()
        lf = filters.LagFilter(1, tau)
        self.data['C'] = lf.filter(self.data['time'], self.data['Craw'])

    def cost_function(self, x, method_fun, Cparameter, Tparameter, lon ,lat):
        self.apply_thermal_lag_correction(*x, Cparameter=Cparameter, Tparameter=Tparameter,
                                          lon=lon, lat=lat)
        upcasts = self.get_upcasts()
        downcasts = self.get_downcasts()
        errors = [method_fun(downcast, upcast) for (downcast, upcast) in zip(downcasts, upcasts)]
        errors = np.hstack([e for e in errors if not e == None])
        print(x, errors)
        return np.mean(errors)
    
    def get_profile_score_S_profile(self, downcast, upcast, parameter = "S", dz=0.01):
        if not self.has_mask:
            self.clear_mask()

        z_d = downcast.pressure
        z_u = upcast.pressure
        zi = np.arange(min(z_d.min(), z_u.min()), max(z_d.max(), z_u.max())+dz, dz)
        mask_i = (np.interp(zi, z_d, downcast.mask)+0.5).astype(int)
        mask_i |= (np.interp(zi, z_u[::-1], upcast.mask[::-1])+0.5).astype(int)
        mask_i = mask_i.astype(bool)
        if np.all(mask_i):
            return None
        v_d = np.interp(zi, z_d, downcast.get(parameter))
        v_u = np.interp(zi, z_u[::-1], upcast.get(parameter)[::-1])
        dx = (v_d.compress(~mask_i)-v_u.compress(~mask_i))**2
        return sum(dx)


    def get_profile_score_ST_diagram(self, p, N=200):
        C = (self.data['Ccor'][p.i_down], self.data['Ccor'][p.i_up])
        T = (self.data['T'][p.i_down], self.data['T'][p.i_up])
        P = (self.data['pressure'][p.i_down]*10, self.data['pressure'][p.i_up]*10)
        CT = (self.data['CT'][p.i_down], self.data['CT'][p.i_up])
        SA = (self.data['S'][p.i_down], self.data['S'][p.i_up])
        pot_rho = (self.data['pot_rho'][p.i_down]*10, self.data['pot_rho'][p.i_up]*10)
        
        rhoi = np.linspace(max(pot_rho[0].min(), pot_rho[1].min()),min(pot_rho[0].max(), pot_rho[1].max()), N)
        idx = [np.argsort(_pot_rho) for _pot_rho in pot_rho]
        CTi = [np.interp(rhoi, y[i], x[i]) for x,y,i in zip(CT, pot_rho, idx)]
        SAi = [np.interp(rhoi, y[i], x[i]) for x,y,i in zip(SA, pot_rho, idx)]
        error = (np.diff(SAi, axis=0)**2 + np.diff(CTi, axis=0)**2)[0]
        if not self.mask is None:
            mask = self.mask[p.i_down]
            mask_i = np.interp(rhoi, pot_rho[0][idx[0]], mask[idx[0]]).astype(bool)
            error = error.compress(~mask_i)
        return error




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

class CTDprofileCalibrate(object):
    ''' CTDprofileCalibrate

    Class to facilitate calibrating the CTD per profile.

    Parameters
    ----------
    lat_c : latitude (decimal degree) or None
    lon_c : longitude (decimal degree) or None
    
    if lat_c == lon_c == None, then the coordinates are retrieved from the gliderdata.

    '''
    def __init__(self, lat_c=None, lon_c=None):
        self.lat_c = lat_c
        self.lon_c = lon_c
        self.thermal_lag_model = {}

    def load_data(self, glider, path, dt=1, short_time_correction=None):
        '''loads required data from particular glider from dbd/ebd files

        Parameters
        ----------
        glider : str 
            id string for storing data loaded. 
        path : str
            path to where dbd and ebd files are found
        dt : float (default 1.0)
            makes time series equidistant with this time interval
        short_time_correction : float or None (default None)
            applies short_time_correction to C and T measurements. If None, no correction is applied.
        
        Note
        ----

        The glider id is soley used to refer to the proper data
        set. It is possible to load different segments of data for a
        single glider to and have each segment calibrated. The glider
        id lets you retrieve the correct data set.
        '''
        dbd = dbdreader.MultiDBD(pattern=path)
        tctd, C, T, D, lat, lon = dbd.get_CTD_sync("m_gps_lat", "m_gps_lon")
        lat_c = self.lat_c or lat.mean()
        lon_c = self.lon_c or lon.mean()
        SA = fast_gsw.SA(C*10, T, D*10, lon_c, lat_c)
        CT = fast_gsw.CT(C*10, T, D*10, lon_c, lat_c)
        pot_rho = fast_gsw.pot_rho(C*10, T, D*10, lon_c, lat_c)
        data = dict(time=tctd, pressure=D, T=T, C=C*10, P=D*10, S = SA, S0=SA.copy(), CT=CT, CT0=CT.copy(),
                    pot_rho=pot_rho, pot_rho0=pot_rho.copy())

        tl = ThermalLag(data, lat=lat_c, lon=lon_c)
        tl.interpolate_data(dt=1)
        tl.split_profiles()
        if short_time_correction:
            tl.apply_short_time_mismatch(short_time_correction)
        self.thermal_lag_model[glider] = (tl, short_time_correction)

    def calibrate(self, glider, z_min=None, z_max=None, **options):
        ''' Calibrates CTD

        Parameters
        ----------
        glider : str
            glider id, see also documentation of load_data()
        z_min : float or None:
            minimum depth of measurements to consider in the cost function
        z_max : float or None:
            maximum depth of measurements to consider in the cost function
        **options: dictionary with key words, passed to the minimisation model.
        
        Returns
        -------
            namedtuple tuple with alpha, beta, short_time_correction and number_of_profiles.
        '''
        tl, short_time_correction = self.thermal_lag_model[glider]
        if not z_min is None or not z_max is None:
            z_min = z_min or 0
            z_max = z_max or 1200
            z = tl.data["P"]
            condition = np.logical_or(z<z_min,
                                      z>z_max)
            tl.clear_mask()
            tl.apply_mask(condition)
            
        alpha, beta = tl.calibrate(initial_values=[0.05, 0.057], **options)
        number_of_profiles = len(tl)
        return CalibrationResult(alpha, beta, short_time_correction, number_of_profiles)

    def get_profiles(self, glider):
        '''
        Convenience method, that returns the profiles for a specific glider id

        Parameters
        ----------
        glider : str
            glider id, see also documentation of load_data()

        Returns
        -------
            ThermalLag instance (derived from ProfileSplitter)
        '''
        prfls, _ = self.thermal_lag_model[glider]
        return prfls


def create_figure():
    f, ax = plt.subplots(1,4)
    for i in range(2,4):
        ax[i].sharey(ax[1])
    return f, ax


def STplots(p, ax, color=None, offset=0, suffix='', **kwds):
    xd, yd = p.get_downcast(f"CT{suffix}", f"S{suffix}")
    xu, yu = p.get_upcast(f"CT{suffix}", f"S{suffix}")
    options = dict(alpha=0.5)
    options.update(kwds)
    if color:
        options['color']=color
    ax[0].plot(yd+offset, xd, **options)
    ax[0].plot(yu+offset, xu, **options)
    ax[0].set_xlabel('S')
    ax[0].set_ylabel('CT (degree Celsius)')
    f = lambda x, y : (x+offset, y)
    ax[1].plot(*f(*p.get_downcast(f"CT{suffix}", "P")), **options)
    ax[2].plot(*f(*p.get_downcast(f"S{suffix}", "P")), **options)
    ax[3].plot(*f(*p.get_downcast(f"pot_rho{suffix}", "P")), **options)
    ax[1].plot(*f(*p.get_upcast(f"CT{suffix}", "P")), ls='--', **options)
    ax[2].plot(*f(*p.get_upcast(f"S{suffix}", "P")), ls='--', **options)
    ax[3].plot(*f(*p.get_upcast(f"pot_rho{suffix}", "P")), ls='--', **options)
    ax[1].set_ylabel('Depth (m)')
    ax[2].set_ylabel('Depth (m)')
    ax[1].set_xlabel('CT (degree C)')
    ax[2].set_xlabel('SA (-)')
    ax[3].set_xlabel('Pot. Density (kg m^{-3})')
    ax[1].yaxis.set_inverted(True)
    ax[2].yaxis.set_inverted(True)
    ax[3].yaxis.set_inverted(True)

