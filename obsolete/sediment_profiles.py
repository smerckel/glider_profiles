from collections import namedtuple
from scipy.signal import csd, coherence, detrend
from scipy.signal._spectral_py import _fft_helper

from . import filters

from . import iterprofiles

profile_integrated = namedtuple("profile_integrated","t C c0 c1 H0 H1 H")


class Profile(object):
    def __init__(self,data,i_down,i_up,
                 T_str='time',P_str='pressure'):
        self.data=data
        self.i_down=i_down
        self.i_up=i_up
        self.i_cast=np.hstack((i_down,i_up))
        self.T_str=T_str
        self.P_str=P_str
        t=self.data[T_str]
        self.t_down=t[self.i_down].mean()
        self.t_up=t[self.i_up].mean()
        self.t_cast=0.5*(self.t_down+self.t_up)
        
    def __call__(self,parameter,co_parameter=None):
        pass

    def __get_cast_data(self,i,parameter,co_parameter,despike=False):
        if co_parameter==None:
            x=self.data[self.T_str][i]
            if parameter in self.data.keys():
                y=self.data[parameter][i]
            elif parameter=="surface":
                y=x*0
            elif parameter=="bed":
                y=x*0+1e9
            else:
              raise ValueError("Unknown parameter!")
        else:
            if parameter in self.data.keys():
                x=self.data[parameter][i]
            elif parameter=="surface":
                x=(i*0).astype(float)
            elif parameter=="bed":
                x=(i*0).astype(float)+1e9
            else:
                raise ValueError("Unknown parameter!")  
            if co_parameter in self.data.keys():
                y=self.data[co_parameter][i]
            elif co_parameter=="surface":
                y=x*0
            elif co_parameter=="bed":
                y=x*0+1e9
            else:
                raise ValueError("Unknown co_parameter!")
        if despike:
            if co_parameter:
                x=self.despike(x)
            y=self.despike(y)
        return x,y
            
    def get_cast(self,parameter,co_parameter=None,despike=False):
        return self.__get_cast_data(self.i_cast,parameter,co_parameter,despike)

    def get_upcast(self,parameter,co_parameter=None,despike=False):
        return self.__get_cast_data(self.i_up,parameter,co_parameter,despike)

    def get_downcast(self,parameter,co_parameter=None,despike=False):
        return self.__get_cast_data(self.i_down,parameter,co_parameter,despike)

    def get_downcast_integrated(self,parameter,levels=[],despike=False,min_values=3, avg_distance=1):
        return self.__get_cast_integrated(parameter,levels,'down',despike,min_values, avg_distance)

    def get_upcast_integrated(self,parameter,levels=[],despike=False,min_values=3, avg_distance=1):
        return self.__get_cast_integrated(parameter,levels,'up',despike,min_values, avg_distance)
    
    def get_downcast_gradient(self,parameter,levels=[],despike=False,min_values=3):
        return self.__get_cast_gradient(parameter,levels,'down',despike,min_values)

    def get_upcast_gradient(self,parameter,levels=[],despike=False,min_values=3):
        return self.__get_cast_gradient(parameter,levels,'up',despike,min_values)
    
    def despike(self,x):
        xp=x.copy()
        n=xp.shape[0]
        for i in range(1,n-1):
            xp[i]=np.median(x[i-1:i+2])
        return xp

    def __get_cast_gradient(self,parameter,levels,direction,despike,min_values):
        if direction=="up":
            get_fun=self.get_upcast

        else:
            get_fun=self.get_downcast
            fc=1
        s=int(direction=="down")*2-1 # stride -1 for upcast, +1 for downcast
        
        t,x=get_fun(parameter)
        if despike:
            x=self.despike(x)
        d=get_fun(self.P_str)[1]*10
        t=t[::s]
        x=x[::s]
        d=d[::s]
        
        i=np.where(np.diff(d)>0)[0]
        iall=np.hstack([i,[i[-1]+1]])
        if levels:
            top_level,bottom_level=get_fun(levels[0],levels[1])
            i_section=iall.compress(np.logical_and(d[iall]>=top_level[iall],
                                                   d[iall]<=bottom_level[iall]))
        else:
            i_section=iall
        if len(i_section)>=min_values:
            dxdz=np.polyfit(x[i_section],d[i_section],1)[0]
        else:
            dxdz=0
        return dxdz
    
    def get_level(self, get_fun, level):
        try:
            _, z = get_fun(level)
        except ValueError:
            z = level
        else:
            z=z.mean()
        return z
    
    def __get_cast_integrated(self,parameter,levels,direction,despike,min_values, avg_distance):
        # avg_distance: to determine the value on a level (top level/ bottom level, a median value is computed within avg_distance
        
        if direction=="up":
            get_fun=self.get_upcast
        else:
            get_fun=self.get_downcast

        s=int(direction=="down")*2-1 # stride -1 for upcast, +1 for downcast
        
        t,x=get_fun(parameter)
        if despike:
            x=self.despike(x)
        d=get_fun(self.P_str)[1]*10
        t=t[::s]
        x=x[::s]
        d=d[::s]
        
        i=np.where(np.diff(d)>0)[0]
        iall=np.hstack([i,[i[-1]+1]])
        if levels:
            try:
                lvl = self.get_level(get_fun, levels)
                top_level = lvl - avg_distance/2
                bottom_level = lvl + avg_distance/2
            except (TypeError, ValueError) as e:
                top_level = self.get_level(get_fun, levels[0])
                bottom_level = self.get_level(get_fun, levels[1])
            i_section=iall.compress(np.logical_and(d[iall]>=top_level,
                                                   d[iall]<=bottom_level))
        else:
            i_section=iall
            top_level = d.min()
            bottom_level = d.max()
        
        I=np.trapz(x[i_section],d[i_section])
        if len(i_section)<min_values:
            I=0
            H=1
            H_top=0
            H_bottom=0
            top_value=0
            bottom_value=0
            tm=t.mean()
        else:
            H=(d[i_section[0]]-d[i_section[-1]])
            tm=t[i_section].mean()
            _i=iall.compress(abs(d[iall]-top_level)<avg_distance/2)
            if len(_i):
                top_value=np.median(x[_i])
            else:
                top_value=0
            _i=iall.compress(abs(d[iall]-bottom_level)<avg_distance/2)
            if len(_i):
                bottom_value=np.median(x[_i])
            else:
                bottom_value=0
            H_top=d[i_section[0]]
            H_bottom=d[i_section[-1]]
        return profile_integrated(tm,I,top_value,bottom_value,H_top,H_bottom, -H)



class SedimentProfileSplitter(iterprofiles.ProfileSplitter):
    def __init__(self,data={},window_size=9,threshold_bar_per_second=1e-3,
                 remove_incomplete_tuples=True):
        '''
        data: dictionary of data to be split in profiles
              should contain T_str (default "time")
                             P_str (default "pressure")
                             and other data
        window_size:
        threshold_bar_per_second: discriminant for detecting profile changes
        remove_incomplete_tuples: if true only when down AND up cast are available,
                                  the profile is retained.
        '''
        super().__init__(data, window_size, thresholde_bar_per_second, remove_incomplte_tuples)


        
    def add_level_timeseries(self,t,z,level_name='pycnocline_depth'):
        ''' Sets a time series of depth and a given name, for example pycnocline.
            These depth levels can be used to limit the integration of profile
            averaged values just to a given layer, such as top-pycnocline, or
            2 m level - 10 m level etc.
        '''
        self.data[level_name]=np.interp(self.data[ProfileSplitter.T_str],t,z)
        if level_name not in self.levels:
            self.levels.append(level_name)

    def get_downcast_integrated(self,parameter,levels=[],despike=False, avg_distance=1):
        ''' get integrated upcast value for parameter at levels. The values at the levels are median values computed over within avg_distance m'''
                
        x=np.array([p.get_downcast_integrated(parameter,levels,despike, avg_distance=avg_distance)
                    for p in self]).T
        s={'t':x[0],
           parameter:x[1],
           'z1':x[4],
           'z0':x[5],
           "%s_z1"%(parameter):x[2],
           "%s_z0"%(parameter):x[3],
           "levels":levels}

        profile_integrated = namedtuple("profile_integrated","t %s c0 c1 H0 H1 H"%(parameter))
        s = profile_integrated(*x)
        return s
    
    def get_upcast_integrated(self,parameter,levels=[],despike=False, avg_distance=1):
        ''' get integrated upcast value for parameter at levels. The values at the levels are median values computed over within avg_distance m'''
        
        x=np.array([p.get_upcast_integrated(parameter,levels,despike, avg_distance=avg_distance)
                    for p in self]).T
        s={'t':x[0],
           parameter:x[1],
           'z1':x[4],
           'z0':x[5],
           "%s_z1"%(parameter):x[2],
           "%s_z0"%(parameter):x[3],
           "levels":levels}

        profile_integrated = namedtuple("profile_integrated","t %s c0 c1 H0 H1 H"%(parameter))
        s = profile_integrated(*x)
        return s

    # some interpolation functions:
    def interpolate_data(self,dt = 1):
        ''' using linear interpolation '''
        tctd = self.data["time"]
        ti = np.arange(tctd.min(), tctd.max()+dt, dt)
        for k, v in self.data.items():
            if k=="time":
                continue
            self.data[k] = np.interp(ti, tctd, v)
        self.data["time"] = ti

    def interpolate_data_shape_preserving(self,dt = 1):
        ''' using cubic shape preserving interpolation '''
        tctd = self.data["time"]
        ti = np.arange(tctd.min(), tctd.max()+dt, dt)
        for k, v in self.data.items():
            if k=="time":
                continue
            f = pchip(tctd, v)
            self.data[k] = f(ti)
        self.data["time"] = ti
        
    def lag_filter_pressure_data(self, delay, other_pressure_parameters = []):
        ti = self.data["time"]
        p = ["pressure"] + other_pressure_parameters
        LF = filters.LagFilter(1,delay)
        for k in p:
            self.data[k] = LF.filter(ti, self.data[k])

    def get_profile(self, t):
        ''' Get nearest profile.

        For given time t, the method returns the profile that is closest in time.

        Parameters:
        -----------
        
        t: scalar | time in s

        Returns:
        --------
        
        profile that is closest in time.
        '''
        return self[self.get_profile_index(t)]
    
    def get_profile_index(self, t):
        ''' Get nearest profile index.

        For given time t, the method returns the profile index that is closest in time.

        Parameters:
        -----------
        
        t: scalar | time in s

        Returns:
        --------
        
        index of profile that is closest in time.
        '''
        cast_times = np.array([_p.t_cast for _p in self])
        idx = np.argmin(np.abs(cast_times-t))
        return idx


class Thermocline(ProfileSplitter):
    def __init__(self,data={},window_size=9,threshold_bar_per_second=1e-3,
                 remove_incomplete_tuples=True):
        ProfileSplitter.__init__(self,data,window_size,threshold_bar_per_second,
                                 remove_incomplete_tuples)

    def add_buoyancy_frequency(self,rho0=1027.,window=15):
        ''' Adds the buoyancy frequency to the self.data dictionary '''
        W=window
        rho=self.data['rho']
        z=self.data[self.P_str]*10
        dz = np.gradient(z)
        drho = np.gradient(rho)
        condition = np.abs(dz)>0.02 # if dz >? 2 cm compute N2, otherwise leave it nan
        i = np.where(condition) 
        _N2=9.81/rho0*drho[i]/dz[i]
        _N2=np.convolve(_N2,np.ones(W)/W,'same')
        N2 = np.zeros_like(z)*np.nan
        N2[i] = _N2
        self.data['N2']=N2

    def thermocline_depth_maxN2(self,
                                direction="down",
                                N2_crit=1e-3,
                                min_depth=5,
                                max_depth=35,
                                max_depth_pct=80):
        ttcl=np.zeros(len(self),float)
        ztcl=np.zeros(len(self),float)

        for j,p in enumerate(self):
            if direction == "down":
                f = p.get_downcast
            elif direction == "up":
                f = p.get_upcast
            else:
                raise ValueError("direction should be up or down.")
            t,P=f(self.P_str)
            _, N2=f("N2")
            condition=np.logical_and(P*10>min_depth,
                                     np.logical_or(P*10<max_depth,P<max_depth_pct/10*P.max()))
            condition=np.logical_and(np.isfinite(N2), condition)
            N2_section,P_section,t_section=np.compress(condition,[N2,P,t],axis=1)
            i=np.argmax(N2_section)
            if N2_section[i]>N2_crit:
                ttcl[j]=t_section[i]
                ztcl[j]=P_section[i]*10
            else:
                ttcl[j]=t_section[0]
                ztcl[j]=np.nan
        return ttcl, ztcl
    
    def thermocline_depth_max_temp_gradient(self,
                                direction="down",
                                Tgrad_crit=0.5, # 0.5 deg/m or more
                                min_depth=5,
                                max_depth=35,
                                max_depth_pct=80):
        ttcl=np.zeros(len(self),float)
        ztcl=np.zeros(len(self),float)

        for j,p in enumerate(self):
            if direction == "down":
                f = p.get_downcast
            elif direction == "up":
                f = p.get_upcast
            else:
                raise ValueError("direction should be up or down.")
            t,P=f(self.P_str)
            _, T=f("T")
            dTdz = np.gradient(T)/np.gradient(P)/10.
            condition=np.logical_and(P*10>min_depth,
                                     np.logical_or(P*10<max_depth,P<max_depth_pct/10*P.max()))
            condition=np.logical_and(np.isfinite(dTdz), condition)
            dTdz_section,P_section,t_section=np.compress(condition,[dTdz,P,t],axis=1)
            i=np.argmax(dTdz_section)
            if dTdz_section[i]>Tgrad_crit:
                ttcl[j]=t_section[i]
                ztcl[j]=P_section[i]*10
            else:
                ttcl[j]=t_section[0]
                ztcl[j]=np.nan
        return ttcl, ztcl

    def calc_thermocline_maxN2(self,N2_crit=1e-3,
                               min_depth=5,
                               max_depth=35,
                               max_depth_pct=80):
        ttcl=np.zeros((2,len(self)),float)
        ztcl=np.zeros((2,len(self)),float)

        for j,p in enumerate(self):
            for k in range(2):
                f=[p.get_downcast,p.get_upcast]
                t,P=f[k](self.P_str)
                N2=f[k]("N2")[1]
                condition=np.logical_and(P*10>min_depth,
                                         np.logical_or(P*10<max_depth,P<max_depth_pct/10*P.max()))
                N2_section,P_section,t_section=np.compress(condition,[N2,P,t],axis=1)
                i=np.argmax(N2_section)
                if N2_section[i]>N2_crit:
                    ttcl[k,j]=t_section[i]
                    ztcl[k,j]=P_section[i]
                else:
                    ttcl[k,j]=t_section[0]
                    ztcl[k,j]=0
        self.add_level_timeseries(ttcl[0],ztcl[0],level_name='pycnocline_depth_downcast')
        self.add_level_timeseries(ttcl[1],ztcl[1],level_name='pycnocline_depth_upcast')
        self.add_level_timeseries(ttcl.mean(axis=0),ztcl.mean(axis=0),level_name='pycnocline_depth')

    def calc_thermocline_temperature(self,dT=0.1):
        ttcl=np.zeros((2,len(self)),float)
        ztcl=np.zeros((2,len(self)),float)

        for j,p in enumerate(self):
            f=[p.get_downcast,p.get_upcast]
            for k in range(2):
                t,P=f[k](self.P_str)
                T=f[k]("T")[1]
                counts,bins=np.histogram(T,10)
                Tc=(bins[1:]+bins[:-1])*0.5
                ttcl[k,j]=t.mean()
                if Tc.ptp()<dT:
                    ztcl[k,j]=np.nan
                else:
                    Tmean=Tc.mean()
                    counts_warm=counts.compress(Tc>Tmean)
                    Tc_warm=Tc.compress(Tc>Tmean)
                    T_warm=Tc_warm[np.argmax(counts_warm)]
                    counts_cold=counts.compress(Tc<Tmean)
                    Tc_cold=Tc.compress(Tc<Tmean)
                    T_cold=Tc_cold[np.argmax(counts_cold)]
                    Tmean=(T_warm+T_cold)/2.
                    i=np.argmin(np.abs(T-Tmean))
                    ztcl[k,j]=P[i]*10
        
        self.add_level_timeseries(ttcl.T.ravel(),ztcl.T.ravel(),level_name='pycnocline_depth')
        return ttcl,ztcl
        
