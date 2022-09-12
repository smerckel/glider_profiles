'''
A module for splitting glider data into profiles

Provides:
      ProfileSplitter()

lucas.merckelbach@hereon.de
'''
from collections import namedtuple

import numpy as np
from scipy.interpolate import interp1d


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
        if co_parameter is None:
            s = ['t', parameter]
        else:
            s = [parameter, co_parameter]
        ProfileData = namedtuple("ProfileData", s)
        return ProfileData(x,y)
            
    def get_cast(self,parameter,co_parameter=None,despike=False):
        return self.__get_cast_data(self.i_cast,parameter,co_parameter,despike)

    def get_upcast(self,parameter,co_parameter=None,despike=False):
        return self.__get_cast_data(self.i_up,parameter,co_parameter,despike)

    def get_downcast(self,parameter,co_parameter=None,despike=False):
        return self.__get_cast_data(self.i_down,parameter,co_parameter,despike)


    def despike(self,x):
        xp = x.copy() # to return, keeps same size.
        y=np.vstack([x[:-2],x[1:-1],x[2:]])
        xp[1:-1] = np.median(y, axis=0)
        return xp
    


    
class ProfileSplitter(list):
    ''' A class to split glider data into profiles

    Typical use:

    import dbdreader
    import profiles.profiles
    

    dbd=dbdreader.MultiDBD(pattern='path/to/some/gliderbinary/files.[ed]bd')
    tmp=dbd.get_sync("sci_ctd41cp_timestamp",["sci_water_temp",
                                          "sci_water_cond",
                                          "sci_water_pressure"])

    t_dummy,tctd,T,C,P=tmp

    data=dict(time=tctd,
               pressure=P,
               # and now the variables. You will reference them by the key
               # name you give them in this dictionary.
               T=T,
               C=C*10, # mS/cm
               P=P*10) # bar

    splitter=profiles.profiles.ProfileSplitter(data=data) # default values should be OK
    splitter.split_profiles()
    temperature_casts=[splitter.get_cast(k,'T') for k in range(len(splitter))]
    '''

    T_str='time'
    P_str='pressure'

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
        list.__init__(self)
        self.set_window_size(window_size)
        self.set_threshold(threshold_bar_per_second)
        self.min_length=50 # at least 50 samples to be in a profile.
        self.required_depth=5.
        self.required_depth_range=15.
        self.data=data
        self.remove_incomplete_tuples=remove_incomplete_tuples
        self.levels=[]
        self.summary={}

    def __getattr__(self, a):
        try:
            return self.data[a]
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'.".format(self.__class__.__name__, a))

    def set_window_size(self,window_size):
        ''' sets window size used in the moving averaged smoother of the pressure rate
        '''
        self.window_size=window_size
        
    def get_window_size(self):
        ''' gets window size used in the moving averaged smoother of the pressure rate
        '''

        return self.window_size

    def set_threshold(self,threshold):
        ''' sets threshold for change in gradient marks end of profile '''
        self.threshold=threshold
        
    def get_threshold(self):
        ''' gets threshold for change in gradient marks end of profile '''
        return self.threshold

    def set_required_depth(self,required_depth):
        ''' Set depth level a profile minimally should have '''
        self.required_depth=required_depth

    def set_required_depth_range(self,required_depth_range):
        ''' Set depth range a profile minimally should have '''
        self.required_depth_range=required_depth_range

    def split_profiles(self,data=None, interpolate=False):
        ''' Splits data into separate profiles. 
        
        Parameters
        ----------
        data : data dictionary or None
            a dictionary with data, and at least "time" and "pressure" fields. If None, then
            the data dictionary supplied to the constructor is used.
        interpolate : boolean (Default: False)
            interpolates time and pressure data prior to splitting the profiles
          
        Notes
        -----
        The method relies on detecting changes in the filtered depth rate. If, however, data
        are provided, that have down casts, or up casts only, then this fails. A workaround
        is to interpolate the time and pressure data first, then split, and remap the indices
        to the original time stamps. This is achieved by setting interpolate to True.

        It is recommnended to leave interpolate equal to False, unless you have data that contain
        either down or up casts.
        
        This method should be called before this object can do anything useful.'''
        self.data=data or self.data
        t=self.data[ProfileSplitter.T_str]
        P=self.data[ProfileSplitter.P_str]
        if interpolate:
            dt = min(np.median(np.diff(t)), 4)
            ti = np.arange(t.min(), t.max()+dt, dt)
            Pi = np.interp(ti, t, P)
            tfun = interp1d(t, np.arange(t.shape[0]))
            ifun = lambda i: self._slice_fun((tfun(ti[i])+0.5).astype(int))
        else:
            ti = t
            Pi = P
        i_down, i_up = self._get_indices(ti,Pi)
        for _i_down, _i_up in zip(i_down, i_up):
            if interpolate:
                s_down = ifun(_i_down)
                s_up = ifun(_i_up)
            else:
                s_down = self._slice_fun(_i_down)
                s_up = self._slice_fun(_i_up)
            self.append(Profile(self.data,
                                np.arange(s_down.start, s_down.stop), # convert slices to index numbers
                                np.arange(s_up.start, s_up.stop),     # other methods rely on this.
                                self.T_str,self.P_str))

    def remove_prematurely_ended_dives(self,threshold=3):
        ''' Remove up/down yo pairs for
            which have a depth that is more than <threshold> m shallower than
            the previous and next dive.

        Parameters
        ----------
        threshold : float {3}
            If profile is <threshold> m shallower than the previous and next profile, it is considered permaturely ended.

        '''
        removables=[]
        for p,c,n in zip(self[:-2],self[1:-1],self[2:]):
            max_depth=(p.get_cast("pressure")[1].max() + \
                       n.get_cast("pressure")[1].max())*0.5
            if c.get_cast("pressure")[1].max()<max_depth-threshold/10:
                removables.append(c)
        for r in removables:
            self.remove(r)

    def get_upcasts(self,parameter, co_parameter=None, despike=False):
        return tuple([p.get_upcast(parameter, co_parameter=None, despike=despike) for p in self])

    def get_dowcasts(self,parameter,co_parameter=None,despike=False):
        return tuple([p.get_downcast(parameter, co_parameter=None, despike=despike) for p in self])

    def get_casts(self,parameter,co_parameter=None,despike=False):
        return tuple([p.get_cast(parameter, co_parameter=co_parameter, despike=despike) for p in self])
    
            
    # Private methods
            
    def _slice_fun(self, idx):
        ''' Returns index list as a slice

        Parameters
        ----------
        idx : array of integers
            data indices
        
        Returns
        -------
        slice : slice object
             slice with intial and last data indices. If idx is empty, a slice(0,0) is returned.
        '''
        if idx.shape[0]:
            return slice(idx[0], idx[-1])
        else:
            return slice(0,0)

    def _get_indices(self,t,P):
        ''' This method de facto splits the profiles by finding the for each
            profile the down cast indices, and up cast indices.

            The method is not intended to be called directly, but from self.split_profiles()
        '''
        _t=t-t[0]
        dT = np.gradient(_t)
        dPdT=np.gradient(P)/dT
        window=np.ones(self.window_size,float)/float(self.window_size)
        dPdT_filtered=np.convolve(dPdT,window,'same')
        idx_down=self._get_casts(dPdT_filtered,P,"down")
        idx_up=self._get_casts(dPdT_filtered,P,"up")
        if self.remove_incomplete_tuples:
            idx_down,idx_up=self._remove_incomplete_tuples(idx_down,idx_up)
        self.dPdT=dPdT_filtered
        return idx_down,idx_up

    def _remove_incomplete_tuples(self,i_down,i_up):
        ''' remove incomplete up/down yo pairs '''
        kd=0
        ku=0
        idx_up=[]
        idx_down=[]
        n_down=len(i_down)
        n_up=len(i_up)
        while 1:
            if kd>=n_down or ku>=n_up:
                break
            c1=i_down[kd][0]<i_up[ku][0] # dive is before climb
            if not c1: 
                ku+=1
                continue
            if kd>=n_down-1:
                break
            c2=i_down[kd+1][0]>i_up[ku][0] # next dive is after climb
            if not c2:
                kd+=1
                continue
            # got a pair
            idx_down.append(i_down[kd])
            idx_up.append(i_up[ku])
            kd+=1
            ku+=1
        return idx_down,idx_up


    def _get_casts(self,dPdT_filtered,P,cast="up"):
        direction=int(cast=="down")*2-1
        idx=np.where(direction*dPdT_filtered>self.threshold)[0]
        k=np.where(np.diff(idx)>1)[0]
        k+=1
        k=np.hstack([[0],k,[len(idx)]])
        jdx=[]
        ignored_profiles=[]
        ptps=[]
        pmaxs=[]
        for i in range(1,len(k)):
            if k[i]-k[i-1]>self.min_length:
                j=idx[k[i-1]:k[i]]
                pmax=P[j].max()
                ptp=P[j].ptp()
                if pmax>self.required_depth*0.1 and ptp>self.required_depth_range*0.1:
                    jdx.append(j)
                else:
                    ignored_profiles.append(i-1)
                    ptps.append(ptp)
                    pmaxs.append(pmax)
        self.summary['ignored_profiles']=ignored_profiles
        self.summary['ptp']=ptps
        self.summary['pmax']=pmaxs
        return jdx

        

import dbdreader

dbd_path = "/home/lucas/gliderdata/nsb3_201907/hd/comet-2019-203-05-000.?bd"
dbd = dbdreader.MultiDBD(dbd_path)
tctd, C, T, D, flntu_turb = dbd.get_CTD_sync("sci_flntu_turb_units")
data = dict(time=tctd, pressure=D, C=C*10, T=T, P=D*10, spm=flntu_turb)

ps = ProfileSplitter(data)
ps.split_profiles()

