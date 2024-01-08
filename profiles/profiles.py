'''
Splits glider data time series into profiles.

Glider data time series can be split in profiles:
1) per down/up cast pair
2) down casts only
3) up casts only

Provides:
      ProfileSplitter()

lucas.merckelbach@hereon.de
'''
#from collections import namedtuple, UserList

import numpy as np
from scipy.interpolate import interp1d

    
class SingleProfile(object):
    def __init__(self, data, s, despike=False):
        self.is_despike_data = despike
        self.data = data
        self.s = s
        
    def __getattr__(self, parameter):
        try:
            d = self.data[parameter]
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'.".format(self.__class__.__name__, a))
        if self.is_despike_data:
            return SingleProfile.despike(d[self.s])
        else:
            return d[self.s]

    @classmethod
    def despike(cls, x):
        xp = x.copy() # to return, keeps same size.
        y=np.vstack([x[:-2],x[1:-1],x[2:]])
        xp[1:-1] = np.median(y, axis=0)
        return xp
    
class ProfileList(object):

    def __init__(self, data, despike=False, profile_factory=None):
        self.data = data
        self.is_despike_data = despike
        self.slices = []
        self.profile_factory = profile_factory or SingleProfile

    def __iter__(self):
        self.__profile_counter = 0
        return self

    def __next__(self):
        pf = self.profile_factory
        if self.__profile_counter < len(self.slices):
            r = pf(self.data, self.slices[self.__profile_counter], despike=self.is_despike_data)
            self.__profile_counter += 1
            return r
        else:
            raise StopIteration
            
    def append(self, s):
        self.slices.append(s)

    def __getattr__(self, parameter):
        try:
            d = self.data[parameter]
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'.".format(self.__class__.__name__, parameter))
        if self.is_despike_data:
            d = self.profile_factory.despike(d)
        return tuple([d[s] for s in self.slices])

    @property
    def parameters(self):
        return tuple(self.data.keys())
    
    
class ProfileSplitter(object):
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

    '''

    T_str='time'
    P_str='pressure'

    def __init__(self,data={},window_size=9,threshold_bar_per_second=1e-3,
                 remove_incomplete_tuples=True, profile_factory=None):
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
        self.set_window_size(window_size)
        self.set_threshold(threshold_bar_per_second)
        self.min_length=50 # at least 50 samples to be in a profile.
        self.required_depth=5.
        self.required_depth_range=15.
        self.data=data
        self.remove_incomplete_tuples=remove_incomplete_tuples
        self.summary={}
        self.indices = []
        self.profile_factory = profile_factory
        if data:
            self.split_profiles()

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
            self.indices.append((s_down, s_up))

    @property
    def nop(self):
        ''' Number of profiles'''
        return len(self.indices)
    
    def remove_prematurely_ended_dives(self,threshold=3):
        ''' Remove up/down yo pairs for
            which have a depth that is more than <threshold> m shallower than
            the previous and next dive.

        Parameters
        ----------
        threshold : float {3}
            If profile is <threshold> m shallower than the previous and next profile, it is considered permaturely ended.

        '''
        disqualified = []
        slices = [slice(s_down.start, s_up.stop) for s_down, s_up in self.indices]
        max_depths = [self.data[ProfileSplitter.P_str][s].max()*10 for s in slices]
        for i, d in enumerate(max_depths):
            if i==0:
                continue
            if d-max_depths[i-1] < -threshold:
                disqualified.append(i)
        self.summary['removed_prematurely_ended_dives'] = disqualified
        if disqualified:
            disqualified.reverse() # start removing from the end.
            for dqfd in disqualified:
                self.indices.pop(dqfd)


    def get_downcasts(self,despike=False):
        return self._get_casts_worker(despike, 0b01)

    def get_upcasts(self,despike=False):
        return self._get_casts_worker(despike, 0b10)
    
    def get_casts(self,despike=False):
        return self._get_casts_worker(despike, 0b11)


    # Private methods
            
    
    def _get_casts_worker(self, despike, direction):
        pl = ProfileList(data=self.data, despike=despike, profile_factory=self.profile_factory)
        for s_down, s_up in self.indices:
            if direction & 0b01:
                start = s_down.start
            else:
                start = s_up.start
            if direction & 0b10:
                stop = s_up.stop
            else:
                stop = s_down.stop
            s = slice(start, stop)
            pl.append(s)
        return pl
            
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
        if len(i_down)==0 or len(i_up)==0: 
            print("Found either only downcasts or only upcasts or nothing at all. Continuing anyway.")
            return i_down, i_up
        # Normal behaviour:
        # downcast i ends before upcast i starts, and upcast i ends before downcast i+1 starts, if present.
        
        i = 0
        k = 0
        error = 0
        while True:
            try:
                d0 = i_down[i]
            except IndexError:
                error|=0b001
            try:
                u0 = i_up[i]
            except IndexError:
                error|=0b010
            if error == 0: # both profiles exist:
                condition0 = d0[-1] < u0[0] # downcast ends before upcast starts
                try:
                    d1 = i_down[i+1]
                except IndexError:
                    error|=0b100
                if error == 0:
                    condition1 = d1[0] > u0[-1]
                else:
                    condition1 = True
                if condition0 and condition1:
                    # all well
                    if error&0b100:
                        # found last pair.
                        break
                    else:
                        i+=1
                        continue
                else:
                    if condition1: # condition0 failed
                        # remove upcast i
                        i_up.pop(i)
                        k+=1
                    if condition0: # condition1 failed.
                        i_down.pop(i)
                        k+=1

            elif error&0b011: # both profiles fail to exists. We're done.
                break
            elif error == 0b010: # no upcast following down cast
                i_down.pop(i)
                k+=1 # counter for removed profiles.
            elif error == 0b001: # no downcast for following upcast.
                raise NotImplementedError("We have no downcast for next upcast. What should I do?")
            error = 0
        self.summary['Number of removed incomplete profiles'] = k
        return i_down,i_up


    def _get_casts(self,dPdT_filtered,P,cast="up"):
        direction=int(cast=="down")*2-1
        idx=np.where(direction*dPdT_filtered>self.threshold)[0]
        k=np.where(np.diff(idx)>1)[0]
        k+=1
        k=np.hstack([[0],k,[len(idx)]])
        jdx=[]
        ignored_profiles=[]
        ptps={}
        pmaxs={}
        profile_data={}
        for i in range(1,len(k)):
            if k[i]-k[i-1]>self.min_length:
                j=idx[k[i-1]:k[i]]
                pmax=P[j].max()
                ptp=P[j].ptp()
                if pmax>self.required_depth*0.1 and ptp>self.required_depth_range*0.1:
                    jdx.append(j)
                else:
                    ignored_profiles.append(i-1)
                    ptps[i-1]=ptp
                    pmaxs[i-1]=pmax
                    profile_data[i-1] = j
        self.summary['ignored_profiles']=ignored_profiles
        self.summary['ptp'] = ptps
        self.summary['pmax'] = pmaxs
        self.summary['profile_data'] = profile_data
        return jdx

        
