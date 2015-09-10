'''
A module for splitting glider data into profiles

Provides:
      ProfileSplitter()
      CrossSpectral()

8 October 2013

lucas.merckelbach@hzg.de

'''
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import dict
from builtins import int
from builtins import range
from future import standard_library
standard_library.install_aliases()

import numpy as np
from functools import reduce

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
        self.set_window_size(window_size)
        self.set_threshold(threshold_bar_per_second)
        self.min_length=50 # at least 50 samples to be in a profile.
        self.required_depth=5.
        self.required_depth_range=15.
        self.data=data
        self.__remove_incomplete_tuples=remove_incomplete_tuples
        self.levels=[]
        self.summary={}
        
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

    def split_profiles(self,data=None):
        ''' splits the data into separate profiles. This method should be called before this object can do anything useful.'''
        self.data=data or self.data
        t=self.data[ProfileSplitter.T_str]
        P=self.data[ProfileSplitter.P_str]
        self.i_down,self.i_up=self.get_indices(t,P)
        self.t_down=np.array([t[i].mean() for i in self.i_down])
        self.t_up=np.array([t[i].mean() for i in self.i_up])
        self.t_cast=0.5*(self.t_down+self.t_up)
        

    def len(self):
        ''' returns number of profiles '''
        return len(self.i_up)

    def __len__(self):
        return len(self.i_up)

    def __get_cast(self,parameter,n):
        j=list(range(self.i_down[n][0],self.i_up[n][-1]+1))
        t=self.data[ProfileSplitter.T_str][j]
        v=self.data[parameter][j]
        return t,v

    def __get_upcast(self,parameter,n):
        ''' gets up profile for parameter for cast n '''
        j=list(range(self.i_up[n][0],self.i_up[n][-1]+1))
        t=self.data[ProfileSplitter.T_str][j]
        v=self.data[parameter][j]
        return t,v

    def __get_downcast(self,parameter,n):
        ''' gets down profile for parameter for cast n '''
        j=list(range(self.i_down[n][0],self.i_down[n][-1]+1))
        t=self.data[ProfileSplitter.T_str][j]
        v=self.data[parameter][j]
        return t,v

    def get_cast(self,n,parameter,co_parameter=None):
        ''' 
        if co_parameter==None:

        gets the down and up profile for parameter for cast n as time series (t,v)

        else:

        gets the down and up profile for parameter for cast n versus co_parameter
            

        '''

        if co_parameter==None:
            return self.__get_cast(parameter,n)
        else:
            return self.__get_cast(parameter,n)[1],self.__get_cast(co_parameter,n)[1]

    def get_upcast(self,n,parameter,co_parameter=None):
        '''
        if co_parameter==None:

        gets the up profile for parameter for cast n as time series (t,v)

        else:

        gets the up profile for parameter for cast n versus co_parameter
        '''

        if co_parameter==None:
            return self.__get_upcast(parameter,n)
        else:
            return self.__get_upcast(parameter,n)[1],self.__get_upcast(co_parameter,n)[1]

    def get_downcast(self,n,parameter,co_parameter=None):
        '''
        if co_parameter==None:

        gets the down profile for parameter for cast n as time series (t,v)

        else:

        gets the down profile for parameter for cast n versus co_parameter
        '''

        if co_parameter==None:
            return self.__get_downcast(parameter,n)
        else:
            return self.__get_downcast(parameter,n)[1],self.__get_downcast(co_parameter,n)[1]
        
            
    def get_upcast_avg(self,parameter,n,min_pitch=None,valid_depths=None,levels=None,
                       integrating_dimension='pressure'):
        '''
        Gets the averaged value of parameter for the n-th upcast.

        min_pitch, valid_depths and levels can be specified to fine tune which part of the
        profile is to be integrated.

        integrating_dimension can be 'pressure' or 'time'

        '''
        return self.__get_updw_cast_avg(parameter,n,'up',min_pitch,valid_depths,levels,
                                        integrating_dimension)

    def get_downcast_avg(self,parameter,n,min_pitch=None,valid_depths=None,levels=None,
                         integrating_dimension='pressure'):
        '''
        Gets the averaged value of parameter for the n-th downcast.

        min_pitch, valid_depths and levels can be specified to fine tune which part of the
        profile is to be integrated.

        integrating_dimension can be 'pressure' or 'time'

        '''

        return self.__get_updw_cast_avg(parameter,n,'down',min_pitch,valid_depths,levels,
                                        integrating_dimension)

    def get_up_avg(self,parameter,min_pitch=None,valid_depths=None,levels=None,
                   integrating_dimension='pressure'):
        ''' gets dpeth averaged values from up ALL profiles for parameter'''
        x=np.vstack([self.get_upcast_avg(parameter,n,min_pitch,valid_depths,levels,
                                         integrating_dimension) 
                     for n in range(self.len())])
        return x

    def get_down_avg(self,parameter,min_pitch=None,valid_depths=None,levels=None,
                     integrating_dimension='pressure'):
        ''' gets dpeth averaged values from ALL down profiles for parameter'''
        x=np.vstack([self.get_downcast_avg(parameter,n,min_pitch,valid_depths,levels,
                                           integrating_dimension) 
                     for n in range(self.len())])
        return x


    def remove_incomplete_tuples(self,i_down,i_up):
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

    def get_indices(self,t,P):
        ''' This method de factor splits the profiles by finding the for each
            profile the down cast indices, and up cast indices.

            The method is not intended to be called directly, but from self.split_profiles()
        '''
        _t=t-t[0]
        dPdT=np.gradient(P)/np.gradient(_t)
        window=np.ones(self.window_size,float)/float(self.window_size)
        dPdT_filtered=np.convolve(dPdT,window,'same')
        idx_down=self.__get_casts(dPdT_filtered,P,"down")
        idx_up=self.__get_casts(dPdT_filtered,P,"up")
        if self.__remove_incomplete_tuples:
            idx_down,idx_up=self.remove_incomplete_tuples(idx_down,idx_up)
        self.dPdT=dPdT_filtered
        return idx_down,idx_up


    def __get_updw_cast_avg(self,parameter,n,cast,min_pitch,valid_depths,levels,
                            integrating_dimension):

        if cast=='up':
            cast_fun=self.__get_upcast
            fc=-1.
        else:
            cast_fun=self.__get_downcast
            fc=1.
        t,v=cast_fun(parameter,n)
        t,P=cast_fun(ProfileSplitter.P_str,n)
        conditions=[]
        if min_pitch:
            pitch=tp,pitch=cast_fun('pitch',n)
            conditions.append(abs(pitch)>min_pitch)
        if valid_depths!=None:
            c1=P>valid_depths[0]/10.
            c2=P<valid_depths[1]/10.
            conditions.append(c1)
            conditions.append(c2)
        if levels!=None:
            # we have to limit the profile given the levels
            if levels[0]=='surface':
                d0=0
            else:
                d0=(cast_fun(levels[0],n)[1]).max()
            if levels[1]=='bed':
                d1=1e9
            else:
                d1=(cast_fun(levels[1],n)[1]).max()
            c=np.logical_and(P*10>=d0,P*10<d1)
            conditions.append(c)
        if conditions:
            c=reduce(np.logical_and,conditions)
            idx=np.where(c)
            if len(idx[0])==0:
                tm=np.nan
                vm=np.nan
            else:
                tm=t[idx].mean()
                if integrating_dimension=='pressure':
                    vm=np.trapz(v[idx],P[idx])/P[idx].ptp()
                else:
                    vm=v[idx].mean()
        else:
            tm=t.mean()
            if integrating_dimension=='pressure':
                vm=np.trapz(v,P)/P.ptp()
            else:
                vm=v.mean()
        return tm,fc*vm

    def __get_casts(self,dPdT_filtered,P,cast="up"):
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

    # to be able to select data according to the mixed layer depth
    def add_level_timeseries(self,t,z,level_name='pycnocline_depth'):
        ''' Sets a time series of depth and a given name, for example pycnocline.
            These depth levels can be used to limit the integration of profile
            averaged values just to a given layer, such as top-pycnocline, or
            2 m level - 10 m level etc.
        '''
        self.data[level_name]=np.interp(self.data[ProfileSplitter.T_str],t,z)
        self.levels.append(level_name)


class CrossSpectral(ProfileSplitter):
    def __init__(self,data={},window_size=9,threshold_bar_per_second=1e-3):
        ProfileSplitter.__init__(self,data,window_size,threshold_bar_per_second)
    
    def ideal_length(self,n):
        return 2**int(np.log2(n))

    def series_length(self):
        n=self.len()
        series_length=[self.get_cast(i,'time')[0].shape[0] for i in range(n)]
        min_series_length=min(series_length)
        sl=self.ideal_length(min_series_length)
        return n,series_length,sl

    def mean_fft(self,parameter,rn,series_length,sl):
        fftC=[]
        for i in rn:
            j0=(series_length[i]-sl)/2
            j1=j0+sl
            C=self.get_cast(i,parameter)[1]
            #Cw=np.hamming(j1-j0)*C[j0:j1]
            fftC.append(np.fft.fft(C[j0:j1]))
        return np.mean(fftC,axis=0)

    def Hs(self,param0,param1,i=None,binsize=1):
        n,series_length,sl=self.series_length()
        if i==None:
            rn=list(range(n))
        else:
            rn=i
        FC=self.mean_fft(param0,rn,series_length,sl)[:sl/2]
        FT=self.mean_fft(param1,rn,series_length,sl)[:sl/2]
        if binsize!=1:
            FC=FC.reshape(-1,binsize).mean(axis=1)
            FT=FT.reshape(-1,binsize).mean(axis=1)
        FCT=FC/FT
        a=FCT.real
        b=FCT.imag
        dT=np.diff(self.data['time']).mean()
        fn=0.5*1./dT
        omega=np.arange(sl/2/binsize)*fn/float(sl/2/binsize)*2.*np.pi
        print("sample length:",sl)
        print("n samples:    ",len(rn))
        return omega,a**2+b**2,np.arctan(b/a)

