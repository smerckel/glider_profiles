import numpy as np
from . import profiles

class Thermocline(profiles.ProfileSplitter):
    Defaults=dict(N2crit=0.1e-3,
                  threshold=3e-2,
                  minimal_gradient=0.06)

    def __init__(self,data={},window_size=9,threshold_bar_per_second=1e-3,
                 remove_incomplete_tuples=True):
        profiles.ProfileSplitter.__init__(self,data,window_size,
                                          threshold_bar_per_second,
                                          remove_incomplete_tuples)

    def __select_value(self,parameter,value):
        if value==None:
            return self.Defaults[parameter]
        else:
            return value

    def calc_rho(self):
        P=[self.get_downcast(i,'pressure')[1]*10 for i in range(self.len())]
        t=[self.get_downcast(i,'pressure')[0] for i in range(self.len())]
        rho=[self.get_downcast(i,'rho')[1] for i in range(self.len())]
        return t,P,rho

    def calc_thermocline_limits(self,threshold=None,minimal_gradient=None):
        threshold=self.__select_value('threshold',threshold)
        minimal_gradient=self.__select_value('minimal_gradient',minimal_gradient)
        
        t,P,rho=self.calc_rho()
        thermocline_limits=[]
        W=20
        for k,(_t,_P,_rho) in enumerate(zip(t,P,rho)):
            drho_dP=np.convolve(np.gradient(_rho)/np.gradient(_P),np.ones(W)/W,'same')
            i_max=np.argmax(drho_dP)
            max_gradient=drho_dP[i_max]
            top=np.nan
            bottom=np.nan
            centre=np.nan
            if max_gradient>minimal_gradient:
                centre=_P[i_max]
                j=np.where(drho_dP[:i_max]<threshold)[0]
                if len(j): # we found at least one. Take the last one
                    top=_P[j[-1]]
                j=np.where(drho_dP[i_max:]<threshold)[0]
                if len(j): # we found at least one. Take the first
                    bottom=_P[i_max+j[0]]
            thermocline_limits.append((top,centre,bottom,max_gradient))
        thermocline_limits=np.array(thermocline_limits).T
        for i in range(3):
            thermocline_limits[i]=self.despike(thermocline_limits[i])
        return thermocline_limits

    def calc_buoyancy_frequency(self,rho0=1027.,window=15):
        t,P,rho=self.calc_rho()
        W=window
        _N2=[9.81/rho0*np.gradient(_rho)/np.gradient(p) for _rho,p in zip(rho,P)]
        N2=[np.convolve(i,np.ones(W)/W,'same') for i in _N2]
        return t,P,rho,N2
        
    def calc_thermocline_maxN2(self,N2_crit=1e-3):
        N2_crit=self.__select_value('N2_crit',N2_crit)
        t,P,rho,N2=self.calc_buoyancy_frequency()
        z_tc=[]
        t_tc=[]
        N2_tc=[]
        for _t,_z,_N2 in zip(t,P,N2):
            z_max=_z.max()
            i=np.where(np.logical_and(_z<0.8*z_max,_z>5))[0]
            t_tc.append(_t.mean())
            if _N2[i].max()>N2_crit:
                idx=np.argmax(_N2[i])
                z_tc.append(_z[i][idx])
                N2_tc.append(_N2[i][idx])
            else:
                z_tc.append(0.)
                N2_tc.append(np.nan)
        return np.array(t_tc),np.array(z_tc),np.array(N2_tc)

    def calc_thermocline_top(self,N2_crit=1e-3):
        N2_crit=self.__select_value('N2_crit',N2_crit)
        t,P,rho,N2=self.calc_buoyancy_frequency()
        z_tc=[]
        t_tc=[]
        N2_tc=[]
        for _t,_z,_N2 in zip(t,P,N2):
            z_max=_z.max()
            i=np.where(np.logical_and(_z<0.95*z_max,_z>5))[0]
            t_tc.append(_t.mean())
            idx=np.where(_N2[i]>N2_crit)[0]
            if len(idx)==0:
                z_tc.append(np.nan)
                N2_tc.append(np.nan)
            else:
                z_tc.append(_z[i][idx[0]])
                N2_tc.append(_N2[i][idx[0]])
        return np.array(t_tc),np.array(z_tc),np.array(N2_tc)

    def despike(self,x):
        nx=len(x)
        y=x.copy()
        for i in range(1,nx-1):
            y[i]=np.median(x[i-1:i+2])
        return y
