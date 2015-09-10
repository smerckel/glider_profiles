import numpy as np
import pylab as pl
import dbdreader
import profiles
import profiles.filters
from scipy.optimize import fmin
import glob
import ndf


class SensorLag(profiles.profiles.ProfileSplitter):
    tau_range=np.arange(1,3,0.05)
    def __init__(self,data):
        profiles.profiles.ProfileSplitter.__init__(self,data=data)
        
    def cost_function_org(self,pd,vd,pu,vu):
        i_sorted=np.argsort(pd)
        pd=pd[i_sorted]
        vd=vd[i_sorted]
        i_sorted=np.argsort(pu)
        pu=pu[i_sorted]
        vu=vu[i_sorted]
        vu_i=np.interp(pd,pu,vu)
        pmin=0.2*pd.max()
        pmax=0.8*pd.max()
        delta=np.compress(np.logical_and(pd>pmin,pd<pmax),[pd,vu_i-vd],axis=1)
        return (delta**2).sum()

    def parameterize(self,x,y,ns=1000,reverse=False):
        t=np.arange(len(x))
        ds=np.sqrt((x[:-1]-x[1:])**2+(y[:-1]-y[1:])**2)
        s=np.hstack([[0],np.cumsum(ds)])
        s/=s[-1]
        si=np.linspace(0,1,ns)
        ti=np.interp(si,s,t)
        xi=np.interp(ti,t,x)
        yi=np.interp(ti,t,y)
        if reverse:
            return xi[::-1],yi[::-1]
        else:
            return xi,yi
        
    def cost_function(self,pd,vd,pu,vu):
        ns=100
        pdi,vdi=self.parameterize(pd,vd,ns,reverse=False)
        pui,vui=self.parameterize(pu,vu,ns,reverse=True)

        a=np.sqrt((pdi[1:]-pdi[:-1])**2+(vdi[1:]-vdi[:-1])**2)
        b=np.sqrt((pui[1:]-pdi[:-1])**2+(vui[1:]-vdi[:-1])**2)
        c=np.sqrt((pdi[1:]-pui[1:])**2+(vdi[1:]-vui[1:])**2)
        s=0.5*(a+b+c)
        A=np.sqrt(s*(s-a)*(s-b)*(s-c)) # heron's formula
        b=np.sqrt((pui[:-1]-pdi[:-1])**2+(vui[:-1]-vdi[:-1])**2)
        c=np.sqrt((pdi[1:]-pui[:-1])**2+(vdi[1:]-vui[:-1])**2)
        s=0.5*(a+b+c)
        A+=np.sqrt(s*(s-a)*(s-b)*(s-c)) # heron's formula
        if 0:
            pl.clf()
            pl.plot(pd,vd)
            pl.plot(pu,vu)
            for j in range(ns):
                pl.plot([pdi[j],pui[j]],[vdi[j],vui[j]],'k',alpha=0.1)
            pl.draw()
            input(A.sum())
        return A.sum()

    def fun(self,tau,pd,pu,vd,vu):
        fltr=profiles.filters.LagFilter(1.,abs(tau))
        if tau>10:
            tau=10.
        if tau<-10:
            tau=-10
        if tau>0:
            pdf=fltr.filter(pd[0],pd[1])
            puf=fltr.filter(pu[0],pu[1])
            vdf=vd[1]
            vuf=vu[1]
        else:
            vdf=fltr.filter(vd[0],vd[1])
            vuf=fltr.filter(vu[0],vu[1])
            pdf=pd[1]
            puf=pu[1]
        return self.cost_function(pdf,vdf,puf,vuf)

    def __estimate_profile_lag(self,pd,pu,vd,vu):
        tau=fmin(self.fun,(2,),args=(pd,pu,vd,vu),disp=0)
        print(tau[0])
        return tau[0]


    def estimate_lag(self,parameter_name):
        tau_range=self.tau_range
        R=np.zeros((len(tau_range),self.len()),float)
        for i,tau in enumerate(tau_range):
            print(tau)
            fltr=profiles.filters.LagFilter(1.,abs(tau))
            if tau<-0.01:
                p=fltr.filter(self.data['time'],self.data[parameter_name])
                self.data['filtered_parameter']=p
                self.data['filtered_pressure']=self.data['pressure']
            elif tau>0.01:
                p=fltr.filter(self.data['time'],self.data['pressure'])
                self.data['filtered_pressure']=p
                self.data['filtered_parameter']=self.data[parameter_name]
            else:
                self.data['filtered_pressure']=self.data['pressure']
                self.data['filtered_parameter']=self.data[parameter_name]
            for k in range(self.len()):
                pd,vd=self.get_downcast(k,'filtered_pressure','filtered_parameter')
                pu,vu=self.get_upcast(k,'filtered_pressure','filtered_parameter')
                R[i,k]=self.cost_function(vd,pd,vu,pu)
        i_min=np.argmin(R,axis=0)
        #self.tau_opt=np.array([tau_range[i] for i in i_min])
        self.tau_opt=tau_range[i_min]
        self.cost=np.min(R,axis=0)


if 0:
    fns_a=glob.glob("/home/lucas/gliderdata/helgoland201407/hd/amadeus-2014*.[de]bd")
    fns_s=glob.glob("/home/lucas/gliderdata/helgoland201407/hd/sebastian-2014*.[de]bd")
else:
    fns_a=glob.glob("/home/lucas/gliderdata/helgoland201308/hd/amadeus-2013*.[de]bd")
    fns_s=glob.glob("/home/lucas/gliderdata/helgoland201308/hd/sebastian-2013*.[de]bd")
fns_a.sort()
fns_s.sort()

fns=dict(a=fns_a[24:530],
         s=fns_s[49:608])

dbd={}
g=['a','s']

for _g in g:
    dbd[_g]=dbdreader.MultiDBD(filenames=fns[_g]) 


T=dict((i,dbd[i].get_sync("sci_ctd41cp_timestamp",["sci_water_temp",
                                                   "sci_water_cond",
                                                   "sci_water_pressure"])) for i in g)

data=dict((i,dict(time=T[i][1],
                  pressure=T[i][4],
                  temperature=T[i][2],
                  conductivity=T[i][3])) 
          for i in g)
SensorLag.tau_range=np.arange(-5,5,0.01)
prf=dict((i,SensorLag(data[i])) for i in g)
for i in prf.values():
    i.split_profiles()
    i.estimate_lag('temperature')

d=ndf.NDF()
d.add_parameter("tau_temperature_amadeus","s",(np.arange(prf['a'].len()),prf['a'].tau_opt),"optimal tau per profile")
d.add_parameter("A_temperature_amadeus","-",(np.arange(prf['a'].len()),prf['a'].cost),"lowest cost per profile")

d.add_parameter("tau_temperature_sebastian","s",(np.arange(prf['s'].len()),prf['s'].tau_opt),"optimal tau per profile")
d.add_parameter("A_temperature_sebastian","-",(np.arange(prf['s'].len()),prf['s'].cost),"lowest cost per profile")


for i in prf.values():
    i.estimate_lag('conductivity')

d.add_parameter("tau_conductivity_amadeus","s",(np.arange(prf['a'].len()),prf['a'].tau_opt),"optimal tau per profile")
d.add_parameter("A_conductivity_amadeus","-",(np.arange(prf['a'].len()),prf['a'].cost),"lowest cost per profile")

d.add_parameter("tau_conductivity_sebastian","s",(np.arange(prf['s'].len()),prf['s'].tau_opt),"optimal tau per profile")
d.add_parameter("A_conductivity_sebastian","-",(np.arange(prf['s'].len()),prf['s'].cost),"lowest cost per profile")

d.set_description("Optimal delay of pressure with respect to temperature and conductivity for helgoland201407 experiment")
d.add_metadata("originator","sensor_lag.py")
#d.save("short_time_mismatch_results.ndf")
d.save("short_time_mismatch_results_2013.ndf")


