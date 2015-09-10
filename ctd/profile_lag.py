import numpy as np
import pylab as pl
from profiles import profiles
import dbdreader
import glob






class Data(object):
    Data={}
    def __init__(self):
        pass

    def read_data(self,fns):
        dbd=dbdreader.MultiDBD(filenames=fns)
        x=dbd.get_sync("sci_ctd41cp_timestamp",["sci_water_temp","sci_water_cond","sci_water_pressure"])
        t,ctdt,T,C,P=x.compress(x[1]>1e8,axis=1)
        data=dict(time=ctdt,pressure=P,C=C*10,T=T,P=P*10)
        return data

    def add_data(self,glider,fns):
        Data.Data[glider]=self.read_data(fns)
                
if 1:
    fns_a=glob.glob("/home/lucas/gliderdata/helgoland201407/hd/amadeus-2014*.[de]bd")
    fns_s=glob.glob("/home/lucas/gliderdata/helgoland201407/hd/sebastian-2014*.[de]bd")
    fns_a.sort()
    fns_s.sort()
    fns=dict(a=fns_a[24:530],
             s=fns_s[49:608])

else:
    fns_a=glob.glob("/home/lucas/gliderdata/helgoland201308/hd/amadeus-2013*.[de]bd")
    fns_s=glob.glob("/home/lucas/gliderdata/helgoland201308/hd/sebastian-2013*.[de]bd")
    fns_a.sort()
    fns_s.sort()
    fns=dict(a=fns_a[194:466],
             s=fns_s[83:320])

D=Data()
D.add_data('a',fns['a'])
D.add_data('s',fns['s'])

g=['a','s']

prf=dict([(k,profiles.CrossSpectral(data=Data.Data[k])) for k in g])
prf['a'].split_profiles()
prf['s'].split_profiles()
