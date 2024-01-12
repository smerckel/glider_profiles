import numpy as np
import pylab as pl
import dbdreader
from profiles import profiles
dbd={}
dbd['a']=dbdreader.MultiDBD(pattern="/home/lucas/gliderdata/helgoland201407/hd/amadeus-2014-215-00-016.[de]bd") # 5 Aug 15:00
dbd['s']=dbdreader.MultiDBD(pattern="/home/lucas/gliderdata/helgoland201407/hd/sebastian-2014-215-00-017.[de]bd") # 5 Aug 15:00
g=['a','s']

T=dict((i,dbd[i].get_xy("sci_ctd41cp_timestamp","sci_water_temp")) for i in g)
P=dict((i,dbd[i].get_xy("sci_ctd41cp_timestamp","sci_water_pressure")) for i in g)
