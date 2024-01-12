from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import dict
from builtins import zip
from future import standard_library
standard_library.install_aliases()
import numpy as np
import dbdreader
import profiles.thermocline
import gsw


dbd=dbdreader.MultiDBD(pattern="/home/lucas/gliderdata/helgoland201407/hd/sebastian-2014-205-00-05*.*")

tmp=dbd.get_sync("sci_ctd41cp_timestamp",["sci_water_pressure",
                                          "sci_water_temp",
                                          "sci_water_cond"])
t,tcdt,pressure,temp,cond=np.compress(tmp[1]>0,tmp,axis=1)

SP=gsw.SP_from_C(cond*10,temp,pressure*10)
SA=gsw.SA_from_SP(SP,pressure*10,7.5,54)
CT=gsw.CT_from_t(SA,temp,pressure*10)
rho=gsw.rho(SA,CT,pressure*10)

data=dict(time=tcdt,pressure=pressure,temp=temp,rho=rho)
tc=profiles.thermocline.Thermocline(data=data)
tc.split_profiles()

R=dict([(k,v) for k,v in zip(['t','P','rho','N2'],tc.calc_buoyancy_frequency())])
x=np.hstack(R['t'])
y=np.hstack(R['P'])
z=np.hstack(R['N2'])

ttc,ztc,N2tc=tc.calc_thermocline_maxN2()
thermocline_limits=tc.calc_thermocline_limits()

