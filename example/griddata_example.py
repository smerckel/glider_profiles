''' A simple demonstration on how to split glider data in profiles

   * First the glider data are read. This can be per segment, or per
     mission, or as many dbd files you want.

   * construct a data dictionary

   * create an instance of the ProfileSplitter class, which takes the
     data dictionary as argument
   * split into profiles

   * now you can get profiles of time or pressure per variable.

'''
import dbdreader
import profiles.iterprofiles
import profiles.griddata
import pylab as pl

# lets read a dbd file (can also be multiple).

dbd=dbdreader.MultiDBD(pattern='/home/lucas/gliderdata/helgoland201407/hd/sebastian-2014-215-00-073.[ed]bd')


# get some CTD data. We need something for time, and pressure. In this
# case, we'll use the time stamp of the ctd, and the pressure from the
# ctd for these.

tmp=dbd.get_sync("sci_ctd41cp_timestamp",["sci_water_temp",
                                          "sci_water_cond",
                                          "sci_water_pressure"])

t_dummy,tctd,T,C,P=tmp
# create a data dictionary. Needs two compulsary keys: time and
# pressure. If you change these, you need to change the labels in the
# ProfileSplitter class too. Best to stick with "time" and "pressure"...

data=dict(time=tctd,
          pressure=P,
          # and now the variables. You will reference them by the key
          # name you give them in this dictionary.
          T=T,
          C=C*10, # mS/cm
          P=P*10) # bar

#The constructor of ProfileSplitter takes 4 optional arguments: 
# data: the data dictionary 
# window_size: 9 sets the window size for smoothing the pressure data
#using a moving average

#threshold_bar_per_second=1e-3 # used in the criteria for limit
#vertical speed. If you use pressure in dbar you'd better adjust this
#parameter correspondingly

# remove_incomplete_tuples =True Each profile has a down and
# upcast. If they are in complete because of an abort or so, they are
# removed from the data pool if this parameter is set to True


ps=profiles.profiles.ProfileSplitter(data=data) # default values should be OK

ps.split_profiles()

print("We have %d profiles"%(len(ps)))

# get time series of 3rd profile of temperature

ts=ps.get_cast(2,'T') # 'T' is the name of the key in data
# or as up cast only

ts_up=ps.get_upcast(2,'T') # 'T' is the name of the key in data
ts_dw=ps.get_downcast(2,'T') # 'T' is the name of the key in data

pl.figure(1)
pl.plot(ts[0],ts[1],label='up and down cast')
pl.plot(ts_up[0],ts_up[1],'^',label='up cast only')
pl.plot(ts_dw[0],ts_dw[1],'v',label='down cast only')
pl.legend()

prs=ps.get_cast(2,'P') # 'P' is the name of the key in data
prs_up=ps.get_upcast(2,'P') 
prs_dw=ps.get_downcast(2,'P') 

pl.figure(2)
pl.plot(prs[0],prs[1],label='up and down cast')
pl.plot(prs_dw[0],prs_dw[1],'v',label='down cast only')
pl.plot(prs_up[0],prs_up[1],'^',label='up cast only')
#
prs=ps.get_cast(3,'P') # 'P' is the name of the key in data
prs_dw=ps.get_downcast(3,'P') 
pl.plot(prs[0],prs[1],label='up and down cast')
pl.plot(prs_dw[0],prs_dw[1],'v',label='down cast only')
pl.legend()



# if you want a profile agains pressure:
# either

ts=ps.get_cast(2,'T') # 'T' is the name of the key in data
prs=ps.get_cast(2,'P') # 'P' is the name of the key in data
# or 'pressure' in stead of 'P'
pl.figure(3)
pl.plot(ts[1],prs[1],label='method one')

#or

tp=ps.get_cast(2,'T','P')

pl.plot(tp[0],-tp[1],label='method two')
pl.legend()

gridder = profiles.griddata.DataGridder(tctd, P, T)

ti, zi, Ci = gridder.griddata(dt=60, dz=0.05)

f, ax = pl.subplots(1,1)
ax.pcolormesh(ti, zi, Ci)
ax.plot(tctd, P, 'k')
ax.set_ylim(4, 0)

pl.show()



