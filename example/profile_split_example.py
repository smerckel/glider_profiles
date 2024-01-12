''' A simple demonstration on how to split glider data in profiles

   * First the glider data are read. This can be per segment, or per
     mission, or as many dbd files you want.

   * construct a data dictionary

   * create an instance of the ProfileSplitter class, which takes the
     data dictionary as argument

   * now you can get profiles of time or pressure per variable.

'''
import os

import dbdreader
from profiles import profiles
import matplotlib.pyplot as plt

# import logging
# logging.basicConfig(level=logging.WARNING)
# logger=logging.getLogger("profiles")
# logger.setLevel(logging.DEBUG)

# lets read a dbd file (can also be multiple).


regex = os.path.join(dbdreader.EXAMPLE_DATA_PATH,"amadeus*.[de]bd")
dbd=dbdreader.MultiDBD(pattern=regex)


# get some CTD data. We need something for time, and pressure. In this
# case, we'll use the time stamp of the ctd, and the pressure from the
# ctd for these.


tctd,C,T,P, bs = dbd.get_CTD_sync("sci_flntu_turb_units")
# create a data dictionary. Needs two compulsary keys: time and
# pressure. If you change these, you need to change the labels in the
# ProfileSplitter class too. Best to stick with "time" and "pressure"...

data=dict(time=tctd,
          pressure=P,
          # and now the variables. You will reference them by the key
          # name you give them in this dictionary.
          T=T,
          C=C*10, # mS/cm
          P=P*10, # bar
          bs=bs)

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


ps=profiles.ProfileSplitter(data=data, profile_factory=profiles.AdvancedProfile) # default values should be OK


# Tell how many profiles we have found.
print("We have %d profiles"%(ps.nop))

# Now let's get some data. We can choose from
#
# get_casts()    : returns up AND down casts
# get_upcasts()  : returns the up casts only
# get_downcast() : returns the down casts only.

casts = ps.get_casts() # up and down casts


# casts is of the type ProfileList. We can ask what parameters are available:

print("Parameters in casts: ")
print(casts.parameters)

# And each of these are accessible as attributes:

print("Mean temperature of cast #2:", casts.T[2].mean())


# Now plot some data.

f, (ax, bx) = plt.subplots(2,1)

scale = 5
for i, (_T, _z) in enumerate(zip(casts.T, casts.P)):
    ax.plot(_T+i*scale, _z)


for i, (_t, _z) in enumerate(zip(casts.time, casts.P)):
    bx.plot(_t, _z)

bx.plot(tctd, P*10, '--')    

for _ax in (ax, bx):
    _ax.yaxis.set_inverted(True)
    _ax.set_ylabel('Depth (m)')
ax.set_xlabel('Temperature, offset by profile (scale 5 ℃/profile) (℃)')
bx.set_xlabel('Time since Epoch (s)')


f, (ax, bx) = plt.subplots(2,1)

scale = 1
for i, cast in enumerate(casts):
    ax.plot(cast.despike('bs', window_size=5)+i*scale, cast.P)


for i, (_t, _z) in enumerate(zip(casts.time, casts.P)):
    bx.plot(_t, _z)

bx.plot(tctd, P*10, '--')    

for _ax in (ax, bx):
    _ax.yaxis.set_inverted(True)
    _ax.set_ylabel('Depth (m)')
ax.set_xlabel(f'Backscatter, offset by profile (scale {scale} NTU/profile) (NTU)')
bx.set_xlabel('Time since Epoch (s)')

plt.show()
