import os

import dbdreader
from profiles import profiles


path = dbdreader.EXAMPLE_DATA_PATH # new in version 0.5.6

regex = os.path.join(path, "amadeus-2014-204-05-000.[de]bd")
dbd=dbdreader.MultiDBD(regex)

tctd,C,T,P = dbd.get_CTD_sync()


data=dict(time=tctd,
pressure=P,
C=C*10, # mS/cm
T=T,
P=P*10) # dbar

 

ps = profiles.ProfileSplitter(data=data,
                              profile_factory=profiles.AdvancedProfile)

print("We have %d profiles"%(ps.nop))

casts = ps.get_downcasts()

for cast in casts:
    _T = cast.T
    _z = -cast.P
    pass

T_all = casts.T
Z_all = casts.P
for _T, _Z in zip(T_all, Z_all):
    _Z*=-1


T_2 = casts.T[2]





T_2_despiked = casts[2].despike("T")

