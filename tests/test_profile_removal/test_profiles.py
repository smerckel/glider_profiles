import numpy as np
import pickle
import matplotlib.pyplot as plt

from profiles import iterprofiles
# this data set contains two consecutive upcasts. The downcast in between was very short and is removed. 
# data = pickle.load(open('data_two_upcasts.pck', 'rb'))
# nop_expected = 41
# removed_profiles_expected = 1

f, (ax, bx) = plt.subplots(1,2)
if 1:
    data = pickle.load(open('data_missing_upcasts.pck', 'rb'))

    nop_expected = 138
    removed_profiles_expected = 1

    ps  = iterprofiles.ProfileSplitter(data, remove_incomplete_tuples=True)


    t = data["time"]
    P = data["pressure"]


    ps.split_profiles()
    nop = ps.nop

    assert(nop==nop_expected)
    assert(ps.summary['Number of removed incomplete profiles']==1)

    dc = ps.get_downcasts()
    uc = ps.get_upcasts()



    for i, (_t, _d) in enumerate(zip(dc.time, dc.D)):
        ax.plot(_t, _d, color=f'C{i%4}', lw=4)

    for i, (_t, _d) in enumerate(zip(uc.time, uc.D)):
        ax.plot(_t, _d, color=f'C{i%4}', lw=4)

    ax.plot(data["time"], data["D"], 'k--', alpha=0.99)

    
if 1:
    data = pickle.load(open('data_two_upcasts.pck', 'rb'))

    nop_expected = 41
    removed_profiles_expected = 1



    ps  = iterprofiles.ProfileSplitter(data, remove_incomplete_tuples=True)


    t = data["time"]
    P = data["pressure"]


    ps.split_profiles()
    nop = ps.nop

    assert(nop==nop_expected)
    assert(ps.summary['Number of removed incomplete profiles']==1)

    dc = ps.get_downcasts()
    uc = ps.get_upcasts()



    for i, (_t, _d) in enumerate(zip(dc.time, dc.D)):
        bx.plot(_t, _d, color=f'C{i%4}', lw=4)

    for i, (_t, _d) in enumerate(zip(uc.time, uc.D)):
        bx.plot(_t, _d, color=f'C{i%4}', lw=4)

    bx.plot(data["time"], data["D"], 'k--', alpha=0.99)
