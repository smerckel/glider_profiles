import sys
sys.path.insert(0, '../')

if not __name__ == '__main__':
    from pytest import fixture
else:
    def fixture(func):
        def inner():
            return func()
        return inner
        

import numpy as np

import dbdreader
import profiles

dbd_path = "/home/lucas/gliderdata/nsb3_201907/hd/comet-2019-203-05-000.?bd"
NOP = 19 # number of profiles of this data file.

@fixture
def load_gliderdata():
    dbd = dbdreader.MultiDBD(dbd_path)
    tctd, C, T, D, flntu_turb = dbd.get_CTD_sync("sci_flntu_turb_units")
    data = dict(time=tctd, pressure=D, C=C*10, T=T, P=D*10, spm=flntu_turb)
    return data


def test_load_gliderdata(load_gliderdata):
    data = load_gliderdata
    ps = profiles.iterprofiles.ProfileSplitter(data)
    ps.split_profiles()
    nop = len(ps)
    assert nop == NOP


def test_get_all_casts(load_gliderdata):
    data = load_gliderdata
    ps = profiles.iterprofiles.ProfileSplitter(data)
    ps.split_profiles()
    C_up = ps.get_upcasts("C")
    Cmean = C_up[0].C.mean()
    assert np.isclose(Cmean, 41.00820866597962)

def test_loop_through_profiles(load_gliderdata):
    data = load_gliderdata
    ps = profiles.iterprofiles.ProfileSplitter(data)
    ps.split_profiles()
    T_means_target = [1563878752.4660194, 1563879239.507901, 1563879731.1192806, 1563880219.1803174,
                      1563880749.6826134, 1563881276.406076, 1563881806.2261407, 1563882319.1111975,
                      1563882863.8103573, 1563883390.7304509, 1563883907.1994038, 1563884403.5452566,
                      1563884904.5165303, 1563885428.5076697, 1563885934.7554157, 1563886436.0727007,
                      1563886952.162765, 1563887449.2640054, 1563887956.6119978]
    T_means = [p.get_downcast("T").t.mean() for p in ps]
    result = np.all([np.isclose(x,y) for x,y in zip(T_means, T_means_target)])
    assert result == True

def test_despike(load_gliderdata):
    data = load_gliderdata
    ps = profiles.iterprofiles.ProfileSplitter(data)
    ps.split_profiles()
    spm_up = ps.get_upcasts("spm", despike=True)
    spm_mean = spm_up[0].spm.mean()
    assert np.isclose(spm_Cmean, 41.00820866597962)


    
if __name__ == '__main__':    
    data = load_gliderdata()
    ps = profiles.iterprofiles.ProfileSplitter(data, remove_incomplete_tuples=False)
    ps.split_profiles()

    C_up = ps.get_upcasts("spm", despike=True)
