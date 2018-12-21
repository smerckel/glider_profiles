from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d

class DataGridder(object):
    '''A class to grid 3D time series data (typically glider data) onto a
grid.  The method used is that the user specifies the grid sizes
(delta time, and delta z), and provides a list of maximum 10
parameters that need to be interpolated. The algorithm dives the data
in blocks, depending on the grid size. Each block is assigned an
average value. Data between blocks are interpolated linearly. 

'''

    def __init__(self,t,z,C):
        ''' 
        Parameters:
        -----------
        t: 1D array (time)
        z: 1D array (depth)
        C: 1D array to be gridded
        '''
        ndim=len(C.shape)
        if ndim!=1:
            raise ValueError('Argument C is not an 1D-array: check it!')
        self.t=t
        self.z=z
        self.C=C
        self.ti = self.zi = self.Ci = None

    def griddata(self, dt=300, dz=0.5, max_span=30*60, zi=None):
        '''griddata
        ========

        Grids data with dt and dz grid sizes. Gaps longer than
        max_span are not interpolated.

        Parameters
        ----------
            dt: grid size in time dimension
            dz: grid size in z dimension
            max_span: time overwich data are allowed to be interpolated.
            zi: interpolated z dimension (optional)

        Returns
        -------
            ti: equidistant time vector
            zi: equidistant z vector
            Ci: 2D array or list of arrays of gridded data.
        '''
        print("")
        print("            It might take a while...")
        # make sure we don't have silly z values above the surface.
        #idx=np.where(self.z<0)[0]
        #self.z[idx]=0.
        #
        ti=np.arange(self.t.min(),self.t.max()+dt,dt)
        if zi is None:
            zi = np.arange(0,self.z.max()+dz,dz)
        nt=ti.shape[0]
        nz=zi.shape[0]
        if nt*nz > 10e6:
            raise ValueError('Too many data points: %d (limit 10e6)'%(nt*nz))
        vi=np.ma.masked_all((nt,nz),float)
        
        fun_t=interp1d(ti,np.arange(nt), bounds_error=False)
        fun_z=interp1d(zi,np.arange(nz), bounds_error=False)
        idx=(fun_t(self.t) + 0.5).astype(int)
        jdx=(fun_z(self.z) + 0.5).astype(int)

        data = defaultdict(list)
        for v,k in enumerate(zip(idx,jdx)):
            data[k].append(v)

        for (i,j),v in data.items():
            vm=np.mean(self.C[v])
            #vm=np.median(self.C[v])
            try:
                vi[i,j]=vm
            except ValueError: # happens when i or j are nan (out of range)
                pass
        vi = self.__interpolate_grid(ti,zi,vi,dz,dt,max_span)
        self.ti=ti
        self.zi=zi
        self.Ci=vi
        return ti,zi,vi

    def __get_blocks(self, v, max_size=10):
        # traverses the matrix along the second dimension and cuts into
        # blocks of cells overwhich data should be interpolated in the first dimension
        # max_span is the criterion used to cut into blocks
        blocks=[]
        new_block=True
        for i in range(len(v)):
            if np.ma.is_masked(v[i]) and not new_block:
                current_block.append(i)
            elif np.ma.is_masked(v[i]) and new_block:
                current_block=[i]
                new_block=False
            elif not new_block:
                if len(current_block)<=max_size and \
                        current_block[0]!=0 and current_block[-1]!=len(v)-1:
                    blocks.append(current_block)
                new_block=True
        return blocks


    def __interpolate_grid(self,ti,zi,vi,dz=0.5,dt=300,max_span=20*60):
        # interpolates in the first dimension (ti) , but only for blocks
        # of data that are considered "continuous", controlled by the max_span setting.
        nt,nz=vi.shape
        max_size=int(np.ceil((max_span/dt)))
        for i in range(nz):
            blocks=self.__get_blocks(vi[:,i],max_size=max_size)
            for block in blocks:
                idx=[block[0]-1,block[-1]+1]
                vi[block,i]=np.interp(block,idx,vi[idx,i])
        return vi
