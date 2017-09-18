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
        C: (list of) 1D array(s) with the parameter to be gridded

        '''
        self.t=t
        self.z=z
        self.C=C
        self.ti = self.zi = self.Ci = None

    def griddata(self, dt=300, dz=0.5, max_span=30*60):
        '''griddata
        ========

        Grids data with dt and dz grid sizes. Gaps longer than
        max_span are not interpolated.

        Parameters
        ----------
            dt: grid size in time dimension
            dz: grid size in z dimension
            max_span: time overwich data are allowed to be interpolated.

        Returns
        -------
            ti: equidistant time vector
            zi: equidistant z vector
            Ci: 2D array or list of arrays of gridded data.

        '''
        # make sure we don't have silly z values above the surface.
        idx=np.where(self.z<0)[0]
        self.z[idx]=0.
        #
        ti=np.arange(self.t.min(),self.t.max()+dt,dt)
        zi=np.arange(0,self.z.max()+dz,dz)
        nt=ti.shape[0]
        nz=zi.shape[0]
        fun_t=interp1d(ti,np.arange(nt))
        fun_z=interp1d(zi,np.arange(nz))
        idx=fun_t(self.t)
        jdx=fun_z(self.z)
        data={}
        for v,k in enumerate(list(zip(idx,jdx))):
            if k in data:
                data[k].append(v)
            else:
                data[k]=[v]
        
        ndim=len(self.C.shape)
        if ndim==1:
            C=self.C.reshape(1,-1)
        else:
            C=self.C.copy()
        nparameters=C.shape[0]
        if nparameters>10:
            raise ValueError('Number of parameters to interpolate exceeds 10 (%d).'%(nparameters))
        if nt*nz*nparameters>10e6:
            raise ValueError('Too many data points: %d (limit 10e6)'%(nt*nz*nparameters))
        vi=[-99*np.ones((nt,nz),float) for i in range(nparameters)]
        for k,v in data.items():
            i,j=k
            for m in range(nparameters):
                vm=np.mean(C[m][v])
                vi[m][int(i),int(j)]=vm
        vi=self.__interpolate_grid(ti,zi,vi,dz,dt,max_span)
        self.ti=ti
        self.zi=zi
        self.Ci=vi
        if nparameters==1:
            return ti, zi, vi[0]
        else:
            return ti,zi,vi

    def __get_blocks(self, v, max_size=10):
        vi=(v==-99).astype(int)
        blocks=[]
        new_block=True
        for i in range(len(v)):
            if vi[i]==1 and not new_block:
                current_block.append(i)
            elif vi[i]==1 and new_block:
                current_block=[i]
                new_block=False
            elif not new_block:
                if len(current_block)<=max_size and \
                        current_block[0]!=0 and current_block[-1]!=len(v)-1:
                    blocks.append(current_block)
                new_block=True
        return blocks


    def __interpolate_grid(self,ti,zi,vi,dz=0.5,dt=300,max_span=20*60):
        nt,nz=vi[0].shape
        max_size=int(np.ceil((max_span/dt)))
        number_of_parameters=len(vi)
        for i in range(nz):
            blocks=self.__get_blocks(vi[0][:,i],max_size=max_size)
            for block in blocks:
                idx=[block[0]-1,block[-1]+1]
                for m in range(number_of_parameters):
                    vi[m][block,i]=np.interp(block,idx,vi[m][idx,i])
        vi_masked=[np.ma.masked_less(_vi,-98).T for _vi in vi]
        return vi_masked




