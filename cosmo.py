from inspect import Attribute
import numpy as np
from scipy.interpolate import interp1d as p1d


class Cosmology:
    '''The cosmology model use in the simulation. 
    
    '''

    def __init__(self, Pkfile, z=0, omegam=0.3, omegacc=0.7, rhocrit=27.75*10e10):
        self.omegam = omegam
        self.omegacc = omegacc
        self.rhocrit = rhocrit
        self.z = z
        self.read_Pk(Pkfile)
        self.a = 1/(1+z)
        self.h0 = self._cap_hubble(1)
        self.hz = self._cap_hubble(self.a)

    def read_Pk(self, fname):
        klist, pklist = np.loadtxt(fname, usecols=(0, 1), unpack=True)
        self.Pk_func = p1d(klist, pklist, kind="linear")

    def _cap_hubble(self, a):
        '''Return the Hubble constant at z. '''
        hubble = np.sqrt(self.omegam/a**3 + self.omegacc)
        return hubble

    def _get_Omegamz(self):
        self.Omegam_z=self.omegam*self.a**3*(self.h0/self.hz)**2

    @property
    def Omegam_z(self):
        self._get_Omegamz()
        return self.Omegam_z
