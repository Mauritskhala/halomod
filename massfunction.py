import numpy as np
from cosmo import Cosmology


class Massfunction:

    def __init__(self, m, Pkfile, z=0, omegam=0.3, omegacc=0.7, rhocrit=27.75*10e10):
        self.cosmo = Cosmology(Pkfile, z, omegam, omegacc, rhocrit)
        self.z = z
        self.m = m

    def nufnu(self, nu):
        ''''''
        norm_A = 0.322
        a_para = 0.707
        p_para = 0.3
        nuprime = np.sqrt(a_para) * nu
        return 2. * norm_A*(1 + 1./np.power(nuprime, 2.*p_para)) \
            * np.sqrt(nuprime**2/2.*np.pi) * np.exp(-nuprime**2/2.)
    
    @property
    def sigma(self):
        pass

    @property
    def dndlnm(self):
        return self.nufnu(self.nu)

    @property
    def rho_0(self):
        return self.cosmo._rhocrit * self.cosmo._omegam
    
    @property
    def reff(self):
        return np.power(self.m/(4.* np.pi/3.* self.rho_0),1/3)