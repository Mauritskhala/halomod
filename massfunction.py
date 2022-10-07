import numpy as np
from cosmo import Cosmology


class Massfunction:

    def __init__(self, m, Pkfile, knum=500, z=0, omegam=0.3, omegacc=0.7, rhocrit=27.75*10e10):
        self.cosmo = Cosmology(Pkfile, z, omegam, omegacc, rhocrit)
        self.z = z
        self.m = m
        self.knum = knum

    def nufnu(self, nu):
        '''Return the fitting function of mass function'''
        norm_A = 0.322
        a_para = 0.707
        p_para = 0.3
        nuprime = np.sqrt(a_para) * nu
        return 2. * norm_A*(1 + 1./np.power(nuprime, 2.*p_para)) \
            * np.sqrt(nuprime**2/2.*np.pi) * np.exp(-nuprime**2/2.)

    @property
    def k(self):
        kmax = 3. / np.min(self.reff)
        kmin = 0.1 / np.max(self.reff)
        return np.logspace(np.log10(kmin),np.log10(kmax),self.knum)

    @property
    def sigma0(self):
        return self.cosmo._sigma0(self.k,self.reff)

    @property
    def nu(self):
        return self.cosmo.delta_c/self.sigma0*self.cosmo.Dgrowth0/self.cosmo.Dgrowth

    @property
    def dlnnu_dlnM(self):
        pass

    @property
    def dndlnm(self):
        return self.rho_0/self.m*self.nufnu(self.nu)*self.dlnnu_dlnM

    @property
    def rho_0(self):
        return self.cosmo._rhocrit * self.cosmo._omegam

    @property
    def reff(self):
        return np.power(self.m/(4. * np.pi/3. * self.rho_0), 1/3)
