import numpy as np
from scipy.integrate import simpson
from cosmo import Cosmology


class Massfunction:

    def __init__(self, m, Pkfile, knum=500, z=0, omegam=0.3, omegacc=0.7, rhocrit=27.75*1e10):
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
        return 2. * norm_A * (1. + 1. / np.power(nuprime, 2.*p_para)) \
            * np.sqrt(nuprime**2. / (2.*np.pi)) * np.exp(- (nuprime ** 2 /2.))

    @property
    def k(self):
        kmax = 10. / np.min(self.reff)
        kmin = .01 / np.max(self.reff)
        return np.logspace(np.log10(kmin), np.log10(kmax), self.knum)

    @property
    def sigma0(self):
        return self.cosmo.sigma_n(self.k, self.reff)

    @property
    def nu(self):
        return self.cosmo.delta_c / self.sigma0 *\
            self.cosmo.Dgrowth0 / self.cosmo.Dgrowth

    @property
    def dlnss_dlnm(self):
        x = np.outer(self.reff, self.k)
        wth = self.cosmo.tophat_kspace(x)
        dwth = self.cosmo.tophat_dw_dx(x)
        pk = self.cosmo.Pk_func_dimless(self.k)
        integ = pk * wth * dwth
        return self.reff / self.sigma0 ** 2 * simpson(integ, self.k, axis=-1)

    @property
    def dlnnu_dlnm(self):
        return -self.dlnss_dlnm
        
    @property 
    def _dlnnudlnm(self):
        nu = self.nu
        mass = self.m
        dlnnu_dlnM = []

        for i in range(len(nu)):
            if i == 0:
                n_left = 0
                n_right = 1
            elif i == (len(nu)-1):
                n_left = len(nu) - 2
                n_right = len(nu) - 1
            else:
                n_left = i - 1
                n_right = i + 1

            derivative = (np.log(nu[n_right])-np.log(nu[n_left])) /    \
                (np.log(mass[n_right]) - np.log(mass[n_left]))

            dlnnu_dlnM.append(derivative)

        return dlnnu_dlnM

    @property
    def dndlnm(self):
        return self.rho_0 / self.m * self.nufnu(self.nu) * self.dlnnu_dlnm

    @property
    def dndm(self):
        return self.dndlnm/self.m

    @property
    def rho_0(self):
        return self.cosmo.rho_0

    @property
    def reff(self):
        return np.power(self.m/(4. * np.pi / 3. * self.rho_0), 1./3.)
