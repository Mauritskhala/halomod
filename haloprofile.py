import numpy as np
from cosmo import Cosmology


class NFW:
    def __init__(
        self,
        m,
        r,
        z=0,
        cosmo="Pk_halomod_Planck15",
        concentration_param=(11, 0.13, 4.15e12),
        at_z=True
    ):
        self.m = m
        self.r = r
        self.z = z
        self.c0 = concentration_param[0]
        self.mbeta = -concentration_param[1]
        self.mstar = concentration_param[2]
        self.cosmo = Cosmology(cosmo,z=z)
        self.at_z = at_z

    def masstoradius(self, m):
        dens = self.cosmo.rho_z if self.at_z else self.cosmo.rho_0
        return (3. * m / (4. * np.pi * self.delta_vir * dens)) ** (1.0 / 3.0)

    def radiustomass(self, r):
        dens = self.cosmo.rho_z if self.at_z else self.cosmo.rho_0
        return 4. * np.pi * r ** 3. * self.delta_vir * dens / 3.

    def _h(self, c):
        return np.log(1.0 + c) - c / (1.0 + c)

    def _f(self, x):
        
        result=np.ones_like(x) / (x * (1 + x) ** 2)
        condition=np.where(x >= self.concentration)
        result[condition]=0
        return result
    
    @property
    def rho(self):
        x=np.outer(self.r,self._r_scale_inv)
        rho=np.transpose(self.rho_scale*self._f(x))
        return rho

    #@property
    #def rho_Fourier_trans(self):
        #pass

    @property
    def _r_vir(self):
        return self.masstoradius(self.m)
    
    @property
    def _r_scale_inv(self):
        return np.ones_like(self._r_scale)/self._r_scale

    @property
    def _r_scale(self):
        return self._r_vir / self.concentration

    @property
    def delta_vir(self):
        return 18 * np.pi ** 2 * (1 + .3999 * (1. / self.cosmo.Om_z - 1) ** .941)

    @property
    def rho_scale(self):
        return self.m / (4 * np.pi * self._r_scale ** 3)/self._h(self.concentration)

    @property
    def concentration(self):
        return self.c0 / (1 + self.z) * (self.m / self.mstar) ** self.mbeta
