from haloprofile import NFW
from massfunction import Massfunction
from poi_HOD import Zehavi05, Zheng05
from scipy.integrate import simpson, quad
import numpy as np
from scipy.fft import irfft, ifft


class FullHOD(Massfunction):
    def __init__(
        self,
        mlist,
        kmodel,
        r,
        z,
        hodparam,
        hodtype=True,
        pkfile="Pk_halomod_Planck15",
        **kwargs
    ):
        '''
        The variable 'kmodel' is used to calculate the power spectrum of tracer. 
        This is not the same 'k' as that used in calculation of mass function.  
        '''
        super().__init__(mlist, pkfile, z=z, **kwargs)
        self.kmodel = kmodel
        self.cosmo = pkfile
        self.r = r
        self.hodtype = hodtype
        self.hodparam = hodparam

    def pk_to_xi(self, r, power):
        intg = self.kmodel * power * np.sin(np.outer(r, self.kmodel))
        norm = 1 / (2 * np.pi ** 2 * r)
        return norm * simpson(intg, self.kmodel)

    @property
    def hod(self):
        if self.hodtype:
            return Zehavi05(self.m, self.hodparam)
        else:
            return Zheng05(self.m, self.hodparam)

    @property
    def profile(self):
        return NFW(self.m, self.r, z=self.z, cosmo=self.cosmo)

    @property
    def ngbar(self):
        return simpson(self.dndm * self.hod.total_occupation, self.m)
    
    @property
    def corr_full(self):
        return self.corr_1halo + self.corr_2halo

    @property
    def corr_1halo_cs(self):
        intg = self.dndm * self.hod.cspair * self.profile.rho.transpose() / self.m / self.ngbar**2
        return simpson(intg, self.m)

    @property
    def corr_1halo_ss(self):
        return self.pk_to_xi(self.r, self.power_ss)

    @property
    def corr_1halo(self):
        return self.corr_1halo_cs + self.corr_1halo_ss

    @property
    def corr_2halo(self):
        return self.pk_to_xi(self.r, self.power_2h)

    @property
    def power_ss(self):
        return simpson(self.dndm * self.hod.sspair * self.profile._u(self.kmodel) ** 2 / self.ngbar ** 2, self.m)

    @property
    def power_2h(self):
        b1eff = simpson(self.dndm * self.hod.total_occupation *
                        np.abs(self.profile._u(self.kmodel)) / self.ngbar, self.m)
        return b1eff ** 2 * self.Pk_func(self.kmodel)
