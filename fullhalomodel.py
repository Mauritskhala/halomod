from haloprofile import NFW
import numpy as np
from massfunction import Massfunction
from poi_HOD import Zehavi05, Zheng05
from scipy.integrate import simpson
from scipy.fft import ifht, fhtoffset
from scipy.interpolate import interp1d


class FullHOD(Massfunction):
    def __init__(
        self,
        mlist=np.logspace(8, 16, 800),
        kmodelmin=-4,
        kmodelmax=4,
        kmodelnum=1000,
        z=0,
        hodparam=(12, 13, 1),
        hodtype=True,
        pkfile="Pk_halomod_Planck15",
        **kwargs
    ):
        '''
        The variable 'kmodel' is used to calculate the power spectrum of tracer. 
        This is not the same 'k' as that used in calculation of mass function.  
        '''
        super().__init__(mlist, pkfile, z=z, **kwargs)
        self.kmodel = np.logspace(kmodelmin, kmodelmax, kmodelnum)
        self.cosmo = pkfile
        self.hodtype = hodtype
        self.hodparam = hodparam

    def pk_to_xi(self, power):
        dln = np.log(self.kmodel[1]/self.kmodel[0])
        intg = self.kmodel ** 1.5 * power
        xi = 1. / (2 * np.pi * self.r) ** 1.5 * ifht(intg, dln, mu=.5)
        return xi

    @property
    def r(self):
        return 1. / self.kmodel[::-1]

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
        intg = self.dndm * self.hod.cspair * self.profile.rho.transpose() / self.m / \
            self.ngbar ** 2
        return simpson(intg, self.m)

    @property
    def corr_1halo_ss(self):
        return self.pk_to_xi(self.power_ss)

    @property
    def corr_1halo(self):
        return self.corr_1halo_cs + self.corr_1halo_ss

    @property
    def corr_2halo(self):
        return self.pk_to_xi(self.power_2h)

    @property
    def power_ss(self):
        return simpson(self.dndm * self.hod.sspair * self.profile._u(self.kmodel) ** 2 / self.ngbar ** 2, self.m)

    @property
    def power_2h(self):
        b1eff = simpson(self.dndm * self.hod.total_occupation * self.bias*
                        np.abs(self.profile._u(self.kmodel)) / self.ngbar, self.m) * self.Dgrowth / self.Dgrowth0
        return b1eff ** 2 * self.Pk_func(self.kmodel)


class Angular(FullHOD):
    def __init__(
        self,
        pzfname,
        deg=np.logspace(-3, np.log10(2), 30),
        urange=np.logspace(-3, 3, 600),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pzfname = pzfname
        self.deg = deg
        self.urange = urange

    def corr_3d(self,r):
        f=interp1d(self.r, self.corr_full)
        return f(r)
    
    def ang_projection(self,xi):
        x = self._comoving_func(self.zrange)
        pintg = (self.patz * self._cap_hubble(1. /
                 (1. + self.zrange)) / 2997.92458) ** 2
        R = np.sqrt(np.add.outer(np.outer(self.theta ** 2, x ** 2), self.urange ** 2)).flatten()
        integrand = np.einsum(
        "kij,i->kij", xi(R,).reshape((len(self.theta), len(x), len(self.urange))), pintg
    )
        return 2 * simpson(simpson(integrand, self.urange), x)
    
    def powerlaw(self,r):
        return r ** -1.8

    @property
    def ang_corr(self):
        return self.ang_projection(self.corr_3d)

    @property
    def zrange(self):
        return np.loadtxt(self.pzfname, usecols=(0,))

    @property
    def patz(self):
        return np.loadtxt(self.pzfname, usecols=(1,))

    @property
    def theta(self):
        return self.deg * np.pi / 180
