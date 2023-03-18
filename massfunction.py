import numpy as np
from scipy.integrate import simpson
from cosmo import Cosmology


class Massfunction(Cosmology):

    def __init__(
        self, 
        m, 
        Pkfile,
        knum=500, 
        z=0, 
        omegam=0.3, 
        omegacc=0.7, 
        rhocrit=27.75e10,
        ff_param=(.322, .707, .3), 
        bias_param=(.707, .3)
    ):
        super().__init__(Pkfile, z, omegam, omegacc, rhocrit,)
        self.m = m
        self.knum = knum
        self.norm_A = ff_param[0]
        self.a_para = ff_param[1]
        self.p_para = ff_param[2]
        self.bias_a = bias_param[0]
        self.bias_p = bias_param[1]

    def nufnu(self, nu):
        '''Return the fitting function of mass function. Here we used the ST mass function( Sheth & Tormen (1999)).'''
        norm_A = self.norm_A
        p_para = self.p_para
        nuprime = np.sqrt(self.a_para) * nu

        return 2. * norm_A * (1. + 1. / np.power(nuprime, 2. * p_para)) \
            * np.sqrt(nuprime ** 2. / (2. * np.pi)) * np.exp(- (nuprime ** 2 / 2.))

    @property
    def k(self):
        '''
        Return the wavenumber k. The range of k is derived by the radii convert from mass. 
        This is to ensure the convergence of :math:`\ frac{\mathrm{d\quadln}\ nu}{\mathrm{d\quadln}\m}`
        '''
        kmax = 100. / np.min(self.reff)
        kmin = .005 / np.max(self.reff)
        return np.logspace(np.log10(kmin), np.log10(kmax), self.knum)

    @property
    def sigma0(self):
        '''Return the mass variance at radii'''
        return self._sigma_n(self.k, self.reff)

    @property
    def nu(self):
        return self.delta_c / self.sigma0 * self.Dgrowth0 / self.Dgrowth

    @property
    def nu2(self):
        return self.nu ** 2

    @property
    def dlnss_dlnm(self):
        x = np.outer(self.reff, self.k)
        wth = self._tophat_kspace(x)
        dwth = self._tophat_dw_dx(x)
        pk = self.Pk_func_dimless(self.k)
        integ = pk * wth * dwth
        return self.reff / self.sigma0 ** 2 * simpson(integ, self.k, axis=-1)

    @property
    def dlnnu_dlnm(self):
        return -self.dlnss_dlnm

    @property
    def _dlnnudlnm(self):
        '''The unused method to getting :math:`\ frac{\mathrm{d\quadln}\ nu}{\mathrm{d\quadln}\m}`.\\
        This way uses finite difference instead.
        '''
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
        '''Return the mass function in form :math:`\ frac{\mathrm{d}n}{\mathrm{d\quadln}\m}`'''
        return self.rho_0 / self.m * self.nufnu(self.nu) * self.dlnnu_dlnm

    @property
    def dndm(self):
        '''Return the mass function in form :math:`\ frac{\mathrm{d}n}{\mathrm{d}\m}`'''
        return self.dndlnm/self.m

    @property
    def reff(self):
        '''Convert the given mass to corresponding radius'''
        return (self.m/(4. * np.pi / 3. * self.rho_0))**(1./3.)

    @property
    def bias(self):
        '''Return the halo bias. Here we used ST99(Sheth & Tormen (1999))'''
        a = self.bias_a
        p = self.bias_p
        return (
            1
            + (a * self.nu2 - 1) / self.delta_c
            + (2 * p / self.delta_c) / (1 + (a * self.nu2) ** p)
        )
