from logging import warning
import numpy as np
from scipy.interpolate import interp1d as p1d
from scipy.integrate import simpson


class Cosmology:
    '''The cosmology model use in the simulation. 

    '''

    def __init__(self, Pkfile, z=0, omegam=0.3, omegacc=0.7, rhocrit=27.75e10):
        self._omegam = omegam
        self._omegacc = omegacc
        self._rhocrit = rhocrit
        self.z = z
        self.Pkfile = Pkfile

    def _cap_hubble(self, a):
        '''Calculate the dimensionless Hubble constant at redshift z. '''
        hubble = np.sqrt(self._omegam / a**3 + self._omegacc)
        return hubble

    def sigma_n(self, klist, r, order=0):
        '''Calculate the :math:`\sigma_n(r)` for given k range. The default order is 0, which means that the result is mass variance.'''
        x = np.outer(r, klist)
        w_tophat = self.tophat_kspace(x)
        dfk = klist ** (2. * order - 1) * \
            self.Pk_func_dimless(klist) * w_tophat ** 2.
        integral = simpson(dfk, klist, axis=-1)
        sigma = np.sqrt(integral)
        return sigma

    def tophat_kspace(self, x):
        '''Calculate the tophat filter in Fourier space'''
        return 3. / x ** 3. * (np.sin(x) - x * np.cos(x))

    def tophat_dw_dx(self, x):
        '''Calculate the differential of tophat filter in Fourier space'''
        return (3. * x * np.cos(x) + (x ** 2. - 3.) * np.sin(x)) / x ** 4.

    @property
    def Pk_func(self):
        '''Return the power spectrum function(linear interpolation of the given power spectrum).'''
        klist, pklist = np.loadtxt(self.Pkfile, usecols=(0, 1), unpack=True)
        Pk_func = p1d(klist, pklist, kind="linear")
        return Pk_func

    @property
    def Pk_func_dimless(self):
        '''Return the dimensionless power spectrum function(linear interpolation of the given power spectrum).'''
        klist, pklist = np.loadtxt(self.Pkfile, usecols=(0, 1), unpack=True)
        pklist = pklist/(2 * np.pi ** 2) * klist ** 3
        Pk_func = p1d(klist, pklist, kind="linear")
        return Pk_func

    @property
    def a(self):
        return 1./(1.+self.z)

    @property
    def h0(self):
        '''Return H0'''
        return self._cap_hubble(1.)

    @property
    def rho_0(self):
        return self._rhocrit*self._omegam

    @property
    def hz(self):
        '''Return dimensionless H at redshift z'''
        return self._cap_hubble(self.a)

    @property
    def Om_z(self):
        '''Return \Omega_z at redshift z'''
        return self._omegam / self.a**3 * (self.h0/self.hz)**2

    @property
    def Occ_z(self):
        '''Return \Omega_c at redshift z'''
        return self._omegacc * (self.h0/self.hz)**2

    @property
    def Dgrowth(self):
        '''Return the linear growth factor at redshift z'''
        return 2.5 * self.Om_z * self.a / (self.Om_z**(4./7.)-self.Occ_z
                                         + (1. + self.Om_z/2.) * (1+self.Occ_z/70))

    @property
    def Dgrowth0(self):
        '''Return the linear growth factor at present'''
        return 2.5 * self._omegam  / (self._omegam**(4./7.)-self._omegacc
                                  + (1. + self._omegam/2.) * (1+self._omegacc/70))

    @property
    def delta_c(self):
        '''Return linear critical density \delta_c at redshift z.\\ 
        We use the fitting formula from Weinberg & Kamionkowski (2003);  
        see also Kitayama & Suto(1996) 
        '''
        return 3. / 20. * (12*np.pi)**(2./3.) * (1 + .013*np.log10(self.Om_z))
