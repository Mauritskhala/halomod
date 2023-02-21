import numpy as np
from scipy.special import erf


class Zehavi05:
    def __init__(
        self,
        mlist,
        param=(12, 13, 1),
    ):
        self.m = mlist
        self.mmin = 10 ** param[0]
        self.m1 = 10 ** param[1]
        self.alpha = param[2]
        mindex = np.where(self.m < self.mmin)
        self.idx = mindex

    @property
    def cen_bar(self):
        nc = np.ones_like(self.m)
        nc[self.idx] = 0
        return nc

    @property
    def satt_bar(self):
        ns = np.zeros_like(self.m)
        ns = (self.m / self.m1) ** self.alpha
        ns[self.idx] = 0
        return ns

    @property
    def total_occupation(self):
        return self.satt_bar + self.cen_bar

    @property
    def cspair(self):
        return self.satt_bar * self.cen_bar

    @property
    def sspair(self):
        return self.satt_bar ** 2


class Zheng05:
    def __init__(
        self,
        mlist,
        param=(12, 1, 13, 0, 1),
        m0=True
    ):
        self.m = mlist
        self.mmin = 10 ** param[0]
        self.sigma_logM = param[1]
        self.m1 = 10 ** param[2]
        self.m0 = 10 ** param[3] if m0 else 0
        self.gamma = param[4]

    @property
    def cen_bar(self):
        return .5 * (1 + erf((self.m-self.mmin) / self.sigma_logM))

    @property
    def satt_bar(self):
        return ((self.m - self.m0) / self.m1) ** self.gamma

    @property
    def total_occupation(self):
        return self.satt_bar + self.cen_bar

    @property
    def cspair(self):
        return self.satt_bar * self.cen_bar

    @property
    def sspair(self):
        return self.satt_bar ** 2
