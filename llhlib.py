import numpy as np
from scipy.integrate import quad


def loglh_maker(fcn, data, xlim=None):
    def objective(pars):
        xlo, xhi = np.min(data), np.max(data) if xlim is None else xlim
        norm = quad(lambda x: fcn(x, *pars), xlo, xhi)
        return -2. * np.sum(np.log(fcn(data, *pars))) + 2. * data.size * np.log(norm[0])
    return objective
