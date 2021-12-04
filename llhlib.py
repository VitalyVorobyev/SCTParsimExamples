import numpy as np
from scipy.integrate import quad


def normalize(fcn, xlo, xhi):
    """ Make PDF normalized """
    def normalized_pdf(x, *pars):
        norm = quad(lambda z: fcn(z, *pars), xlo, xhi)[0]
        return fcn(x, *pars) / norm
    return normalized_pdf


def loglh_maker(fcn, data):
    """ fcn - normalized PDF """
    return lambda pars: -2. * np.sum(np.log(fcn(data, *pars)))
