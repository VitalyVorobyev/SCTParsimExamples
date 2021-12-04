"""
Math fuctions to fit data
"""

import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    """ Gaussian lineshape """
    xi = (x - x0) / sigma
    return np.exp(-0.5 * xi**2) / np.sqrt(2. * np.pi) / sigma


def double_normal(x: np.ndarray, x0: float, sigma_peak: float,
                  sigma_tail: float, fpeak: float) -> np.ndarray:
    """ Double Gaussian lineshape """
    return fpeak * normal(x, x0, sigma_peak) +\
        (1. - fpeak) * normal(x, x0, sigma_tail)


def crystal_ball(x: np.ndarray, x0: float, sigma: float,
                 alpha: float, n: float) -> np.ndarray:
    """ The Crystal Ball function """
    xi = (x - x0) / sigma
    nabsa = n / np.abs(alpha)
    mask = xi > -alpha
    f = np.empty(x.shape)
    f[mask] = np.exp(-0.5 * xi[mask]**2)
    f[~mask] = nabsa**n * np.exp(-0.5 * alpha**2) *\
        (nabsa - np.abs(alpha) - xi[~mask])**-n
    return f


def gauss_exp(x, x0, sigma, k) -> np.ndarray:
    """ The GaussExp function """
    xi = (x - x0) / sigma
    f = np.empty(x.shape)
    mask = xi > -k
    f[mask] = np.exp(-0.5 * xi[mask]**2)
    f[~mask] = np.exp(0.5 * k**2 + k * xi[~mask])
    return f / np.sqrt(2. * np.pi) / sigma


def exp_gauss_exp(x, x0, sigma, kl, kh) -> np.ndarray:
    """ The two-sided GaussExp function """
    xi = (x - x0) / sigma

    if isinstance(x, float):
        if xi < -kl:
            res = np.exp(0.5 * kl**2 + kl * xi)
        elif kh < xi:
            res = np.exp(0.5 * kh**2 - kh * xi)
        else:
            res = np.exp(-0.5 * xi**2)
        return np.clip(res / np.sqrt(2. * np.pi) / sigma, 1e-5, 1e10)

    f = np.empty(x.shape)
    maskl = xi < -kl
    maskh = kh < xi
    maskm = ~maskl & ~maskh
    f[maskm] = np.exp(-0.5 * xi[maskm]**2)
    f[maskl] = np.exp(0.5 * kl**2 + kl * xi[maskl])
    f[maskh] = np.exp(0.5 * kh**2 - kh * xi[maskh])
    return np.clip(f / np.sqrt(2. * np.pi) / sigma, 1e-5, 1e10)


def exp_double_gauss_exp(x, x0, sigma_peak, sigma_tail, fpeak, kl, kh)\
        -> np.ndarray:
    """ Two-sided GaussExp plus Gaussian with the same mean """
    return fpeak * normal(x, x0, sigma_peak) +\
        (1. - fpeak) * exp_gauss_exp(x, x0, sigma_tail, kl, kh)


def test_plot():
    x0, sigma, alpha, n, k, kh = 0, 1, 1, 6, 1.2, 1.2
    x = np.linspace(-6, 6, 250)
    plt.plot(x, crystal_ball(x, x0, sigma, alpha, n), label='CB')
    plt.plot(x, gauss_exp(x, x0, sigma, k), label='GaussExp')
    plt.plot(x, exp_gauss_exp(x, x0, sigma, k, kh), label='ExpGaussExp')
    plt.grid()
    plt.xlim(-6, 6)
    plt.ylim(0, 1.05)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_plot()
