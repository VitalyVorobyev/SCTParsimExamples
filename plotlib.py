import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 14


def poisson_hist(data, figax=None, xlabel=None, leglabel=None):
    fig, ax = plt.subplots(figsize=(8, 6)) if figax is None else figax
    hist, bins = np.histogram(data, bins=100)
    errs = np.sqrt(hist)
    bins = 0.5 * (bins[1:] + bins[:-1])
    ax.errorbar(bins, hist, yerr=errs, linestyle='none',
                marker='o', markersize=3, label=leglabel)
    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')
    ax.set_ylim((0, 1.05 * np.max(hist)))
    ax.set_xlim((bins[0], bins[-1]))
    if leglabel:
        ax.legend(fontsize=20)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    fig.tight_layout()
    return fig, ax, hist, errs, bins


def fit_plot(fcn, params, data, xlabel):
    fig, ax, hist, errs, bins = poisson_hist(data, xlabel=xlabel)
    norm = np.sum(hist) * (bins[1] - bins[0])
    x = np.linspace(bins[0], bins[-1], 250)
    y = fcn(x, *params) * norm
    ax.plot(x, y)
    return fig, ax, hist, errs, bins


def plot_scatter(x1, x2, xlabel, ylabel, pltlabel=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x1, x2, s=1, label=pltlabel)
    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    fig.tight_layout()
    return fig, ax
