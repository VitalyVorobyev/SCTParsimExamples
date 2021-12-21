import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 14
from colors import kelly_gen
import itertools


def xy_calo_plot(crys):
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=False)
    cgen = kelly_gen()
    colors = [next(cgen) for _ in range(2)]

    colit = itertools.cycle(colors)

    ax[0].axis('equal')
    for a in ax:
        a.minorticks_on()
        a.grid(which='major')
        a.grid(which='minor', linestyle=':')

    xyperm = [
        [0, 1, 5, 6, 0],
        [1, 0, 6, 5, 1]
    ]

    zpperm = [
        [0, 1, 3, 2, 0],
        [1, 0, 2, 3, 1]
    ]

    for cry in crys:
        col = next(colit)
        permidx = 0 if np.allclose([cry[0, 0]], [cry[0, 1]]) else 1
        cxy, czphi = cry[xyperm[permidx]], cry[zpperm[permidx]]
        ax[0].plot(cxy[:, 0], cxy[:, 1], color=col)
        ax[1].plot(czphi[:, 2], np.arctan2(czphi[:, 0], czphi[:, 1]), color=col)

    plt.tight_layout()
    return fig
