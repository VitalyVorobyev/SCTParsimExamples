import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 14
from colors import kelly_gen
import itertools


def xy_calo_plot(crys):
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=False)
    cgen = kelly_gen()
    colit = itertools.cycle([next(cgen) for _ in range(2)])

    ax[0].axis('equal')
    for a in ax:
        a.minorticks_on()
        a.grid(which='major')
        a.grid(which='minor', linestyle=':')

    xyperm = [[0, 1, 5, 6, 0],
              [1, 0, 6, 5, 1]]

    zpperm = [[0, 1, 3, 2, 0],
              [1, 0, 2, 3, 1]]

    for cry in crys:
        col = next(colit)
        permidx = 0 if np.allclose([cry[0, 0]], [cry[0, 1]]) else 1
        cxy, czphi = cry[xyperm[permidx]], cry[zpperm[permidx]]
        ax[0].plot(cxy[:, 0], cxy[:, 1], color=col)
        ax[1].plot(czphi[:, 2], np.arctan2(czphi[:, 0], czphi[:, 1]), color=col)

    plt.tight_layout()
    return fig


def encap_plot(crys, left=False):
    fig = plt.figure(figsize=(6, 6))

    cgen = kelly_gen()
    colit = itertools.cycle([next(cgen) for _ in range(2)])

    plt.axis('equal')
    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    perm = [3, 2, 0, 1, 3] if left else\
           [2, 3, 1, 0, 2, 6, 7, 5, 4, 6]

    for cry in crys:
        col = next(colit)
        cxy = cry[perm]
        plt.plot(cxy[:, 0], cxy[:, 1], color=col)
        # break

    plt.tight_layout()
    return fig


clus = np.array([
    [[  19.93168986,  651.24754345, 1290.        ],
     [ -86.98829045,  642.38790129, 1290.        ],
     [  19.93168986,  696.59858595, 1265.81756232],
     [ -94.45282671,  687.12041468, 1265.81756232],
     [  19.93168986,  802.19599214, 1583.77783574],
     [-111.83358852,  791.27760989, 1583.77783574],
     [  19.93168986,  858.08538459, 1553.97605666],
     [-121.03268017,  846.40474387, 1553.97605666]],
    [[ -86.98829045,  642.38790129, 1290.        ],
     [  19.94434512,  651.26019871, 1290.01265526],
     [ -94.45282671,  687.12041468, 1265.81756232],
     [  19.63508056,  696.30197666, 1265.52095302],
     [-111.83358852,  791.27760989, 1583.77783574],
     [  18.91497507,  801.17927735, 1582.76112095],
     [-121.03268017,  846.40474387, 1553.97605666],
     [  18.53384584,  856.68754057, 1552.57821263]],
    [[  19.93168986,  651.24754345, 1290.        ],
     [  19.95700037,  651.27285396, 1290.02531051],
     [  19.93168986,  696.59858595, 1265.81756232],
     [  19.33847126,  696.00536736, 1265.22434373],
     [  19.93168986,  802.19599214, 1583.77783574],
     [  17.89826028,  800.16256256, 1581.74440615],
     [  19.93168986,  858.08538459, 1553.97605666],
     [  17.13600181,  855.28969654, 1551.18036861]],
])

encap_plot(clus[1:])
plt.show()


# [[ -40.77919325  364.22973665 1290.        ]
#  [  81.03916752  364.48971092 1290.25997427]
#  [ -32.45814317  314.36441733 1303.3126484 ]
#  [  81.03916752  322.94544168 1311.89367275]
#  [ -56.0268922   455.60418697 1607.16339898]
#  [  81.03916752  440.61646228 1592.1756743 ]
#  [ -45.61833717  393.22913633 1623.81579647]
#  [  81.03916752  388.64996668 1619.23662681]]

# [[  81.03916752  364.48971092 1290.25997427]
#  [ -40.77919325  364.22973665 1290.        ]
#  [ -45.61833717  393.22913633 1623.81579647]
#  [  81.03916752  440.61646228 1592.1756743 ]
#  [  81.03916752  364.48971092 1290.25997427]]

# [[-162.89381371  328.31669028 1290.        ]
#  [ -48.51091546  370.22531939 1290.25997427]
#  [-138.01964069  284.30458442 1303.3126484 ]
#  [ -34.30193854  331.18647615 1311.89367275]
#  [-208.47386649  408.96556679 1607.16339898]
#  [ -74.54779787  441.76106589 1592.1756743 ]
#  [-177.35950038  353.91212747 1623.81579647]
#  [ -56.7742096   392.92853344 1619.23662681]]

# [[ -48.51091546  370.22531939 1290.25997427]
#  [-162.89381371  328.31669028 1290.        ]
#  [-177.35950038  353.91212747 1623.81579647]
#  [ -74.54779787  441.76106589 1592.1756743 ]
#  [ -48.51091546  370.22531939 1290.25997427]]