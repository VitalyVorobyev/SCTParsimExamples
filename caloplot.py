import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 14
from colors import kelly_gen
import itertools


def sweep_barrel_calo_plot(crys):
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

    xycrys = crys[:, xyperm[0],: ]
    tpcrys = crys[:, zpperm[0], :]

    phi = np.arctan2(tpcrys[:, :, 0], tpcrys[:, :, 1])
    the = np.arctan2(tpcrys[:, :, 2], np.sqrt(tpcrys[:, :, 0]**2 + tpcrys[:, :, 1]**2))
    for cry, ph, th in zip(xycrys, phi, the):
        col = next(colit)
        ax[0].plot(cry[:, 0], cry[:, 1], color=col)
        ax[1].plot(th, ph, color=col)

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

    permf = [3, 2, 0, 1, 3] if left else\
            [2, 3, 1, 0, 2]

    for cry in crys:
        col = next(colit)
        cxyf = cry[permf]
        plt.plot(cxyf[:, 0], cxyf[:, 1], color=col)
        # plt.plot(cry[2, 0], cry[2, 1], 'o', color=col)
        # plt.plot(cry[0, 0], cry[0, 1], 'v', color=col)

    plt.tight_layout()
    return fig


def adjust_plot(ref, cry0, cry1, sigma, left=False):
    fig = plt.figure(figsize=(6, 6))

    plt.axis('equal')
    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    permf = [3, 2, 0, 1, 3] if left else\
            [2, 3, 1, 0, 2]

    for cry, ls in zip([ref, cry0, cry1], ['-', '-', '--']):
        cxyf = cry[permf]
        plt.plot(cxyf[:, 0], cxyf[:, 1], ls)
    
    for i, m in zip(permf[:-1], 'od^v'):
        plt.plot([ref[i, 0]], [ref[i, 1]], m, label=f'{i}')
        # plt.plot([cry0[i, 0]], [cry0[i, 1]], m)
        plt.plot([cry1[i, 0]], [cry1[i, 1]], m)

    norm = sigma[1]
    norm = norm / np.sqrt(np.sum(norm**2)) * 50
    # plt.plot([sigma[0][0], norm[0]], [sigma[0][1], norm[1]], linewidth=3)
    plt.plot([sigma[0][0], sigma[0][0] + norm[0]],
             [sigma[0][1], sigma[0][1] + norm[1]], linewidth=3)

    plt.legend()

    plt.tight_layout()
    return fig


clus = np.array([
    [[  89.74711344,  417.97153602, 1290.        ],
     [  20.        ,  417.97153602, 1290.        ],
     [  81.51277146,  368.62582987, 1305.25896859],
     [  20.        ,  368.62582987, 1305.25896859],
     [ 107.03585553,  521.57728729, 1603.37968733],
     [  20.        ,  521.57728729, 1603.37968733],
     [  96.76040521,  459.99988946, 1622.42101133],
     [  20.        ,  459.99988946, 1622.42101133]],
    [[  80.77919325,  364.22973665, 1290.        ],
     [  20.        ,  364.22973665, 1290.        ],
     [  72.45814317,  314.36441733, 1303.3126484 ],
     [  20.        ,  314.36441733, 1303.3126484 ],
     [  96.0268922 ,  455.60418697, 1607.16339898],
     [  20.        ,  455.60418697, 1607.16339898],
     [  85.61833717,  393.22913633, 1623.81579647],
     [  20.        ,  393.22913633, 1623.81579647]],
    [[  80.77919325,  364.22973665, 1290.        ],
     [  20.        ,  364.22973665, 1290.        ],
     [  89.01977468,  413.61283377, 1276.81609146],
     [  20.        ,  413.61283377, 1276.81609146],
     [  96.0268922 ,  455.60418697, 1607.16339898],
     [  20.        ,  455.60418697, 1607.16339898],
     [ 106.33479137,  517.37604015, 1590.67203838],
     [  20.        ,  517.37604015, 1590.67203838]],
    ])

# encap_plot(clus[:-1])
# encap_plot(clus[[0,2]])
# plt.show()


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
