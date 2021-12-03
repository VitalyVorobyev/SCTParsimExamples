import numpy as np
import matplotlib.pyplot as plt


def data_path():
    return '/'.join(['media', 'vitaly', '4759e668-4a2d-4997-8dd2-eb4d25313d90',
                     'vitaly', 'CTau', 'Data'])


def allowed_energies():
    return [0, 100, 250, 500, 750, 1000, 1250,
            1500, 1750, 2000, 2250, 2500, 2750, 3000]


def allowed_keys():
    return ['pi0', 'gamma']


def data_file(key, energy):
    assert key in allowed_keys()
    assert energy in allowed_energies()
    return '/'.join([data_path(), f'clusters_{key}_mom_{energy}_MeV.dat'])


def parse_data(key, energy):
    oneclu, twoclu = [], []
    with open(data_file(key, energy), 'r') as ifile:
        clusters, clu, cluidx = [], [], None
        ifile.readline()
        eventcnt = 0

        for lnum, line in enumerate(ifile):
            if line.strip() == 'new event':
                eventcnt += 1
                if len(clu) == 25:
                    clusters.append(clu)

                if len(clusters) == 2:
                    twoclu.append(clusters)
                elif clusters:
                    oneclu.append(clusters[0])
                clusters, clu, cluidx = [], [], None
            else:
                idx, eclu, zidx, phidx = map(float, line.strip().split())
                idx, zidx, phidx = map(int, [idx, zidx, phidx])
                if idx == cluidx:
                    clu.append([eclu, zidx, phidx])
                else:
                    if len(clu) == 25:
                        clusters.append(clu)
                    clu = [[eclu, zidx, phidx]]
                    cluidx = idx
    return np.array(oneclu), np.array(twoclu), eventcnt


def draw_cluster(clu):
    plt.figure(figsize=(12, 8))
    plt.xlim((0, 120))
    plt.ylim((0, 115))
    plt.minorticks_on()
    plt.grid(which='both')
    plt.scatter(clu[:, 1], clu[:, 2], s=clu[:, 0] * 100)
    plt.tight_layout()


def draw_energy_spectrum(eclu, key, epcl, xlbl):
    plt.figure(figsize=(8, 6))
    plt.minorticks_on()
    plt.grid(which='both')
    plt.hist(eclu, bins=100, histtype='step')
    plt.title(f'{key} {epcl} MeV', fontsize=16)
    plt.xlabel(f'{xlbl} (GeV)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/{key}_{epcl}_{xlbl.replace(" ", "_")}.png')
