import numpy as np
import matplotlib.pyplot as plt
import os


def data_path():
    return '/media/vitaly/4759e668-4a2d-4997-8dd2-eb4d25313d90/vitaly/CTau/Data'


def allowed_energies():
    return [0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]


def allowed_keys():
    return ['pi0', 'gamma']


def data_file(key, energy, path=None):
    if path is None:
        path = data_path()
    assert key in allowed_keys()
    assert energy in allowed_energies()
    return '/'.join([path, f'clusters_{key}_mom_{energy}_MeV.dat'])


def build_cluster(clu, size=5):
    clu = np.array(clu)
    bclu = np.zeros((size, size))
    eout = 0.
    idx = np.argmax(clu[:, 0])
    cz, cp = map(int, clu[idx, 1:])
    evals = clu[:, 0]
    zvals = clu[:, 1].astype(int) - cz + size // 2
    pvals = clu[:, 2].astype(int) - cp + size // 2
    for e, z, phi in zip(evals, zvals, pvals):
        if z > 0 and z < size and phi > 0 and phi < size:
            bclu[z, phi] = e
        else:
            eout += e
    print(f'bclu shape: {bclu.shape}, eout: {eout:.3f}, eclu: {bclu.sum():.3f}')
    return bclu, eout / bclu.sum()


def parse_data(key, energy, path, clusize=5, evtmax=10**10):
    oneclu, twoclu = [], []
    fname = data_file(key, energy, path)
    assert os.path.isfile(fname)
    with open(fname, 'r') as ifile:
        clusters, clu, cluidx = [], [], None
        ifile.readline()
        eventcnt = 0

        for _, line in enumerate(ifile):
            if line.strip() == 'new event':
                eventcnt += 1
                if eventcnt > evtmax:
                    break
                if clu:
                    nclu, leaks = build_cluster(clu, clusize)
                    if leaks > 0.05:
                        print(f'leaks: {leaks:.3f}')
                    clusters.append(nclu)

                if len(clusters) > 1:
                    twoclu.append(clusters[:2])
                elif clusters:
                    oneclu.append(clusters[0])
                clusters, clu, cluidx = [], [], None
            else:
                idx, eclu, zidx, phidx = map(float, line.strip().split())
                idx, zidx, phidx = map(int, [idx, zidx, phidx])
                if idx == cluidx:
                    clu.append([eclu, zidx, phidx])
                else:
                    if clu:
                        nclu, leaks = build_cluster(clu, clusize)
                        if leaks > 0.05:
                            print(f'leaks: {leaks:.3f}')
                        clusters.append(nclu)
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


if __name__ == '__main__':
    cone, ctwo, neve = parse_data('pi0', 2000, './Data', clusize=5, evtmax=2)
    # print(cone[0])
