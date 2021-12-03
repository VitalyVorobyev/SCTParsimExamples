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
    idx = np.argmax(clu[:, 0])
    cz, cp = map(int, clu[idx, 1:])
    evals = clu[:, 0]
    zvals = clu[:, 1].astype(int) - cz + size // 2
    pvals = clu[:, 2].astype(int) - cp + size // 2
    eout = 0.
    for e, z, phi in zip(evals, zvals, pvals):
        if z >= 0 and z < size and phi >= 0 and phi < size:
            bclu[z, phi] = e
        else:
            eout += e
    bclu = np.append(bclu.ravel(), [eout, cz, cp])
    return bclu


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
                    clusters.append(build_cluster(clu, clusize))

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
                        clusters.append(build_cluster(clu, clusize))
                    clu = [[eclu, zidx, phidx]]
                    cluidx = idx
    return np.array(oneclu), np.array(twoclu), eventcnt


def cluster_to_matrix(clu, local=False):
    size = int(np.sqrt(clu.size - 1)) if local else int(np.sqrt(clu.size - 3))
    z = np.empty(size**2)
    p = np.empty(size**2)
    if local:
        e = clu[:-1]
        for row in range(size):
            z[row * size:(row +  1) * size] = row
            p[row * size:(row +  1) * size] = np.arange(size)
    else:
        e = clu[:-3]
        for row in range(size):
            z[row * size:(row +  1) * size] = row - size // 2
            p[row * size:(row +  1) * size] = np.arange(size) + clu[-1] - size // 2
    return z, p, e


def draw_cluster(clu, fullgrid=True):
    plt.figure(figsize=(12, 8))
    if fullgrid:
        plt.xlim((0, 120))
        plt.ylim((0, 115))
    plt.minorticks_on()
    plt.grid(which='both')

    z, p, e = cluster_to_matrix(clu)
    plt.scatter(z, p, s=e * 100)
    plt.tight_layout()


def draw_cluster_local(clu):
    plt.figure(figsize=(8, 8))
    plt.xticks(range(5))
    plt.yticks(range(5))
    plt.grid()
    z, p, e = cluster_to_matrix(clu, local=True)
    plt.scatter(z, p, s=e * 2500)
    plt.tight_layout()


def draw_two_clusters(clu1, clu2):
    plt.figure(figsize=(12, 8))
    plt.xlim((0, 120))
    plt.ylim((0, 115))
    plt.minorticks_on()
    plt.grid(which='both')

    z1, p1, e1 = cluster_to_matrix(clu1)
    z2, p2, e2 = cluster_to_matrix(clu2)
    plt.scatter(z1, p1, s=e1 * 100)
    plt.scatter(z2, p2, s=e2 * 100)
    plt.tight_layout()


def draw_energy_spectrum(clusters, key, epcl, xlbl):
    plt.figure(figsize=(8, 6))
    plt.minorticks_on()
    plt.grid(which='both')
    plt.hist(clusters[:,:-2].sum(axis=1), bins=100, histtype='step', label='Cluster energy')
    plt.hist(clusters[:,-3], bins=100, histtype='step', label='Beyond 5x5')
    plt.title(f'{key} {epcl} MeV', fontsize=16)
    plt.xlabel(f'{xlbl} (GeV)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{key}_{epcl}_{xlbl.replace(" ", "_")}.png')


def draw_predictions(pred, labels, type1, type2, key):
    plt.figure(figsize=(8, 6))
    plt.minorticks_on()
    plt.grid(which='both')
    plt.hist(pred[labels==1], bins=100, histtype='step', label=type1)
    plt.hist(pred[labels==0], bins=100, histtype='step', label=type2)
    plt.legend()
    plt.xlabel(f'Classifier score', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/prediction_{key}.png')


if __name__ == '__main__':
    cone, ctwo, neve = parse_data('pi0', 2000, data_path(), clusize=5)
    print(cone.shape)
    # draw_cluster(cone[0])
    draw_cluster_local(cone[0][:-2])
    # draw_energy_spectrum(cone, 'pi0', 2000, 'Energy')
    plt.show()
