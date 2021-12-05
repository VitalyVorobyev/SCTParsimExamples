import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def data_path():
    return '/'.join(['media', 'vitaly', '4759e668-4a2d-4997-8dd2-eb4d25313d90',
                     'vitaly', 'CTau', 'Data'])


def allowed_energies():
    return [0, 100, 250, 500, 750, 1000, 1250,
            1500, 1750, 2000, 2250, 2500, 2750, 3000]


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
            z[row * size:(row + 1) * size] = row
            p[row * size:(row + 1) * size] = np.arange(size)
    else:
        e = clu[:-3]
        for row in range(size):
            z[row * size:(row + 1) * size] = row - size // 2
            p[row * size:(row + 1) * size] =\
                np.arange(size) + clu[-1] - size // 2
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


def draw_cluster_local(clu, key):
    plt.figure(figsize=(9, 8))
    plt.xticks(range(5))
    plt.yticks(range(5))
    plt.xlabel(r'Crystal $z$ index', fontsize=16)
    plt.ylabel(r'Crystal $\phi$ index', fontsize=16)
    plt.grid()
    z, p, e = cluster_to_matrix(clu, local=True)
    plt.scatter(z, p, s=e * 2500)
    plt.scatter(5, 2, s=clu[-1] * 2500, label='Beyond 5x5')

    for i in range(6):
        plt.plot([-0.5 + i, -0.5 + i], [-0.5, 4.5], 'k-', linewidth=1.5)
        plt.plot([-0.5, 4.5], [-0.5 + i, -0.5 + i], 'k-', linewidth=1.5)

    plt.legend(fontsize=20, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'plots/cluster_{key}.png')


def draw_two_clusters(clu1, clu2, local=False):
    plt.figure(figsize=(12, 8))
    if not local:
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
    energy = np.sqrt(0.13957**2 + epcl**2 * 10**-6)
    plt.figure(figsize=(8, 6))
    plt.minorticks_on()
    plt.grid(which='both')
    h, b, _ = plt.hist(clusters[:, :-2].sum(axis=1), bins=100,
                       histtype='step', label='Cluster energy')
    plt.hist(clusters[:, -3], bins=100, histtype='step', label='Beyond 5x5')
    plt.ylim((0, 1.2 * np.max(h)))
    plt.title(f'{key} {energy:.3f} MeV', fontsize=16)
    plt.xlabel(f'{xlbl} (GeV)', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{key}_{epcl}_{xlbl.replace(" ", "_")}.png')


def draw_predictions(pred, labels, key, type1='gamma', type2='pi0'):
    rascore = roc_auc_score(labels, pred)
    plt.figure(figsize=(9, 6))
    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.hist(pred[labels == 1], bins=100, histtype='step',
             density=True, label=type1)
    plt.hist(pred[labels == 0], bins=100, histtype='step',
             density=True, label=type2)
    plt.legend(fontsize=16)
    plt.title(f'{key} ROC AUC Score = {rascore:.3f}', fontsize=16)
    plt.xlabel('Classifier score', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/prediction_{key}.png')


def preprocess_data(data):
    X = data[:, :-2]
    sums = X.sum(axis=1)
    return X / sums.reshape(-1, 1), sums


def make_data(pidata, gadata):
    pidata0, _ = preprocess_data(pidata)
    gadata0, _ = preprocess_data(gadata)
    X = np.vstack((pidata0, gadata0))
    y = np.concatenate((
        np.zeros(pidata0.shape[0], dtype=int),
        np.ones(gadata0.shape[0], dtype=int)
    ))
    return train_test_split(X, y, test_size=0.2, random_state=47)


if __name__ == '__main__':
    cone, ctwo, neve = parse_data('pi0', 2000, data_path(), clusize=5)
    print(cone.shape)
    # draw_cluster(cone[0])
    draw_cluster_local(cone[0][:-2])
    # draw_energy_spectrum(cone, 'pi0', 2000, 'Energy')
    plt.show()
