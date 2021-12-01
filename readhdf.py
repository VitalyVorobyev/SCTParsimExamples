import pandas as pd
import h5py

fname = 'hdf/tupdnkpi.hdf'
# data = pd.read_hdf(fname, 'data')

with h5py.File(fname, 'r') as ifile:
    data = ifile['data']
    print(data.keys())
