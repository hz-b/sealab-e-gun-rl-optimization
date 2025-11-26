import xarray as xr
import argparse
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../../data/bbp_10m_*.nc')
args = parser.parse_args()

z = xr.open_mfdataset('../../data/bbp_ds_10m_*.nc', combine='nested', concat_dim='sample',parallel=True)
x = z['X']
y = z['Y']

# cut out nan lines
mask = ~np.isnan(x).any(axis=1)
x = x[mask]
y = y[mask]

#ds = xr.Dataset({'X': (['sample', 'x'], x), 'Y': (['sample', 'y'], y)})
#ds.to_netcdf('../../data/bbp_merged.nc')

f = h5py.File("/mnt/work/xfel/bessy/berlinPro/bbp_merged.hdf5", "w")
f.create_dataset('Y',data=np.transpose(x.values))
f.create_dataset('X',data=np.transpose(y.values))
f.close()
