from surrogate import H5Dataset
import torch
import matplotlib.pyplot as plt
from simulation import simulation_parallel

ds = H5Dataset('datasets/bbp_ds_2m_merged_v2.h5', raw=True)

repetitions = 1
limit = 100

mask = torch.isnan(ds.y_norm[:limit]).any(dim=1)
mask2 = (torch.abs(ds.y_norm[:limit]) > 30).any(dim=1)
mask3 = (ds.x_norm[:limit,4] < -5) & (ds.x_norm[:limit, 8] > -5)

tryout = torch.repeat_interleave(ds.x_norm[:limit][mask&mask3][0].unsqueeze(0), repetitions, dim=0)
print(simulation_parallel(tryout.cpu())[:,:4])

