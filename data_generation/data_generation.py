import sys
from simulation import sample, simulation
from multiprocessing import Pool
import torch
import os
import h5py
import tqdm

sample_length = 4 
core_count = 10 
num_tasks = 250  # core_count * sample_length 

def f(sample_id):
    x = sample(sample_id, sample_length)
    y = simulation(x)
    return (x,y)

if __name__ == '__main__':
    job_id = int(sys.argv[1])

    file_path = 'data/bbp_ds_10m_' + str(job_id) + '.h5'
    if os.path.exists(file_path):
        sys.exit(0)

    tasks = range(job_id * num_tasks, (job_id+1) * num_tasks)

    with Pool(core_count) as p:
        sample_list = list(tqdm.tqdm(p.imap(f, tasks), total=len(tasks)))

    x = []
    y = []
    print("sent them all")
    for sample in sample_list:
        x.append(sample[0])
        y.append(sample[1])
    print("collected them")

    x = torch.cat(x, 0).numpy()
    y = torch.cat(y, 0).numpy()

    with h5py.File(file_path, 'w') as f:
            f.create_dataset('X', data=x)
            f.create_dataset('Y', data=y)
