import h5py
import numpy as np
import glob
import re
import os
from tqdm import tqdm

# Use glob to find all matching files
file_pattern = 'data/bbp_ds_10m_*.h5'
files = glob.glob(file_pattern)

print(len(files))

unpresent_files = []

for i in range(1, 2001):
    filename = f"data/bbp_ds_10m_{i}.h5"
    if filename not in files:
        unpresent_files.append(filename)

# Print result
print("Missing files:")
for f in unpresent_files:
    print(f)

# Sort files numerically based on the number in the filename
def extract_number(filename):
    match = re.search(r'bbp_ds_10m_(\d+)\.h5', os.path.basename(filename))
    return int(match.group(1)) if match else -1

files.sort(key=extract_number)

X_list = []
Y_list = []

for filename in tqdm(files):
    with h5py.File(filename, 'r') as f:
        X_list.append(f['X'][:])
        Y_list.append(f['Y'][:])

# Merge along axis 0
X_merged = np.concatenate(X_list, axis=0)
Y_merged = np.concatenate(Y_list, axis=0)

# Write merged data
output_file = '../datasets/bbp_ds_2m_merged_v2.h5'
with h5py.File(output_file, 'w') as f_out:
    f_out.create_dataset('X', data=X_merged, compression='gzip')
    f_out.create_dataset('Y', data=Y_merged, compression='gzip')
