# __QBunch__ 0.1e-3
# __zSol__ 0.4625
# __Start__ [-24:-5]*1e-4 wie oben __Gun_field__
# __Stop__ 1.737
# __Zoff__ identical with __Gun_Field__ *1e-4 (pseudo integer)
# Y = measured values (meas. rms. hor. beam size (mm), meas. rms. ver. beam size (mm), meas. hor. beam position (mm), meas. ver. beam position (mm), meas. avr. beam momentum (MeV/c)=

import re
import os
import hashlib
import torch
import subprocess
import pandas as pd
import numpy as np
from collections import OrderedDict

sim_Y_labels = ["Horizontal beam size [mm]", "Vertical beam size [mm]", "Horizontal beam position [mm]", "Vertical beam postion [mm]", "Average beam momentum [MeV/c]"]

limits = OrderedDict([
    ("__pulselength__", [0.6e-3, 4e-3]),           # x[0]
    ("__spotsize__", [0.2, 0.8]),                  # x[1]
    ("__Gun_Epeak__", [9, 18]),                    # x[2]
    ("__Gun_Bias__", [3, 5]),                      # x[3]
    ("__Gun_Field__", [-24, -5]),                  # x[4] (integer, limits included)
    ("__FF__", [-0.5, 0.5]),                       # x[5]
    ("__SpotXPos__", [-1.5, 1.5]),                 # x[6]
    ("__SpotYPos__", [-1.5, 1.5]),                 # x[7]
    ("__Gun_Phase__", [-10, 70]),                  # x[8]
    ("__BSol__", [-0.1, 0.1]),                     # x[9]
    ("__SolXPos__", [-4e-3, 4e-3]),                # x[10]
    ("__SolYPos__", [-4e-3, 4e-3]),                # x[11]
    ("__rotX__", [-30e-3, 30e-3]),                 # x[12]
    ("__rotY__", [-30e-3, 30e-3])                  # x[13]
])

# Define which keys are integers
int_keys = ["__Gun_Field__"]

# Separate keys
all_keys = list(limits.keys())
cont_keys = [k for k in all_keys if k not in int_keys]

# Min and max tensors for continuous keys
cont_min = torch.tensor([limits[k][0] for k in cont_keys])
cont_max = torch.tensor([limits[k][1] for k in cont_keys])

def sample(first_sample_id, sample_length):
    samples = []

    for seed in range(first_sample_id, first_sample_id + sample_length):
        torch.manual_seed(seed)

        # Sample continuous values
        rand_cont = torch.rand(len(cont_keys))
        cont_sample = cont_min + (cont_max - cont_min) * rand_cont

        # Sample integer value(s)
        while True:
            gun_field = torch.randint(limits["__Gun_Field__"][0], limits["__Gun_Field__"][1]+1, (1,))
            if gun_field.item() not in (-21, -23):
                break
        gun_field = gun_field.float()

        # Create a dictionary to hold all sampled values
        sample_dict = {k: None for k in all_keys}

        # Insert sampled values into correct positions
        for i, k in enumerate(cont_keys):
            sample_dict[k] = cont_sample[i]
        sample_dict["__Gun_Field__"] = gun_field[0]

        # Build final ordered tensor sample
        sample_tensor = torch.tensor([sample_dict[k] for k in all_keys])
        samples.append(sample_tensor)

    return torch.stack(samples)

def replace_variables(input_string, replace):
    pattern = '|'.join(sorted(re.escape('__'+k+'__') for k in replace))
    return re.sub(pattern, lambda m: str(replace.get(m.group(0)[2:-2])), input_string)
    
def read_config_file(file_path):
    config_file = open(file_path, 'r')
    config_string = config_file.read()
    config_file.close()
    return config_string

def write_config_file(file_path, string):
    output_file = open(file_path, 'w')
    output_file.write(string)
    output_file.close()

def read_output_file(file_path):
    df = pd.read_csv(file_path, sep=' +', engine='python', header=None)
    sel_df = df[df[9] >= 0]
    
    hor_beam_size = np.std(sel_df[0])*1e3
    ver_beam_size = np.std(sel_df[1])*1e3
    hor_beam_position = np.mean(sel_df[0])*1e3
    ver_beam_position = np.mean(sel_df[1])*1e3
    
    if df[9][0] >= 0:
        sel_df = sel_df[1:]

    avr_beam_momentum = np.mean(sel_df[5]) + df[5][0]/1e6
    return torch.tensor([hor_beam_size, ver_beam_size, hor_beam_position, ver_beam_position, avr_beam_momentum], dtype=torch.float32)
    
def simulation(parameters, scratch_dir='/tmp', simulation_dir='./simulation'):
    results_list = []
    for x in parameters:
        x = x.numpy()
        this_config_name = hashlib.sha256(str(x).encode()).hexdigest()
        config_string = read_config_file(os.path.join(simulation_dir, 'Generator_Setup.in'))
        
        dist_file_path = os.path.join(scratch_dir, this_config_name+'_dist')
        replace_generator = {'Dist' : dist_file_path, 'QBunch' : .1e-3, 'pulselength' : x[0], 'spotsize' : x[1]}
        new_string = replace_variables(config_string, replace_generator)
        generator_config_path = os.path.join(scratch_dir, this_config_name+'_gen.in')

        write_config_file(generator_config_path, new_string)

        replace_astra = {'CavFields' : os.path.join(simulation_dir, 'CavFields'), 'Gun_Epeak' : x[2], 'Gun_Bias' : x[3], 'FF' : x[5], 'SpotXPos' : x[6], 'SpotYPos' : x[7], 'SolXPos' : x[10], 'SolYPos' : x[11], 'rotX' : x[12], 'rotY' : x[13], 'Gun_Phase' : x[8], 'BSol' : x[9], 'Gun_Field' : "%.f" % x[4], 'Zoff' : x[4]*1e-4, 'SpotZPos' : x[4]*1e-4, 'Start' : x[4]*1e-4, 'Stop' : 1.737, 'zSol' : 0.4625 }
        replace = {**replace_generator, **replace_astra}

        config_string = read_config_file(os.path.join(simulation_dir, 'ASTRA_Setup.in'))
        new_string = replace_variables(config_string, replace)
        astra_config_path = os.path.join(scratch_dir, this_config_name+'_astra.in')
        write_config_file(astra_config_path, new_string)
        
        log_path = os.path.join(scratch_dir, this_config_name+'.log')
        log_file = open(log_path, "w")
        subprocess.run([os.path.join(simulation_dir, 'generator'), generator_config_path], stdout=log_file)
        astra_return_code = subprocess.run([os.path.join(simulation_dir, 'Astra'), astra_config_path], stdout=log_file).returncode
        
        astra_output_path = os.path.join(scratch_dir, this_config_name+'_astra.0174')
        if not os.path.isfile(astra_output_path):
            results_list.append(torch.zeros(6, dtype=torch.float32) * float("nan"))
        else:
            results_list.append(read_output_file(astra_output_path))

        
        for file_path in [dist_file_path, generator_config_path, astra_config_path, astra_output_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
    return torch.stack(results_list)

if __name__ == '__main__':
    print(simulation(sample(2,1)))
