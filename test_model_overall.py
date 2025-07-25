from data_generation.simulation import simulation_parallel
import torch
from evaluate import *
from model import RandomIterableDataset
from surrogate import H5Dataset
import os
import pickle

network_outputs_list = []
repetitions = 2
device=torch.device('cpu')
model, critic_net = load_model_critic_net(device)
ds = RandomIterableDataset(repetitions, 8, 10000000, device)
h5ds = H5Dataset(os.path.join('datasets','bbp_ds_10m_merged.h5'))

for i, state in enumerate(tqdm(ds, total=repetitions)):
    with torch.no_grad():
        state = state.unsqueeze(0)
        expanded_actions, expanded_states = critic_net.expand_action_states(state, model(state))
        merged_input = torch.cat([expanded_states, expanded_actions], dim=1)
        merged_un_z_scored_input = h5ds.un_z_score(merged_input.cpu())
        output = simulation_parallel(merged_un_z_scored_input[:1])
        with open('outputs/test_model_overall_'+str(i)+'.pkl', 'wb') as handle:
            pickle.dump(output, handle)
