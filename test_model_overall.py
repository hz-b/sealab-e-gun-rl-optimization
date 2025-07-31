from data_generation.simulation import simulation_parallel
import torch
from evaluate import *
from model import RandomIterableDataset
from surrogate import H5Dataset
import os
import pickle

network_outputs_list = []
repetitions = 2
device=torch.device('cuda')
model, critic_net = load_model_critic_net(device)
ds = RandomIterableDataset(repetitions, 8, 10000000, device)
h5ds = H5Dataset(os.path.join('datasets','bbp_ds_10m_merged.h5'))

mode = "ga" # "model" or "ga"

print("MODE:", mode)
for i, state in enumerate(tqdm(ds, total=repetitions)):
    with torch.no_grad():
        state = state.unsqueeze(0)
        if mode == "ga":
            _, action, _, _ = eval_evotorch_GA(state.cuda(), 100)
            action= action.unsqueeze(0)
        if mode == "model":
            action = model(state)
        print(action)
        print(critic_net.denormalize_reward(critic_net(action, state)))

        expanded_actions, expanded_states = critic_net.expand_action_states(state, action)
        merged_input = torch.cat([expanded_states, expanded_actions], dim=1)
        merged_un_z_scored_input = h5ds.un_z_score(merged_input.cpu())
        output = simulation_parallel(merged_un_z_scored_input)
        with open('outputs/test_model_overall_'+mode+'_'+str(i)+'.pkl', 'wb') as handle:
            pickle.dump(output, handle)
