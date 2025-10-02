from data_generation.simulation import simulation_parallel
import torch
from evaluate import *
from model import RandomIterableDataset
from surrogate import H5Dataset
import os
import pickle

def optimize_state(state, mode, model, normalizer, device):
    with torch.no_grad():
        state = state.unsqueeze(0)
        if mode == "ga":
            _, action, _, _ = eval_evotorch_GA(state, 100)
            action= action.unsqueeze(0)
        if mode == "model":
            action = model(state)
        print(action)
        print(critic_net.denormalize_reward(critic_net(action, state)))

        expanded_actions, expanded_states = critic_net.expand_action_states(state, action)
        merged_input = torch.cat([expanded_states, expanded_actions], dim=1)
        merged_unscored_input = normalizer.unscore_x(merged_input.cpu())
        output = simulation_parallel(merged_unscored_input)
        return output

if __name__ == "__main__":
    repetitions = 2
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, critic_net = load_model_critic_net(device)
    ds = RandomIterableDataset(repetitions, 8, 10000000, device)
    normalizer = critic_net.model.normalizer
    
    mode = "ga" # "model" or "ga"
    
    print("MODE:", mode)
    for i, state in enumerate(tqdm(ds, total=repetitions)):
        output = optimize_state(state, mode, model, normalizer, device)
        with open('outputs/test_model_overall_'+mode+'_'+str(i)+'.pkl', 'wb') as handle:
            pickle.dump(output, handle)
