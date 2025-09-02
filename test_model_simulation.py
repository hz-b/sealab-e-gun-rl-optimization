import torch
from evaluate import *
from tqdm.auto import tqdm
from model import RandomModel
from simulation import simulation_parallel


take_model = True
sample_length = 100

model, critic_net = load_model_critic_net(torch.device('cuda'))
ds = RandomIterableDataset(sample_length, 8, 420000000, torch.device('cuda'), fixed_seed=True)
solution_vector_list = []

for i in tqdm(ds, leave=False):
    state = i.unsqueeze(0)
    if take_model:
        with torch.no_grad():
            best_solution = model(state)
    else:
        result, best_solution, _, _ = eval_evotorch_GA(state, 100, popsize=200, stdev=0.01, tournament_size=64, eta=8, cross_over_rate=1.0)
        best_solution = best_solution.unsqueeze(0)
    integral_parameters = torch.tensor([[0.4368, 0.7263]], device=state.device)
    solution_vector = critic_net.model.normalizer.unscore_x(torch.hstack([state, integral_parameters, best_solution]))
    solution_vector_list.append(solution_vector[0])
solution_tensor = torch.stack(solution_vector_list)

print(simulation_parallel(solution_tensor.cpu())[:, :4])
