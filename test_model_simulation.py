import torch
from optimizers import *
from tqdm.auto import tqdm
from model import RandomModel
from simulation import simulation_parallel
import sys

# Default to False if no argument is given
arg = sys.argv[1] if len(sys.argv) > 1 else "0"

# Set take_model based on argument
take_model = arg == "1"

print(f"take_model = {take_model}")

sample_length = 100

model, critic_net = load_model_critic_net(torch.device('cuda'))
ds = RandomIterableDataset(sample_length, 8, 420000000, model.device, fixed_seed=True)
solution_vector_list = []

for i in tqdm(ds, leave=False):
    state = i.unsqueeze(0)
    if take_model:
        with torch.no_grad():
            best_solution = model(state)
    else:
        result, best_solution, _, _ = eval_evotorch_GA(state)
        best_solution = best_solution.unsqueeze(0)
    integral_parameters = torch.tensor([[0.4368, 0.7263]], device=state.device)
    solution_vector = critic_net.model.normalizer.unscore_x(torch.hstack([state, integral_parameters, best_solution]))
    solution_vector_list.append(solution_vector[0])
solution_tensor = torch.stack(solution_vector_list)

output = simulation_parallel(solution_tensor.cpu())[:, :4]

add_str = "model" if take_model else "ga"
torch.save(output, 'outputs/simulated_example'+add_str+'.pt')

# Find rows with any NaNs
nan_mask = torch.isnan(output).any(dim=1)

# Count NaN rows
num_nan_rows = nan_mask.sum().item()

# Total number of rows
total_rows = output.size(0)

# Filter out rows with NaNs
clean_output = output[~nan_mask]

# Calculate percentage
nan_percent = 100.0 * num_nan_rows / total_rows

# Print results
print(f"Removed {num_nan_rows} NaN row(s) out of {total_rows} ({nan_percent:.2f}%)")

result = torch.stack([
    torch.abs(clean_output[:, 0] - clean_output[:, 1]),
    torch.abs(clean_output[:, 2]),
    torch.abs(clean_output[:, 3])
], dim=0)


print("f1-f2", result[0].mean().item(), "[mm]")
print("f3", result[1].mean().item(), "[mm]")
print("f4", result[2].mean().item(), "[mm]")
