import torch
from optimizers import *
from tqdm.auto import tqdm
from model import RandomModel
from simulation import simulation_parallel
import sys
import numpy as np
from scipy import stats

# Default to False if no argument is given
arg = sys.argv[1] if len(sys.argv) > 1 else "0"

if arg == "1" or arg == "0":
    # Set take_model based on argument
    take_model = arg == "1"

    print(f"take_model = {take_model}")

    sample_length = 100

    model, critic_net = load_model_critic_net(torch.device('cuda'))
    ds = RandomIterableDataset(sample_length, 8, 420000000, model.device, fixed_seed=True)
    solution_vector_list = []

    for i,j in enumerate(tqdm(ds, leave=False)):
        state = j.unsqueeze(0)
        if take_model:
            with torch.no_grad():
                best_solution = model(state)
        else:
            result, _, _, best_solution = eval_ga(state, seed=42+i, return_best=True)
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


    print("f1 - f2:", f"{result[0].mean().item():.3f} ± {result[0].std().item():.3f} [mm]")
    print("f3     :", f"{result[1].mean().item():.3f} ± {result[1].std().item():.3f} [mm]")
    print("f4     :", f"{result[2].mean().item():.3f} ± {result[2].std().item():.3f} [mm]")

if arg == "2":
    # Paths to the output files
    file_ga = 'outputs/simulated_examplega.pt'
    file_model = 'outputs/simulated_examplemodel.pt'

    # Load tensors
    out_ga = torch.load(file_ga)
    out_model = torch.load(file_model)

    # Convert to NumPy
    out_ga = out_ga.cpu().numpy()
    out_model = out_model.cpu().numpy()

    # Remove rows with NaNs
    mask_ga = ~np.isnan(out_ga).any(axis=1)
    mask_model = ~np.isnan(out_model).any(axis=1)
    out_ga = out_ga[mask_ga]
    out_model = out_model[mask_model]

    print(f"GA: {len(out_ga)} valid samples")
    print(f"Model: {len(out_model)} valid samples")

    # Compute derived quantities: |f1 - f2|, |f3|, |f4|
    ga_diff = np.abs(out_ga[:, 0] - out_ga[:, 1])
    model_diff = np.abs(out_model[:, 0] - out_model[:, 1])
    ga_f3 = np.abs(out_ga[:, 2])
    model_f3 = np.abs(out_model[:, 2])
    ga_f4 = np.abs(out_ga[:, 3])
    model_f4 = np.abs(out_model[:, 3])

    metrics = {
        "f1-f2": (ga_diff, model_diff),
        "f3": (ga_f3, model_f3),
        "f4": (ga_f4, model_f4)
    }

    # Perform Welch’s t-test for each derived quantity
    for name, (ga_vals, model_vals) in metrics.items():
        t_stat, p_val = stats.ttest_rel(ga_vals, model_vals, nan_policy='omit')
        print(f"\n{name}:")
        print(f"  GA mean ± std     = {ga_vals.mean():.4f} ± {ga_vals.std():.4f}")
        print(f"  Model mean ± std  = {model_vals.mean():.4f} ± {model_vals.std():.4f}")
        print(f"  t = {t_stat:.4f}, p = {p_val:.4e}")
        if p_val < 0.05:
            print("  → Significant difference (p < 0.05)")
        else:
            print("  → No significant difference (p ≥ 0.05)")
