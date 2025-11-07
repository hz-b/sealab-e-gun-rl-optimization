from optimizers import *
from model import RandomIterableDataset
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import sys
from matplotlib.ticker import FuncFormatter
from matplotlib.pylab import cycler

def space_thousands(x, pos):
    return f"{int(x):,}".replace(",", "\u202f")

def eval_iterative_single_param(eval_fn, param_name, param_info, repetitions=5, niter=100, init_seed=8000000):
    result_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RandomIterableDataset(repetitions, 8, 50000000, device, fixed_seed=True)

    result_dict[param_name] = {}

    for val in tqdm(param_info["values"], desc=f"{param_name}", leave=False):
        run_progresses = []

        for i, state in enumerate(tqdm(ds, total=repetitions, leave=False)):
            state = state.unsqueeze(0)
            kwargs = {param_name: val}

            with torch.no_grad():
                progress, _, _ = eval_fn(state, niter=niter, seed=init_seed+i, **kwargs)

            run_progresses.append(progress)

        progresses_tensor = torch.stack(run_progresses)
        mean_result = progresses_tensor[:, -1].mean()
        print(f"{param_name} = {val} → Mean final loss: {mean_result:.6f}")

        result_dict[param_name][f"{val}"] = {
            "mean_result": mean_result,
            "progresses": progresses_tensor
        }

    return result_dict

def plot_result_dict(result_dict, param_info, optimizer_name, param_name):
    plt.figure(figsize=(8, 3))
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    custom_colors = default_colors[:2] + default_colors[3:]
    plt.rcParams['axes.prop_cycle'] = cycler(color=custom_colors)
    display_name = param_info.get("label", param_name)
    param_results = result_dict[param_name]
    all_means = []

    for param_value_str, data in param_results.items():
        progresses = data['progresses'].cpu()
        mean_progress = progresses.mean(dim=0)
        std_progress = progresses.std(dim=0)

        all_means.append(mean_progress)
        x = range(len(mean_progress))
        plt.plot(x, mean_progress, label=f"{display_name}={param_value_str}", alpha=0.8)
        plt.fill_between(x, mean_progress - std_progress, mean_progress + std_progress, alpha=0.2)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(space_thousands))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(space_thousands))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel("Iteration [#]", fontsize=16)
    plt.ylabel(r"$\mathcal{L}_l(\mathbf{x})$", fontsize=16)

    if "scale" in param_info:
        plt.yscale(param_info["scale"])

    all_means_tensor = torch.stack(all_means)
    ymin = all_means_tensor.min().item()

    if "scale" in param_info and param_info["scale"] == "log":
        ymin = max(ymin * 0.8, 1e-8)
        plt.ylim(bottom=ymin)

    loc = param_info.get("loc", "best")
    plt.legend(fontsize=12, loc=loc)

    plt.grid(True)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{optimizer_name}_{param_name}_srf.pdf")
    plt.show()


if __name__ == "__main__":
    job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0               
    optimize_dict = {
        "SA": (
            eval_sa,
            {
                "step_size": {
                    "values": [0.01, 0.1, 0.2, 0.5, 1.0],
                    "label": r"$\eta$",
                    "scale": "log",
                    "loc": "lower left",
                },
                "T_start": {
                    "values": [0.001, 0.01, 0.1],
                    "label": r"$t_\mathrm{start}$",
                    "scale": "log",
                },
                "cooling_schedule": {
                    "values": ['exp', 'linear'],
                    "label": r"$\text{schedule}$",
                    "scale": "log",
                },
            }
        ),
        "GD": (
            eval_gd,
            {
                "lr": {
                    "values": [0.001, 0.5, 1.0, 5.0, 10.0],
                    "label": r"$\eta$",
                    "scale": "log",
                    "loc": "upper right",
                },
            }
        ),
        "GA": (
            eval_ga,
            {
                "num_candidates": {
                    "values": [10, 100, 200, 500],
                    "label": r"$p$",
                     "scale": "log",
                 },
                 "tournament_size": {
                     "values": [1, 3, 5, 10, 15, 20],
                     "label": r"$k_t$",
                     "scale": "log",
                 },
                 "mutation_rate": {
                     "values": [0.001, 0.01, 0.05, 0.1, 0.2],
                     "label": r"$r_m$",
                     "scale": "log",
                 },
                 "mutation_scale": {
                     "values": [0.001, 0.01, 0.05, 0.1, 0.2],
                     "label": r"$s_m$",
                     "scale": "log",
                 },
                 "sbx_eta": {
                     "values": [1, 5, 10, 50, 100],
                     "label": r"$\eta$",
                     "scale": "log",
                     "loc": "lower left",
                 },
                 "sbx_crossover_rate": {
                     "values": [0.1, 0.3, 0.5, 0.8, 0.9],
                     "label": r"$r_c$",
                     "scale": "log",
                     "loc": "lower left",
                },
            }
        ),
        "BLOP": (
            eval_blop,
            {
                "acq": {
                    "values": ["ei", "lcb"],
                    "label": r"$a$",
                    "scale": "log",
                    "loc": "upper right",
                },
                "warm_up_iterations": {
                    "values": [16, 32, 64],
                    "label": r"$l_\mathrm{warm}$",
                     "scale": "log",
                    "loc": "upper right",
                },
                 "transform": {
                     "values": [None, "log"],
                     "label": r"$t$",
                     "scale": "log",
                    "loc": "upper right",
                 },
                 "ucb_beta": {
                     "values": [1.0, 5.0, 10.0, 20.0, 50.],
                     "label": r"$\beta$",
                     "scale": "log",
                    "loc": "upper right",
                 },
            }
        ),
    }
    

    # Flatten job list → [(optimizer_name, eval_fn, param_name, param_info), ...]
    job_list = []
    for opt_name, (eval_fn, param_grid) in optimize_dict.items():
        for param_name, param_info in param_grid.items():
            job_list.append((opt_name, eval_fn, param_name, param_info))

    if job_id >= len(job_list):
        raise ValueError(f"Invalid job ID {job_id}. Only {len(job_list)} jobs available.")

    opt_name, eval_fn, param_name, param_info = job_list[job_id]

    print(f"Running Job ID {job_id}: {opt_name} – {param_name}")
    result_dict = eval_iterative_single_param(eval_fn, param_name, param_info, repetitions=100, niter=150)
    plot_result_dict(result_dict, param_info, opt_name, param_name)

