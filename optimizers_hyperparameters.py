from optimizers import *
from model import RandomIterableDataset
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import sys
from matplotlib.ticker import FuncFormatter

def space_thousands(x, pos):
    return f"{int(x):,}".replace(",", "\u202f")

def eval_iterative_single_param(eval_fn, param_name, param_info, repetitions=5, niter=100, init_seed=8000000):
    result_dict = {}
    device = torch.device('cpu')
    ds = RandomIterableDataset(repetitions, 8, 50000000, device)

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
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.xlabel("Iteration [#]")
    plt.ylabel(r"$\mathcal{L}_h(\mathbf{x})$")

    if "scale" in param_info:
        plt.yscale(param_info["scale"])

    all_means_tensor = torch.stack(all_means)
    ymin = all_means_tensor.min().item()

    if "scale" in param_info and param_info["scale"] == "log":
        ymin = max(ymin * 0.8, 1e-8)
        plt.ylim(bottom=ymin)

    loc = param_info.get("loc", "best")
    plt.legend(fontsize=11, loc=loc)

    plt.grid(True)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{optimizer_name}_{param_name}.pdf")
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
                    "values": [0.1, 1.0, 10.0, 100.0],
                    "label": r"$T_0$",
                    "scale": "log",
                },
                "cooling_schedule": {
                    "values": ['exp', 'linear'],
                    "label": r"$\text{schedule}$",
                    "scale": "log",
                },
            }
        ),
        # "GA": (
        #     eval_ga,
        #     {
        #         "population_size": {...},
        #         ...
        #     }
        # )
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
    result_dict = eval_iterative_single_param(eval_fn, param_name, param_info, repetitions=10, niter=1000)
    plot_result_dict(result_dict, param_info, opt_name, param_name)

