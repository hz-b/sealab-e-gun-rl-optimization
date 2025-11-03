import logging
import pickle
import time
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap, jacrev
import torch.utils.benchmark as benchmark

import matplotlib.pyplot as plt
import numpy as np

from blop import DOF, Objective, Agent
from bluesky import RunEngine
from databroker import Broker

from critic import Critic
import random
from model import RandomModel, RandomIterableDataset

from scipy.optimize import minimize, dual_annealing
from scipy.stats import ttest_rel

from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import (
    SimulatedBinaryCrossOver,
    GaussianMutation,
)
from evotorch.logging import StdOutLogger

from evaluate_nn import get_checkpoint_path

def eval_scipy(state, niter, seed=42, method="Powell", device=torch.device('cpu'), eval_mode=False):
    if seed is not None:
        seed_all(seed)
    initial_action = torch.rand((4), device=device)
    state = state.to(device)
    critic_net = load_critic(device)
    optimization_values = []
    validity_values = []
    
    def y_const(x):
        value = critic_net(torch.tensor(x, device=device, dtype=torch.float).view(1, -1), state, eval_mode=eval_mode)
        if eval_mode:
            value, validity = value
            validity_values.append(validity)
        optimization_values.append(value.mean(-1).mean(-1))
        callback_times.append(time.time())
        return value.mean().item()

    callback_times = []
    start_time = time.time()
    
    res = minimize(
        fun = y_const,
        x0 = initial_action,
        method = method,
        bounds = [(0., 1.), (0., 1.), (0., 1.), (0., 1.)],
        options = {'maxiter': niter, "disp": False},
    )

    while len(optimization_values) < niter:
        optimization_values.append(optimization_values[-1])

    end_time = time.time()
    elapsed_time = end_time - start_time
    best_losses = torch.stack(optimization_values[:niter])
    if eval_mode:
        _, best_idx = best_losses.min(dim=0)
        best_losses = best_losses, torch.stack(validity_values[:niter])[best_idx]
    return best_losses, elapsed_time, calculate_iter_durations(start_time, callback_times, niter)

def eval_sa(
    state,
    niter,
    seed=42,
    step_size=0.1,
    T_start=100.0,
    T_end=1e-3,
    cooling_schedule='exp',
    verbose=True,
    eval_mode=False,
):
    seed_all(seed)
    # initial action
    dim = 4  # action dimension
    device = state.device
    x = torch.rand(dim, device=device)
    critic_net = load_critic(device)
    
    callback_times = []
    validity_values = []

    def energy_fn(x):
        x = x.clamp(0.0, 1.0)
        value = critic_net(x.view(1, -1), state, eval_mode=eval_mode)
        if eval_mode:
            value, validity = value
            validity_values.append(validity)
        callback_times.append(time.time())
        return value.mean()
    
    best_x = x.clone()
    current_energy = energy_fn(x)
    best_energy = current_energy.clone()
    best_losses = [best_energy.item()]
    accepted = 0

    if verbose:
        print(f"[SA] Initial energy: {current_energy.item():.6f}")
        print(f"[SA] Running with step_size={step_size}, T_start={T_start}, schedule={cooling_schedule}, niter={niter}, seed={seed}")

    def temperature(t):
        if cooling_schedule == 'exp':
            return T_start * (T_end / T_start) ** (t / niter)
        elif cooling_schedule == 'linear':
            return T_start - t * (T_start - T_end) / niter
        else:
            raise ValueError("Unknown cooling schedule")

    start_time = time.time()

    for t in tqdm(range(niter), leave=False):
        T = temperature(t)
        perturbation = torch.randn_like(x) * step_size
        x_new = (x + perturbation).clamp(0.0, 1.0)
        energy_new = energy_fn(x_new)

        delta_E = energy_new - current_energy
        accept_prob = torch.exp(-delta_E / T).clamp(max=1.0)
        rand_val = torch.rand(1, device=device)

        if delta_E < 0 or rand_val < accept_prob:
            x = x_new
            current_energy = energy_new
            accepted += 1
            if energy_new < best_energy:
                best_energy = energy_new
                best_x = x_new.clone()

        best_losses.append(best_energy.item())

        # Optional per-interval logging
        if verbose and t % (niter // 10) == 0 and t > 0:
            current_acceptance_rate = accepted / t
            print(
                f"[SA] Iter {t:4d} | "
                f"T={T:.4g} | "
                f"Current Energy={current_energy.item():.6f} | "
                f"Best={best_energy.item():.6f} | "
                f"Acceptance Rate={current_acceptance_rate * 100:.2f}%"
            )


    end_time = time.time()
    elapsed_time = end_time - start_time
    acceptance_rate = accepted / niter

    if verbose:
        print(f"[SA] Finished in {elapsed_time:.2f} seconds")
        print(f"[SA] Final best energy: {best_energy.item():.6f}")
        print(f"[SA] Acceptance rate: {acceptance_rate * 100:.2f}%")
    best_losses = torch.tensor(best_losses[:niter])
    if eval_mode:
        _, best_idx = best_losses.min(dim=0)
        best_losses = best_losses, torch.stack(validity_values[:niter])[best_idx]
    return best_losses, elapsed_time, calculate_iter_durations(start_time, callback_times, niter)

def eval_gd(state, niter, seed=42, lr=0.1, eval_mode=False):
    seed_all(seed)
    initial_action = torch.rand((1, 4), device=state.device, requires_grad=True)
    if state.device.type == "cuda":
        warmup_gpu(state.device)

    if state.device.type == "cuda":
        torch.cuda.synchronize()

    critic_net = load_critic(state.device)

    optimizer = optim.SGD([initial_action], lr=lr)
    optimization_values = []
    validity_values = []

    start_time = time.time()
    callback_times = []
    for _ in range(niter):
        optimizer.zero_grad()
        with torch.enable_grad():
            value = critic_net(initial_action.view(1, -1), state, eval_mode=eval_mode)
            if eval_mode:
                value, validity = value
                validity_values.append(validity)
            loss = value.mean()
            optimization_values.append(value.mean().detach())
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                initial_action.clamp_(0.0, 1.0)
            
            callback_times.append(time.time())
    if state.device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    best_losses = torch.cummin(torch.tensor(optimization_values, device=state.device), dim=0).values
    if eval_mode:
        _, best_idx = best_losses.min(dim=0)
        best_losses = best_losses, torch.stack(validity_values[:niter])[best_idx]
    return best_losses, elapsed_time, calculate_iter_durations(start_time, callback_times, niter)

def eval_ga(state, niter=1000, seed=42, num_candidates=100, mutation_scale=0.01, mutation_rate=0.01, tournament_size=20, sbx_eta=5, sbx_crossover_rate=0.5, return_best=False, eval_mode=False):
    seed_all(seed)
    assert num_candidates > 2 # else we have problems on crossover
    init_problem_one = state.repeat_interleave(num_candidates, dim=0)
    if state.device.type == "cuda":
        warmup_gpu(state.device)

    if state.device.type == "cuda":
        torch.cuda.synchronize()
    
    logging.getLogger("evotorch").setLevel(logging.WARNING)
    critic_net = load_critic(state.device)
    
    optimization_values = []
    validity_values = []
    def critic_problem(x):
        if sbx_crossover_rate != 1.0:
            init_problem = state.repeat_interleave(x.shape[0], dim=0)
        else:
            init_problem = init_problem_one
        output = critic_net(x.clone(), init_problem, clamping=False, penalize_forbidden_actions=True, eval_mode=eval_mode)
        if eval_mode:
            output, validity = output
        scalar = output.mean(dim=1)
        if x.shape[0] == num_candidates:
            optimization_values.append(output)
            if eval_mode:
                validity_values.append(validity)
        callback_times.append(time.time())
        return scalar
                                          
    prob = Problem(
        ["min"],
        critic_problem,
        initial_bounds=(0.0, 1.0),
        solution_length=4,
        vectorized=True,
        device=state.device
    )
    
    # Works like NSGA-II for multiple objectives
    ga = GeneticAlgorithm(prob, 
        operators=[
            SimulatedBinaryCrossOver(
                    prob,
                    tournament_size=tournament_size,
                    cross_over_rate=sbx_crossover_rate,
                    eta=sbx_eta,
                ),
                GaussianMutation(prob, stdev=mutation_scale, mutation_probability=mutation_rate)
            ],
        popsize=num_candidates)

    callback_times = []
    start_time = time.time()
    ga.run(niter)
    output = torch.stack(optimization_values)

    best_indices = output.mean(dim=-1).argmin(dim=1)
    best_solution = ga.status["pop_best"].values.clone()

    if state.device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    best_losses = output.mean(dim=-1)[torch.arange(output.shape[0]), best_indices]
    iter_durations = calculate_iter_durations(start_time, callback_times, niter)
    if return_best:
        return out, elapsed_time, iter_durations, best_solution
    if eval_mode:
        _, best_idx = best_losses.min(dim=0)
        best_losses = best_losses, torch.stack(validity_values[:niter])[best_idx]
    return best_losses, elapsed_time, iter_durations

def eval_blop(state, niter=1000, warm_up_iterations=32, bo_iterations=150, acq="lcb", ucb_beta=5.0, transform="log", seed=None, num_candidates=1, device=None, eval_mode=False):
    if seed is not None:
        seed_all(seed)
    device = state.device
    if acq == "lcb":
        acq = "ucb"
    if acq == "qlcb":
        acq = "qucb"
        
    # Setup BLoP optimizer components
    db = Broker.named("temp")
    RE = RunEngine({})
    RE.subscribe(db.insert)
    
    critic_net = load_critic(state.device)

    ndims = 4
    dofs = [DOF(name=str(i), search_domain=(0., 1.)) for i in range(ndims)]
    
    objectives = [
        Objective(name="l_loss", transform=transform, description="Sealab", target="min"),
        Objective(name="is_invalid", constraint=(-torch.inf, 0), transform=None)
    ]
    
    # Logging
    losses_list = []
    validity_values = []
    callback_times = []
    progress = tqdm(total=niter, desc="BLoP Optimization", leave=False)

    def objective_function(action):
        progress.update(1)
        
        # Compute main loss
        loss = critic_net(action, state.repeat_interleave(action.shape[0], dim=0), eval_mode=eval_mode)
        if eval_mode:
            loss, validity = loss
            validity_values.append(validity)
        is_invalid = torch.all(loss == 1000., dim=1).int().tolist()
        loss = loss.mean(dim=-1)
        losses_list.append(loss.min())
        loss = loss.tolist()
        loss = loss if isinstance(loss, list) else [loss]
        is_invalid = is_invalid if isinstance(is_invalid, list) else [is_invalid]
        callback_times.append(time.time())
        return loss, is_invalid

    def digestion(df):
        param_tensor = torch.tensor([df[str(i)] for i in range(ndims)], device=device)
        df["l_loss"], df["is_invalid"] = objective_function(param_tensor.T)
        return df

    agent = Agent(dofs=dofs, objectives=objectives, digestion=digestion, db=db)

    # Time tracking
    start_time = time.time()

    # Warmup phase
    RE(agent.learn("quasi-random", iterations=warm_up_iterations, n=num_candidates))

    if num_candidates > 1 and not acq.startswith("q"):
        acq = "q" + acq
        print("Switched to quasi-acq, since you picked a num_candidates larger than 1, which only works with quasi-acqs.")
    
    # Main optimization phase
    if acq in ("qucb", "ucb"):
        RE(agent.learn(acq, iterations=bo_iterations-warm_up_iterations, n=num_candidates, beta=ucb_beta))
    else:
        RE(agent.learn(acq, iterations=bo_iterations-warm_up_iterations, n=num_candidates))

    end_time = time.time()
    elapsed_time = end_time - start_time

    while len(losses_list) < niter:
        losses_list.append(losses_list[-1])
    
    best_losses = torch.cummin(torch.stack(losses_list[:niter]), dim=0).values
    iter_durations = calculate_iter_durations(start_time, callback_times, niter)
    if eval_mode:
        best_losses = best_losses, torch.stack(validity_values[:niter])[-1]
    return best_losses, elapsed_time, iter_durations

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def warmup_gpu(device):
    a = torch.randn(3000, 3000, device=device)
    b = torch.randn(3000, 3000, device=device)
    torch.cuda.synchronize()
    _ = torch.mm(a, b)
    torch.cuda.synchronize()

def benchmark_model(model, input_count=4, samples=1):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    
    # Set model to eval mode and move to device
    if isinstance(model, nn.Module):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    #input_shape = (samples, input_count)
    # Dummy input
    dummy_input = torch.randn(samples, input_count).to(device)
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Use benchmark.Timer
    timer = benchmark.Timer(
        stmt='model(dummy_input)',
        globals={'model': model, 'dummy_input': dummy_input},
        label='Model Inference',
    )
    
    result = timer.timeit(1000)
    print(result)  # Automatically shows time per run and other stats

def calculate_iter_durations(start_time, callback_times, niter):
    callback_times = [start_time] + callback_times  # prepend total start
    iter_durations = [t2 - start_time for t1, t2 in zip(callback_times[:-1], callback_times[1:])]

    while len(iter_durations) < niter:
        iter_durations.append(iter_durations[-1])
    return iter_durations[:niter]


def plot_time_comparison(outputs, network_outputs=None, real_time=False):
    clrs = list(plt.cm.tab10.colors)
    clrs[3], clrs[0] = clrs[0], clrs[3]
    clrs[-1], clrs[2] = clrs[2], clrs[-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    fontsize = 24
    fontsize_small = 18

    if real_time:
        x_string = "Elapsed time [s]"
    else:
        x_string = "Evaluation count [#]"
    ax.set_xlabel(x_string, fontsize=fontsize)
    ax.set_ylabel("Mean $\\mathcal{L}_l$", fontsize=fontsize)
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=fontsize_small)
    ax.tick_params(axis='y', labelsize=fontsize_small)
    
    l = outputs[next(iter(outputs))][0].shape[1]  # number of steps

    if real_time:
        max_time = 0.
        for _, _, time_vector in outputs.values():
            max_time = max(max_time, time_vector.mean(dim=0).max()).cpu()
        x = [0, max_time]
    else:
        x = range(l)

    # Track global min for y-axis
    global_min = float('inf')

    if network_outputs is not None:
        mean = network_outputs[0].mean()
        std = network_outputs[0].mean(dim=1).std()
        mean_val = mean.cpu().item()
        global_min = min(global_min, mean_val)

        ax.plot(x, [mean.cpu() for i in x], label="Decision Model", color=clrs[0], linestyle=(0, (5, 1)), zorder=4)
        ax.fill_between(x, (mean - std).cpu(), (mean + std).cpu(), alpha=0.25, facecolor=clrs[0])
        ax.scatter([0], mean.cpu(), color=clrs[0], s=100, zorder=5)
    
    
    for i, (key, (value, _, _, time_vector)) in enumerate(outputs.items()):
        # Mean over all features
        tracked = value  # (runs, steps)
    
        # Compute the cumulative minimum for each run
        best_so_far = torch.cummin(tracked, dim=1).values  # shape: (runs, steps)
    
        # Mean and std across runs
        mean = best_so_far.mean(dim=0)  # shape: (steps,)
        std = best_so_far.std(dim=0)
        mean_min = mean.min().cpu().item()
        global_min = min(global_min, mean_min)

        if real_time:
            x = time_vector.mean(dim=0).cpu()
        else:
            x = range(l)
        
        ax.plot(x, mean.cpu(), label=key, color=clrs[i+1], linestyle='solid')
        ax.fill_between(x, (mean - std).cpu(), (mean + std).cpu(), alpha=0.25, facecolor=clrs[i+1])

    # Set lower y-limit based on global_min (ignore std)
    if global_min < float('inf'):
        padding_factor = 0.9  # 10% padding below
        ax.set_ylim(bottom=global_min * padding_factor)

    ax.legend(fontsize=fontsize_small)
    plt.savefig('outputs/time_comparison.pdf', dpi=300, bbox_inches="tight")


def print_time_to_match(outputs, network_outputs):
    for key, (value, _, _, _) in outputs.items():
        compare = value.min(dim=1).values
        cummin, _ = torch.cummin(value, dim=1)
        matching_bool = cummin.cpu() <= network_outputs[0].mean(1).unsqueeze(1).cpu()
        matching_bool_sum = matching_bool.any(dim=1)
        iterations_until_matched = (~matching_bool).sum(dim=1)
        print(key, "& $", matching_bool_sum.sum().item(),'/',len(compare), "$ &", f"${iterations_until_matched.float().mean().item():.2f}" , '\\pm', f"{iterations_until_matched.float().std().item():.2f}$ \\\\")

from scipy.stats import ttest_rel

def print_comparison_table(outputs, network_outputs):
    def format_value(val, min_val, higher_is_better=False):
        s = f"{val:.6f}"
        # Bold the "best" (lowest or highest) value depending on metric direction
        if higher_is_better:
            return f"\\mathbf{{{s}}}" if val == max_val else s
        else:
            return f"\\mathbf{{{s}}}" if val == min_val else s

    def print_line(key, tensor, validity, time,
                   metric_sig=False, validity_sig=False, time_sig=False,
                   min_mean=None, min_validity=None, min_time=None):
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()

        valid_mean = validity.mean().item()
        valid_std = validity.std().item()

        time_mean = time.mean().item()
        time_std = time.std().item()

        mean_str = format_value(mean_val, min_mean)
        valid_str = format_value(valid_mean*100, min_validity)
        time_str = format_value(time_mean, min_time)

        # Add significance markers
        metric_dagger = "\\dagger" if metric_sig else ""
        validity_dagger = "\\dagger" if validity_sig else ""
        time_dagger = "\\dagger" if time_sig else ""

        print(
            f"{key} & "
            f"${mean_str}\\pm{std_val:.4f}{metric_dagger}$ & "
            f"${valid_str}\\pm{valid_std*100:.4f}{validity_dagger}$ & "
            f"${time_str}\\pm{time_std:.4f}{time_dagger}$ \\\\"
        )

    metric_means = []
    validity_means = []
    time_means = []

    # Extract decision model data
    decision_tensor = network_outputs[0].mean(1)
    decision_validity = (~network_outputs[1]).float().mean(dim=-1)
    decision_time = network_outputs[2]

    metric_means.append(decision_tensor.mean().item())
    validity_means.append(decision_validity.mean().item())
    time_means.append(decision_time.mean().item())

    # Store others
    for key, (value, validity, time, _) in outputs.items():
        compare = value.min(dim=1).values
        invalidities = (~validity).float().mean(dim=-1)

        metric_means.append(compare.mean().item())
        validity_means.append(invalidities.mean().item())
        time_means.append(time.mean().item())

    # Find bests for bolding
    min_metric_mean = min(metric_means)
    min_validity_mean = min(validity_means)
    min_time_mean = min(time_means)

    # Print header
    print("\\textbf{Model} & \\textbf{Metric} & \\textbf{Invalid [\\%]} & \\textbf{Time} \\\\")

    # Print Decision Model (baseline)
    print_line(
        'Decision Model', decision_tensor, decision_validity, decision_time,
        min_mean=min_metric_mean, min_validity=min_validity_mean, min_time=min_time_mean
    )

    # Print comparisons
    for key, (value, validity, time, _) in outputs.items():
        compare = value.min(dim=1).values
        invalidities = (~validity).float().mean(dim=-1)
        
        print(decision_tensor.shape, compare.shape)

        # Significance tests
        metric_p = ttest_rel(decision_tensor.cpu(), compare.cpu()).pvalue
        validity_p = ttest_rel(decision_validity.cpu(), invalidities.cpu()).pvalue
        time_p = ttest_rel(decision_time.cpu(), time.cpu()).pvalue

        metric_sig = metric_p <= 0.01
        validity_sig = validity_p <= 0.01
        time_sig = time_p <= 0.01

        print_line(
            key, compare, invalidities, time,
            metric_sig=metric_sig, validity_sig=validity_sig, time_sig=time_sig,
            min_mean=min_metric_mean, min_validity=min_validity_mean, min_time=min_time_mean
        )


def plot_evaluation_accuracy(outputs, network_outputs):
    str_f = "{:.6f}"
    plt.tight_layout()
    fig = plt.figure(figsize = (28,5))
    plt.rcParams.update({'font.size': 20})
    ax_list = []
    
    for i, (key, (value, _, _, _)) in enumerate(outputs.items()):
        ax = fig.add_subplot(1,len(outputs),i+1)
        ax_list.append(ax)
        ax.set_title(key)
        ax.set_xlabel("Decision Model")
        if i == 1:
            ax.set_ylabel("Optimal reward [arb.u.]")
        lower_limit = 0.03 #2e-1
        upper_limit = 9e-3
        bins = torch.logspace(torch.log10(torch.tensor(upper_limit)), torch.log10(torch.tensor(lower_limit)), 50)
        hist = ax.hist2d(network_outputs[0].mean(1).cpu(), value.cpu()[:,-1], bins = bins.cpu(), vmin = 0, vmax = 25, cmap='hot')
        ax.plot([lower_limit, upper_limit], [lower_limit, upper_limit], 'tab:cyan')
        if i+1 != 1:
            ax.axes.get_yaxis().set_visible(False)
        ax.set_xscale('symlog')
        ax.set_xlim((lower_limit,upper_limit,))
        ax.set_yscale('symlog')
        ax.set_ylim((lower_limit,upper_limit,))
    fig.colorbar(hist[3], ax=ax_list, label="Count [#]")
    plt.savefig('outputs/linear_int_rew_comp.pdf',dpi=300, bbox_inches = "tight")

def plot_comparison_scatter(outputs, network_outputs=None):
    keys = []
    times = []
    mses = []
    fontsize=13
    if network_outputs is not None:
        # Baseline: Deep Learning model
        dl_mse = network_outputs[0].mean(1)
        dl_time = network_outputs[2].mean()
        keys.append("Decision Model")
        times.append(dl_time.item())
        mses.append(dl_mse.mean().item())

    # Others
    for key, (value, _, time, _) in outputs.items():
        compare = value.min(dim=1).values
        mean_mse = compare.mean().item()
        mean_time = time.mean().item()
        keys.append(key)
        times.append(mean_time)
        mses.append(mean_mse)

    # Compute axis limits with +10% padding
    time_min, time_max = min(times), max(times)
    mse_min, mse_max = min(mses), max(mses)

    time_padding = 0.1 * (time_max - time_min) if time_max > time_min else 0.1
    mse_padding = 0.1 * (mse_max - mse_min) if mse_max > mse_min else 0.1

    # Plot
    plt.figure(figsize=(8, 4.7))
    plt.scatter(times, mses, s=50)

    for i, label in enumerate(keys):
        plt.annotate(label.replace(' ', '\n'), (times[i], mses[i]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=fontsize)

    plt.xlim(time_min - time_padding, time_max + time_padding*2)
    plt.ylim(mse_min - mse_padding, mse_max + mse_padding*1.9)

    plt.xlabel("Mean Evaluation Time [s]", fontsize=fontsize)
    plt.gcf().text(0.223, 0.045, r"$\leftarrow$ Lower is better", ha='center', c="dimgray", fontsize=fontsize)
    plt.ylabel(r"Mean $\mathcal{L}_l$", fontsize=fontsize)
    plt.gcf().text(0.02, 0.28, r"$\leftarrow$ Lower is better", va='center', rotation=90, c="dimgray", fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/comparison_scatter.pdf',dpi=300, bbox_inches = "tight")
    
def plot_attribute(model, attribute_index = 5):
    ds = RandomIterableDataset(100000, 8, 10000000, model.device)
    z = torch.stack([element for element in ds]).reshape(500, -1, 8)
     
    attribute_index = attribute_index
    l = torch.linspace(0.,1., z.shape[0], device=model.device)
    
    # replace all random values from attribute_index with linspace
    for i in range(z.shape[1]):
        z[:, i, attribute_index] = l
    
    with torch.no_grad():
        y = model(z)
    
    fig, ax = plt.subplots(figsize=(10,6))
    clrs = list(plt.cm.tab10.colors)
    for i in range(y.shape[2]):
        sub_y = y[:,:,i].cpu()
        var = sub_y.var(dim = 1)
        mean = sub_y.mean(dim = 1)
        ax.plot(l.cpu(), mean, label = get_labels('action')[i], c=clrs[i])
        ax.fill_between(l.cpu(), mean-var, mean+var,alpha=0.25, facecolor=clrs[i])
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.xlabel(get_labels('obs')[attribute_index]+ " [normalized]", fontsize=20)
    plt.ylabel("Action value [normalized]", fontsize=20)
    label = get_labels('obs')[attribute_index].replace(" ", "_")
    std_str = str(ds.stddev).replace(".", "_")
    plt.savefig(f"outputs/{label}{std_str}.pdf")

def get_labels(category:str):
    labels = ['Laser pulse length', 'Laser spot size', 'Gun peak field', 'Gun DC bias field', 'Field flatness', 'Laser horizontal position', 'Laser vertical position', 'Solenoid horizontal position', 'Solenoid vertical position', 'Solenoid angle y-axis', 'Solenoid angle x-axis', 'Emission phase', 'Solenoid strength', 'Cathode position', 'Average horizontal beam size', 'Average vertical beam size', 'Horizontal beam position', 'Vertical beam position', 'Average beam momentum']
    state_labels = labels[:7] + labels[11:14]
    if category == "obs":
        return state_labels[:7] + [state_labels[9]]
    if category == "action":
        return labels[7:11]
    if category == "target":
        return labels[14:]
    else:
        raise Exception("Category not found.")
    
def jac_std_avg(model, stddev=.2):
    x = torch.empty((1000, 8), device=model.device)
    torch.nn.init.trunc_normal_(x, mean=0.5, std=stddev, a=-0.5/stddev, b=0.5/stddev)
    
    
    
    # Ensure policy is in eval mode and x requires grad
    model.eval()
    x.requires_grad_(True)
    
    # Define a single-sample function
    def single_policy(xi):
        return model(xi.unsqueeze(0)).squeeze(0)  # xi: (8,) -> output: (4,)
    
    # Use vmap to compute jacobian for each sample: shape (1000, 4, 8)
    jacobian = vmap(jacrev(single_policy))(x)
    
    jac = jacobian.detach().cpu()  # shape: (1000, 4, 8)
    
    # Plot mean of Jacobian
    plt.figure(figsize=(10, 6))
    im = plt.imshow(jac.mean(0).cpu(), cmap='hot')
    plt.xticks(range(8), get_labels("obs"), rotation=45, ha='right')
    plt.yticks(range(4), get_labels("action"))
    plt.colorbar(im, label='Count [#]')
    plt.tight_layout()
    plt.savefig(f'outputs/jac_avg_{stddev}.pdf', dpi=300, bbox_inches="tight")
    
    # Plot std of Jacobian
    plt.figure(figsize=(10, 6))
    im = plt.imshow(jac.std(0).cpu(), cmap='hot')
    plt.xticks(range(8), get_labels("obs"), rotation=45, ha='right')
    plt.yticks(range(4), get_labels("action"))
    plt.colorbar(im, label='Count [#]')
    plt.tight_layout()
    plt.savefig(f'outputs/jac_std_{stddev}.pdf', dpi=300, bbox_inches="tight")

def load_critic(device):
    return Critic(device=device)

def load_model_critic_net(device, path='zez828tm'):
    critic_net = load_critic(device)
    path = get_checkpoint_path(path)
    model = RandomModel.load_from_checkpoint(path, critic_net=critic_net,  map_location=device).to(device)
    model.eval()
    return model, critic_net

def evaluation(repetitions=1000, niter=100, device=torch.device('cuda')):
    outputs_list = []
    network_outputs_list = []
    network_times_list = []
    model, critic_net = load_model_critic_net(device)
    ds = RandomIterableDataset(repetitions, 8, 60000000, device)

    for seed, state in enumerate(tqdm(ds, total=repetitions)):
        state = state.unsqueeze(0)

        if state.device.type == "cuda":
            warmup_gpu(state.device)
    
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
    
        with torch.no_grad():
            policy_action = model(state)

        if state.device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time
            
        network_outputs_list.append(critic_net(policy_action, state, eval_mode=True))
        network_times_list.append(elapsed_time)
        
        outputs = {
            "Powell’s Method": eval_scipy(state, niter, seed=seed, eval_mode=True),
            "SA": eval_sa(state, niter, seed=seed, eval_mode=True),
            "GD": eval_gd(state, niter, seed=seed, eval_mode=True),
            "GA": eval_ga(state, niter, seed=seed, eval_mode=True, sbx_crossover_rate=0.3),
            "BLOP": eval_blop(state, niter, seed=seed, eval_mode=True, ucb_beta=5)
        }
        outputs_list.append(outputs)
    outputs = {}
    for key in outputs_list[0]:
        outputs[key] = torch.stack([entry[key][0][0] for entry in outputs_list], dim=0), torch.stack([entry[key][0][1] for entry in outputs_list], dim=0), torch.tensor([entry[key][1] for entry in outputs_list], device=device), torch.tensor([entry[key][2] for entry in outputs_list], device=device)
    network_outputs = torch.stack([entry[0] for entry in network_outputs_list]).squeeze(1), torch.stack([entry[1] for entry in network_outputs_list]).squeeze(1), torch.tensor(network_times_list, device=device)

    with open("outputs/eval_dict.pkl", "wb") as f:
        pickle.dump(outputs, f)

    with open("outputs/network_eval_dict.pkl", "wb") as f:
        pickle.dump(network_outputs, f)

    return outputs, network_outputs, model, critic_net

if __name__ == "__main__":
    outputs, network_outputs, model, critic_net = evaluation(repetitions=1000, niter=150)
    
    plot_time_comparison(outputs, network_outputs)

    plot_evaluation_accuracy(outputs, network_outputs)
    
    plot_comparison_scatter(outputs, network_outputs)

    print_time_to_match(outputs, network_outputs)

    print_comparison_table(outputs, network_outputs)

    jac_std_avg(model)

    benchmark_model(critic_net.model, 14)

    benchmark_model(critic_net.model, 14, samples=100000)

    plot_attribute(model)

