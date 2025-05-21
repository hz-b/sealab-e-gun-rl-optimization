import logging
import pickle
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap, jacrev

import matplotlib.pyplot as plt
import optuna

from critic import Critic
from model import RandomModel, RandomIterableDataset

from scipy.optimize import minimize, dual_annealing
from scipy.stats import ttest_rel

from evotorch import Problem
from evotorch.algorithms import SteadyStateGA, SNES
from evotorch.operators import (
    SimulatedBinaryCrossOver,
    GaussianMutation,
)
from evotorch.logging import StdOutLogger


def eval_optuna(state, n_trials=100):
    def eval_critic_solution(solution, state, critic_net):
        # solution: numpy array of shape (4,)
        solution_tensor = torch.tensor(solution, dtype=torch.float32, device=state.device).unsqueeze(0)
        init_problem = state.repeat(1, 1)
        output = critic_net(solution_tensor, init_problem)  # shape: (1, 3)
        return output

    eval_history = []
    def optuna_objective(trial, state):
        # Sample 4 parameters between 0.0 and 1.0
        solution = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(4)]
        
        critic_net = Critic(device=state.device)
        solution = eval_critic_solution(solution, state, critic_net)
        eval_history.append(solution)
        return solution.mean().item()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective(trial, state), n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_solution = torch.tensor([best_params[f"x{i}"] for i in range(4)], device=state.device)
    return torch.stack(eval_history).squeeze(1)


def eval_evotorch(state, niter):
    logging.getLogger("evotorch").setLevel(logging.WARNING)
    critic_net = Critic(device=state.device)
    popsize=200
    init_problem = state.repeat_interleave(popsize, dim=0)
    #print(init_problem.shape)
    optimization_values = []
    def critic_problem(x):
        output = critic_net(x, init_problem)
        optimization_values.append(output)
        return output
                                          
    prob = Problem(
        # Two objectives, both minimization
        ["min", "min", "min"],
        critic_problem,
        initial_bounds=(0.0, 1.0),
        solution_length=4,
        vectorized=True,
        device=state.device
    )
    
    # Works like NSGA-II for multiple objectives
    ga = SteadyStateGA(prob, popsize=popsize)
    ga.use(
        SimulatedBinaryCrossOver(
            prob,
            tournament_size=2,
            cross_over_rate=1.0,
            eta=8,
        )
    )
    ga.use(GaussianMutation(prob, stdev=0.03))
    
    ga.run(niter//2)
    output = torch.stack(optimization_values)

    best_indices = output.mean(dim=-1).argmin(dim=1)
    return output[torch.arange(niter), best_indices]

def eval_evotorch_single(state, niter):
    logging.getLogger("evotorch").setLevel(logging.WARNING)

    critic_net = Critic(device=state.device)
    popsize = 200
    init_problem = state.repeat_interleave(popsize, dim=0)

    optimization_values = []

    def critic_problem(x):
        output = critic_net(x, init_problem)  # shape: (batch, 3)
        scalar = output.mean(dim=1)           # shape: (batch,) — single objective
        optimization_values.append(output)
        return scalar

    problem = Problem(
        "min",  # Single-objective
        critic_problem,
        initial_bounds=(0.0, 1.0),
        solution_length=4,
        vectorized=True,
        device=state.device
    )

    searcher = SNES(problem, popsize=popsize, stdev_init=10.0)

    searcher.run(num_generations=niter)

    output = torch.stack(optimization_values)  # shape: (niter, popsize, 3)

    # Instead of selecting best post-hoc, just return all
    best_indices = output.mean(dim=-1).argmin(dim=1)
    return output[torch.arange(niter), best_indices]

def eval_scipy_annealing(state, niter, visit=2.62, accept=-5):
    device = state.device
    critic_net = Critic(device=device)
    
    optimization_values = []

    def y_const(x):
        value = critic_net(torch.tensor(x, device=device, dtype=torch.float).view(1, -1), state)
        optimization_values.append(value)
        return value.mean().item()
    
    bounds = [(0., 1.), (0., 1.), (0., 1.), (0., 1.)]
    
    # Run dual annealing
    res = dual_annealing(
        func=y_const,
        bounds=bounds,
        maxiter=niter,
        maxfun=niter,
        visit=visit,
        accept=accept,
        no_local_search=True  # disables local search to perform normal simulated annealing
    )
    
    # Pad values if fewer than niter
    while len(optimization_values) < niter:
        optimization_values.append(optimization_values[-1])
    
    return torch.stack(optimization_values[:niter]).squeeze(1)


def eval_scipy(method, state, niter, device=torch.device('cpu')):
    state = state.to(device)
    critic_net = Critic(device=device)
    initial_action = torch.rand((4), device=device)
    
    optimization_values = []
    def y_const(x):
        value = critic_net(torch.tensor(x, device=device, dtype=torch.float).view(1, -1), state)
        optimization_values.append(value)
        return value.mean().item()

    res = minimize(
        fun = y_const,
        x0 = initial_action,
        method = method,
        bounds = [(0., 1.), (0., 1.), (0., 1.), (0., 1.)],
        options = {'maxiter': niter, "disp": False}
    )
    while len(optimization_values) < niter:
        optimization_values.append(optimization_values[-1])
    return torch.stack(optimization_values[:niter]).squeeze(1)

def eval_torch_sgd(state, niter, device=torch.device('cpu')):
    state = state.to(device)
    critic_net = Critic(device=device)
    action = torch.rand((1, 4), device=device, requires_grad=True)  # Needs grad to be optimized
    lr = 0.1

    optimizer = optim.SGD([action], lr=lr)
    optimization_values = []

    for _ in range(niter):
        optimizer.zero_grad()
        value = critic_net(action.view(1, -1), state)
        loss = value.mean()
        optimization_values.append(value.detach())
        loss.backward()
        optimizer.step()

    return torch.stack(optimization_values).squeeze(1)

def plot_time_comparison(output_data, policy_value):
    clrs = list(plt.cm.tab10.colors)
    clrs[3], clrs[0] = clrs[0], clrs[3]
    clrs[-1], clrs[2] = clrs[2], clrs[-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    fontsize = 24
    fontsize_small = 18
    ax.set_xlabel("Evaluation count [#]", fontsize=fontsize)
    ax.set_ylabel("Reward [arb.u.]", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize_small)
    ax.tick_params(axis='y', labelsize=fontsize_small)
    
    l = output_data[next(iter(output_data))].shape[1]  # number of steps
    
    ax.plot(range(l), [policy_value for i in range(l)], label="Deep Learning", color=clrs[0], linestyle=(0, (5, 1)))
    
    for i, (key, value) in enumerate(output_data.items()):
        # Mean over all features
        tracked = value.mean(-1)  # (runs, steps)
    
        # Compute the cumulative minimum for each run
        best_so_far = torch.cummin(tracked, dim=1).values  # shape: (runs, steps)
    
        # Mean and std across runs
        mean = best_so_far.mean(dim=0)  # shape: (steps,)
        std = best_so_far.std(dim=0)
    
        ax.plot(range(l), mean.cpu(), label=key, color=clrs[i+1], linestyle='solid')
        ax.fill_between(range(l), (mean - std).cpu(), (mean + std).cpu(), alpha=0.25, facecolor=clrs[i+1])
    
    ax.plot([-1, l], [0, 0], color='lightgrey', lw=2, linestyle='dotted', alpha=0.7)
    ax.scatter([0], policy_value, color=clrs[0], s=100, zorder=5)
    ax.legend(fontsize=fontsize_small)
    plt.savefig('outputs/time_comparison.pdf', dpi=300, bbox_inches="tight")

def print_time_to_match(outputs, network_outputs):
    for key, value in outputs.items():
        compare = value.mean(2).min(dim=1).values
        cummin, _ = torch.cummin(value.mean(2), dim=1)
        matching_bool = cummin <= network_outputs.mean(1).unsqueeze(1)
        matching_bool_sum = matching_bool.sum(dim=1)
        iterations_until_matched = (~matching_bool).sum(dim=1)
        print(key, "& $", matching_bool_sum.sum().item(),'/',len(compare), "$ &", f"${iterations_until_matched.float().mean().item():.2f}" , '\\pm', f"{iterations_until_matched.float().std().item():.2f}$ \\\\")

def print_comparison_table(outputs, network_outputs):
    def print_line(key, tensor, sig=''):
        mean = f"${tensor.mean():.3f}"#.replace("e-0", "e-").replace("e+0", "e+")
        std = f"{tensor.std():.4f}"#.replace("e-0", "e-").replace("e+0", "e+")
        print(key, "&", f"{mean}\\pm{std}", sig, "$ \\\\")
    
    print_line('Deep Learning', network_outputs)
    
    for key, value in outputs.items():
        compare = value.mean(2).min(dim=1).values
        sig = ''
        result = ttest_rel(network_outputs.mean(1).cpu(), compare.cpu()).pvalue
        if (result<=0.99):
            sig = "\\dagger"
        print_line(key, compare, sig)
        
def plot_evaluation_accuracy(outputs, network_outputs):
    str_f = "{:.6f}"
    plt.tight_layout()
    fig = plt.figure(figsize = (28,5))
    plt.rcParams.update({'font.size': 20})
    ax_list = []
    
    for i, (key, value) in enumerate(outputs.items()):
        ax = fig.add_subplot(1,5,i+1)
        ax_list.append(ax)
        ax.set_title(key)
        ax.set_xlabel("Deep Learning")
        if i == 1:
            ax.set_ylabel("Optimal reward [arb.u.]") 
        bins = torch.logspace(torch.log10(torch.tensor(5e-5)), torch.log10(torch.tensor(1e-1)), 50)
        hist = ax.hist2d(network_outputs.mean(1).cpu(), value.mean(2).cpu()[:,-1], bins = bins.cpu(), vmin = 0, vmax = 25, cmap='hot')
        ax.plot([1e-1, 1e-5], [1e-1, 1e-5], 'tab:cyan')
        if i+1 != 1:
            ax.axes.get_yaxis().set_visible(False)
        ax.set_xscale('symlog', linthresh = 1e-6, subs = range(2,10))
        ax.set_xlim((1e-1,5e-5,))
        ax.set_yscale('symlog', linthresh = 1e-6, subs = range(2,10))
        ax.set_ylim((1e-1,5e-5,))
    fig.colorbar(hist[3], ax=ax_list, label="Count [#]")
    plt.savefig('outputs/linear_int_rew_comp.pdf',dpi=300, bbox_inches = "tight")

def plot_attribute(model, attribute_index = 5):
    ds = RandomIterableDataset(500000, 8, 10000000, model.device)
    z = torch.stack([element for element in ds]).reshape(500, -1, 8)
    
    
    attribute_index = 5
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
    plt.legend()
    plt.xlabel(get_labels('obs')[attribute_index]+ " [normalized]", fontsize=20)
    plt.ylabel("Policy action value [normalized]", fontsize=20)
    plt.savefig("outputs/"+get_labels('obs')[attribute_index] + str(ds.stddev) + '.pdf')

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

def load_model_critic_net(device, path="outputs/berlinpro/pt5s96kz/checkpoints/epoch=24999-step=200000.ckpt"):
    critic_net = Critic(device=device)
    model = RandomModel.load_from_checkpoint(path, critic_net=critic_net,  map_location=device).to(device)
    model.eval()
    return model, critic_net
    
def evaluation(repetitions=1000, niter=100, device=torch.device('cuda')):
    outputs_list = []
    network_outputs_list = []
    model, critic_net = load_model_critic_net(device)
    ds = RandomIterableDataset(repetitions, 8, 10000000, device)
    
    for state in tqdm(ds, total=repetitions):
        state = state.unsqueeze(0)
        outputs = {
            "Powell": eval_scipy("Powell", state, niter),
            "Evotorch": eval_evotorch(state, niter),
            "SNES": eval_evotorch_single(state, niter),
            "SGD": eval_torch_sgd(state, niter),
            "TPE": eval_optuna(state, niter),
            "Simulated Annealing": eval_scipy_annealing(state, niter)
        }
        outputs_list.append(outputs)
        with torch.no_grad():
            policy_action = model(state)
        network_outputs_list.append(critic_net(policy_action, state))
        
    outputs = {}
    for key in outputs_list[0]:
        outputs[key] = torch.stack([entry[key] for entry in outputs_list], dim=0)
    
    network_outputs = torch.stack(network_outputs_list).squeeze(1)

    with open("outputs/eval_dict.pkl", "wb") as f:
        pickle.dump(outputs, f)

    with open("outputs/network_eval_dict.pkl", "wb") as f:
        pickle.dump(network_outputs, f)

    return outputs, network_outputs, model

if __name__ == "__main__":
    outputs, network_outputs, model = evaluation(niter=100)
    
    plot_time_comparison(outputs, network_outputs.mean().item())

    plot_evaluation_accuracy(outputs, network_outputs)

    print_time_to_match(outputs, network_outputs)

    print_comparison_table(outputs, network_outputs)

    jac_std_avg(model)

    plot_attribute(model)

