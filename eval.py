from scipy.optimize import fmin, fmin_powell, basinhopping
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from critic import Critic
from model import RandomModel
import torch.multiprocessing as mp

from scipy.optimize import minimize

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
        options = {'maxiter': niter}
    )
    return optimization_values

def eval_torch_sgd(state, niter, device=torch.device('cpu')):
    state = state.to(device)
    critic_net = Critic(device=device)
    action = torch.randn((1, 4), device=device, requires_grad=True)  # Needs grad to be optimized
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

    return optimization_values

def eval_opt(seed):
    device = torch.device('cpu')
    torch.manual_seed(seed)
    state = torch.randn((1,8), device=device)
    niter = 100
    result_dict = {
            "SGD": eval_torch_sgd(state, niter),
            "Powell's": eval_scipy("Powell", state, niter),
    }
    return result_dict

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
    
        ax.plot(range(l), mean, label=key, color=clrs[i+1], linestyle='solid')
        ax.fill_between(range(l), mean - std, mean + std, alpha=0.25, facecolor=clrs[i+1])
    
    ax.plot([-1, l], [0, 0], color='lightgrey', lw=2, linestyle='dotted', alpha=0.7)
    ax.scatter([0], policy_value, color=clrs[0], s=100, zorder=5)
    ax.legend(fontsize=fontsize_small)
    plt.savefig('time_comparison.pdf', dpi=300, bbox_inches="tight")
    #plt.show()
    
if __name__ == "__main__":
    device = torch.device('cpu')
    state = torch.randn((1,8), device=device)
    niter = 10
    
    critic_net = Critic(device=device)
    model = RandomModel.load_from_checkpoint("outputs/berlinpro/8ao5e725/checkpoints/epoch=455-step=7296.ckpt", critic_net=critic_net,  map_location=device).to(device)
    model.eval()  # Set to eval mode
    
    with torch.no_grad():
        policy_action = model(state)
    
    policy_value = critic_net(policy_action, state).mean()
    
    reps = 2
    workers = 3
    
    
    mp.set_start_method("spawn")
    
    with mp.Pool(3) as p:
        results = p.map(eval_opt, range(reps))
    
    from collections import defaultdict
    
    buffer = defaultdict(list)
    
    for run in results:
        for method, tensors in run.items():
            step_tensor = torch.cat(tensors, dim=0)  # (steps, features)
            buffer[method].append(step_tensor)       # List[(steps, features)]
    
    output = {}
    for method, runs in buffer.items():
        stacked = torch.stack(runs)         # (num_runs, steps, features)
        output[method] = stacked
    
    # `output` is the dictionary you wanted
    #print(output['SGD'].shape)
    plot_time_comparison(output, policy_value)
