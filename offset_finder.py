from surrogate import BerlinPro2
import torch
from critic import Critic
from evotorch.algorithms import SNES, CMAES, GeneticAlgorithm
from evotorch import Problem
from evotorch.operators import OnePointCrossOver, MultiPointCrossOver, GaussianMutation, SimulatedBinaryCrossOver
import logging
from tqdm.auto import tqdm
from simulation import sim_Y_labels, simulation_parallel


def generate_configurations(model, fine_model, validity_classifier, max_offset=0.2, offset_count=100, seed=42):
    torch.manual_seed(seed)
    # Generate configurations
    uncompensated_parameters = torch.rand((14), device=model.device)  # shape: (14,)
    offsets = torch.rand((offset_count, 14), device=model.device) * 2 * max_offset - max_offset
    target_parameters = offsets + uncompensated_parameters  # shape: (100, 14)
    
    # Step 1: Filter within [0, 1]
    valid_mask = (target_parameters >= 0) & (target_parameters <= 1)
    valid_rows = valid_mask.all(dim=-1)  # shape: (100,)
    
    # Step 2: Apply first filter to parameters and offsets
    filtered_parameters = target_parameters[valid_rows]  # shape: (N, 14)
    filtered_offsets = offsets[valid_rows]  # shape: (N, 14) ← these are the ones you want to keep

    with torch.no_grad():
        model_output = model(filtered_parameters)  # shape: (N, 1)
    within_border = (abs(model.normalizer.unscore_y(model_output)) <30).all(-1).squeeze(-1)  # shape: (N,)
    
    # Step 3: Use classifier on already range-valid configs
    with torch.no_grad():
        validity_scores = validity_classifier(model.normalizer.unscore_x(filtered_parameters))  # shape: (N, 1)
    validity = (validity_scores > 0.5).squeeze(-1)  # shape: (N,)
    
    # Step 4: Apply second filter to parameters and offsets
    final_target_parameters = filtered_parameters[validity&within_border][:1]  # shape: (M, 14)
    final_offsets = filtered_offsets[validity&within_border][:1]        # shape: (M, 14)


    # Step 5: Predict with the model
    with torch.no_grad():
        experiment_output = fine_model(final_target_parameters)
    if final_offsets.shape[0] == 0:
        raise Exception("No fitting offset found.")
    return uncompensated_parameters, final_offsets, experiment_output

def optimize_evotorch_ga(
    model, validity_classifier, observed_experiment, uncompensated_parameters, fine_model=None, iterations=1000, 
    num_candidates=100, tournament_size=1, mutation_rate=0.05, mutation_scale=0.05, crossover_rate=0.5, eta=None, crossover_points=10, seed=42
    ):
    torch.manual_seed(seed)
    # Define the loss function
    def loss(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tensor_sum = x + uncompensated_parameters ###model.rescale_offset(x) → model
            compensated_rays = model(tensor_sum)
            y = model.normalizer.unscore_y(compensated_rays)
            if fine_model is not None:
                limit_y_mask = (abs(y[:, :4]) < 30).all(dim=1)
                compensated_rays = compensated_rays.clone()
                fine_model_outputs = fine_model(tensor_sum[limit_y_mask])
                rescored_fine_model_outputs = model.normalizer.score_y(fine_model.normalizer.unscore_y(fine_model_outputs))
                compensated_rays[limit_y_mask] = rescored_fine_model_outputs
            loss_orig = ((compensated_rays - observed_experiment) ** 2).mean(-1)#.mean(0).mean(0).mean(-1) #cus_loss(compensated_rays, observed_rays) #
        return loss_orig
    
    def constrained_loss(x: torch.Tensor) -> torch.Tensor:
        penalty = torch.sum(((x + uncompensated_parameters) < 0.) | ((x + uncompensated_parameters) > 1.0), dim=-1).float()
        total_loss = loss(x) + 1.0 * penalty  # Add large penalty for violations

        with torch.no_grad():
            validity_scores = validity_classifier(model.normalizer.unscore_x(x + uncompensated_parameters))  # shape: (N, 1)
            validity = (validity_scores > 0.5).squeeze(-1)  # shape: (N,)
            total_loss = total_loss + 1.0 * ~validity
    
        return total_loss
    logging.getLogger("evotorch").setLevel(logging.WARNING)
    
    # Define the problem
    problem = Problem(
        "min",
        constrained_loss,
        initial_bounds=(0., 1.),
        solution_length=14,###model.mutable_parameter_count,
        vectorized=True,
        device="cuda" if torch.cuda.is_available() else "cpu",  # Enable for GPU support
    )

    operators=[]
    if eta==None:
        operators.append(MultiPointCrossOver(problem, tournament_size=tournament_size, cross_over_rate=crossover_rate, num_points=crossover_points))
    else:
        operators.append(SimulatedBinaryCrossOver(problem, tournament_size=tournament_size, eta=eta, cross_over_rate=crossover_rate))
    
    operators.append(GaussianMutation(problem, stdev=mutation_scale, mutation_probability=mutation_rate))

    # Create the searcher
    searcher = GeneticAlgorithm(
    problem,
    popsize=num_candidates,
    operators = operators
    )

    # Set up tqdm for progress tracking
    num_generations = iterations
    loss_history = []
    with tqdm(total=num_generations, desc="Evotorch Progress", leave=False) as pbar:
        for generation in range(num_generations):
            searcher.step()  # Perform a single optimization step
            pbar.update(1)  # Update the progress bar
            best = searcher.status["pop_best"]
            best_item = best.evals.item()
            pbar.set_postfix({"loss": best_item})
            loss_history.append(best_item)
    best_loss = best.evals.item()
    loss_min_params = best.values + uncompensated_parameters
    return loss_min_params.squeeze(-1), best_loss, loss_history

def rmse_simulated_target_compensated(model, fine_model, validity_classifier, sample_count=1):
    unscore_y = model.normalizer.unscore_y
    score_y = model.normalizer.score_y
    
    score_x = model.normalizer.score_x
    unscore_x = model.normalizer.unscore_x
    
    fine_score_y = fine_model.normalizer.unscore_y
    fine_score_y = fine_model.normalizer.score_y
    fine_score_x = fine_model.normalizer.score_x
    unscored_target_parameters_list = []
    unscored_compensated_parameters_list = []
    
    for i in range(sample_count):
        uncompensated_parameters, offsets, observed_experiment = generate_configurations(model, validity_classifier, offset_count=1000, seed=i+1)
        compensated_parameters, best_loss, _ = optimize_evotorch_ga(model, validity_classifier, observed_experiment, uncompensated_parameters, fine_model=fine_model, iterations=3000, num_candidates=10000)
        unscored_target_parameters_list.append(unscore_x(uncompensated_parameters + offsets).squeeze(0))
        unscored_compensated_parameters_list.append(unscore_x(compensated_parameters))
    unscored_target_parameters = torch.stack(unscored_target_parameters_list)
    unscored_compensated_parameters = torch.stack(unscored_compensated_parameters_list)
    
    simulation_input = torch.cat((unscored_target_parameters, unscored_compensated_parameters))
    simulation_output = simulation_parallel(simulation_input.cpu())
    simulated_target = simulation_output[:sample_count]
    simulated_compensated = simulation_output[sample_count:]
    
    mask = torch.isnan(simulated_target).any(dim=1) | torch.isnan(simulated_compensated).any(dim=1)
    simulated_target = simulated_target[~mask]
    simulated_compensated = simulated_compensated[~mask]
    
    rmse = ((simulated_target - simulated_compensated) ** 2).mean(dim=0).sqrt()
    std = ((simulated_target - simulated_compensated) ** 2).std(dim=0).sqrt()
    #print("RMSE", rmse, "NaNs", (mask).sum().item())
    nan_count = mask.sum().item()
    return rmse, std, nan_count

def simulation_like(model, x):
    #return model(x)
    model.normalizer.to(model.device)
    simulation_result = simulation_parallel(model.normalizer.unscore_x(x.cuda()).cpu())[:, :4].to(model.device)
    return model.normalizer.score_y(simulation_result)

def compare_label_surrogate_simulation(model, x, y=None, label="sur_sim"):
    with torch.no_grad():
        if y is not None:
            target = model.normalizer.unscore_y(y.cuda())
            torch.save(target, "outputs/label_"+label+".pt")
            print("Label says", target)
        surrogate = model.normalizer.unscore_y(model(x.cuda()))
        torch.save(surrogate, "outputs/surrogate_"+label+".pt")
        print("Surrogate says", surrogate)
        simulation_result = model.normalizer.unscore_y(simulation_like(model, x.cuda()))
        torch.save(simulation_result, "outputs/simulation_"+label+".pt")
        print("Simulation says", simulation_result)
        return simulation_result

def generate_stacked_configurations(model, fine_model, validity_classifier, offset_count=100000, num_iterations=100, **args):
    sets_list = []
    experiment_output_list = []
    uncompensated_parameters_list = []
    final_offsets_list = []
    
    # Loop for a certain number of iterations
    for i in range(num_iterations):
        # Generate configurations (replace with your actual function)
        uncompensated_parameters, final_offsets, experiment_output = generate_configurations(
            model, fine_model, validity_classifier, offset_count=offset_count, seed=i, **args)
    
        # Stack the outputs in the lists
        uncompensated_parameters_list.append(uncompensated_parameters)
        final_offsets_list.append(final_offsets[0])
        experiment_output_list.append(experiment_output[0])
    
    # After the loop, convert the lists to tensors and stack them
    uncompensated_parameters_tensor = torch.stack(uncompensated_parameters_list)
    final_offsets_tensor = torch.stack(final_offsets_list)
    experiment_output_tensor = torch.stack(experiment_output_list)

    return uncompensated_parameters_tensor, final_offsets_tensor, experiment_output_tensor

class JointModel():
    def __init__(self, model, fine_model=None, validity_classifier=None):
        super().__init__()
        self.model = model
        self.fine_model=fine_model
        self.validity_classifier=validity_classifier
        self.model.normalizer.to(model.device)
        self.fine_model.normalizer.to(fine_model.device)
        
    def __call__(self, sample, clone=False):
        model_score_sample = self.model.normalizer.score_x(sample)
        with torch.no_grad():
            model_score_output = self.model(model_score_sample)

        output = self.model.normalizer.unscore_y(model_score_output)
        
        if clone and (self.fine_model is not None or self.validity_classifier is not None):
            output = output.clone()

        if self.fine_model is not None:
            limit_y_mask = (abs(output) < 30).all(dim=1)

            with torch.no_grad():
                fine_model_outputs = self.fine_model(self.fine_model.normalizer.score_x(sample[limit_y_mask]))
            output[limit_y_mask] = fine_model.normalizer.unscore_y(fine_model_outputs)
            
        if self.validity_classifier is not None:
            with torch.no_grad():
                validity_scores = self.validity_classifier(sample)
            validity = (validity_scores > 0.5).squeeze(-1)
            output[~validity] = torch.nan
        
        return output

if __name__ == "__main__":
    critic = Critic()
    validity_classifier = critic.validity_classifier
    fine_model = critic.fine_surrogate
    model = critic.model
    torch.manual_seed(142)

    jm = JointModel(model=model, fine_model=fine_model, validity_classifier=validity_classifier)
    sample = jm.model.normalizer.unscore_x(torch.rand((100, 14), device=model.device))

    print(jm(sample))
    sim_output = simulation_parallel(sample.cpu())[:, :4]
    print(sim_output)
    torch.save(sim_output, "outputs/simulation_simple_test.pt")
    exit(0)
    
    #rmse, std, nan_count = rmse_simulated_target_compensated(critic.model, critic.fine_surrogate, critic.validity_classifier, sample_count=2)
    #print("RMSE:", rmse, "Std:", std, "NaN#", nan_count)
    uncompensated_parameters, final_offsets, experiment_output = generate_stacked_configurations(model, fine_model, validity_classifier, num_iterations=200, offset_count=100000)
    experiment_simulation_output = compare_label_surrogate_simulation(fine_model, uncompensated_parameters+final_offsets, experiment_output, label="blueprint")
    scored_experiment_simulation_output = model.normalizer.score_y(experiment_simulation_output)
    mask = ~torch.isnan(experiment_simulation_output).any(dim=1) & ~(torch.abs(experiment_simulation_output)>30).any(dim=1)

    loss_min_params_list = []
    for i, entry in enumerate(scored_experiment_simulation_output[mask]):
        loss_min_params, _, _ = optimize_evotorch_ga(model, validity_classifier, entry, uncompensated_parameters[mask][i], fine_model)
        loss_min_params_list.append(loss_min_params)
    loss_min_tensor = torch.stack(loss_min_params_list)
    
    fine_scored_experiment_simulation_output = fine_model.normalizer.score_y(experiment_simulation_output[mask])
    compare_label_surrogate_simulation(fine_model, loss_min_tensor, fine_scored_experiment_simulation_output, label="result")