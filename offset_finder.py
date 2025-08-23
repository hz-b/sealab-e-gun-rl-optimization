from surrogate import BerlinPro2
import torch
from critic import Critic
from evotorch.algorithms import SNES, CMAES, GeneticAlgorithm
from evotorch import Problem
from evotorch.operators import OnePointCrossOver, MultiPointCrossOver, GaussianMutation, SimulatedBinaryCrossOver
import logging
from tqdm.auto import tqdm
from simulation import sim_Y_labels, simulation_parallel


def generate_configurations(model, validity_classifier, max_offset=0.2, offset_count=100, seed=42):
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
    within_border = (abs(model.dataset.un_z_score_y(model_output)) <30).all(-1).squeeze(-1)  # shape: (N,)
    
    # Step 3: Use classifier on already range-valid configs
    with torch.no_grad():
        validity_scores = validity_classifier(filtered_parameters)  # shape: (N, 1)
    validity = (validity_scores > 0.5).squeeze(-1)  # shape: (N,)
    
    # Step 4: Apply second filter to parameters and offsets
    final_target_parameters = filtered_parameters[validity&within_border][:1]  # shape: (M, 14)
    final_offsets = filtered_offsets[validity&within_border][:1]        # shape: (M, 14)


    # Step 5: Predict with the model
    with torch.no_grad():
        experiment_output = model(final_target_parameters)
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
            y = model.dataset.un_z_score_y(compensated_rays)
            if fine_model is not None:
                limit_y_mask = (abs(y[:, :4]) < 30).all(dim=1)
                #print(y[limit_y_mask].shape)
                compensated_rays = compensated_rays.clone()
                fine_model_outputs = fine_model(tensor_sum[limit_y_mask])
                rescored_fine_model_outputs = model.dataset.z_score_y(fine_model.dataset.un_z_score_y(fine_model_outputs))
                compensated_rays[limit_y_mask] = rescored_fine_model_outputs
            loss_orig = ((compensated_rays - observed_experiment) ** 2).mean(-1)#.mean(0).mean(0).mean(-1) #cus_loss(compensated_rays, observed_rays) #
            #print(loss_orig.shape)
        return loss_orig
    
    def constrained_loss(x: torch.Tensor) -> torch.Tensor:
        penalty = torch.sum(((x + uncompensated_parameters) < 0.) | ((x + uncompensated_parameters) > 1.0), dim=-1).float()
        total_loss = loss(x) + 1.0 * penalty  # Add large penalty for violations

        with torch.no_grad():
            validity_scores = validity_classifier(x + uncompensated_parameters)  # shape: (N, 1)
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
    model.prepare_data()
    model.dataset.change_z_score_device(model.device)
    fine_model.prepare_data()
    fine_model.dataset.change_z_score_device(model.device)
    
    un_z_score_y = model.dataset.un_z_score_y
    z_score_y = model.dataset.z_score_y
    
    z_score = model.dataset.z_score
    un_z_score = model.dataset.un_z_score
    
    fine_un_z_score_y = fine_model.dataset.un_z_score_y
    fine_z_score_y = fine_model.dataset.z_score_y
    fine_z_score = fine_model.dataset.z_score
    un_z_scored_target_parameters_list = []
    un_z_scored_compensated_parameters_list = []
    
    for i in range(sample_count):
        uncompensated_parameters, offsets, observed_experiment = generate_configurations(model, validity_classifier, offset_count=1000, seed=i+1)
        compensated_parameters, best_loss, _ = optimize_evotorch_ga(model, validity_classifier, observed_experiment, uncompensated_parameters, fine_model=fine_model, iterations=3000, num_candidates=10000)
        un_z_scored_target_parameters_list.append(un_z_score(uncompensated_parameters + offsets).squeeze(0))
        un_z_scored_compensated_parameters_list.append(un_z_score(compensated_parameters))
    un_z_scored_target_parameters = torch.stack(un_z_scored_target_parameters_list)
    un_z_scored_compensated_parameters = torch.stack(un_z_scored_compensated_parameters_list)
    
    simulation_input = torch.cat((un_z_scored_target_parameters, un_z_scored_compensated_parameters))
    simulation_output = simulation_parallel(simulation_input.cpu())
    simulated_target = simulation_output[:sample_count]
    simulated_compensated = simulation_output[sample_count:]
    
    mask = torch.isnan(simulated_target).any(dim=1) & torch.isnan(simulated_compensated).any(dim=1)
    simulated_target = simulated_target[~mask]
    simulated_compensated = simulated_compensated[~mask]
    
    rmse = ((simulated_target - simulated_compensated) ** 2).mean(dim=0).sqrt()
    #print("RMSE", rmse, "NaNs", (mask).sum().item())
    nan_count = mask.sum().item()
    return rmse, nan_count

if __name__ == "__main__":
    fine_model_path = "outputs/berlinpro_surrogate/berlinpro_surrogate/zmz50ufb/checkpoints/epoch=9999-step=2530000.ckpt"
    critic = Critic(surrogate="outputs/berlinpro_surrogate/berlinpro_surrogate/jlp0mkw3/checkpoints/epoch=9999-step=3650000.ckpt", fine_surrogate=fine_model_path)
    
    validity_classifier = critic.validity_classifier
    fine_model = critic.fine_surrogate
    model = critic.model
    
    rmse, nan_count = rmse_simulated_target_compensated(model, fine_model, validity_classifier, sample_count=100)
    print("RMSE:", rmse, "NaN#", nan_count)
