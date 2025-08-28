import torch
import os
from validity_classifier import ValidityClassifier
from surrogate import BerlinPro2, H5Dataset

class Critic:
    def __init__(self, surrogate='outputs/berlinpro_surrogate/berlinpro_surrogate/7w2za3qa/checkpoints/epoch=9999-step=3650000.ckpt', validity_classifier='outputs/berlinpro_validity/berlinpro_validity/r7hmmg07/checkpoints/epoch=49-step=18200.ckpt', fine_surrogate='outputs/berlinpro_surrogate/berlinpro_surrogate/q2hke2pp/checkpoints/epoch=9999-step=2530000.ckpt', device=None, grid_resolution=20):
        self.model = BerlinPro2.load_from_checkpoint(surrogate, map_location=device)
        self.model.freeze()
        self.model.eval()
        if validity_classifier is not None:
            self.validity_classifier = ValidityClassifier.load_from_checkpoint(validity_classifier, map_location=device)
            self.validity_classifier.freeze()
            self.validity_classifier.eval()
            if device is not None:
                self.validity_classifier = self.validity_classifier.to(device)
        if fine_surrogate is not None:
            self.fine_surrogate =  BerlinPro2.load_from_checkpoint(fine_surrogate, map_location=self.model.device)
            self.fine_surrogate.freeze()
            self.fine_surrogate.eval()

        if device is not None:
            self.model = self.model.to(device)
        
        # Precompute the phase/solenoid grid
        self.N2 = grid_resolution ** 2
        grid_phase = torch.linspace(0.1, 0.9, grid_resolution, device=self.model.device)
        grid_solenoid = torch.linspace(0.6, 0.9, grid_resolution, device=self.model.device)
        mesh_phase, mesh_solenoid = torch.meshgrid(grid_phase, grid_solenoid, indexing='ij')
        self.grid_phase_flat = mesh_phase.reshape(-1, 1)        # (N^2, 1)
        self.grid_solenoid_flat = mesh_solenoid.reshape(-1, 1)  # (N^2, 1)
        self.output_min = self.model.normalizer.y_min
        self.output_max = self.model.normalizer.y_max
        self.normalized_zero = -self.output_min / (self.output_max - self.output_min)
   

    def minmax_diff(self, norm):
        diff_min = 0.0
        diff_max = max(norm(self.output_max[0] - self.output_min[1]), norm(self.output_min[0]-self.output_max[1]))
        return diff_min, diff_max
        
    def calculate_reward(self, output_array, norm=torch.abs):
        output_inv = (self.output_max - self.output_min)*output_array + self.output_min  
        diff = norm(output_inv[:,0] - output_inv[:,1])
        
        diff_min, diff_max = self.minmax_diff(norm=norm)
        
        normalized = (diff - diff_min) / (diff_max - diff_min)
        value = torch.vstack([norm(output_array[:,2]-self.normalized_zero[2]), norm(output_array[:,3]-self.normalized_zero[3]), normalized]).T
        return value

    def denormalize_reward(self, reward, norm=torch.abs):
        diff_min, diff_max = self.minmax_diff(norm=norm)
        mins = torch.tensor([self.output_min[2].item(), self.output_min[3].item(), diff_min], device=reward.device)
        maxs = torch.tensor([self.output_max[2].item(), self.output_max[3].item(), diff_max], device=reward.device)
        return (maxs - mins) * reward

    def compute_integrated_reward(self, expanded_actions, expanded_states, norm=torch.abs, penalize_invalid=True, penalize_forbidden_actions=False):
        merged_input = torch.cat([expanded_states, expanded_actions], dim=1)
        output = self.model(merged_input)
        if hasattr(self, "fine_surrogate"):
            limit_y_mask = (abs(self.model.normalizer.unscore_y(output)) <30).all(-1).squeeze(-1)  # shape: (N,)
            #print(limit_y_mask.shape)
            fine_model_outputs = self.fine_surrogate(merged_input[limit_y_mask])
            rescored_fine_model_outputs = self.model.normalizer.unscore_y(self.fine_surrogate.normalizer.unscore_y(fine_model_outputs))
            output = output.clone()
            output[limit_y_mask] = rescored_fine_model_outputs
        reward =  self.calculate_reward(output, norm=norm)
        if penalize_forbidden_actions:
            forbidden_actions_mask = (expanded_actions < 0.) | (expanded_actions > 1.)
            forbidden_actions_mask = forbidden_actions_mask.any(dim=1)
            reward_copy = reward.clone()
            reward_copy[forbidden_actions_mask] = 1000.
            return reward_copy

        if penalize_invalid:
            validity_scores = self.validity_classifier(merged_input)
            validity = (validity_scores > 0.5).squeeze(-1)
            reward_copy = reward.clone()
            reward_copy[~validity] = 1000.
            return reward_copy
        return reward

    def expand_action_states(self, action_batch, state_batch):
        batch_size = state_batch.shape[0]

        # Process state
        state = state_batch.squeeze(1)  # (batch_size, 8)
        state_tiled = state.repeat_interleave(self.N2, dim=0)

        phase_repeated = self.grid_phase_flat.repeat(batch_size, 1)
        solenoid_repeated = self.grid_solenoid_flat.repeat(batch_size, 1)

        expanded_states = torch.cat([state_tiled, phase_repeated, solenoid_repeated], dim=1)

        # Process actions
        expanded_actions = action_batch.repeat_interleave(self.N2, dim=0)
        return expanded_actions, expanded_states

    def __call__(self, action_batch, state_batch, clamping=True, norm=torch.abs, penalize_invalid=True, penalize_forbidden_actions=False):
        """
        action_batch: (batch_size, 4)
        state_batch: (batch_size, 1, 8)
        Returns: (batch_size, 3)
        """
        
        if clamping:
            action_batch = torch.clamp(action_batch, min=0.0, max=1.0)

        expanded_actions, expanded_states = self.expand_action_states(action_batch, state_batch)
        # Get reward
        reward_output = self.compute_integrated_reward(expanded_actions, expanded_states, norm=norm, penalize_invalid=penalize_invalid, penalize_forbidden_actions=penalize_forbidden_actions)  # (batch_size * N^2, 3)
        rewards = reward_output.view(state_batch.shape[0], self.N2, 3)

        # Aggregate
        rewards_mean = rewards.mean(dim=1)  # (batch_size, 3)
        return rewards_mean
