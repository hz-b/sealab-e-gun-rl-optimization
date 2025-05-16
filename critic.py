import torch
from surrogate.light_net import BerlinPro2

class Critic:
    def __init__(self, checkpoint='surrogate/surrogate_model.ckpt', device=None, grid_resolution=20):
        self.model = BerlinPro2.load_from_checkpoint(checkpoint, map_location=device)
        self.model.freeze()
        if device is not None:
            self.model = self.model.to(device)

        self.output_min = torch.tensor([5.5178498e-05,  4.9179042e-05, -2.9998908e-02, -2.2358654e-01, 2.3033355e+05], device=self.model.device)
        self.output_max = torch.tensor([2.9990217e-02, 2.9933929e-02, 2.9999450e-02, 2.9999496e-02, 2.5222615e+06], device=self.model.device)
        self.epsilon = 5e-5

        # Precompute the phase/solenoid grid
        self.N = grid_resolution
        grid_phase = torch.linspace(0., 0.9, self.N, device=self.model.device)
        grid_solenoid = torch.linspace(0.6, 0.9, self.N, device=self.model.device)
        mesh_phase, mesh_solenoid = torch.meshgrid(grid_phase, grid_solenoid, indexing='ij')
        self.grid_phase_flat = mesh_phase.reshape(-1, 1)        # (N^2, 1)
        self.grid_solenoid_flat = mesh_solenoid.reshape(-1, 1)  # (N^2, 1)

    def smooth_function(self, x):
        value = torch.min(torch.abs(x), torch.tensor(self.epsilon, dtype=x.dtype, device=x.device))
        return torch.reshape(value, (-1,1))

    def calculate_reward(self, output_array):
        output_inv = (self.output_max - self.output_min)*output_array + self.output_min  
        #x_pos_smoothed = self.smooth_function(output_inv[:,2])
        #y_pos_smoothed = self.smooth_function(output_inv[:,3])
        #sizes_smoothed = self.smooth_function(output_inv[:,0] - output_inv[:,1])
        sizes = output_inv[:,0] - output_inv[:,1]
        sizes_min = self.output_min[0]-self.output_max[1]
        sizes_max = self.output_max[0]-self.output_min[1]
        sizes_inv = (sizes_max - sizes_min)*sizes + sizes_min

        value = torch.vstack([torch.abs(output_inv[:,2]), torch.abs(output_inv[:,3]), torch.abs(sizes_inv)]).T
        return value

    def compute_integrated_reward(self, expanded_actions, expanded_states):
        merged_input = torch.cat([expanded_states[:, :7], expanded_actions, expanded_states[:, -3:]], dim=1)
        output = self.model(merged_input)
        return self.calculate_reward(output)

    def __call__(self, action_batch, state_batch):
        """
        action_batch: (batch_size, 4)
        state_batch: (batch_size, 1, 8)
        Returns: (batch_size, 3)
        """
        batch_size = state_batch.shape[0]
        N2 = self.N ** 2

        # Process state
        state = state_batch.squeeze(1)  # (batch_size, 8)
        base = state[:, :7]
        base_tiled = base.repeat_interleave(N2, dim=0)

        phase_repeated = self.grid_phase_flat.repeat(batch_size, 1)
        solenoid_repeated = self.grid_solenoid_flat.repeat(batch_size, 1)
        param8 = state[:, 7].repeat_interleave(N2).unsqueeze(1)

        expanded_states = torch.cat([base_tiled, phase_repeated, solenoid_repeated, param8], dim=1)

        # Process actions
        expanded_actions = action_batch.repeat_interleave(N2, dim=0)

        # Get reward
        reward_output = self.compute_integrated_reward(expanded_actions, expanded_states)  # (batch_size * N^2, 3)
        rewards = reward_output.view(batch_size, N2, 3)

        # Aggregate
        rewards_mean = rewards.mean(dim=1)  # (batch_size, 3)
        return rewards_mean
