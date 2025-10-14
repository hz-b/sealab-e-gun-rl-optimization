import torch
from torch.utils.data import IterableDataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.cli import LightningCLI
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch import optim, nn
import torch.nn.functional as F
from critic import Critic
import random
import wandb
from typing import Optional
from model_helpers import create_sequential

class RandomIterableDataset(IterableDataset):
    def __init__(self, num_samples, input_dim, seed, device, stddev=.2, fixed_seed=False):
        super().__init__()
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.seed = seed
        self.stddev = stddev
        self.device = device
        self.fixed_seed=fixed_seed

    def __iter__(self):
        torch.manual_seed(self.seed)
        if not self.fixed_seed:
            self.seed = self.seed+1
        x = torch.empty((self.num_samples, self.input_dim), device=self.device)
        torch.nn.init.trunc_normal_(x, mean=0.5, std=self.stddev, a=-0.5/self.stddev, b=0.5/self.stddev)
        x.clamp_(min=0.0, max=1.0)
        for i in range(self.num_samples):
            yield x[i]

    def __len__(self):
        return self.num_samples

class RandomModel(L.LightningModule):
    def _get_activation(self, name):
        if name is None:
            return nn.Identity()
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "mish":
            return nn.Mish()
        elif name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {name}")
            
    def __init__(self, input_dim:int=8, output_dim:int=4, critic_net:Critic=None, neuron_factor:int=500, layer_size:Optional[int]=None, 
             learning_rate:float=1e-4, optimizer:str='adam', lr_scheduler:str=None, shrink_factor:str="lin", 
             activation:str="relu", last_activation:Optional[str]=None, batch_norm:bool=False, patience:int=3, loss_norm="abs", **kwargs):
        super().__init__()
        self.neuron_factor=neuron_factor
        self.shrink_factor=shrink_factor
        self.activation = self._get_activation(activation)
        self.last_activation = self._get_activation(last_activation)

        if layer_size is None:
            self.model = nn.Sequential(nn.Linear(input_dim, self.neuron_factor*5), nn.LazyBatchNorm1d() if batch_norm else nn.Identity(), self.activation,
                             nn.Linear(self.neuron_factor*5, self.neuron_factor*2), nn.LazyBatchNorm1d() if batch_norm else nn.Identity(), self.activation,
                             nn.Linear(self.neuron_factor*2, self.neuron_factor*1), nn.LazyBatchNorm1d() if batch_norm else nn.Identity(), self.activation,
                             nn.Linear(self.neuron_factor*1, output_dim), self.last_activation
                            )
        else:
            self.model = create_sequential(input_dim, output_dim, layer_size, blow=neuron_factor, shrink_factor=shrink_factor, activation_function=self.activation, last_activation=self.last_activation, batch_norm=batch_norm)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.learning_rate = learning_rate
        self.layer_size = layer_size
        if critic_net is None:
            critic_net = Critic()
        self.critic_net = critic_net
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patience = patience
        self.save_hyperparameters()
        if loss_norm == "l2":
            self.loss_norm = lambda x: x**2
        else:
            self.loss_norm = torch.abs
        print(self.model)

    def forward(self, x):
        return ((self.model(x)+1.)/2.)

    def training_step(self, batch, batch_idx):
        x = batch
        rewards_mean = self.critic_net(self(x), x, penalize_invalid=False, norm=self.loss_norm)
        self.log("x_pos_loss", rewards_mean[:,0].mean())
        self.log("y_pos_loss", rewards_mean[:,1].mean())
        self.log("size_loss", rewards_mean[:,2].mean())
        loss = rewards_mean.mean()
        if isinstance(self.last_activation, nn.Sigmoid):
            loss = loss * 0.8 + 0.1
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rewards_mean = self.critic_net(self(batch), batch, penalize_invalid=False, norm=self.loss_norm).mean()
        self.log("val_loss", rewards_mean, prog_bar=True)
        return rewards_mean

    def configure_optimizers(self):
        if self.optimizer == "adam_w":
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler == "exp":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ExponentialLR(optimizer, gamma=0.895),
                    "frequency": 1,
                },
            }
        if self.lr_scheduler == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, patience=self.patience),
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        if self.lr_scheduler is not None:
            raise Exception("Defined LR scheduler not found.")

        return optimizer

class RandomDataModule(L.LightningDataModule):
    def __init__(self, input_dim=8, output_dim=4, num_samples=100000, batch_size=32, seed=42, device=None, val_samples=100000, val_seed=20000042):
        super().__init__()
        if device is None:
            device = torch.device('cuda')
        self.dataset = RandomIterableDataset(num_samples, input_dim, seed, device)
        self.val_dataset = RandomIterableDataset(val_samples, input_dim, val_seed, device, fixed_seed=True)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Allow CLI override of W&B settings
        parser.add_argument("--wandb_name", type=str, default="ref6")
        parser.add_argument("--wandb_project", type=str, default="berlinpro_decision_model")
        parser.add_argument("--offline", action="store_true", help="Run W&B in offline mode")
        parser.set_defaults(trainer={"max_epochs": 250, "log_every_n_steps": 500})

    def before_fit(self):
        # Set W&B logger
        wandb_logger = WandbLogger(
            name=self.config.fit.wandb_name,
            project=self.config.fit.wandb_project,
            save_dir="outputs",
            offline=self.config.fit.offline,
        )
        self.trainer.logger = wandb_logger

        # Add LearningRateMonitor
        self.trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    def after_fit(self):
        # Optionally close W&B run
        wandb.finish()


if __name__ == "__main__":
    cli = CustomCLI(
        RandomModel,
        RandomDataModule,
        seed_everything_default=42,
        save_config_callback=None,  # or keep default if you want .yaml configs saved
    )
