import torch
from torch.utils.data import IterableDataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch import optim, nn
import torch.nn.functional as F
from critic import Critic
import random
import wandb
from model_helpers import ScaledSigmoid, create_sequential

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
    def __init__(self, input_dim=8, output_dim=4, critic_net=Critic(), neuron_factor=500, layer_size=None, learning_rate=1e-4, optimizer='adam', lr_scheduler=None, shrink_factor="lin", activation=nn.ReLU(), last_activation=None, batch_norm=False):
        super().__init__()
        self.neuron_factor=neuron_factor
        self.shrink_factor=shrink_factor
        self.last_activation = nn.Identity() if last_activation is None else last_activation
        self.activation = activation
        if layer_size is None:
            self.model = nn.Sequential(nn.Linear(input_dim, self.neuron_factor*5), nn.LazyBatchNorm1d() if batch_norm else nn.Identity(), activation,
                             nn.Linear(self.neuron_factor*5, self.neuron_factor*2), nn.LazyBatchNorm1d() if batch_norm else nn.Identity(), activation,
                             nn.Linear(self.neuron_factor*2, self.neuron_factor*1), nn.LazyBatchNorm1d() if batch_norm else nn.Identity(), activation,
                             nn.Linear(self.neuron_factor*1, output_dim), self.last_activation
                            )
        else:
            self.model = create_sequential(input_dim, output_dim, layer_size, blow=neuron_factor, shrink_factor=shrink_factor, activation_function=activation, last_activation=last_activation, batch_norm=batch_norm)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.learning_rate = learning_rate
        self.layer_size = layer_size
        self.critic_net = critic_net
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.save_hyperparameters()
        print(self.model)

    def forward(self, x):
        return ((self.model(x)+1.)/2.)
    


    def training_step(self, batch, batch_idx):
        x = batch
        rewards_mean = self.critic_net(self(x), x)
        self.log("x_pos_loss", rewards_mean[:,0].mean())
        self.log("y_pos_loss", rewards_mean[:,1].mean())
        self.log("size_loss", rewards_mean[:,2].mean())
        loss = rewards_mean.mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rewards_mean = self.critic_net(self(batch), batch).mean()
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
                    "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        if self.lr_scheduler is not None:
            raise Exception("Defined LR scheduler not found.")

        return optimizer

class RandomDataModule(L.LightningDataModule):
    def __init__(self, num_samples, input_dim, output_dim, batch_size, seed, device, val_samples, val_seed):
        super().__init__()
        self.dataset = RandomIterableDataset(num_samples, input_dim, seed, device)
        self.val_dataset = RandomIterableDataset(val_samples, input_dim, val_seed, device, fixed_seed=True)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

if __name__ == "__main__":
    batch_size = 32
    num_samples = 100000
    seed = 42
    
    wandb_logger = WandbLogger(
            name="ref6", project="berlinpro", save_dir='outputs', offline=False
        )
    
    wandb.finish()
    
    critic_net = Critic()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = L.Trainer(max_epochs=250, log_every_n_steps=500, accelerator=str(critic_net.model.device.type), logger=wandb_logger, callbacks=[lr_monitor])
    
    model = RandomModel(critic_net=critic_net)
    dm = RandomDataModule(num_samples, model.input_dim, model.output_dim, batch_size, seed, device=critic_net.model.device, val_samples=100000, val_seed=seed+20000000)
    
    
    trainer.fit(model, datamodule=dm)
