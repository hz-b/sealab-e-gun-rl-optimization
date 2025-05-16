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

class RandomIterableDataset(IterableDataset):
    def __init__(self, num_samples, input_dim, seed, device):
        super().__init__()
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.base_seed = seed
        self.seed = seed
        self.stddev = .2
        self.device = device

    def __iter__(self):
        torch.manual_seed(self.seed)
        self.seed = self.seed + 1
        # Batch-wise generation is much faster
        x = torch.empty((self.num_samples, self.input_dim), device=self.device)
        torch.nn.init.trunc_normal_(x, mean=0.5, std=self.stddev, a=-0.5/self.stddev, b=0.5/self.stddev)
        for i in range(self.num_samples):
            yield x[i]

# LightningModule
class RandomModel(L.LightningModule):
    def __init__(self, input_dim=8, output_dim=4, critic_net=Critic(), neuron_factor=200):
        super().__init__()
        self.neuron_factor=neuron_factor
        self.model = nn.Sequential(nn.Linear(input_dim, self.neuron_factor*5), nn.ReLU(),
                         nn.Linear(self.neuron_factor*5, self.neuron_factor*2), nn.ReLU(),
                         nn.Linear(self.neuron_factor*2, self.neuron_factor*1), nn.ReLU(),
                         nn.Linear(self.neuron_factor*1, output_dim), nn.Tanh(),
                        )
        self.optimizer = 'adam'
        self.lr_scheduler = None#'plateau'
        self.learning_rate = 1e-4
        self.critic_net = critic_net
        self.input_dim = input_dim
        self.output_dim = output_dim
        print(self.model)

    def forward(self, x):
        return ((self.model(x)+1.)/2.)

    def training_step(self, batch, batch_idx):
        x = batch
        #loss = batch_integral_reward(self(x), x, self.log).mean()
        rewards_mean = self.critic_net(self(x), x)
        self.log("x_pos_loss", rewards_mean[:,0].mean())
        self.log("y_pos_loss", rewards_mean[:,1].mean())
        self.log("size_loss", rewards_mean[:,2].mean())
        loss = rewards_mean.mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

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
                    "monitor": "train_loss",
                    "frequency": 1,
                },
            }
        if self.lr_scheduler is not None:
            raise Exception("Defined LR scheduler not found.")

        return optimizer

# DataModule that updates seed per epoch
class RandomDataModule(L.LightningDataModule):
    def __init__(self, num_samples, input_dim, output_dim, batch_size, seed, device):
        super().__init__()
        self.dataset = RandomIterableDataset(num_samples, input_dim, seed, device)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

# Usag
if __name__ == "__main__":
    batch_size = 64
    num_samples = 1000
    seed = 42
    
    wandb_logger = WandbLogger(
            name="test_no_smoothing_weighted_32_1e4_plateau", project="berlinpro", save_dir='outputs', offline=False
        )
    
    wandb.finish()
    
    critic_net = Critic()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = L.Trainer(max_epochs=1000, log_every_n_steps=100, accelerator=str(critic_net.model.device.type), logger=wandb_logger, callbacks=[lr_monitor])
    
    model = RandomModel(critic_net=critic_net)
    dm = RandomDataModule(num_samples, model.input_dim, model.output_dim, batch_size, seed, device=critic_net.model.device)
    
    
    trainer.fit(model, datamodule=dm)
