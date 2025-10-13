import math
import torch
import torch.nn as nn
import lightning as pl
import h5py
import os
import sys
import wandb
import numpy as np
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything
from argparse import ArgumentParser
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_generation')))
from simulation import sim_Y_labels
import matplotlib.pyplot as plt
from distutils.util import strtobool
from normalizer import Normalizer
#sns.set(style="darkgrid")

class H5Dataset(Dataset):
    def __init__(self, path, limit_y=True, raw=False, max_len=None):
        data = h5py.File(path,'r')

        x = data['X'][:max_len]
        y = data['Y'][:max_len, :4]
        
        data.close()
        del data
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if raw:
            self.x_norm = x
            self.y_norm = y
            return

        limit_y_mask = (abs(y[:, :4]) < 30).all(dim=1)
        isnan_mask = torch.isnan(y[:, :4]).any(dim=1)
        if limit_y:
            selection_mask = ~isnan_mask & limit_y_mask
        else:
            selection_mask = ~isnan_mask
        # score on all x values (also nan and out of bound y, they may be used)
        # score y according to limit and not on nans for y
        self.normalizer = Normalizer(x, y[selection_mask], method="minmax")
        
        self.x_norm = self.normalizer.score_x(x[selection_mask])
        self.y_norm = self.normalizer.score_y(y[selection_mask])
        
    def __len__(self):
        return self.x_norm.shape[0]

    def __getitem__(self, idx):
        return self.x_norm[idx], self.y_norm[idx]

class BerlinPro2(pl.LightningModule):
    def __init__(self, hparams):
        super(BerlinPro2, self).__init__()
        self.num_workers = self.get_cpu_count()
        self.save_hyperparameters(hparams)
        #self.hparams = hparams
        self.net = self.create_sequential(14, 4, self.hparams.layer_size, blow=self.hparams.blow, shrink_factor=self.hparams.shrink_factor)
        self.val_x = []
        self.val_y = []
        self.val_y_hat = []
        self.test_x = []
        self.test_y = []
        self.test_y_hat = []
        self.normalizer = None
        
    def prepare_data(self):
        self.dataset = H5Dataset(self.hparams.data_path, limit_y=self.hparams.limit_y)
        self.normalizer = self.dataset.normalizer

        train_size = int(0.6 * len(self.dataset))
        val_size = int(0.2 * len(self.dataset))
        test_size = len(self.dataset) - (train_size + val_size)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size, test_size])
        
    def forward(self, x):
        return self.net(x)
    
    def create_sequential(self, input_length, output_length, layer_size, blow=0, shrink_factor="log"):
        layers = [input_length]
        blow_disabled = blow == 1 or blow == 0
        if not blow_disabled:
            layers.append(input_length*blow)

        if shrink_factor == "log":
            add_layers = torch.logspace(math.log(layers[-1], 10), math.log(output_length,10), steps=layer_size+2-len(layers), base=10).long()
            # make sure the last element is correct, even though rounding
            add_layers[-1] = output_length
        elif shrink_factor == "lin":
            add_layers = torch.linspace(layers[-1], output_length, steps=layer_size+2-len(layers)).long()
        else:
            shrink_factor = float(shrink_factor)
            new_length = layer_size+1-len(layers)
            add_layers = (torch.ones(new_length)*layers[-1] * ((torch.ones(new_length) * shrink_factor) ** torch.arange(new_length))).long()
            layers = torch.cat((torch.tensor([input_length]), add_layers))
            layers = torch.cat((layers, torch.tensor([output_length])))
    
        if not blow_disabled:
            layers = torch.tensor([layers[0]])
            layers = torch.cat((layers, add_layers))
        else:
           layers = add_layers

        nn_layers = []
        for i in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[i].item(), layers[i+1].item()))
            if not i == len(layers)-2:
                nn_layers.append(nn.ReLU())
                nn_layers.append(nn.BatchNorm1d(layers[i+1].item()))
        return nn.Sequential(*nn_layers)
    @staticmethod
    def get_cpu_count():
        slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
        if slurm_cpus:
            return int(slurm_cpus)
        else:
            return multiprocessing.cpu_count()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y, y_hat)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = nn.MSELoss()(y_hat, y)
        self.test_x.append(x)
        self.test_y.append(y)
        self.test_y_hat.append(y_hat)
        
    def on_test_epoch_end(self):
        x = torch.cat([i for i in self.test_x])
        y = torch.cat([i for i in self.test_y])
        y_hat = torch.cat([i for i in self.test_y_hat])
        
        per_sample_loss = nn.MSELoss(reduction='none')(y_hat, y)
        per_sample_loss = per_sample_loss.view(per_sample_loss.size(0), -1).mean(dim=1)
        self.log("test_loss/test_loss", per_sample_loss.mean())
        self.log("test_loss/test_loss_std", per_sample_loss.std())
        un_z_scored_y = self.normalizer.unscore_y(y.cpu())
        un_z_scored_y_hat = self.normalizer.unscore_y(y_hat.cpu())
        
        feature_mses = []
        for i in range(y.shape[1]):
            feature_mse = nn.MSELoss(reduction="none")(un_z_scored_y[:, i], un_z_scored_y_hat[:, i])
            feature_mses.append(feature_mse)
            self.log("feature_rmse/"+sim_Y_labels[i].replace(" ", "_").replace("/", "\\"), feature_mse.mean().sqrt())
            self.log("feature_rmse/"+sim_Y_labels[i].replace(" ", "_").replace("/", "\\")+"_std", feature_mse.std().sqrt())
        
        if not self.hparams.limit_y:
            l_30_feature_mses = []
            l_30_mask = (abs(un_z_scored_y[:, :4]) < 30).all(dim=1)
            
            feature_mses = []
            for i in range(y.shape[1]):
                l_30_feature_mse = nn.MSELoss(reduction="none")(un_z_scored_y[l_30_mask, i], un_z_scored_y_hat[l_30_mask, i])
                l_30_feature_mses.append(l_30_feature_mse)
                self.log("feature_rmse_<_30/"+sim_Y_labels[i].replace(" ", "_").replace("/", "\\"), l_30_feature_mse.mean().sqrt())
                self.log("feature_rmse_<_30/"+sim_Y_labels[i].replace(" ", "_").replace("/", "\\")+"_std", l_30_feature_mse.std().sqrt())
                
        self.test_x.clear()
        self.test_y.clear()
        self.test_y_hat.clear()

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.val_x.append(x)
        self.val_y.append(y)
        self.val_y_hat.append(y_hat)
        return val_loss

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 10 == 0:
            x = torch.cat([i for i in self.val_x])
            y = torch.cat([i for i in self.val_y])
            y_hat = torch.cat([i for i in self.val_y_hat])
            
            output = {}
            plot_data_count = 1000
            
            un_z_scored_y = self.normalizer.unscore_y(y.cpu())
            un_z_scored_y_hat = self.normalizer.unscore_y(y_hat.cpu())

            feature_mses = []
            for i in range(y.shape[1]):
                feature_mse = nn.MSELoss(reduction="none")(un_z_scored_y[:, i], un_z_scored_y_hat[:, i])
                feature_mses.append(feature_mse)
                self.log("feature_rmse/"+sim_Y_labels[i].replace(" ", "_").replace("/", "\\"), feature_mse.mean().sqrt())

            feature_mses_tensor = torch.stack(feature_mses, dim=1)  # shape: [num_samples, num_features]

            # Compute the mean and std across samples for each feature
            mean_rmse_per_feature = feature_mses_tensor.mean(dim=0).sqrt()  # Mean RMSE per feature
            std_rmse_per_feature = feature_mses_tensor.std(dim=0).sqrt()  # Std RMSE per feature

            
            l_30_feature_mses = []
            l_30_mask = (abs(un_z_scored_y[:, :4]) < 30).all(dim=1)
            
            feature_mses = []
            for i in range(y.shape[1]):
                l_30_feature_mse = nn.MSELoss(reduction="none")(un_z_scored_y[l_30_mask, i], un_z_scored_y_hat[l_30_mask, i])
                l_30_feature_mses.append(l_30_feature_mse)
                self.log("feature_rmse_<_30/"+sim_Y_labels[i].replace(" ", "_").replace("/", "\\"), l_30_feature_mse.mean().sqrt())

            l_30_feature_mses_tensor = torch.stack(l_30_feature_mses, dim=1)  # shape: [num_samples, num_features]

            # Compute the mean and std across samples for each feature
            l_30_mean_rmse_per_feature = l_30_feature_mses_tensor.mean(dim=0).sqrt()  # Mean RMSE per feature
            l_30_std_rmse_per_feature = l_30_feature_mses_tensor.std(dim=0).sqrt()  # Std RMSE per feature

                
            y_data = un_z_scored_y[:plot_data_count]
            y_hat_data = un_z_scored_y_hat[:plot_data_count]
            
            for i in range(y.shape[1]):
                mask = y_data[:,i] < 30000000
                joint = sns.jointplot(x=y_data[:,i][mask], y=y_hat_data[:,i][mask], kind='scatter').set_axis_labels("Real", "Predicted")
                joint.ax_joint.set_ylim(bottom=y_data[:, i][mask].min(), top=y_data[:, i][mask].max())
                joint.ax_joint.set_title(sim_Y_labels[i])
                joint.ax_joint.plot([y_data[:,i][mask].min(), y_data[:,i][mask].max()], [y_data[:,i][mask].min(), y_data[:,i][mask].max()], color="r")
                rmse_str = f"RMSE: {mean_rmse_per_feature[i]:.4f} ± {std_rmse_per_feature[i]:.4f}, y < 30: {l_30_mean_rmse_per_feature[i]:.4f} ± {l_30_std_rmse_per_feature[i]:.4f}"
                joint.ax_joint.text(
                0.5, -0.15, rmse_str, 
                transform=joint.ax_joint.transAxes, 
                ha='center', va='top'
                )
                plt.tight_layout()
                path = os.path.join(self.logger.save_dir, "berlinpro_surrogate", self.logger.experiment.id)
                os.makedirs(path, exist_ok=True)

                plt.savefig(os.path.join(path, 'jointplot_'+str(i+1)+'.pdf'))
                
                wandb.log({sim_Y_labels[i].replace(" ", "_").replace("/", "\\"): wandb.Image(joint.fig)})
                joint.fig.clf()
                plt.close(joint.fig)
                errorplot = sns.jointplot(x=y_data[:,i][mask], y=y_data[:,i][mask]-y_hat_data[:,i][mask], color="g").fig
                plt.tight_layout()
                plt.savefig(os.path.join(path, 'line_plot_'+str(i+1)+'.pdf'))
                plt.close(errorplot)

            self.val_x.clear()
            self.val_y.clear()
            self.val_y_hat.clear()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['normalizer'] = self.normalizer

    def on_load_checkpoint(self, checkpoint):
        self.normalizer = checkpoint['normalizer']

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        model = super().load_from_checkpoint(checkpoint_path, map_location=map_location, **kwargs)

        if hasattr(model, 'normalizer') and model.normalizer is not None:
            model.normalizer.to(model.device)

        return model
        
    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        if self.hparams.patience is not None:
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=self.hparams.patience,
                    factor=0.1,
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        elif self.hparams.lr_decay_gamma is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.hparams.lr_decay_gamma
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer


    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.hparams.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('name', type=str)
        parser.add_argument('--transfer_ckpt_path', type=str)
        parser.add_argument('--layer_size', default=5, type=int)
        parser.add_argument('--blow', default=143., type=float)
        parser.add_argument('--shrink_factor', default="log", type=str)
        parser.add_argument('--learning_rate', default=0.0001, type=float)

        # data
        parser.add_argument('--data_path', default='datasets/bbp_ds_2m_merged_v2.h5', type=str)
        parser.add_argument('--output_dir', default='outputs', type=str)
        parser.add_argument('--limit_y', type=lambda x: bool(strtobool(x)), default=False)

        # training params (opt)
        parser.add_argument('--batch_size', default=1024, type=int)
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--optimizer', default='adam', type=str)
        parser.add_argument('--patience', default=500, type=int, help='Patience for ReduceLROnPlateau scheduler. If None, scheduler is not used.')
        parser.add_argument('--lr_decay_gamma', default=None, type=float, help='Decay Gamma for ExponentialLR scheduler. If None, scheduler is not used.')
        return parser

if __name__ == '__main__':
    seed_everything(42)
    parser = BerlinPro2.add_model_specific_args(ArgumentParser(add_help=False))
    transfer_ckpt_path = parser.parse_args().transfer_ckpt_path
    if transfer_ckpt_path is not None:
        model = BerlinPro2.load_from_checkpoint(transfer_ckpt_path)
        new_lr = parser.parse_args().learning_rate
        if new_lr is not None:
            model.hparams.learning_rate = new_lr
    else:
        model = BerlinPro2(parser.parse_args())
    
    print(model.net)
    
    logger = WandbLogger(name=model.hparams.name, offline=False, project="berlinpro_surrogate", save_dir=os.path.join(model.hparams.output_dir, "berlinpro_surrogate"))
    lr_monitor = LearningRateMonitor()
        
    trainer = pl.Trainer(fast_dev_run=False, check_val_every_n_epoch=1, max_epochs=10000, logger=logger, precision=32, accelerator="gpu" if model.hparams.gpus > 0 else "cpu", devices=model.hparams.gpus if model.hparams.gpus > 0 else 1, callbacks=[lr_monitor])
    trainer.fit(model)
    trainer.test(ckpt_path='last')
