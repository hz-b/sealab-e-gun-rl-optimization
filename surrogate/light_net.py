import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import h5py
import os
import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt

#sns.set(style="darkgrid")

class MinMaxDataset(Dataset):
    def __init__(self, x, y):

        # Compute min, max, and range for features
        self.min = torch.min(self.x, dim=0).values
        self.max = torch.max(self.x, dim=0).values
        self.z = self.max - self.min

        # Compute min, max, and range for targets
        self.minY = torch.min(self.y, dim=0).values
        self.maxY = torch.max(self.y, dim=0).values
        self.zY = self.maxY - self.minY

        # Apply normalization
        self.x_norm = self.z_score(self.x)
        self.y_norm = self.z_score_y(self.y)

    def z_score(self, x):
        return (x - self.min) / self.z

    def un_z_score(self, x):
        return x * self.z + self.min

    def z_score_y(self, y):
        return (y - self.minY) / self.zY

    def un_z_score_y(self, y):
        return y * self.zY + self.minY

    def __len__(self):
        return len(self.x_norm)

    def __getitem__(self, idx):
        return self.x_norm[idx], self.y_norm[idx]

class ZScoreDataset(Dataset):
    def __init__(self, x, y):
        self.mean = np.mean(x.T, axis=1)
        self.std = np.std(x.T, axis=1)
        self.yMean = np.mean(y.T, axis=1)
        self.yStd = np.std(y.T, axis=1)
        
    def z_score(self, x):
        return (x - self.mean) / self.std

    def un_z_score(self, x):
        return x * self.std + self.mean

    def z_score_y(self, y):
        return (y - self.yMean) / self.yStd

    def un_z_score_y(self, y):
        return y * self.yStd + self.yMean

class H5Dataset(MinMaxDataset):
    def __init__(self, path, transform=None, target_transform=None, omit_outliers=False, outlier_replacement=[100., 1000., 100., 100., 100.]):
        self.transform = transform
        self.target_transform = target_transform
        self.data = h5py.File(path,'r')
        if(self.data['X'].shape[0] != 14):
            self.x = self.data['X'][[0,1,4,5,7,8,9,10,11,12,13,15,16,6]].T
        else:
            self.x = self.data['X'][:].T
        self.y = self.data['Y'][:5].T
        
        self.data.close()
        
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()

        mask = (abs(self.y.T[0][:]) < 0.03) & (abs(self.y.T[1][:]) < 0.03) & (abs(self.y.T[2][:]) < 0.03) & (abs(self.y.T[3][:]) < 0.03)
        
        if omit_outliers:
            self.x = self.x[mask]
            self.y = self.y[mask]
        else:
            self.y[~mask] = torch.tensor(outlier_replacement, device=self.x.device).repeat(self.y[~mask].shape[0], 1)
        
        super().__init__(self.x, self.y)
        self.x = self.z_score(self.x)
        self.y = self.z_score_y(self.y)
        

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return self.x[idx], self.y[idx]

class BerlinPro2(pl.LightningModule):
    def __init__(self, hparams):
        super(BerlinPro2, self).__init__()
        self.save_hyperparameters(hparams)
        #self.hparams = hparams
        os.makedirs(os.path.join(self.hparams.output_dir, self.hparams.name), exist_ok=True)
        self.net = self.create_sequential(14, 5, self.hparams.layer_size, blow=self.hparams.blow, shrink_factor=self.hparams.shrink_factor)
        self.val_x = []
        self.val_y = []
        self.val_y_hat = []
        
    def prepare_data(self):
        self.dataset = H5Dataset(os.path.join(self.hparams.data_root,'bbp_merged.hdf5'))

        train_size = int(0.6 * len(self.dataset))
        val_size = int(0.2 * len(self.dataset))
        test_size = len(self.dataset) - (train_size + val_size)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size, test_size])
        #second_set = H5Dataset('../data/CombineData_MLP_4_9.hdf5', transform=self.get_transforms()[0], target_transform=self.get_transforms()[1])
        #self.train_dataset = MultiDataset(self.train_dataset, second_set)
        
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
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", val_loss)
        self.val_x.append(x)
        self.val_y.append(y)
        self.val_y_hat.append(y_hat)

    def on_validation_epoch_end(self):
        x = torch.cat([i for i in self.val_x])
        y = torch.cat([i for i in self.val_y])
        y_hat = torch.cat([i for i in self.val_y_hat])
        
        output = {}
        plot_data_count = 1000
        for i in range(y.shape[1]):
            feature_loss = nn.MSELoss(reduction="mean")(y[:,i], y_hat[:,i])
            output["feature_loss_"+str(i)] = feature_loss

        y_data = self.dataset.un_z_score_y(y[:plot_data_count].cpu())
        y_hat_data = self.dataset.un_z_score_y(y_hat[:plot_data_count].cpu())
        for i in range(y.shape[1]):
            mask = y_data[:,i] < 0.03
            if (~mask).all():
                continue
            joint = sns.jointplot(x=y_data[:,i][mask], y=y_hat_data[:,i][mask], kind='scatter').set_axis_labels("real", "predicted")
            joint.ax_joint.set_ylim(bottom=y_data[:, i][mask].min(), top=y_data[:, i][mask].max())
            joint.ax_joint.plot([y_data[:,i][mask].min(), y_data[:,i][mask].max()], [y_data[:,i][mask].min(), y_data[:,i][mask].max()], color="r")
            plt.tight_layout()
            plt.savefig('outputs/'+self.hparams.name+'/jointplot_'+str(i+1)+'.pdf')
            wandb.log({"good_"+str(i): wandb.Image(joint.fig)})
            joint.fig.clf()
            plt.close(joint.fig)
            errorplot = sns.jointplot(x=y_data[:,i][mask], y=y_data[:,i][mask]-y_hat_data[:,i][mask], color="g").fig
            plt.tight_layout()
            plt.savefig('outputs/'+self.hparams.name+'/line_plot_'+str(i+1)+'.pdf')
            plt.close(errorplot)

        self.val_x.clear()
        self.val_y.clear()
        self.val_y_hat.clear()

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            return [torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)]
        elif self.hparams.optimizer == 'sgd':
            return [torch.optim.SGD(model.parameters(), lr=self.hparams.learning_rate, momentum=0.9)]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.on_gpu)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.on_gpu)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.on_gpu)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('name', type=str)
        parser.add_argument('--layer_size', default=5, type=int)
        parser.add_argument('--blow', default=143., type=float)
        parser.add_argument('--shrink_factor', default="log", type=str)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default='../datasets', type=str)
        parser.add_argument('--output_dir', default='outputs', type=str)

        # training params (opt)
        parser.add_argument('--batch_size', default=2048, type=int)
        parser.add_argument('--num_workers', default=os.cpu_count(), type=int)
        parser.add_argument('--gpus', default=0, type=int)
        parser.add_argument('--optimizer', default='adam', type=str)
        return parser

if __name__ == '__main__':
    pl.trainer.seed_everything(42)
    parser = BerlinPro2.add_model_specific_args(ArgumentParser(add_help=False))
    model = BerlinPro2(parser.parse_args())
    
    print(model.net)
    
    logger = WandbLogger(name=model.hparams.name, project="berlinpro_surrogate", save_dir=model.hparams.output_dir)
    
    trainer = pl.Trainer(fast_dev_run=False, check_val_every_n_epoch=10, max_epochs=1000, num_nodes=1, logger=logger, precision=32)
    trainer.fit(model)
    trainer.test()
