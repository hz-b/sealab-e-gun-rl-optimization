import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import h5py
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

class MinMaxDataset(Dataset):
    def __init__(self, x, y):
        self.max = np.amax(x.T, axis=1)
        self.min = np.amin(x.T, axis=1)
        self.z = self.max - self.min

        self.maxY = np.amax(self.y.T, axis=1)
        self.minY = np.amin(self.y.T, axis=1)
        self.zY = self.maxY - self.minY

    def z_score(self, x):
        return (x - self.min) / self.z

    def un_z_score(self, x):
        return x * self.z + self.min

    def z_score_y(self, y):
        return (y - self.minY) / self.zY

    def un_z_score_y(self, y):
        return y * self.zY + self.minY

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
        
class MultiDataset(ZScoreDataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        self.x = self.dataset1[:][0].numpy()
        self.y = self.dataset1[:][1].numpy()
        super().__init__(self.x, self.y)
        
    def __getitem__(self, idx):
        if(idx >= len(self.dataset1)):
            return self.dataset2[idx-len(self.dataset1)]
        else:
            return self.dataset1[idx]
    def __len__(self):
        return len(self.dataset1)+len(self.dataset2)

class H5Dataset(MinMaxDataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = h5py.File(path,'r')
        if(self.data['X'].shape[0] != 14):
            self.x = self.data['X'][[0,1,4,5,7,8,9,10,11,12,13,15,16,6]].T
        else:
            self.x = self.data['X'][:].T
        self.y = self.data['Y'][:5].T
        
        self.data.close()
                
        mask = (abs(self.y.T[0][:]) < 0.03) & (abs(self.y.T[1][:]) < 0.03) & (abs(self.y.T[2][:]) < 0.03) & (abs(self.y.T[3][:] < 0.03))
        self.x = self.x[mask]
        self.y = self.y[mask]
        super().__init__(self.x, self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.z_score(self.x[idx])
        y = self.z_score_y(self.y[idx])
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return torch.from_numpy(x).float(),torch.from_numpy(y).float()

class BerlinPro2(pl.LightningModule):
    def __init__(self, hparams):
        super(BerlinPro2, self).__init__()
        self.save_hyperparameters(hparams)
        #self.hparams = hparams
        self.net = self.create_sequential(14, 5, self.hparams.layer_size, blow=self.hparams.blow, shrink_factor=self.hparams.shrink_factor)
        
    def prepare_data(self):
        #self.dataset = H5Dataset(os.path.join(self.hparams.data_root,'CombineData_MLP_10-15_16(200)_17(200)_INF1-6.hdf5'))
        #first_set = H5Dataset(os.path.join(self.hparams.data_root,'CombineData_MLP_10-15_16(200)_17(200)_INF1-6.hdf5'))
        second_set = H5Dataset(os.path.join(self.hparams.data_root,'bbp_merged.hdf5'))
        self.dataset = second_set
        #self.dataset = MultiDataset(first_set, second_set)

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
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = nn.MSELoss()(y_hat, y)
        return {'s_test_loss': test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['s_test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = nn.MSELoss()(y_hat, y)
        return {'s_val_loss': val_loss, 'y': y, 'y_hat': y_hat, 'x': x}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['s_val_loss'] for x in outputs]).mean()

        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        output = {}
        plot_data_count = 1000
        for i in range(y.shape[1]):
            feature_loss = nn.MSELoss(reduction="mean")(y[:,i], y_hat[:,i])
            output["feature_loss_"+str(i)] = feature_loss

        y_data = self.dataset.un_z_score_y(y[:plot_data_count].cpu())
        y_hat_data = self.dataset.un_z_score_y(y_hat[:plot_data_count].cpu())
        for i in range(y.shape[1]):
            joint = sns.jointplot(y_data[:,i], y_hat_data[:,i], kind='scatter').set_axis_labels("real", "predicted")
            joint.ax_joint.plot([y_data[:,i].min(), y_data[:,i].max()], [y_data[:,i].min(), y_data[:,i].max()], color="r")
            #self.logger.experiment.add_figure("jointplot"+str(i+1),joint.fig)
            plt.tight_layout()
            plt.savefig('lightning_logs/'+self.hparams.name+'/jointplot_'+str(i+1)+'.pdf')
            plt.close(joint.fig)
            errorplot = sns.jointplot(y_data[:,i], y_data[:,i]-y_hat_data[:,i], color="g").fig
            #self.logger.experiment.add_figure("errorplot"+str(i+1),errorplot)
            plt.tight_layout()
            plt.savefig('lightning_logs/'+self.hparams.name+'/line_plot_'+str(i+1)+'.pdf')
            plt.close(errorplot)
        output['val_loss'] = val_loss
        return {'val_loss':val_loss,'log':output}

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
        parser.add_argument('--data_root', default='/mnt/work/xfel/bessy/berlinPro', type=str)

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
    
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=model.hparams.name,
        name='lightning_logs'
    )
    trainer = pl.Trainer(fast_dev_run=False, limit_train_batches=1.0, limit_val_batches=1.0, num_nodes=1, gpus=model.hparams.gpus, logger=logger, early_stop_callback=False, precision=32)
    trainer.fit(model)
    trainer.test()
