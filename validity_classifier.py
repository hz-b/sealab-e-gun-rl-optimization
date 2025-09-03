import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from surrogate import H5Dataset
        
class ValidityClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            #nn.Sigmoid()
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        y = y.unsqueeze(-1)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        y = y.unsqueeze(-1)
        loss = self.loss_fn(pred, y)
        acc = ((pred > 0.5) == y.bool()).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        y = y.unsqueeze(-1)
        loss = self.loss_fn(pred, y)
        acc = ((pred > 0.5) == y.bool()).float().mean()
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
        
    def prepare_data(self):
        full_dataset = H5Dataset(self.hparams.data_path, raw=True)
        
        isnan_mask = torch.isnan(full_dataset.y_norm[:, :4]).any(dim=1)
        validity = (~isnan_mask).float()

        class ValidityDataset(Dataset):
            def __init__(self, x, labels):
                self.x = x
                self.labels = labels

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                return self.x[idx], self.labels[idx]

        dataset = ValidityDataset(full_dataset.x_norm, validity)       
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('name', type=str)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--data_path', type=str, default='datasets/bbp_ds_10m_merged.h5')
        parser.add_argument('--max_epochs', type=int, default=50)
        parser.add_argument('--output_dir', default='outputs', type=str)
        parser.add_argument('--gpus', type=int, default=1)
        return parser

if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser = ValidityClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    model = ValidityClassifier(hparams)
    wandb_logger = WandbLogger(name=model.hparams.name, project="berlinpro_validity", save_dir=os.path.join(model.hparams.output_dir, "berlinpro_validity"))

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
        accelerator="gpu" if hparams.gpus > 0 else "cpu",
        devices=hparams.gpus if hparams.gpus > 0 else 1,
        log_every_n_steps=10,
    )

    trainer.fit(model)
    trainer.test(ckpt_path='best')
