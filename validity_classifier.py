import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from surrogate import H5Dataset
from model_helpers import create_sequential

class ValidityClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = create_sequential(14, 1, self.hparams.layer_size, blow = self.hparams.blow_to/14, shrink_factor="log")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.test_losses = []
        self.test_accs = []

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
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        y = y.unsqueeze(-1)
        loss = self.loss_fn(pred, y)
        acc = ((pred > 0.5) == y.bool()).float().mean()
        self.test_losses.append(loss.item())
        self.test_accs.append(acc.item())
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        losses = torch.tensor(self.test_losses)
        std_loss = torch.std(losses)
        self.log('test_loss_std', std_loss)
        accs = torch.tensor(self.test_accs)
        std_accs = torch.std(accs)
        self.log('test_acc_std', std_accs)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.min_lr
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        print( validity.sum().item(), "/", len(full_dataset), "data points are valid.")

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
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('name', type=str)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--layer_size', type=int, default=3)
        parser.add_argument('--blow_to', type=int, default=256)
        parser.add_argument('--shrink_factor', type=str, default='log')
        parser.add_argument('--data_path', type=str, default='datasets/bbp_ds_2m_merged_v2.h5')
        parser.add_argument('--max_epochs', type=int, default=500)
        parser.add_argument('--output_dir', type=str, default='outputs')
        parser.add_argument('--upscale_exp', type=int, default=7)
        parser.add_argument('--gpus', type=int, default=1)

        # Scheduler-specific args
        parser.add_argument('--lr_patience', type=int, default=5, help='Patience for LR scheduler')
        parser.add_argument('--lr_factor', type=float, default=0.95, help='Factor by which the LR will be reduced')
        parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR')

        return parser

if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser = ValidityClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    model = ValidityClassifier(hparams)

    wandb_logger = WandbLogger(
        name=model.hparams.name,
        project="berlinpro_validity",
        save_dir=os.path.join(model.hparams.output_dir, "berlinpro_validity")
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
        accelerator="gpu" if hparams.gpus > 0 else "cpu",
        devices=hparams.gpus if hparams.gpus > 0 else 1,
        log_every_n_steps=10,
    )

    trainer.fit(model)
    trainer.test(ckpt_path='last')

