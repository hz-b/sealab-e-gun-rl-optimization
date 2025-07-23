import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from surrogate import H5Dataset  # import from your file
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
        
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
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        y = y.unsqueeze(-1)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self._val_outputs = []  # reset buffer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        y = y.unsqueeze(-1)
        #print(pred.shape, y.shape)
        loss = self.loss_fn(pred, y)
        acc = ((pred > 0.5) == y.bool()).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        # Store output manually
        self._val_outputs.append({
            'x': x.detach().cpu(),
            'pred': pred.detach().cpu(),
            'label': y.detach().cpu()
        })

    def on_validation_epoch_end(self):
        if not hasattr(self, "_val_outputs") or len(self._val_outputs) == 0:
            return

        limit = 1000
        all_x = torch.cat([o['x'] for o in self._val_outputs], dim=0)[:limit]
        all_preds = torch.cat([o['pred'] for o in self._val_outputs], dim=0)[:limit]
        all_labels = torch.cat([o['label'] for o in self._val_outputs], dim=0)[:limit]

        pred_labels = (all_preds > 0.5).float()
        correct_mask = (pred_labels == all_labels).squeeze(-1)
        incorrect_mask = ~correct_mask

        test_mask = (all_labels == 1.).squeeze(-1)
        print(all_x[test_mask][0], "results in", pred_labels[test_mask][0])
        #print(all_x.shape)
        
        x = all_x.numpy()
        print("Running t-SNE on validation features...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        x_tsne = tsne.fit_transform(x)
        print("xtsne", x_tsne.shape)
        print("correctmask", correct_mask.shape)
        print("pred_labels all_labels", pred_labels.shape, all_labels.shape)

        plt.figure(figsize=(8, 6))
        plt.scatter(x_tsne[correct_mask, 0], x_tsne[correct_mask, 1],
                    c='green', label='Correct', alpha=0.5, s=10)
        plt.scatter(x_tsne[incorrect_mask, 0], x_tsne[incorrect_mask, 1],
                    c='red', label='Incorrect', alpha=0.5, s=10)
        plt.title("t-SNE of Validation Set: Correct vs Incorrect Predictions")
        plt.legend()
        plt.grid(True)

        fig_path = "outputs/val_tsne_correct_incorrect.png"
        plt.savefig(fig_path)
        plt.close()

        if isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({"val_tsne_scatter": wandb.Image(fig_path)})

        # Clear buffer
        self._val_outputs.clear()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def prepare_data(self):

        full_dataset = H5Dataset(self.hparams.data_path, omit_outliers=False)
        un_z_scored_y = full_dataset.un_z_score_y(full_dataset.y)
        # Define binary label: valid = 1, invalid = 0
        #validity = (
        #    (torch.abs(un_z_scored_y[:,0]) < 30) &
        #    (torch.abs(un_z_scored_y[:,1]) < 30) &
        #    (torch.abs(un_z_scored_y[:,2]) < 30) &
        #    (torch.abs(un_z_scored_y[:,3]) < 30)
        #).float()
        
        #print(f"{validity.sum().item()} out of {full_dataset.y[:, 0].shape[0]} samples are valid.")
        
        isnan_mask = torch.isnan(full_dataset.y[:, :4]).any(dim=1)
        validity = isnan_mask.float()
        #print(f"{isnan_mask.sum().item()} out of {full_dataset.y[:, 0].shape[0]} samples are nan.")
        #print("full_dataset", full_dataset.x[:].max())

        class ValidityDataset(Dataset):
            def __init__(self, x, labels):
                self.x = x
                self.labels = labels

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                return self.x[idx], self.labels[idx]

        dataset = ValidityDataset(full_dataset.x, validity)
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        #print("trainsize", train_size, "val_size", val_size)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
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
    wandb_logger = WandbLogger(project="berlinpro_validity", name="validity_clf", save_dir=model.hparams.output_dir)

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
        accelerator="gpu" if hparams.gpus > 0 else "cpu",
        devices=hparams.gpus if hparams.gpus > 0 else 1,
        log_every_n_steps=10,
    )

    trainer.fit(model)
