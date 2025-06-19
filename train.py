import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 100, image_size: int = 64):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        return image


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_dataset = DummyDataset(self.cfg['data']['train_samples'],
                                          self.cfg['data']['image_size'])
        self.val_dataset = DummyDataset(self.cfg['data']['val_samples'],
                                        self.cfg['data']['image_size'])
        self.test_dataset = DummyDataset(self.cfg['data']['test_samples'],
                                         self.cfg['data']['image_size'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg['train']['batch_size'],
            num_workers=self.cfg['train']['num_workers']
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg['train']['batch_size'],
            num_workers=self.cfg['train']['num_workers']
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg['train']['batch_size'],
            num_workers=self.cfg['train']['num_workers']
        )


class SimpleAutoEncoder(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LitModel(pl.LightningModule):
    def __init__(self, image_size: int, lr: float, warmup_steps: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleAutoEncoder(image_size)
        self.lr = lr
        self.warmup_steps = warmup_steps

    def training_step(self, batch, batch_idx):
        recon = self.model(batch)
        loss = F.mse_loss(recon, batch)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        recon = self.model(batch)
        loss = F.mse_loss(recon, batch)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min(1.0, (step + 1) / self.warmup_steps)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }
        return optimizer


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train HiDream model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg.get('seed', 42))

    dm = DummyDataModule(cfg)
    logger = WandbLogger(project=cfg.get('wandb_project', 'hidream_train'))

    model = LitModel(
        cfg['data']['image_size'],
        cfg['train']['lr'],
        cfg['train']['warmup_steps']
    )

    trainer = pl.Trainer(
        max_steps=cfg['train']['max_steps'],
        logger=logger,
        precision=cfg['train']['precision'],
        accelerator=cfg['train']['accelerator'],
        devices=cfg['train']['devices'],
        strategy=cfg['train']['strategy'],
        gradient_clip_val=cfg['train']['gradient_clip_val'],
        log_every_n_steps=cfg['train']['log_every_n_steps'],
    )
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=cfg['train'].get('resume_from_checkpoint')
    )
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
