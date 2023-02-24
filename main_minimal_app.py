import os
import sys
import numpy as np
from scipy import stats
import torch
from torch import nn
from mvhg.pt_fmvhg import MVHG

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

import hydra
from dataclasses import dataclass, field
from typing import List, Dict
from omegaconf import MISSING, OmegaConf
from hydra.core.config_store import ConfigStore


@dataclass
class MyHGConf:
    m: List[int] = field(default_factory=lambda: [200, 200, 200])
    n: int = 180
    w: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    seed: int = 0
    n_samples: int = 1000
    batch_size: int = 1
    learning_rate: float = 0.001
    temperature: float = 0.5
    n_epochs: int = 20


cs = ConfigStore.instance()
cs.store(name="config", node=MyHGConf)


class HGBasicModule(pl.LightningModule):
    def __init__(self, lr, m_all, n, temperature, device):
        super().__init__()
        self.n_classes = m_all.shape[0]
        self.lr = lr
        self.m_all = m_all
        self.n = n
        self.temperature = temperature
        self.log_omega = torch.nn.parameter.Parameter(torch.zeros(1, self.n_classes))
        self.mvhg = MVHG(device=device)
        # Save hyperparameters
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        _, b_data, b_labels = batch
        x_data = []
        for c in range(0, self.n_classes):
            x_data.append(b_data[0, c])
        x_out = self(x_data)
        loss = 0.0
        for c in range(0, self.n_classes):
            loss += (x_data[c] - x_out[c]) ** 2
        self.log("train_loss", loss)
        for c in range(0, self.n_classes):
            log_w_c = self.log_omega[0, c]
            self.log("log_w_" + str(c), log_w_c)
        return loss

    def validation_step(self, batch, batch_idx):
        _, b_data, b_labels = batch
        x_data = []
        w_gt = b_labels[0]
        for c in range(0, self.n_classes):
            x_data.append(b_data[0, c])
        x_out = self(x_data)
        loss = 0.0
        for c in range(0, self.n_classes):
            loss += (x_data[c] - x_out[c]) ** 2
        w_l_norm = self.log_omega.exp() / sum(self.log_omega.exp())
        w_gt_norm = w_gt.unsqueeze(0) / sum(w_gt)
        mse_w = ((w_gt_norm - w_l_norm) ** 2).mean()
        self.log(
            "validation_mse_w",
            mse_w,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log("validation_loss", loss)
        return loss

    def forward(self, x):
        mvhg_out = self.mvhg(
            self.m_all,
            self.n,
            self.log_omega,
            self.temperature,
            add_noise=True,
            hard=True,
        )
        x_out = mvhg_out[1]
        return x_out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99),
        }


def create_data(m_all, n, w_all, num_samples):
    n_classes = m_all.shape[0]
    w_all = w_all.unsqueeze(0).repeat(num_samples, 1)
    n = n.unsqueeze(0).repeat(num_samples, 1)
    n_out = np.zeros((num_samples, 1))
    rvs_all_classes_ref = np.zeros((num_samples, n_classes))
    for c in range(n_classes - 1):
        m_i = m_all[c].squeeze(0).repeat(num_samples, 1)
        m_i = m_i.cpu().numpy()
        m_rest = m_all[c + 1 :].sum().squeeze(0).repeat(num_samples, 1)
        m_rest = m_rest.cpu().numpy()
        n_i = n.cpu().numpy() - n_out
        w_i = w_all[:, c].cpu().numpy()
        w_rest_enum = (m_all[c + 1 :] * w_all[:, c + 1 :]).sum(dim=1, keepdims=True)
        w_rest_enum = w_rest_enum.cpu().numpy()
        w_rest_denom = m_rest
        w_rest = (w_rest_enum / w_rest_denom).flatten()
        w = w_i / w_rest
        M = m_i + m_rest
        x_i = stats.nchypergeom_fisher.rvs(
            M.flatten(), m_i.flatten(), n_i.flatten(), w, size=num_samples
        )
        n_out += np.expand_dims(x_i, axis=1)
        rvs_all_classes_ref[:, c] = x_i
    rvs_all_classes_ref[:, -1] = (n.cpu().numpy() - n_out).flatten()
    return rvs_all_classes_ref


def get_dataloader(samples, w_all, batch_size):
    n_samples_train = int(samples.shape[0] * 0.8)
    train_samples = samples[:n_samples_train]
    train_samples = torch.tensor(train_samples)
    val_samples = samples[n_samples_train:]
    val_samples = torch.tensor(val_samples)

    train_labels = w_all.unsqueeze(0).repeat(train_samples.shape[0], 1)
    val_labels = w_all.unsqueeze(0).repeat(val_samples.shape[0], 1)

    dataset = torch.utils.data.TensorDataset(
        torch.arange(train_samples.shape[0]), train_samples, train_labels
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    dataset = torch.utils.data.TensorDataset(
        torch.arange(val_samples.shape[0]), val_samples, val_labels
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True, num_workers=4
    )
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyHGConf):
    pl.utilities.seed.seed_everything(cfg.seed)

    dir_logs = "./minimal_app_mvhg/"
    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)

    seed = cfg.seed
    device = "cpu"
    m = torch.tensor(cfg.m)
    n = torch.tensor(cfg.n)
    n = n.unsqueeze(0)
    w = torch.tensor(cfg.w)
    n_samples = cfg.n_samples
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    temp = cfg.temperature
    data = create_data(m, n, w, n_samples)
    train_loader, val_loader = get_dataloader(data, w, batch_size)

    model = HGBasicModule(learning_rate, m, n, temp, device)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    wandb_logger = WandbLogger(
        project="minimal_app_mvhg",
        save_dir=dir_logs,
        group="nSamples_"
        + str(cfg.n_samples)
        + "_w1_"
        + str(float(w[1].item()))
        + "_nEpochs_"
        + str(cfg.n_epochs),
    )
    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        devices=1,
        accelerator="cpu",
        logger=wandb_logger,
        # auto_lr_find=True,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    run_experiment()
