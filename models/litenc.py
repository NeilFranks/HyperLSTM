import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def compute_loss(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        loss0 = self.compute_loss(batch)
        self.manual_backward(loss0)
        optimizer.first_step(zero_grad=True)
        self.log("train_loss", loss0)

        # From SAM example: https://github.com/davda54/sam
        loss1 = self.compute_loss(batch)
        self.manual_backward(loss1)
        optimizer.second_step(zero_grad=True)
        return loss0

    def configure_optimizers(self):
        optimizer = SAM(self.parameters(), torch.optim.SGD, rho=0.05,
                        adaptive=False, lr=3e-4)
        return optimizer
