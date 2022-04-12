import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM


class LSTMWrapper(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, proj_size=output_size
        )

        self.automatic_optimization = False

    def forward(self, x, h, c):
        # in lightning, forward defines the prediction/inference actions
        return self.lstm(x, (h, c))

    def compute_loss(self, batch):
        x, y = batch
        yh, _ = self(x.float(), h, c)
        return F.cross_entropy(yh, y)

    def training_step(self, batch):
        # From SAM example: https://github.com/davda54/sam
        optimizer = self.optimizers()
        loss0 = self.compute_loss(batch)
        self.manual_backward(loss0)
        optimizer.first_step(zero_grad=True)
        self.log("train_loss", loss0)

        loss1 = self.compute_loss(batch)
        self.manual_backward(loss1)
        optimizer.second_step(zero_grad=True)
        return loss0

    def configure_optimizers(self):
        optimizer = SAM(
            self.parameters(),
            torch.optim.Adam,
            rho=0.05,
            adaptive=False,
            lr=3e-4
        )
        return optimizer
