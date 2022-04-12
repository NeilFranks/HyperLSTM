import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM

class LSTMWrapper(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size, batch_size=1):
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size  = batch_size
        super().__init__()
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, proj_size=output_size)
        self.automatic_optimization = False

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        h0 = torch.randn(1, self.batch_size, self.output_size)
        c0 = torch.randn(1, self.batch_size, self.hidden_size)
        return self.lstm(x, (h0, c0))

    def compute_loss(self, batch):
        x, y = batch
        yh, _ = self(x.float())
        return F.cross_entropy(yh, y)

    def training_step(self, batch, batch_idx):
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
        optimizer = SAM(self.parameters(), torch.optim.Adam, rho=0.05,
                        adaptive=False, lr=3e-4)
        return optimizer
