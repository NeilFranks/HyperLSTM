import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM

class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.automatic_optimization = False

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def compute_loss(self, batch):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
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
