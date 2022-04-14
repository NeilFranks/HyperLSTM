import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM


class SequenceWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def forward(self, x):
        raise NotImplementedError('Define in derived class')

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

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        return loss

    def configure_optimizers(self):
        optimizer = SAM(self.parameters(), torch.optim.Adam, rho=0.05,
                        adaptive=False, lr=3e-4)
        return optimizer
