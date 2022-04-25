import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM


class SequenceWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.automatic_optimization = False
        # self.lr = 3e-4

    def forward(self, x):
        raise NotImplementedError('Define in derived class')

    def compute_loss(self, batch):
        x, y = batch
        y_hat, _ = self(x.float())
        return F.binary_cross_entropy_with_logits(
            torch.squeeze(y_hat).type(torch.FloatTensor),
            torch.squeeze(y).type(torch.FloatTensor)
        )

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch)
    #     # From SAM example: https://github.com/davda54/sam
    #     optimizer = self.optimizers()
    #     loss0 = self.compute_loss(batch)
    #     # self.manual_backward(loss0)
    #     # optimizer.first_step(zero_grad=True)

    #     optimizer.zero_grad()
    #     torch.set_grad_enabled(True)
    #     loss0.backward()
    #     optimizer.step()
    #     self.log("train_loss", loss0)

    #     # loss1 = self.compute_loss(batch)
    #     # self.manual_backward(loss1)
    #     # optimizer.second_step(zero_grad=True)
    #     return loss0

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch)

    def configure_optimizers(self):
        # optimizer = SAM(self.parameters(), torch.optim.Adam, rho=0.05,
        #                 adaptive=False, lr=self.lr)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001
        )
        return optimizer
