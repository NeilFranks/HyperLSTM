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
        # mask = x[:, :, -1] # b, t, w
        # mask_index = int(torch.sum(mask))
        # y     =     y[:, :mask_index, :]
        # y_hat = y_hat[:, :mask_index, :]
        return F.binary_cross_entropy_with_logits(
            torch.squeeze(y_hat).type(torch.FloatTensor),
            torch.squeeze(y).type(torch.FloatTensor)
        )

    # def compute_loss(self, batch):
    #     x, y = batch
    #     y_hat, _ = self(x.float())
    #     return F.binary_cross_entropy_with_logits(
    #         torch.squeeze(y_hat).type(torch.FloatTensor),
    #         torch.squeeze(y).type(torch.FloatTensor)
    #     )

    def training_step(self, batch, batch_idx):
        # This is the default loss code
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss
        # And if automatic optimization is off..
        # self.manual_backward(loss)

        # From SAM example: https://github.com/davda54/sam
        # optimizer = self.optimizers()
        # loss0 = self.compute_loss(batch)
        # self.manual_backward(loss0)
        # optimizer.first_step(zero_grad=True)
        # self.log("train_loss", loss0)
        # loss1 = self.compute_loss(batch)
        # self.manual_backward(loss1)
        # optimizer.second_step(zero_grad=True)
        # return loss0

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch)

    def configure_optimizers(self):
        # optimizer = SAM(self.parameters(), torch.optim.Adam, rho=0.05,
        #                 adaptive=False, lr=self.lr)
        # Use this if we don't want SAM
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001
        )
        return optimizer
