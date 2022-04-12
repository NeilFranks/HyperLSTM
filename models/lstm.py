import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM

class LSTMWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        h0 = torch.randn(2, 3, 20) # From docs
        c0 = torch.randn(2, 3, 20)
        return self.lstm(x, h0, c0)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch # x is an input, y is label
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        loss = F.mse_loss(self(x), y) # Assuming MSE is appropriate to the dataset
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = SAM(self.parameters(), torch.optim.Adam, rho=0.05,
                        adaptive=False, lr=3e-4)
        return optimizer
