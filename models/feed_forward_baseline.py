import torch
from torch import nn
import torch.nn.functional as F
from torch import vmap

import pytorch_lightning as pl
from .sequence_wrapper import SequenceWrapper

class FeedForwardBaseline(SequenceWrapper):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super().__init__()
        self.first = nn.Linear(input_size,  hidden_size)
        self.mid = []
        for _ in range(n_layers):
            self.mid.append(nn.Linear(hidden_size, hidden_size))
        self.final = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        h = vmap(self.first)(x)
        h = F.gelu(h)
        for layer in self.mid:
            h = vmap(layer)(h)
            h = F.gelu(h)
        o = vmap(self.final)(h)
        o = F.gelu(o)
        return o

    def compute_loss(self, batch):
        x, y = batch
        y_hat = self(x.float())
        y_hat = torch.squeeze(y_hat).type(torch.FloatTensor)
        y     = torch.squeeze(y).type(torch.FloatTensor)

        mask = x[:, :, -1] # b, t, w (temporal mask for padded sequences)
        tensor_bce = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
        masked_bce = torch.mul(mask, tensor_bce) # elementwise (hadamard) product
        return torch.mean(masked_bce)
