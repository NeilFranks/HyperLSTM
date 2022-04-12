import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM
from .hyperlstm import HyperLSTM
from .sequence_wrapper import SequenceWrapper

class HyperLSTMWrapper(SequenceWrapper):
    def __init__(self, input_size, output_size, hidden_size, hyper_size, n_z, n_layers, batch_size):
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size  = batch_size
        self.hyper_size  = hyper_size
        self.n_z         = n_z
        self.n_layers    = n_layers
        super().__init__()
        self.hyper_lstm = HyperLSTM(self.input_size, self.hidden_size, self.hyper_size,
                                    self.n_z, self.n_layers)
        self.l0 = nn.Linear(self.hidden_size, self.output_size)
        self.automatic_optimization = False

    def forward(self, x):
        x = torch.permute(x, (2, 0, 1))
        # in lightning, forward defines the prediction/inference actions
        # h0  = torch.randn(self.batch_size, self.hidden_size)
        # c0  = torch.randn(self.batch_size, self.hidden_size)
        # hh0 = torch.randn(self.batch_size, self.hyper_size)
        # ch0 = torch.randn(self.batch_size, self.hyper_size)
        # self.hyper_lstm(x, (h0, c0, hh0, ch0))
        seq, state = self.hyper_lstm(x, None)
        return F.relu(self.l0(seq)), state

    def compute_loss(self, batch):
        ''' Loss for parity task '''
        x, y = batch
        yh, _ = self(x.float())
        yh = yh[0, :, :]
        return F.cross_entropy(yh, torch.squeeze(y))
