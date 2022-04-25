import torch
from torch import nn
import torch.nn.functional as F
from .hyperlstm import HyperLSTM
from .sequence_wrapper import SequenceWrapper


class HyperLSTMWrapper(SequenceWrapper):
    def __init__(self, input_size, output_size, hidden_size, hyper_size, n_z, n_layers, batch_size):
        super().__init__()
        self.hyper_lstm = HyperLSTM(
            input_size,
            hidden_size,
            hyper_size,
            n_z,
            n_layers
        )
        self.l0 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        seq, state = self.hyper_lstm(x, None)
        return F.relu(self.l0(seq)), state

