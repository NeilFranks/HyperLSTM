from torch import nn
import torch.nn.functional as F
from .hyperlstm import HyperLSTM
from .sequence_wrapper import SequenceWrapper


class HyperLSTMWrapper(SequenceWrapper):
    def __init__(self, input_size, output_size, hidden_size, hyper_size, n_z, n_layers, sequence_length, seed, batch_size):
        super().__init__(seed=seed, batch_size=batch_size)
        self.hyper_lstm = HyperLSTM(
            input_size,
            hidden_size,
            hyper_size,
            n_z,
            n_layers
        )
        self.l0 = nn.Linear(hidden_size, 1)
        self.l1 = nn.Linear(sequence_length, output_size)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        seq, state = self.hyper_lstm(x, None)
        step0 = self.l0(seq)
        step1 = self.l1(step0.squeeze())
        return step1, state
