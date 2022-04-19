import torch
from torch import nn
from .sequence_wrapper import SequenceWrapper


class LSTMWrapper(SequenceWrapper):
    def __init__(self, input_size, output_size, hidden_size, batch_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        super().__init__()
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, proj_size=output_size)
        self.automatic_optimization = False

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        h0 = torch.randn(1, self.batch_size, self.output_size)
        c0 = torch.randn(1, self.batch_size, self.hidden_size)
        return self.lstm(x, (h0, c0))
