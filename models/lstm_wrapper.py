from torch import nn
import torch.nn.functional as F
from .sequence_wrapper import SequenceWrapper


class LSTMWrapper(SequenceWrapper):
    def __init__(self, input_size, output_size, hidden_size, batch_size, seed):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        super().__init__(seed=seed, batch_size=batch_size)
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, proj_size=output_size)
        self.l0 = nn.LazyLinear(output_size)
        # self.automatic_optimization = False

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        seq, state = self.lstm(x, None)
        return F.relu(self.l0(seq))[:, -1], state
