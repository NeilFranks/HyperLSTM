import numpy as np
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ParityDataset(Dataset):
    datapoints     : int
    rng            : object = None
    length         : int = 16
    zeros          : bool = False

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()

    def __len__(self):
        return self.datapoints

    def __getitem__(self, i):
        inx, label = self.encode(*self.generate())
        return inx, label

    def generate(self):
        sequence = self.rng.integers(low=0, high=2, size=self.length)
        if not self.zeros: # Use -1 encoding instead of 0
            sequence = sequence * 2 - np.ones(self.length)
            parity = np.sum(sequence) == 0
        else:
            parity = np.abs(np.sum(sequence)) % 2 == 0
        return sequence, parity

    def encode(self, sequence, parity):
        if parity:
            label = np.array([1], dtype=np.float32)
        else:
            label = np.array([0], dtype=np.float32)
        label = np.expand_dims(label, -1)
        return np.expand_dims(sequence, -1), label

if __name__ == '__main__':
    n = 1024
    length = 16
    parity = ParityDataset(n, length=length, zeros=False)
    for i in range(100):
        x, y = parity[i]
        print(x.shape)
        print(y.shape)
        1/0
