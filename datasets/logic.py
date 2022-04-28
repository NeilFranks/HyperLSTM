import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch

from pprint import pprint
from copy import copy, deepcopy
from collections import defaultdict

from time import time
from dataclasses import dataclass, field
from typing import Any

OPS = ['NOR', 'XQ', 'ABJ', 'XOR', 'NAND', 'AND', 'OR', 'XNOR', 'If/then', 'Then/if']
TRUTH_TABLE = np.array([
    # NOR, XQ, ABJ, XOR, NAND, AND, OR, XNOR, If/then, Then/if
    [   1,  0,   0,   0,    1,   0,  0,    1,       1,       1], # F F
    [   0,  1,   0,   1,    1,   0,  1,    0,       1,       0], # F T
    [   0,  0,   1,   1,    1,   0,  1,    0,       0,       1], # T F
    [   0,  0,   0,   0,    0,   1,  1,    1,       1,       1], # T T
    ])

import numpy as np
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass, field
from typing import Any

@dataclass
class LogicDataset:
    ''' Representation:
            [a1 a2 b1..b10] where:
                - a1 and a2 are bits,
                - b1..b10 are each len-10 one-hot vecs encoding operators o1..o10
            Then, the output is the sequential application of operators o1..o10 to a1 and a2 '''
    datapoints     : int
    rng            : object = None
    length         : int = 8
    n_ops          : int = 10
    base           : int = 2 # Binary
    zeros          : bool = False
    repeat         : int = 1

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()

    def __len__(self):
        return self.datapoints

    def __getitem__(self, i):
        inx, label = self.encode(*self.generate())
        return inx, label

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.n_ops = len(OPS)

    def generate(self):
        # https://stackoverflow.com/questions/41069825/convert-binary-01-numpy-to-integer-or-binary-string
        inp = self.rng.integers(low=0, high=2, size=self.base) # Binary ops
        ops = self.rng.integers(low=0, high=self.n_ops, size=self.length)

        post = inp.copy()
        encs = []
        for op in ops:
            truth_i = post.dot(1 << np.arange(post.size)[::-1])
            new_bit = TRUTH_TABLE[truth_i, op]
            encoding = np.array(F.one_hot(torch.tensor([op]), self.n_ops))
            encs.append(encoding)
            post = np.array([post[-1], new_bit])
        return inp, encs, post

    def encode(self, inp, encs, post):
        full = np.zeros((self.n_ops+self.base, self.length + 1))
        full[:self.base, 0] = inp
        # offset = lambda i : i * self.n_ops + self.base
        for i, enc in enumerate(encs):
            full[self.base:, i+1] = enc
        return full.T, np.expand_dims(post, 0)

from time import sleep
if __name__ == '__main__':
    n = 1024
    length = 8
    logic = LogicDataset(n, length=length)
    for i in range(1):
        inx, lbl = logic[i]
        print(inx.shape)
        print(lbl.shape)
