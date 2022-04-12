import sys, os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from models   import *
from datasets import *

def main(*args):
    scale = 1024
    length = 16
    hidden_size = 64
    output_size = 1

    parity = ParityDataset(scale * 10, length=length, zeros=False)
    train, val = random_split(parity, [scale * 9, scale]) # Intentionally small

    model = LSTMWrapper(length, output_size, hidden_size)

    pl.seed_everything(42, workers=True)
    trainer = pl.Trainer(deterministic=True)
    trainer.fit(model, DataLoader(train), DataLoader(val))

if __name__ == '__main__':
    main(sys.argv[1:])
