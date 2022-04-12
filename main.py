import sys, os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from models   import *
from datasets import *

def main(*args):
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()

    pl.seed_everything(42, workers=True)
    trainer = pl.Trainer(deterministic=True)
    trainer.fit(autoencoder, DataLoader(train), DataLoader(val))

if __name__ == '__main__':
    main(sys.argv[1:])
