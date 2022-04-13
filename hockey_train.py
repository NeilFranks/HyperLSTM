import sys
from numpy import full
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from datasets.hockey import HockeyDataset

from models import *
from datasets import *


def main(*args):
    full_dataset = HockeyDataset("data/standardized_data.csv")

    # split dataset into train and test
    TRAINING_SIZE = int(0.7*len(full_dataset))

    train_dataset, validation_dataset = random_split(
        full_dataset,
        [TRAINING_SIZE, len(full_dataset)-TRAINING_SIZE]
    )

    # we now have datasets pointing to varying-length sequences of games
    # within each sequence, the same team is either the home team or away team for every game

    input_size = full_dataset[0][0].shape[1]
    hidden_size = 64
    hyper_size = hidden_size // 2
    output_size = 1
    n_z = full_dataset[0][0].shape[1]
    n_layers = 10
    # batch size has to be 1 because sequences are different lengths (maybe theres another way to fix this)
    batch_size = 1

    model = HyperLSTMWrapper(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        hyper_size=hyper_size,
        n_z=n_z,
        n_layers=n_layers,
        batch_size=batch_size
    )

    trainer = pl.Trainer(accelerator="gpu")
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=batch_size, num_workers=4),
        DataLoader(validation_dataset, batch_size=batch_size, num_workers=4)
    )


if __name__ == '__main__':
    main(sys.argv[1:])
