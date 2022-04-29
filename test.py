import sys
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

from models import *
from datasets import *
from checkpointer import *

from train import dataset_split

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

features = [
    "Year", "Month", "Day",
    "Home_ID",
    "Home_wins_last10",
    "Home_wins_VERSUS_last2",
    "Home_goals_lastGame", "Home_assists_lastGame",
    "Home_GA_startingGoalie", "Home_SA_startingGoalie",
    "Home_GA_allGoalies", "Home_SA_allGoalies",
    "Away_ID",
    "Away_wins_last10",
    "Away_wins_VERSUS_last2",
    "Away_goals_lastGame", "Away_assists_lastGame",
    "Away_GA_startingGoalie", "Away_SA_startingGoalie",
    "Away_GA_allGoalies", "Away_SA_allGoalies",
    "Home_Won"
]


def main(seed, *args):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if seed:
        pl.seed_everything(seed, workers=True)

    # look at sequences of length 10
    sequence_length = 10
    full_dataset, train_dataset, validation_dataset = dataset_split(
        sequence_length=sequence_length
    )

    # we now have datasets pointing to varying-length sequences of games
    # within each sequence, the same team is either the home team or away team for every game

    input_size = full_dataset[0][0].shape[1]
    hidden_size = 16
    hyper_size = int(hidden_size*0.75)
    output_size = 1
    n_z = full_dataset[0][0].shape[1]
    n_layers = 2
    batch_size = 8  # Really helps with stability, trust me :)

    model = HyperLSTMWrapper(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        hyper_size=hyper_size,
        n_z=n_z,
        n_layers=n_layers,

        sequence_length=sequence_length,

        seed=seed,
        batch_size=batch_size
    )

    trainer = pl.Trainer(
        accelerator=DEVICE,
        # log_every_n_steps=1,
        max_epochs=100000,
        # logger=csv_logger,
        devices=1
    )

    trainer.test(
        model,
        dataloaders=[
            DataLoader(
                validation_dataset, batch_size=batch_size, num_workers=5
            ),
            DataLoader(
                validation_dataset, batch_size=batch_size, num_workers=5
            ),
        ],
        ckpt_path="csv_data/hockey/version_310/checkpoints/N-Step-Checkpoint_epoch=167_global_step=88800.ckpt"
    )


if __name__ == '__main__':
    main(None, sys.argv[1:])

    # seed = 5187
    # main(seed, sys.argv[1:])
