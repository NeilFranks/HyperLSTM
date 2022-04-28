import sys
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from sklearn.model_selection import train_test_split

from models import *
from datasets import *
from checkpointer import *

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

    # constructing dataset
    full_dataset = HockeyDataset(
        "data/standardized_data.csv",
        features,
        sequence_length=sequence_length,
        restrict_to_years=[e-1918 for e in range(2005, 2016)]
    )

    # get all the y
    y = [e[1] for e in full_dataset]
    print(f"home team won {round(float(100*sum(y)/len(y)), 2)}% of the time.")

    # split dataset into train and test
    train_dataset, validation_dataset = train_test_split(
        full_dataset,
        train_size=0.8,
        test_size=0.2,
        random_state=313,  # so split is reproducible
        shuffle=True,
        stratify=y
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

    csv_logger = CSVLogger(
        'csv_data',
        name='hockey',
        flush_logs_every_n_steps=1,
    )

    trainer = pl.Trainer(
        accelerator=DEVICE,
        log_every_n_steps=1,
        max_epochs=100000,
        logger=csv_logger,
        callbacks=[CheckpointEveryNSteps(save_step_frequency=100)],
        devices=1,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset, batch_size=batch_size, num_workers=5, shuffle=True
        ),
        val_dataloaders=DataLoader(
            validation_dataset, batch_size=batch_size, num_workers=5
        ),
        # ckpt_path="csv_data/hockey/version_296/checkpoints/N-Step-Checkpoint_epoch=80_global_step=18600.ckpt"
    )


if __name__ == '__main__':
    main(None, sys.argv[1:])

    # seed = 5187
    # main(seed, sys.argv[1:])

    # # Train with global seeds
    # for i in range(10):
    #     print(f"\n\n\tUsing global seed {seed}\n\n")
    #     try:
    #         main(seed, sys.argv[1:])
    #     except:
    #         print(f"\n\n\t{seed} is no good for a seed\n\n")

    #     # fight randomness with randomness....
    #     seed = random.randrange(0, 6666)
