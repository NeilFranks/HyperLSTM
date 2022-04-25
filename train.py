import sys
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

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
    "Away_GA_allGoalies", "Away_SA_allGoalies"
]


def main(*args):
    full_dataset = HockeyDataset(
        "data/standardized_data.csv",
        features,
        pad_length=20,
        # only get games which occured from 1950 to 1960
        restrict_to_years=[e-1918 for e in range(2010, 2023)]
    )
    # full_dataset = PCAHockeyDataset("data/standardized_data.csv", pad_length=20,
    #         n_components=32)
    # full_dataset = HockeyDataset("data/standardized_data.csv", features, pad_length=20)
    # full_dataset = ParityDataset(10240, length=4) # Small reasonable parity ds

    # split dataset into train and test
    l = len(full_dataset)
    train_p = int(0.8*l)          # (80%)
    val_p = int(0.1*l)          # (10%)

    # Hacky mode where we overfit a batch
    # train_p = 128
    # val_p = 128
    test_p = l - train_p - val_p  # (last ~10%)

    # k = 20
    train_dataset, validation_dataset, test_dataset = random_split(
        full_dataset,
        [train_p, val_p, test_p]
    )

    # we now have datasets pointing to varying-length sequences of games
    # within each sequence, the same team is either the home team or away team for every game

    input_size = full_dataset[0][0].shape[1]
    # hidden_size = 64
    hidden_size = 16
    hyper_size = hidden_size // 2
    output_size = 1
    # output_size = 2
    n_z = full_dataset[0][0].shape[1]
    n_layers = 1
    batch_size = 128  # Really helps with stability, trust me :)
    # batch_size = 32  # Really helps with stability, trust me :)

    model = HyperLSTMWrapper(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        hyper_size=hyper_size,
        n_z=n_z,
        n_layers=n_layers,
        batch_size=batch_size
    )

    # model = LSTMWrapper(
    #     input_size=input_size,
    #     output_size=output_size,
    #     hidden_size=hidden_size,
    #     batch_size=batch_size
    # )

    # model = FeedForwardBaseline(
    #     input_size=input_size,
    #     output_size=output_size,
    #     hidden_size=hidden_size,
    #     batch_size=batch_size,
    #     n_layers=3
    # )

    csv_logger = CSVLogger(
        'csv_data',
        name='hockey',
        flush_logs_every_n_steps=1
    )

    # Let's call this our default seed
    # pl.seed_everything(2022, workers=True)
    # Here is me testing if seeds affect init
    # pl.seed_everything(0, workers=True) # Confirmed: seed affects init (0.498 loss)
    # pl.seed_everything(1, workers=True) # Confirmed: seed affects init (0.512 loss)
    trainer = pl.Trainer(
        accelerator=DEVICE,
        log_every_n_steps=1,
        # max_steps=1024 * 10,
        max_epochs=100000,
        logger=csv_logger,
        callbacks=[CheckpointEveryNSteps(save_step_frequency=1024)],
        devices=1
        # auto_lr_find=True
    )

    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=batch_size, num_workers=8),
        DataLoader(validation_dataset, batch_size=batch_size, num_workers=8),
        # ckpt_path="lightning_logs/version_21/checkpoints/N-Step-Checkpoint_epoch=3_global_step=0.ckpt"
    )


if __name__ == '__main__':
    main(sys.argv[1:])
