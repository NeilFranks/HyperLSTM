import os
import sys
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from datasets.hockey import HockeyDataset
from datasets.parity import ParityDataset

from models import *
from datasets import *
from checkpointer import *

def main(*args):
    # full_dataset = HockeyDataset("data/standardized_data.csv")
    full_dataset = ParityDataset(10240, length=4) # Small reasonable parity ds

    # split dataset into train and test
    l = len(full_dataset)
    train_p = int(0.8*l)
    val_p   = int(0.1*l)
    test_p  = int(0.1*l)

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
    # batch size has to be 1 because sequences are different lengths (maybe theres another way to fix this)
    # batch_size = 1
    batch_size = 128 # Really helps with stability, trust me :)

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

    csv_logger = CSVLogger(
        'csv_data',
        name='hockey',
        flush_logs_every_n_steps=1
    )

    pl.seed_everything(2022, workers=True)
    trainer = pl.Trainer(
        accelerator='cpu',
        log_every_n_steps=1,
        max_steps=1024,
        logger=csv_logger,
        callbacks=[CheckpointEveryNSteps(save_step_frequency=1024)]
        # auto_lr_find=True
    )

    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=batch_size, num_workers=4),
        DataLoader(validation_dataset, batch_size=batch_size, num_workers=4),
        # ckpt_path="lightning_logs/version_21/checkpoints/N-Step-Checkpoint_epoch=3_global_step=0.ckpt"
    )


if __name__ == '__main__':
    main(sys.argv[1:])
