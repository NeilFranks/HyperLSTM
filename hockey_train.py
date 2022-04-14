import os
import sys
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from datasets.hockey import HockeyDataset

from models import *
from datasets import *


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(
                trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def main(*args):
    full_dataset = HockeyDataset("data/standardized_data.csv")

    # split dataset into train and test
    TRAINING_SIZE = int(0.7*len(full_dataset))

    train_dataset, validation_dataset, _ = random_split(
        full_dataset,
        # [TRAINING_SIZE, len(full_dataset)-TRAINING_SIZE]
        [10, 5, len(full_dataset) - 15]
    )

    # we now have datasets pointing to varying-length sequences of games
    # within each sequence, the same team is either the home team or away team for every game

    input_size = full_dataset[0][0].shape[1]
    hidden_size = 64
    hyper_size = hidden_size // 2
    output_size = 1
    n_z = full_dataset[0][0].shape[1]
    n_layers = 3
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

    csv_logger = CSVLogger(
        'csv_data',
        name='hockey',
        flush_logs_every_n_steps=1
    )

    pl.seed_everything(414, workers=True)
    trainer = pl.Trainer(
        accelerator='gpu',
        log_every_n_steps=1,
        max_steps=1000,
        logger=csv_logger,
        callbacks=[CheckpointEveryNSteps(save_step_frequency=1000)]
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
