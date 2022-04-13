import sys
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from models import *
from datasets import *


def main(*args):
    scale = 1024
    length = 16
    hidden_size = 64
    hyper_size = hidden_size // 2
    output_size = 2
    n_z = 16
    n_layers = 1
    batch_size = 64

    parity = ParityDataset(scale * 10, length=length, zeros=False)
    train, val = random_split(
        parity, [scale * 9, scale])  # Intentionally small

    # model = LSTMWrapper(1, output_size, hidden_size, batch_size)
    model = HyperLSTMWrapper(
        input_size=1,
        output_size=output_size,
        hidden_size=hidden_size,
        hyper_size=hyper_size,
        n_z=n_z,
        n_layers=n_layers,
        batch_size=batch_size
    )

    pl.seed_everything(42, workers=True)
    trainer = pl.Trainer(deterministic=True)
    trainer.fit(
        model,
        DataLoader(train, batch_size=batch_size, num_workers=4),
        DataLoader(val, batch_size=batch_size, num_workers=4)
    )


if __name__ == '__main__':
    main(sys.argv[1:])
