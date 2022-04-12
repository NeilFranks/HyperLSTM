import sys, os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from models   import *
from custom_datasets import *
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

AVAIL_GPUS = min(1, torch.cuda.device_count())

def main(*args):
    pl.seed_everything(42, workers=True)
    dm = GLUEDataModule(model_name_or_path="albert-base-v2", task_name="cola")
    dm.setup("fit")
    model = GLUETransformer(
        model_name_or_path="albert-base-v2",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
    )

    trainer = pl.Trainer(max_epochs=1, gpus=AVAIL_GPUS)
    trainer.fit(model, datamodule=dm)

    # test = dataset = load_dataset('glue', 'mrpc', split='train')
    # print(test)
    # 1/0
    # dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    # train, val = random_split(dataset, [55000, 5000])

    # # model = LitAutoEncoder()
    # model = MNISTModel()

    # trainer = pl.Trainer(deterministic=True)
    # trainer.fit(model, DataLoader(train), DataLoader(val))

if __name__ == '__main__':
    main(sys.argv[1:])
