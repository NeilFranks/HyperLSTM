import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceWrapper(pl.LightningModule):
    def __init__(self, seed, batch_size):
        super().__init__()
        # self.automatic_optimization = False
        # self.lr = 3e-4

        self.last_loss = None
        self.val_loss = None

        self.logged_seed_and_co = False
        self.seed = seed
        self.batch_size = batch_size

    def forward(self, x):
        raise NotImplementedError('Define in derived class')

    def compute_loss(self, batch):
        x, y = batch
        y_hat, _ = self(x.float())
        y_hat = torch.squeeze(y_hat).type(torch.FloatTensor).to(DEVICE)
        y = torch.squeeze(y).type(torch.FloatTensor).to(DEVICE)

        tensor_bce = F.binary_cross_entropy_with_logits(
            y_hat,
            y,
        )

        # to find accuracy, round predictions to either to 0 or 1 and count proportion of correct predictions
        rounded_predictions = torch.tensor(
            [0 if abs(e-0) < abs(e-1) else 1 for e in y_hat]
        ).to(DEVICE)
        accuracy = sum(
            y.type(torch.int16) == rounded_predictions.type(torch.int16)
        )/len(y)

        return tensor_bce, accuracy

    def log_seed_trainp_batchsize(self):
        if self.seed:
            self.log("seed", float(self.seed))
        self.log(
            "train_size",
            float(
                len(self.trainer.train_dataloader.dataset.datasets)
            )
        )
        self.log("batch_size", float(self.batch_size))
        self.log(
            "val_size",
            float(
                len(self.trainer.val_dataloaders[0].dataset)
            )
        )

        self.logged_seed_and_co = True

    def training_step(self, batch, batch_idx):
        if not self.logged_seed_and_co:
            self.log_seed_trainp_batchsize()

        # This is the default loss code
        loss, accuracy = self.compute_loss(batch)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.log("lr", self.optimizers().param_groups[0]['lr'])

        # if loss == self.last_loss:
        #     raise Exception("Training loss stagnated... something aint right")

        self.last_loss = loss
        return loss
        # And if automatic optimization is off..
        # self.manual_backward(loss)

        # From SAM example: https://github.com/davda54/sam
        # optimizer = self.optimizers()
        # loss0 = self.compute_loss(batch)
        # self.manual_backward(loss0)
        # optimizer.first_step(zero_grad=True)
        # self.log("train_loss", loss0)
        # loss1 = self.compute_loss(batch)
        # self.manual_backward(loss1)
        # optimizer.second_step(zero_grad=True)
        # return loss0

    def validation_step(self, batch, batch_idx):
        val_loss, val_accuracy = self.compute_loss(batch)
        self.log("val_loss", val_loss)
        self.log("val_accuracy", val_accuracy)
        # self.log("metric_to_track", val_loss)
        return self.val_loss

    def test_step(self, batch, batch_idx, dataset_idx):
        test_loss, test_accuracy = self.compute_loss(batch)
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_accuracy)
        return test_loss

    def configure_optimizers(self):
        # optimizer = SAM(self.parameters(), torch.optim.Adam, rho=0.05,
        #                 adaptive=False, lr=self.lr)
        # Use this if we don't want SAM
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.0002
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     # mode='max',
        #     factor=0.5,
        #     patience=20,
        #     min_lr=0.00005
        # )

        return (
            {
                "optimizer": optimizer,
                # "lr_scheduler": {
                #     "scheduler": scheduler,

                #     # whatever we log as "metric_to_track" is found by the scheduler
                #     "monitor": "metric_to_track",

                #     # track it every epoch
                #     "interval": "epoch",
                # },
            }
        )
