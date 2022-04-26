import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .sam import SAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceWrapper(pl.LightningModule):
    def __init__(self, seed, train_p, batch_size):
        super().__init__()
        # self.automatic_optimization = False
        # self.lr = 3e-4

        self.last_loss = None
        self.val_loss = None

        self.logged_seed_and_co = False
        self.seed = seed
        self.train_p = train_p
        self.batch_size = batch_size

    def forward(self, x):
        raise NotImplementedError('Define in derived class')

    def compute_loss(self, batch):
        x, y = batch
        y_hat, _ = self(x.float())
        y_hat = torch.squeeze(y_hat).type(torch.FloatTensor).to(DEVICE)
        y = torch.squeeze(y).type(torch.FloatTensor).to(DEVICE)

        # mask = x[:, :, -1]  # b, t, w (temporal mask for padded sequences)
        tensor_bce = F.binary_cross_entropy_with_logits(
            y_hat, y, reduction='none'
        )
        # elementwise (hadamard) product
        # masked_bce = torch.mul(mask, tensor_bce)
        # return torch.mean(masked_bce)

        return torch.mean(tensor_bce)

    # def compute_loss(self, batch):
    #     x, y = batch
    #     y_hat, _ = self(x.float())
    #     return F.binary_cross_entropy_with_logits(
    #         torch.squeeze(y_hat).type(torch.FloatTensor),
    #         torch.squeeze(y).type(torch.FloatTensor)
    #     )

    def log_seed_trainp_batchsize(self):
        if self.seed:
            self.log("seed", float(self.seed))
        self.log("train_p", float(self.train_p))
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
        loss = self.compute_loss(batch)

        self.log("train_loss", loss)
        if self.val_loss:
            self.log("lr", self.optimizers().param_groups[0]['lr'])
            self.log("val_loss", self.val_loss)
            self.log("metric_to_track", self.val_loss)

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
        self.val_loss = self.compute_loss(batch)
        return self.val_loss

    def test_step(self, batch, batch_idx, dataset_idx):
        loss = self.compute_loss(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = SAM(self.parameters(), torch.optim.Adam, rho=0.05,
        #                 adaptive=False, lr=self.lr)
        # Use this if we don't want SAM
        optimizer = torch.optim.Adam(
            self.parameters(),
            # lr=0.0003
            lr=0.003
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.75,
            patience=25,
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,

                    # whatever we log as "metric_to_track" is found by the scheduler
                    "monitor": "metric_to_track",

                    # track it every epoch
                    "interval": "epoch",
                },
            }
        )
