import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .lstm import LSTMWrapper
from .sam import SAM


class HyperLSTMCell(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, output_size: int):
        super().__init__()

        # NOTE: might just be able to use input_size in place of n_z everywhere... right???

        # make the hyper network, which is just a smaller LSTM
        # the input is the output of the main LSTM (h) concatenated with the normal input (x)
        self.hyper_network = LSTMWrapper(
            input_size=hidden_size+input_size,
            hidden_size=hyper_size,
            output_size=output_size
        )

        # we want the output of these linear operations to have 4 components; i, f, g, o
        self.z_h = nn.Linear(hyper_size, 4*n_z)
        self.z_x = nn.Linear(hyper_size, 4*n_z)
        self.z_b = nn.Linear(hyper_size, 4*n_z, bias=False)

        # transform z into weights
        self.d_h = nn.ModuleList(
            [
                nn.Linear(n_z, hidden_size, bias=False)
                for _ in range(4)
            ]
        )
        self.d_x = nn.ModuleList(
            [
                nn.Linear(n_z, hidden_size, bias=False)
                for _ in range(4)
            ]
        )
        self.d_b = nn.ModuleList(
            [
                nn.Linear(n_z, hidden_size)
                for _ in range(4)
            ]
        )

        # weight matrices
        self.w_h = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(hidden_size, hidden_size))
                for _ in range(4)
            ]
        )
        self.w_x = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(hidden_size, input_size))
                for _ in range(4)
            ]
        )

        # layer normalization
        self.layer_norm = nn.ModuleList(
            [
                nn.LayerNorm(hidden_size)
                for _ in range(4)
            ]
        )
        self.layer_norm_c = nn.LayerNorm(hidden_size)

    def forward(self, x, h, c, h_hat, c_hat):

        # input for the hyper network is h concatenated with x
        x_hat = torch.cat((h, x), dim=-1)

        # use the hyper network to get what we need to change the weights on the main LSTM
        h_hat, c_hat = self.hyper_network(x_hat, h_hat, c_hat)

        # calculate new weights
        z_h = self.z_h(h_hat).chunk(4, dim=-1)
        z_x = self.z_x(h_hat).chunk(4, dim=-1)
        z_b = self.z_b(h_hat).chunk(4, dim=-1)

        ifgo = []
        for index in range(4):
            ifgo.append(
                self.layer_norm[index](
                    # calculate d_h
                    self.d_h[index](z_h[index]) * torch.einsum(
                        'ij,bj->bi', self.w_h[index], h
                    ) +
                    # calculate d_x
                    self.d_x[index](z_x[index]) * torch.einsum(
                        'ij,bj->bi', self.w_x[index], x
                    ) +
                    # calculate d_b
                    self.d_b[index](z_b[index])
                )
            )
        i, f, g, o = ifgo

        c_next = torch.sigmoid(f)*c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(
            self.layer_norm_c(c_next)
        )

        return h_next, c_next, h_hat, c_hat

    def training_step(self, batch):
        # From SAM example: https://github.com/davda54/sam
        optimizer = self.optimizers()
        loss0 = self.compute_loss(batch)
        self.manual_backward(loss0)
        optimizer.first_step(zero_grad=True)
        self.log("train_loss", loss0)

        loss1 = self.compute_loss(batch)
        self.manual_backward(loss1)
        optimizer.second_step(zero_grad=True)
        return loss0

    def configure_optimizers(self):
        optimizer = SAM(
            self.parameters(), torch.optim.Adam, rho=0.05,
            adaptive=False, lr=3e-4)
        return optimizer


class HyperLSTM(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size

        # have layers of cells
        self.cells = nn.ModuleList(
            # first cell takes real input
            [
                HyperLSTMCell(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    hyper_size=hyper_size,
                    n_z=n_z
                )
            ] +
            # every subsequent cell takes input from last cell
            [
                HyperLSTMCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    hyper_size=hyper_size,
                    n_z=n_z
                )
                for _ in range(n_layers - 1)
            ]
        )

    def forward(self, x, h=None, c=None, h_hat=None, c_hat=None):
        n_steps, batch_size = x.shape[:2]

        # if no state was given, initialize with all zeros
        if h is None:
            h = [
                x.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.n_layers)
            ]
        if c is None:
            c = [
                x.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.n_layers)
            ]
        if h_hat is None:
            h_hat = [
                x.new_zeros(batch_size, self.hyper_size)
                for _ in range(self.n_layers)
            ]
        if c_hat is None:
            c_hat = [
                x.new_zeros(batch_size, self.hyper_size)
                for _ in range(self.n_layers)
            ]

        # use the input to generate some output
        outs = []
        for step in range(n_steps):
            inputs = x[step]

            # send it on through the layers
            for layer in range(self.n_layers):
                h[layer], c[layer], h_hat[layer], c_hat[layer] = self.cells[layer](
                    inputs,
                    h[layer],
                    c[layer],
                    h_hat[layer],
                    c_hat[layer]
                )

                # input for next layer is the output for this layer
                inputs = h[layer]

            # remember last output
            outs.append(h[-1])

        outputs = torch.stack(outputs)

        return outputs, (h, c, h_hat, c_hat)
