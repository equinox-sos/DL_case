import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, insize, outsize):
        super(Model, self).__init__()
        self.l1 = nn.Linear(insize, outsize)

    def forward(self, x):
        return self.l1(x)


class PL_Model(pl.LightningModule):
    def __init__(self, insize, outsize):
        super(PL_Model, self).__init__()
        self.model = Model(insize, outsize)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return F.mse_loss(y, y_hat)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return F.mse_loss(y, y_hat)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)