import torch
from torch import nn
import pytorch_lightning as pl


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        pass


class PL_Model(pl.LightningModule):
    def __init__(self):
        super(PL_Model, self).__init__()
        self.model = Model()

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)