from turtle import forward
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
import halfkp
import cupy as cp

L1 = 256
L2 = 32

class NNUE(pl.LightningModule):
  def __init__(self, psqt_buckets=8, ls_buckets=8):
    super(NNUE, self).__init__()
    # piece square table buckets
    self.psqt_buckets = psqt_buckets
    # layer stack buckets
    self.ls_buckets = ls_buckets
    # single perspective nets for each side
    # account for features separately
    self.input = nn.Linear(halfkp.NUM_INPUTS, L1)
    self.l1 = nn.Linear(2*L1, L2)
    self.l2 = nn.Linear(L2, L2)
    self.output = nn.Linear(L2, 1)

  def forward(self, us, them, white, black) -> nn.Linear:
    white = self.input(white)
    black = self.input(black)
    input_layer = nn.functional.relu(us * torch.cat([white, black], dim=1) + them * torch.cat([black, white], dim=1))
    l1 = nn.functional.relu(self.l1(input_layer))
    l2 = nn.functional.relu(self.l2(l1))
    return self.output(l2)

  def _step(self, loss_type, batch) -> Tensor:
    us, them, white, black, outcome, score = batch
    net2score = 600
    in_scaling = 410
    out_scaling = 361

    q = (self(us, them, white, black) * net2score / out_scaling).sigmoid()
    t = outcome
    p = (score / in_scaling).sigmoid()
    
    pt = p + t
    loss = torch.pow(torch.abs(pt - q), 2.6).mean()

    self.log(loss_type, loss)
    return loss

  def training_step(self, batch, _) -> Tensor:
    return self._step('training_loss', batch)

  def validation_step(self, batch, _) -> None:
    self._step('validation_loss', batch)

  # Use simple Adadelta optimizer
  def configure_optimizers(self) -> torch.optim.Adadelta:
    optimizer = torch.optim.Adadelta(self.parameters(), lr=1, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.3)

    return [optimizer], [scheduler]
