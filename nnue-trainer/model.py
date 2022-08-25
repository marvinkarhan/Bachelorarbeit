import pytorch_lightning as pl
import torch
from torch import Tensor, nn
import halfkp
import numpy as np

L1 = 256
L2 = 32

class NNUE(pl.LightningModule):
  def __init__(self):
    super(NNUE, self).__init__()
    # single perspective nets for each side
    # account for features separately
    self.input = nn.Linear(halfkp.NUM_INPUTS, L1)
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L2)
    self.output = nn.Linear(L2, 1)

  def forward(self, us, them, white, black) -> nn.Linear:
    white = self.input(white)
    black = self.input(black)
    # clipped relu activation function (0.0, 1.0) 
    input_layer = torch.clamp(((us * torch.cat([white, black], dim=1)) + (them * torch.cat([black, white], dim=1))), 0.0, 1.0)
    l1 = torch.clamp(self.l1(input_layer), 0.0, 1.0)
    l2 = torch.clamp(self.l2(l1), 0.0, 1.0)
    return self.output(l2)

  def _step(self, loss_type, batch) -> Tensor:
    us, them, white, black, wdl_outcome, score = batch
    # hyperparameters tuned by stockfish accounting for the data provided by them
    # determines the shape of the sigmoid
    net2score = 600
    # in and out scaling differentiates because the input score gets added to the game out come and still should match the net eval 
    in_scaling = 410
    out_scaling = 361

    # calculate the win draw loss by forward activation of the net and scale it to levels adjusted to normal cp value/scores provided by the training data
    # use sigmoid to convert centipawn to 1, 0, -1 values (wdl)
    # could use a lambda to increase/decrease the influence of net evaluation and score given by the input data 
    wdl_eval = (self(us, them, white, black) * net2score / out_scaling).sigmoid()
    # also scale the score of the input data to wdl
    wdl_score = (score / in_scaling).sigmoid()
    
    wdl_target = wdl_score + wdl_outcome
    # exponent tuning from stockfish, values >2 choosing more precision over accuracy
    loss = torch.pow(torch.abs(wdl_target - wdl_eval), 2.6).mean()

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
