from turtle import forward
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
import halfkp
import cupy as cp

L1_size = 256
L2_size = 32

class NNUE(pl.LightningModule):
  def __init__(self, psqt_buckets=8, ls_buckets=8):
    super(NNUE, self).__init__()
    # piece square table buckets
    self.psqt_buckets = psqt_buckets
    # layer stack buckets
    self.ls_buckets = ls_buckets
    # single perspective nets for each side
    # account for features separately
    self.white_input = nn.Linear(halfkp.NUM_INPUTS, L1_size)
    self.black_input = nn.Linear(halfkp.NUM_INPUTS, L1_size)
    self.l1 = nn.Linear(2*L1_size, L2_size)
    self.l2 = nn.Linear(L2_size, L2_size)
    self.output = nn.Linear(L2_size, 1)

  def forward(self, us, them, white_inputs, black_inputs) -> nn.Linear:
    white = self.white_input(white_inputs)
    black = self.black_input(black_inputs)
    input_layer = nn.functional.relu(us * torch.cat([white, black], dim=1) + them * torch.cat([black, white], dim=1))
    l1 = nn.functional.relu(self.l1(input_layer))
    l2 = nn.functional.relu(self.l2(l1))
    return self.output(l2)

  def _step(self, batch) -> Tensor:
    us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices = batch
    output = self(us, them, white_values, black_values)
    loss = nn.functional.mse_loss(output, convert_to_centipawn(score))
    

  def training_step(self, batch, _) -> Tensor:
    # print('training_step: ', us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices)
    pass

  def validation_step(self, batch, _) -> None:
    print('\nlen: ',len(batch), '---', type(batch))
    us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices = batch
    print('\n\n\nvalidation_step: ', white_indices.size())
    # self._step(batch)
    exit()
    # self.log('training_loss', loss)

  # Use simple Adadelta optimizer
  def configure_optimizers(self) -> torch.optim.Adadelta:
    return torch.optim.Adadelta(self.parameters(), lr=1.0)
    
def convert_to_centipawn(x, alpha=0.0016):
  # return (x * alpha).sigmoid()
  return x * 600
