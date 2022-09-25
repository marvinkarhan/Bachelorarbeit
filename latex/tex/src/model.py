from symbol import parameters
from torch import nn
import torch


feature_transformer = nn.Linear(41024, 256)
affine_transformer1 = nn.Linear(2 * 256, 32)
affine_transformer2 = nn.Linear(32, 32)
output = nn.Linear(32, 1)

def forward(us, them, white, black):
  white = feature_transformer(white)
  black = feature_transformer(black)
  # clipped relu activation function (0.0, 1.0) 
  ft = torch.clamp((us * torch.cat([white, black], dim=1)) + (them * torch.cat([black, white], dim=1)), 0.0, 1.0)
  at1 = torch.clamp(affine_transformer1(ft), 0.0, 1.0)
  at2 = torch.clamp(affine_transformer2(at1), 0.0, 1.0)
  return output(at2)

lambda_ = 1

def training_step(self, batch):
  us, them, white, black, wdl_outcome, cp_score = batch
  net_to_cp_score = 600
  scaling = 361

  q = (forward(us, them, white, black) * net_to_cp_score / scaling).sigmoid()
  t = wdl_outcome
  p = (cp_score / scaling).sigmoid()

  epsilon = 1e-12
  p_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
  t_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
  p_loss = -(p * q.log() + (1.0 - p) * (-q).log())
  t_loss = -(t * q.log() + (1.0 - t) * (-q).log())
  result  = lambda_ * p_loss    + (1.0 - lambda_) * t_loss
  entropy = lambda_ * p_entropy + (1.0 - lambda_) * t_entropy
  loss = result.mean() - entropy.mean()
  return loss

parameters = []

torch.optim.Adadelta(parameters)