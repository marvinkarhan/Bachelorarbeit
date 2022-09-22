from torch import nn
import torch


feature_transformer = nn.Linear(41024, 256)
affine_transformer1 = nn.Linear(2 * 256, 32)
affine_transformer2 = nn.Linear(32, 32)
output = nn.Linear(32, 1)

def forward(us, them, feature_transformer_white, feature_transformer_black):
  feature_transformer_white = feature_transformer(feature_transformer_white)
  feature_transformer_black = feature_transformer(feature_transformer_black)
  # clipped relu activation function (0.0, 1.0) 
  feature_transformer_ = torch.clamp(((us * torch.cat([feature_transformer_white, feature_transformer_black], dim=1)) + (them * torch.cat([feature_transformer_black, feature_transformer_white], dim=1))), 0.0, 1.0)
  affine_transformer1_ = torch.clamp(affine_transformer1(feature_transformer_), 0.0, 1.0)
  affine_transformer2_ = torch.clamp(affine_transformer2(affine_transformer1_), 0.0, 1.0)
  return output(affine_transformer2_)