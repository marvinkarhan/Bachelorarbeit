import sys
import os
import model
import torch
import util
import argparse
from torch import nn


# returns a bytearray with the model coalesced in it
# and the weights/bias quantized
def model_to_bin(nnue: model.NNUE) -> bytearray:
  buffer = bytearray()
  nnue.train(False)

  input_to_buffer(buffer, nnue.input)
  layer_to_buffer(buffer, nnue.l1)
  layer_to_buffer(buffer, nnue.l2)
  layer_to_buffer(buffer, nnue.output, True)

  return buffer


# Add the feature transformer (input) to the bytearray
def input_to_buffer(buffer: bytearray, input: nn.Linear):
  bias = input.bias.data
  bias = bias.mul(127).round().to(torch.int16)
  buffer.extend(bias.flatten().numpy().tobytes())

  # transpose data to fit nnue model
  weight = input.weight.data.t()
  weight = weight.mul(127).round().to(torch.int16)
  buffer.extend(weight.flatten().numpy())

def layer_to_buffer(buffer: bytearray, layer: nn.Linear, is_output=False):
  # following scales discoverd by stockfish
  # layers are stored as int8 weights, and int32 biases
  kWeightScaleBits = 6
  kActivationScale = 127.0
  if not is_output:
    kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
  else:
    kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
  kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
  kMaxWeight = 127.0 / kWeightScale # roughly 2.0

  bias = layer.bias.data
  bias = bias.mul(kBiasScale).round().to(torch.int32)
  buffer.extend(bias.flatten().numpy().tobytes())

  weight = layer.weight.data
  weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
  buffer.extend(weight.flatten().numpy().tobytes())

  # load file and convert to binary
def save_ckpt(ckpt_path: str, output: str, full_path = False):
  nnue = model.NNUE.load_from_checkpoint(ckpt_path)
  cwd = os.path.dirname(os.path.realpath(__file__))
  out_path = output if full_path else f'{cwd}/{output}'
  with open(out_path, 'wb') as f:
    f.write(model_to_bin(nnue))

def main():
  parser = argparse.ArgumentParser(description='Parse ckpt to nnue.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('input', nargs='?', default='last', help='Checkpoint path to a .ckpt file. Can be either a path to a ckpt or "last".')
  parser.add_argument('output', nargs='?', default='default.nnue', help='Name of output .nnue file.')
  args = parser.parse_args()
  if args.input == 'last':
    args.input = util.last_ckpt()
  # read data inputs
  if not args.input.endswith('.ckpt'):
    raise Exception(f'{args.input} does not end with .ckpt')
  if not args.output.endswith('.nnue'):
    raise Exception(f'{args.output} does not end with .nnue')
  util.validate_path(args.input)
  save_ckpt(args.input, args.output)

if __name__ == '__main__':
  main()