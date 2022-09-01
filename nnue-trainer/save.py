import sys
import os
import model
import torch
import util
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
  # read data inputs
  if len(sys.argv) < 3:
    raise Exception('Input input .ckpt file and output .nnue filename.')
  else:
    if not os.path.exists(sys.argv[1]) and sys.argv[1] != 'latest':
      raise Exception('{0} does not exist'.format(sys.argv[1]))
    if not sys.argv[1].endswith('.ckpt') and sys.argv[1] != 'latest':
      raise Exception('{0} does not end with .ckpt'.format(sys.argv[1]))
    if not sys.argv[2].endswith('.nnue'):
      raise Exception(f'{sys.argv[2]} does not end with .nnue')
  input = util.last_ckpt() if sys.argv[1] == 'latest' else sys.argv[1]
  output = sys.argv[2]
  save_ckpt(input, output)

if __name__ == '__main__':
  main()