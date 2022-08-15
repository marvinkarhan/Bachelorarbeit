import sys
import os
import model
import torch
import numpy
from torch import Tensor, nn


def ascii_hist(name, x, bins=6):
  N,X = numpy.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

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
  ascii_hist('input bias:', bias.numpy())

  # transpose data to fit nnue model
  weight = input.weight.data.t()
  weight = weight.mul(127).round().to(torch.int16)
  buffer.extend(weight.flatten().numpy())
  ascii_hist('input weight:', weight.numpy())

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
  ascii_hist('layer bias:', bias.numpy())

  weight = layer.weight.data
  weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
  buffer.extend(weight.flatten().numpy().tobytes())
  ascii_hist('layer weight:', weight.numpy())


def main():
  # read data inputs
  if len(sys.argv) < 3:
    raise Exception('Input input .ckpt file and output .nnue filename.')
  else:
    if not os.path.exists(sys.argv[1]):
      raise Exception(f'{sys.argv[1]}  does not exist')
    if not sys.argv[1].endswith('.ckpt'):
      raise Exception(f'{sys.argv[1]} does not end with .ckpt')
    if not sys.argv[2].endswith('.nnue'):
      raise Exception(f'{sys.argv[2]} does not end with .nnue')
  input, output = sys.argv[1], sys.argv[2]
  # load file and convert to binary
  nnue = model.NNUE.load_from_checkpoint(input)
  cwd = os.path.dirname(os.path.realpath(__file__))
  # nnue.join_layers().tofile(f'{cwd}/{output}')
  with open(f'{cwd}/{output}', 'wb') as f:
    f.write(model_to_bin(nnue))

if __name__ == '__main__':
  main()