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

def transformInput(white_indices: Tensor, white_values: Tensor, black_indices: Tensor, black_values: Tensor) -> Tensor:
  device = white_indices.device
  size, max_active_features = white_indices.shape

def _find_nearest_divisor(value, target):
    divisors = []
    for i in range(1, value+1):
        if value % i == 0:
            divisors.append((i, abs(target-i)))
    divisors.sort(key=lambda x:x[1])
    return divisors[0][0]

_num_threads_forward_cache = dict()
def _get_num_threads_for_forward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(output_size, optimal_num_threads)

    return _num_threads_forward_cache[output_size]

_feature_transformer_slice_forward_kernel_cache = dict()
def make_feature_transformer_slice_forward_kernel(max_active_features, output_size):
    '''
        @param: max_active_features
            The maximum number of features that are active
            (non-zero) for a single position. This value determines
            the shape of the inputs.
            This value is of type uint32_t.

        @param: output_size
            The number of outputs. Must match the shape of weights
            and biases.
            This value is of type uint32.
    '''
    num_threads = _get_num_threads_for_forward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_features, output_size, num_threads)
    if key not in _feature_transformer_slice_forward_kernel_cache:
        kernel = cp.RawKernel(r'''

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__

/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: feature_indices
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing indices of active features for each position
        in a batch. Feature index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the feature_indices matrix.
        The type for feature indices is int32_t.

    @param: feature_values
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing the values (arity) of the corresponding
        feature index in feature_indices.
        The type for the feature value (arity) is float32.

    @param: weight
        The weight matrix of shape (NUM_INPUTS, output_size).
        Weights must be of type float32.

    @param: bias
        The bias vector of shape (output_size,).
        Bias values must be of type float32.

    @param: output
        An output matrix of shape (BATCH_SIZE, output_size).
        It may not be initialized, bias is always copied
        to the output first.
        Output values must have type float32.
*/
void feature_transformer_slice_forward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const float*   const weight,
    const float*   const bias,
          float*   const output
) {{
    __shared__
          float          shared_output[{output_size}];

    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       slice_offset        = threadIdx.x * {output_thread_slice_size};

          float*   const output_slice        = output + block_idx * {output_size} + slice_offset;
    const float*   const bias_slice          = bias                               + slice_offset;
          float*         shared_output_slice = shared_output                      + slice_offset;

    const int32_t* const feature_index_row   = feature_indices + block_idx * {max_active_features};
    const float*   const feature_value_row   = feature_values  + block_idx * {max_active_features};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_slice[s] = bias_slice[s];
    }}

    for (uint32_t k = 0; k < {max_active_features}; ++k)
    {{
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        if (feature_index != -1)
        {{
            const float* const weight_slice = weight + feature_index * {output_size} + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
            {{
                shared_output_slice[s] += weight_slice[s] * feature_value;
            }}
        }} else break;
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        output_slice[s] = shared_output_slice[s];
    }}
}}

'''.format(
                max_active_features=max_active_features,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size),
            'feature_transformer_slice_forward')
        kernel.compile()
        _feature_transformer_slice_forward_kernel_cache[key] = _kernel_with_threads(kernel, (num_threads,))
    return _feature_transformer_slice_forward_kernel_cache[key]

def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)
    return f