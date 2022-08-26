import enum
import sys
import os
import util
import model
import torch
import pytorch_lightning as pl

FEATURE_SET_NAME = 'HalfKP'
BATCH_SIZE = 1000
NUM_WORKERS = 1
SMART_FEN_SKIPPING = False
RANDOM_FEN_SKIPPING = 0
EPOCH_SIZE = 100000000
VAL_SIZE = 1000000
MAX_EPOCHS = 800

def main():
  # read data inputs
  if len(sys.argv) < 3:
    raise Exception('Input training and validation datasets.')
  else:
    if not os.path.exists(sys.argv[1]):
      raise Exception('{0} does not exist'.format(sys.argv[1]))
    if not sys.argv[2].endswith('.ckpt') and sys.argv[2] != 'latest':
      raise Exception('{0} does not end with .ckpt'.format(sys.argv[2]))
  filename, net_filename = sys.argv[1], util.last_ckpt() if sys.argv[2] == 'latest' else sys.argv[2]
  nnue = model.NNUE.load_from_checkpoint(net_filename)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  nnue.to(device)
  
  data_loader, _ = util.make_data_loaders(filename, filename, FEATURE_SET_NAME, NUM_WORKERS, BATCH_SIZE, SMART_FEN_SKIPPING, RANDOM_FEN_SKIPPING, device, EPOCH_SIZE, VAL_SIZE)

  for batch in data_loader:
    # print(nnue(batch))
    us, them, white, black, wdl_outcome, score = batch
    eval = nnue(us, them, white, black) * 600 / 361
    for i, e in enumerate(eval):
      print(f'eval: {e.item()} -- score: {score[i].item() / 361}')
    # for data in batch:
    #   print(data)
    return


if __name__ == '__main__':
  main()