import nnue_dataset
import sys
import os
import torch
import model
import util
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch.utils.data import DataLoader


FEATURE_SET_NAME = 'HalfKP'
BATCH_SIZE = 2**14
NUM_WORKERS = 8
SMART_FEN_SKIPPING = True
RANDOM_FEN_SKIPPING = 10
EPOCH_SIZE = 100000000
VAL_SIZE = 1000000
MAX_EPOCHS = 800


# using data loader from the Gary Linscott (SF NNUE) https://github.com/glinscott/nnue-pytorch/blob/master/train.py
def make_data_loaders(train_filename, val_filename, feature_set_name, num_workers, batch_size, filtered, random_fen_skipping, main_device, epoch_size, val_size):
  # Epoch and validation sizes are arbitrary
  train_infinite = nnue_dataset.SparseBatchDataset(feature_set_name, train_filename, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)
  val_infinite = nnue_dataset.SparseBatchDataset(feature_set_name, val_filename, batch_size, num_workers=num_workers, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def main():  
  # read data inputs
  if len(sys.argv) < 3:
    raise Exception('Input training and validation datasets.')
  else:
    if not os.path.exists(sys.argv[1]):
      raise Exception('{0} does not exist'.format(sys.argv[1]))
    if not os.path.exists(sys.argv[2]):
      raise Exception('{0} does not exist'.format(sys.argv[2]))
    if len(sys.argv) > 3 and sys.argv[3] != 'latest' and not os.path.exists(sys.argv[3]):
      raise Exception('{0} does not exist or ist named wrong'.format(sys.argv[3]))
  train_filename, val_filename = sys.argv[1], sys.argv[2]

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train, val = make_data_loaders(train_filename, val_filename, FEATURE_SET_NAME, NUM_WORKERS, BATCH_SIZE, SMART_FEN_SKIPPING, RANDOM_FEN_SKIPPING, device, EPOCH_SIZE, VAL_SIZE)

  # continue from ckpt
  ckpt_path = None
  if len(sys.argv) > 3:
    ckpt_path = util.last_ckpt() if sys.argv[3] == 'latest' else sys.argv[3]
    nnue = model.NNUE.load_from_checkpoint(ckpt_path)
  else:
    nnue = model.NNUE()

  logger = loggers.TensorBoardLogger('logs/')
  trainer = pl.Trainer(logger=logger, max_epochs=MAX_EPOCHS, accelerator='gpu', devices=1)
  if ckpt_path:
    trainer.fit(nnue, train, val, ckpt_path=ckpt_path)
  else:
    trainer.fit(nnue, train, val)

if __name__ == '__main__':
  main()