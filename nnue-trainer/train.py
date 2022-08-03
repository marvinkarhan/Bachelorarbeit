import chess
from torch.utils.data import DataLoader
import nnue_dataset
import visualize
import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
import model


FEATURE_SET_NAME = 'HalfKP'
BATCH_SIZE = 2**13
NUM_WORKERS = 12
SMART_FEN_SKIPPING = True
RANDOM_FEN_SKIPPING = 3
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
  train_filename, val_filename = sys.argv[1], sys.argv[2]

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train, val = make_data_loaders(train_filename, val_filename, FEATURE_SET_NAME, NUM_WORKERS, BATCH_SIZE, SMART_FEN_SKIPPING, RANDOM_FEN_SKIPPING, device, EPOCH_SIZE, VAL_SIZE)

  # visualize.visualize_data_loader(train)

  nnue = model.NNUE()

  logger = loggers.TensorBoardLogger('logs/')
  trainer = pl.Trainer(logger=logger, max_epochs=MAX_EPOCHS, accelerator='gpu', devices=1)
  trainer.fit(nnue, train, val)

if __name__ == '__main__':
  main()