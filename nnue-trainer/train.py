import sys
import os
import torch
import warnings
import model
import util
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

# disable dataloader warning
warnings.filterwarnings("ignore", ".*does not have many workers.*")

FEATURE_SET_NAME = 'HalfKP'
BATCH_SIZE = 2**14
NUM_WORKERS = 8
SMART_FEN_SKIPPING = True
RANDOM_FEN_SKIPPING = 10
EPOCH_SIZE = 100000000
VAL_SIZE = 1000000

def main():
  parser = argparse.ArgumentParser(description='Train NNUE.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('train_file', help='Training data path in .binpack format.')
  parser.add_argument('val_file', help='Validation data path in .binpack format.')
  parser.add_argument('--resume', type=str, dest='ckpt_path', help='Resume from a checkpoint (.ckpt). Can be either a path to a ckpt or "last".')
  parser.add_argument('--epochs', default=800, dest='max_epochs', type=int, help='Max Epochs.')
  parser.add_argument('--device', default=0, dest='device_index', type=int, help='Specify which gpu to use.')
  parser.add_argument('--lambda', default=1.0, dest='lambda_', type=int, help='Ratio of eval to wdl score in percent. 1.0 means only eval score and 0 means only wdl score. (0.0 - 1.0)')
  args = parser.parse_args()
  # validate path inputs
  util.validate_path(args.train_file, args.val_file)

  # continue from ckpt
  ckpt_path = args.ckpt_path
  if ckpt_path:
    if ckpt_path == 'last':
      ckpt_path = util.last_ckpt()
    util.validate_path(ckpt_path)
    nnue = model.NNUE.load_from_checkpoint(ckpt_path)
  else:
    nnue = model.NNUE(args.lambda_)

  logger = loggers.TensorBoardLogger('logs/')
  ckpt_callback = ModelCheckpoint(save_top_k=-1, every_n_epochs=25)
  trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator='gpu', devices=1, callbacks=[ckpt_callback, ModelCheckpoint()])

  device_index = args.device_index if args.device_index else str(trainer.strategy.root_device.index)
  device = torch.device('cpu') if trainer.strategy.root_device.index is None else f'cuda:{device_index}'

  train, val = util.make_data_loaders(args.train_file, args.val_file, FEATURE_SET_NAME, NUM_WORKERS, BATCH_SIZE, SMART_FEN_SKIPPING, RANDOM_FEN_SKIPPING, device, EPOCH_SIZE, VAL_SIZE)

  if ckpt_path:
    trainer.fit(nnue, train, val, ckpt_path=ckpt_path)
  else:
    trainer.fit(nnue, train, val)

if __name__ == '__main__':
  main()