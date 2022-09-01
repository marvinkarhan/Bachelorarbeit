import sys
import os
import torch
import model
import util
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

FEATURE_SET_NAME = 'HalfKP'
BATCH_SIZE = 2**14
NUM_WORKERS = 8
SMART_FEN_SKIPPING = True
RANDOM_FEN_SKIPPING = 10
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
    if not os.path.exists(sys.argv[2]):
      raise Exception('{0} does not exist'.format(sys.argv[2]))
    if len(sys.argv) > 3 and sys.argv[3] != 'latest' and not os.path.exists(sys.argv[3]):
      raise Exception('{0} does not exist or ist named wrong'.format(sys.argv[3]))
  train_filename, val_filename = sys.argv[1], sys.argv[2]

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train, val = util.make_data_loaders(train_filename, val_filename, FEATURE_SET_NAME, NUM_WORKERS, BATCH_SIZE, SMART_FEN_SKIPPING, RANDOM_FEN_SKIPPING, device, EPOCH_SIZE, VAL_SIZE)

  # continue from ckpt
  ckpt_path = None
  if len(sys.argv) > 3:
    ckpt_path = util.last_ckpt() if sys.argv[3] == 'latest' else sys.argv[3]
    nnue = model.NNUE.load_from_checkpoint(ckpt_path)
  else:
    nnue = model.NNUE()

  logger = loggers.TensorBoardLogger('logs/')
  ckpt_callback = ModelCheckpoint(save_top_k=-1, every_n_epochs=25)
  trainer = pl.Trainer(logger=logger, max_epochs=MAX_EPOCHS, accelerator='gpu', devices=1,callbacks=[ckpt_callback, ModelCheckpoint()])
  if ckpt_path:
    trainer.fit(nnue, train, val, ckpt_path=ckpt_path)
  else:
    trainer.fit(nnue, train, val)

if __name__ == '__main__':
  main()