import glob
import os
import nnue_dataset
from torch.utils.data import DataLoader

def last_ckpt() -> str:
  list_ckpt_paths = glob.glob('./logs/lightning_logs/*/checkpoints/*.ckpt')
  ckpt_path = max(list_ckpt_paths, key=os.path.getctime)
  return ckpt_path

def ckpt_paths(run = -1) -> str:
  ckpt_dir_glob = f'./logs/lightning_logs/{"*" if int(run) < 0 else "version_" + run}/checkpoints/'
  list_run_ckpt_path = glob.glob(ckpt_dir_glob)
  ckpt_paths = glob.glob(f'{max(list_run_ckpt_path, key=os.path.getctime)}*.ckpt')
  if len(ckpt_paths) <= 0:
    raise Exception('No ckpts found!')
  return ckpt_paths

def validate_path(*paths: str):
  for path in paths:
    if not os.path.exists(path):
      raise Exception('{0} does not exist'.format(path))

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