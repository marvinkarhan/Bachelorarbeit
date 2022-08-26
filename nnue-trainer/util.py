import glob
import os

def last_ckpt() -> str:
  list_ckpt_paths = glob.glob('./logs/lightning_logs/*/checkpoints/*.ckpt')
  ckpt_path = max(list_ckpt_paths, key=os.path.getctime)
  print(f'Loading ckpt: {ckpt_path}')
  return ckpt_path
