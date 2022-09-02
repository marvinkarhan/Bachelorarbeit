import os
import util
import save
import shutil
import re
import argparse
from typing import List

RELATIVE_DEFAULT_ENGINE = '../../../../uci-engine'
OPENING_BOOK_EPD = 'lib/books/UHO_V3_6mvs_+090_+099.epd'
CUTECHESS_CLI = './lib/cutechess-cli/cutechess-cli'
ORDO = './lib/ordo/ordo-linux64'

DIR = 'estimate_elo'

# using cutechess (cli): https://github.com/cutechess/cutechess

class Engine:
  def __init__(self, file = RELATIVE_DEFAULT_ENGINE, name = '', options=[]):
    self.file = file
    self.name = name
    self.option = "".join([f'option.{o[0]}={o[1]} ' for o in options])

  def to_string(self):
    string = f' -engine cmd={self.file} '
    if self.name:
      string += f'name={self.name} '
    if self.option:
      string += self.option
    return string

class Tournament:
  def __init__(self, run: int, engines: List[Engine], concurrency: int, games: int, time: int, increment = 0):
    self.run = run
    self.engines = engines
    self.concurrency = concurrency
    self.games = games
    self.tc = f'{time}+{increment}' if increment else f'{time}'
   
  def start(self):
    out_dir = get_out_dir(self.run).replace('./', '')
    cmd = (
      f'{CUTECHESS_CLI} -each proto=uci tc={self.tc} dir={out_dir} restart=on '
      f'{"".join([e.to_string() for e in self.engines])} '
      f'-event Estimate_Elo_on_run_{self.run} '
      f'-tournament gauntlet '
      f'-games {self.games} '
      f'-rounds 1 '
      f'-pgnout {out_dir}/out.pgn '
      f'-openings file={OPENING_BOOK_EPD} format=epd order=random '
      f'-concurrency {self.concurrency} '
      f'-resign movecount=3 score=1000 '
      f'-draw movenumber=40 movecount=8 score=10 '
    )
    print(cmd)
    return os.system(cmd)

def get_out_dir(run: int):
  ckpt_paths = util.ckpt_paths(run)
  out_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_paths[0])), DIR)
  return out_dir

def convert_ckpts(run: int):
  ckpt_paths = util.ckpt_paths(run)
  out_dir = get_out_dir(run)
  out_paths = []

  for ckpt_path in ckpt_paths:
    epoch = re.findall(r"epoch=\d*", ckpt_path)
    out_name = (epoch[0] if len(epoch) > 0 else os.path.basename(ckpt_path).replace('.ckpt', '')) + '.nnue'
    out_path = os.path.join(out_dir, out_name)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    save.save_ckpt(ckpt_path, out_path, True)
    out_paths.append(out_name)
  return out_paths

def setup_engines(nets):
  engines = [Engine(name='master')]
  engines += [Engine(name=net.replace('.nnue', ''), options=[['nnueFile', net]]) for net in nets]
  return engines

def estimate_elo_with_ordo(run: int, concurrency: int):
  out_dir = get_out_dir(run)
  pgn_file = f'{out_dir}/out.pgn'
  ordo_file = f'{out_dir}/ordo_rating.txt'
  cmd = (
    f'{ORDO} '
    f'-q ' # quiet
    f'-a 0 ' # avg to 0 in order to diff to master
    f'-A master ' # show result relative to master
    f'-D ' # adjust for draw rate
    f'-W ' # adjust for white advantage
    f'-s 100 ' # use 100 simulations to estimate error
    f'-p {pgn_file} '
    f'-o {ordo_file} '
  )
  print(cmd)
  return os.system(cmd)

def clean_out_dir(run: int):
  out_dir = get_out_dir(run)
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

def main():
  parser = argparse.ArgumentParser(description='Plays a gauntlet tournament comparing a runs nets to master.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--run', default=-1, type=int, help='Specify which run to test. If none is omitted, it defaults to the last run.')
  parser.add_argument('--games', default=200, type=int, help='Games in gauntlet tournament per engine, against master.')
  parser.add_argument('--concurrency', default=6, type=int, help='Number of Threads for running the tournament.')
  parser.add_argument('--tc', default=10, type=int, help='Number of Seconds for each Game.')
  parser.add_argument('--clean', action='store_true', help='Clean previous elo estimation files, if there are any.')
  args = parser.parse_args()
  # clean up if requested
  if args.clean:
    clean_out_dir(args.run)
  # convert ckpts of specified run
  nets = convert_ckpts(args.run)
  # get list of Engines
  engines = setup_engines(nets)
  # run a gauntlet tournament against master
  tournament = Tournament(args.run, engines, args.concurrency, args.games, args.tc, 0.05)
  tournament.start()
  # estimate the elo with ordo
  estimate_elo_with_ordo(args.run, args.concurrency)

if __name__ == '__main__':
  main()
