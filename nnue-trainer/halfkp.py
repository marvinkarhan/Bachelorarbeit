import chess
import torch

# halfkp represents (piece_square, piece_type, piece_color)
# allows the understanding of the pieces in relation to the king

NUM_SQUARES = 64
# piece squares = 10 kings are not included
NUM_PIECE_TYPES = 10
NUM_PLANES = NUM_SQUARES * NUM_PIECE_TYPES + 1
NUM_INPUTS = NUM_PLANES * NUM_SQUARES

# get a square 
def orient(active_side: bool, square: int):
  return (63 * (not active_side)) ^ square

def halfkp_index(active_side: bool, king_square: int, square: int, piece: chess.Piece):
  piece_index = (piece.piece_type - 1) * 2 + (piece.color != active_side)
  return 1 + orient(active_side, square) + piece_index * NUM_SQUARES + king_square * NUM_PLANES

def get_piece_indices(active_side: bool, board: chess.Board):
  indices = torch.zeros(NUM_INPUTS)
  for square, piece in board.piece_map().items():
    if piece.piece_type != chess.KING:
      indices[halfkp_index(active_side, orient(active_side, board.king(active_side)), square, piece)] = 1

# returns feature set for both sides
def get_halfkp_features(board: chess.Board):
  return (get_piece_indices(board.turn), get_piece_indices(not board.turn))