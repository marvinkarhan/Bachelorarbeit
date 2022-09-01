# halfkp represents (piece_square, piece_type, piece_color)
# allows the understanding of the pieces in relation to the king

NUM_SQUARES = 64
# piece squares = 10 kings are not included
NUM_PIECE_TYPES = 10
NUM_PLANES = NUM_SQUARES * NUM_PIECE_TYPES + 1
NUM_INPUTS = NUM_PLANES * NUM_SQUARES