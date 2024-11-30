import chess
import numpy as np
import random

def encode_piece(piece_type:int, piece_color:bool) -> np.ndarray:
    """ Encode a chess piece from a int to a tuple with shape==(12,)
    Examples:
        >>> encode_piece(5, False)
        array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int8)
    """
    enconded_piece = np.zeros((12,), dtype=np.int8)
    piece_type_color = piece_type - 1 + 6*piece_color
    enconded_piece[piece_type_color] = 1
    return enconded_piece

def get_chess_grid(board:chess.Board) -> np.array:
    # Initialize an 8x8 matrix filled with empty strings
    grid = np.zeros((8,8,12), dtype=np.int8)

    # Map square indices to rows and columns
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)  # Convert to matrix row (0 is the top)
        col = chess.square_file(square)     # Convert to matrix column (0 is 'a')
        encoded_piece = encode_piece(piece.piece_type, piece.color)
        grid[row][col] = encoded_piece     # Use the piece's symbol

    return grid

def action_sample(board:chess.Board):
        legal_moves = list(board.board.generate_legal_moves())
        
        if len(legal_moves) == 0:
            return None

        move = random.choice(legal_moves)
        return move.from_square, move.to_square