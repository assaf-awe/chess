import numpy as np
import torch
import chess

INT_TO_PIECE = {
    1: chess.PAWN,
    2: chess.KNIGHT,
    3: chess.BISHOP,
    4: chess.ROOK,
    5: chess.QUEEN,
    6: chess.KING
}

def matrix_to_board(matrix):
    """
    Convert 8x8 numpy array or tensor to chess.Board
    matrix: integers -6..-1 for black, 1..6 for white
    """
    # ensure numpy array
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().squeeze().numpy()
    
    board = chess.Board.empty()  # empty board

    for r in range(8):
        for c in range(8):
            val = matrix[r, c]
            if val == 0:
                continue  # empty square

            color = chess.WHITE if val > 0 else chess.BLACK
            piece_type = INT_TO_PIECE[abs(val)]
            square = chess.square(c, 7-r)  # chess.square(file, rank) with rank 0=bottom
            board.set_piece_at(square, chess.Piece(piece_type, color))
    
    return board