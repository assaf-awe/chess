import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pickle 
import numpy as np 
from pathlib import Path
import zstandard as zstd
import io
import chess
import chess.pgn
import pandas as pd


class ChessDataset(Dataset):
    def __init__(self, filename):
        # load pandas once
        with open(filename, "rb") as f:
            df = pickle.load(f)

        # convert everything to NumPy arrays at once
        self.matrices = np.stack(df.matrix.values)             # shape: (N, 8, 8)
        self.turns    = df.turn.values                         # shape: (N,)
        self.total_turns = df.turns.values                     # shape: (N,)
        self.white_castling = np.stack(df.white_castling.values).astype(np.float32).tolist()  # shape: (N, 2)
        self.black_castling = np.stack(df.black_castling.values).astype(np.float32).tolist()  # shape: (N, 2)

        # vectorized winner conversion
        self.winners = np.zeros(len(df), dtype=np.int8)
        self.winners[df.winner.values == 'white'] = 1
        self.winners[df.winner.values == 'black'] = -1

        del df  # free pandas memory

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        winner = self.winners[idx]
        turn = self.turns[idx]
        steps = self.total_turns[idx] - turn
        brd_state = self.matrices[idx]

        if turn % 2 == 0:  # black turn
            brd_state = np.flipud(-brd_state)
            winner = -winner

        n = torch.tensor(brd_state+7, dtype=torch.int64).unsqueeze(0)
        brd_state3d = torch.zeros(15,8,8, dtype=torch.float32)
        brd_state3d.scatter_(0, n, 1.0)
        brd_state3d = brd_state3d[[0,1,2,3,4,5,6,8,9,10,11,12,13,14],:,:]

        # castling flags
        if turn % 2 == 1:  # white turn
            brd_state3d[-1,7,0] = self.white_castling[idx][0]
            brd_state3d[-1,7,7] = self.white_castling[idx][1]
            brd_state3d[0,0,0] = self.black_castling[idx][0]
            brd_state3d[0,0,7] = self.black_castling[idx][1]
        else:
            brd_state3d[-1,7,0] = self.black_castling[idx][0]
            brd_state3d[-1,7,7] = self.black_castling[idx][1]
            brd_state3d[0,0,0] = self.white_castling[idx][0]
            brd_state3d[0,0,7] = self.white_castling[idx][1]

        result = torch.tensor((winner, steps), dtype=torch.float32)
        return brd_state3d, result


def get_dataset(ds_select):
    if ds_select=='kaggle':
        print('loading kaggle')
        train_ds = ChessDataset("ds/train_states_ds.pkl")
        test_ds = ChessDataset("ds/test_states_ds.pkl")
    elif ds_select == 'fish':
        print('loading fish')
        train_ds = ChessDataset("ds/fish_train_states_ds.pkl")
        test_ds = ChessDataset("ds/fish_test_states_ds.pkl")
    elif ds_select == 'fish40':
        print('loading fish40')
        train_ds = ChessDataset("ds/fish40_train_states_ds.pkl")
        test_ds = ChessDataset("ds/fish40_test_states_ds.pkl")
    elif ds_select == 'large':
        print('loading fish40_large')
        train_ds = ChessDataset("ds/fish40_large_train_states_ds.pkl")
        test_ds = ChessDataset("ds/fish40_large_test_states_ds.pkl")        
    elif ds_select == 'lichess100k':
        print('loading lichess100k')
        train_ds = ChessDataset("ds/lichess100k_train_states_ds.pkl")
        test_ds = ChessDataset("ds/lichess100k_test_states_ds.pkl")      
    elif ds_select == 'lichess500k':
        print('loading lichess500k  ')
        train_ds = ChessDataset("ds/lichess500k_train_states_ds.pkl")
        test_ds = ChessDataset("ds/lichess500k_test_states_ds.pkl") 
    elif ds_select == 'lichess499k':
        print('loading lichess499k  ')
        train_ds = ChessDataset("ds/lichess499k_train_states_ds.pkl")
        test_ds = ChessDataset("ds/lichess499k_test_states_ds.pkl") 
    elif ds_select == 'lichess498k':
        print('loading lichess498k  ')
        train_ds = []
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_0.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_1.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_2.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_3.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_4.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_5.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_6.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_7.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_8.pkl"))
        train_ds.append(ChessDataset("ds498k/lichess_train_states_ds_9.pkl"))
        test_ds = ChessDataset("ds/lichess499k_test_states_ds.pkl") 
    DS = train_ds, test_ds
    return DS

def get_np_board(board_):
    matrix = np.zeros((8, 8), dtype=np.int8)
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }
    for square, piece in board_.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        value = piece_values[piece.piece_type]
        matrix[row, col] = value if piece.color == chess.WHITE else -value    
    return matrix

def pgn_result_to_winner(result):
    """Convert PGN result string to winner string"""
    if result == "1-0":
        return 'white'
    elif result == "0-1":
        return 'black'
    elif result == "1/2-1/2":
        return 'draw'
    else:
        return 'draw'  # default for unknown results

def pgn_result_to_victory_status(result, termination=None):
    """Convert PGN result and termination to victory_status"""
    if termination:
        term_lower = termination.lower()
        if 'time' in term_lower:
            return 'outoftime'
        elif 'resign' in term_lower:
            return 'resign'
        elif 'mate' in term_lower:
            return 'mate'
        elif 'draw' in term_lower:
            return 'draw'
    # Default based on result
    if result == "1/2-1/2":
        return 'draw'
    else:
        return 'mate'  # default for decisive games

def safe_int_rating(rating_str, default=0):
    """Safely convert rating string to int, handling '?' and other non-numeric values"""
    if not rating_str or rating_str == '?' or rating_str == '':
        return default
    try:
        return int(rating_str)
    except (ValueError, TypeError):
        return default

def process_lichess_pgn(pgn_file_path, start_gameid=0, max_games=None, start_skip=0):
    """Process games from a compressed lichess PGN file"""
    states = []
    sog = []
    gameid = start_gameid
    games_processed = 0
    
    # Decompress and open the file
    dctx = zstd.ZstdDecompressor()
    
    with open(pgn_file_path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            while True:
                if max_games and games_processed >= max_games:
                    break
                    
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                
                # Skip games that ended due to timeout
                termination = game.headers.get('Termination', '')
                if 'time' in termination.lower() and 'forfeit' in termination.lower():
                    games_processed += 1
                    continue
                
                result = game.headers.get('Result', '*')
                if result == '*':  # Incomplete game
                    games_processed += 1
                    continue
                
                # Extract metadata
                white_rating = safe_int_rating(game.headers.get('WhiteElo', '0'))
                black_rating = safe_int_rating(game.headers.get('BlackElo', '0'))
                victory_status = pgn_result_to_victory_status(result, termination)
                winner = pgn_result_to_winner(result)
                
                # Skip if out of time
                if victory_status == 'outoftime':
                    games_processed += 1
                    continue
                
                sog.append(len(states))
                
                # Process moves
                board = game.board()
                move_count = 0
                
                # Count total moves first
                mainline_moves_list = list(game.mainline_moves())
                total_turns = len(mainline_moves_list)
                
                # Process moves
                board = game.board()
                move_count = 0
                
                for move in mainline_moves_list:
                    board.push(move)
                    move_count += 1
                    
                    if move_count < start_skip:
                        continue
                    
                    state = []
                    state.append(gameid)  # gameid
                    state.append(move_count)  # turn
                    state.append(total_turns)  # total turns
                    state.append(victory_status)
                    state.append(winner)
                    state.append(white_rating)
                    state.append(black_rating)
                    state.append([board.has_queenside_castling_rights(chess.WHITE), board.has_kingside_castling_rights(chess.WHITE)])
                    state.append([board.has_queenside_castling_rights(chess.BLACK), board.has_kingside_castling_rights(chess.BLACK)])
                    state.append(get_np_board(board))
                    states.append(state)
                
                gameid += 1
                games_processed += 1
                
                if games_processed % 1000 == 0:
                    print(f'Processed {games_processed} games from lichess, {len(states)} states so far')
    
    print(f'Finished processing {games_processed} games from lichess PGN file')
    return states, sog

def process_path(path, outpath, max_games = 1000, start_skip = 0):
    print(f"Path is: {path}")

    # Process all lichess PGN files in data folder
    print('Processing lichess PGN files...')
    data_path = Path(path)
    lichess_files = sorted(data_path.glob("*.zst"))

    if len(lichess_files) == 0:
        print('Warning: No .zst files found in data folder')
    else:
        print(f'Found {len(lichess_files)} .zst file(s) in data folder')
        
        for file_idx, lichess_file in enumerate(lichess_files):
            print(f'Processing file {file_idx + 1}/{len(lichess_files)}: {lichess_file.name}')
            lichess_states, lichess_sog = process_lichess_pgn(str(lichess_file), start_gameid=0, max_games=max_games, start_skip=start_skip)
            
            # Adjust indices in lichess_sog to account for previously accumulated states
                    
            print(f'Generated {len(lichess_states)} states from {lichess_file.name}')
        
            # Create DataFrame
            states_df = pd.DataFrame(lichess_states, columns=["gameid", "turn", "turns", 'victory_status', 'winner', 
                                                            'white_rating', 'black_rating',
                                                            'white_castling', 'black_castling', 
                                                            'matrix', ])

            with open(f"{outpath}_{file_idx}.pkl", "wb") as f:
                pickle.dump(states_df, f)

def main():
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <path> <out_filename> <games> <skip>")
        sys.exit(1)

    path = sys.argv[1]
    outpath = sys.argv[2]
    games = int(sys.argv[3])
    skips = int(sys.argv[4])
    process_path(path, outpath, max_games = games, start_skip = skips)

if __name__ == "__main__":
    main()



