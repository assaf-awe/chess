import argparse
import random
import re
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

EVAL_REGEX = re.compile(r"\[%eval\s+([^\]]+)\]")
CLOCK_REGEX = re.compile(r"\[%clk\s+([^\]]+)\]")

STATE_COLUMNS = [
    "gameid",
    "turn",
    "turns",
    "victory_status",
    "winner",
    "white_rating",
    "black_rating",
    "white_castling",
    "black_castling",
    "matrix",
    "result",
    "termination",
    "rating_mean",
    "rating_diff",
    "rating_abs_diff",
    "white_rating_diff",
    "black_rating_diff",
    "time_control",
    "time_control_base",
    "time_control_inc",
    "time_control_class",
    "num_ply",
    "num_moves",
    "white_title",
    "black_title",
    "white_is_bot",
    "black_is_bot",
    "eco",
    "opening",
    "variant",
    "rated",
    "eval_available",
    "eval_cp",
    "eval_mate",
    "eval_type",
    "eval_raw",
    "clock_sec",
    "clock_str",
]

TURN_INDEX = STATE_COLUMNS.index("turn")
GAMEID_INDEX = STATE_COLUMNS.index("gameid")


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

def _load_dataset_path(path_str):
    path = Path(path_str)
    if path.is_dir():
        files = sorted(path.glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"No .pkl files found in {path}")
        return [ChessDataset(str(file_path)) for file_path in files]
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return [ChessDataset(str(path))]

def get_dataset(train_path, val_path):
    print(f"loading train from {train_path}")
    print(f"loading val from {val_path}")
    train_ds = _load_dataset_path(train_path)
    val_ds = _load_dataset_path(val_path)
    return train_ds, val_ds

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

def pgn_result_to_victory_status(result, termination=None, board_is_mate=None, board_is_stalemate=None):
    """Convert PGN result and termination to victory_status"""
    if termination:
        term_lower = termination.lower()
        if 'time' in term_lower:
            return 'outoftime'
        elif 'resign' in term_lower:
            return 'resign'
        elif 'stalemate' in term_lower:
            return 'stalemate'
        elif 'abandon' in term_lower or 'abort' in term_lower:
            return 'abandoned'
        elif 'mate' in term_lower:
            return 'mate'
        elif 'draw' in term_lower:
            return 'draw'
    # Default based on result
    if result == "1/2-1/2":
        if board_is_stalemate:
            return 'stalemate'
        return 'draw'
    if result == "*":
        return 'unknown'
    if board_is_mate:
        return 'mate'
    if board_is_stalemate:
        return 'stalemate'
    return 'resign'

def safe_int_rating(rating_str, default=0):
    """Safely convert rating string to int, handling '?' and other non-numeric values"""
    if not rating_str or rating_str == '?' or rating_str == '':
        return default
    try:
        return int(rating_str)
    except (ValueError, TypeError):
        return default

def parse_eval_comment(comment):
    """Parse [%eval ...] comment tags into numeric evals."""
    if not comment:
        return None
    match = EVAL_REGEX.search(comment)
    if not match:
        return None
    raw = match.group(1).strip()
    token = raw.split()[0]
    if "," in token:
        token = token.split(",", 1)[0]
    if "/" in token:
        token = token.split("/", 1)[0]
    if token.startswith("#"):
        mate_str = token[1:]
        try:
            mate = int(mate_str)
        except ValueError:
            return None
        return {
            "eval_available": True,
            "eval_cp": None,
            "eval_mate": mate,
            "eval_type": "mate",
            "eval_raw": raw,
        }
    try:
        cp_pawns = float(token)
    except ValueError:
        return None
    return {
        "eval_available": True,
        "eval_cp": cp_pawns * 100.0,
        "eval_mate": None,
        "eval_type": "cp",
        "eval_raw": raw,
    }

def parse_clock_comment(comment):
    """Parse [%clk ...] comment tags into seconds."""
    if not comment:
        return None, None
    match = CLOCK_REGEX.search(comment)
    if not match:
        return None, None
    clock_str = match.group(1).strip()
    parts = clock_str.split(":")
    try:
        parts = [int(part) for part in parts]
    except ValueError:
        return None, clock_str
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds, clock_str

def classify_time_control(base_seconds, inc_seconds):
    if base_seconds is None:
        return None
    increment = inc_seconds or 0
    estimated = base_seconds + 40 * increment
    if estimated < 29:
        return "ultrabullet"
    if estimated < 179:
        return "bullet"
    if estimated < 479:
        return "blitz"
    if estimated < 1499:
        return "rapid"
    return "classic"

def parse_time_control(time_control):
    if not time_control or time_control in ("-", "?"):
        return None, None, None
    base_str, inc_str = time_control, "0"
    if "+" in time_control:
        base_str, inc_str = time_control.split("+", 1)
    base = safe_int_rating(base_str, default=None)
    inc = safe_int_rating(inc_str, default=None)
    return base, inc, classify_time_control(base, inc)

def sample_state_rows(state_rows, states_per_game, method, rng):
    if not states_per_game or states_per_game <= 0:
        return state_rows
    if len(state_rows) <= states_per_game:
        return state_rows
    method = (method or "all").lower()
    if method == "all":
        return state_rows
    if method == "first":
        return state_rows[:states_per_game]
    if method == "last":
        return state_rows[-states_per_game:]
    if method == "uniform":
        if states_per_game == 1:
            indices = [len(state_rows) // 2]
        else:
            indices = np.linspace(0, len(state_rows) - 1, states_per_game, dtype=int)
        return [state_rows[i] for i in indices]
    if method == "random":
        indices = sorted(rng.sample(range(len(state_rows)), states_per_game))
        return [state_rows[i] for i in indices]
    raise ValueError(f"Unknown state_sample_method: {method}")

def update_ply_hist(ply_hist, total_turns, start_ply=1):
    if total_turns < start_ply:
        return ply_hist
    if ply_hist is None:
        ply_hist = np.zeros(total_turns + 1, dtype=np.int64)
    elif total_turns >= len(ply_hist):
        new_hist = np.zeros(total_turns + 1, dtype=np.int64)
        new_hist[: len(ply_hist)] = ply_hist
        ply_hist = new_hist
    ply_hist[start_ply : total_turns + 1] += 1
    return ply_hist

def sample_states_global_ply(states, target_total, rng, ply_hist=None):
    if target_total is None or target_total <= 0 or target_total >= len(states):
        return states
    by_ply = {}
    for row in states:
        ply = row[TURN_INDEX]
        by_ply.setdefault(ply, []).append(row)
    if ply_hist is not None and len(ply_hist) > 0:
        weights = {ply: int(ply_hist[ply]) for ply in by_ply.keys() if ply < len(ply_hist)}
        total_weight = sum(weights.values())
        if total_weight > 0:
            desired = {
                ply: int(round(weights.get(ply, 0) / total_weight * target_total))
                for ply in by_ply.keys()
            }
        else:
            total = len(states)
            desired = {
                ply: int(round(len(rows) / total * target_total))
                for ply, rows in by_ply.items()
            }
    else:
        total = len(states)
        desired = {
            ply: int(round(len(rows) / total * target_total))
            for ply, rows in by_ply.items()
        }
    for ply, rows in by_ply.items():
        if desired[ply] > len(rows):
            desired[ply] = len(rows)
    remainder = target_total - sum(desired.values())
    if remainder > 0:
        candidates = sorted(
            by_ply.keys(), key=lambda p: len(by_ply[p]) - desired[p], reverse=True
        )
        i = 0
        while remainder > 0 and candidates:
            ply = candidates[i % len(candidates)]
            if desired[ply] < len(by_ply[ply]):
                desired[ply] += 1
                remainder -= 1
            i += 1
            if i > len(candidates) * 2:
                break
    elif remainder < 0:
        remainder = -remainder
        candidates = sorted(by_ply.keys(), key=lambda p: desired[p], reverse=True)
        i = 0
        while remainder > 0 and candidates:
            ply = candidates[i % len(candidates)]
            if desired[ply] > 0:
                desired[ply] -= 1
                remainder -= 1
            i += 1
            if i > len(candidates) * 2:
                break

    sampled = []
    for ply, rows in by_ply.items():
        k = desired.get(ply, 0)
        if k <= 0:
            continue
        if k >= len(rows):
            sampled.extend(rows)
        else:
            sampled.extend(rng.sample(rows, k))
    return sampled

def rebuild_gameids_and_sog(states):
    if not states:
        return [], []
    states_sorted = sorted(states, key=lambda r: (r[GAMEID_INDEX], r[TURN_INDEX]))
    new_states = []
    sog = []
    current_old = None
    new_gameid = -1
    for row in states_sorted:
        old_gameid = row[GAMEID_INDEX]
        if old_gameid != current_old:
            current_old = old_gameid
            new_gameid += 1
            sog.append(len(new_states))
        row[GAMEID_INDEX] = new_gameid
        new_states.append(row)
    return new_states, sog

def process_lichess_pgn(
    pgn_file_path,
    start_gameid=0,
    max_games=None,
    start_skip=0,
    game_sample_method="first",
    states_per_game=None,
    state_sample_method="all",
    require_eval=True,
    evals_only_states=True,
    random_seed=None,
    skip_terminations=None,
):
    """Process games from a compressed lichess PGN file."""
    states = []
    sog = []
    gameid = start_gameid
    games_seen = 0
    eligible_games = 0
    rng = random.Random(random_seed)
    ply_hist = None

    skip_terms = [term.lower() for term in (skip_terminations or [])]
    game_sample_method = (game_sample_method or "first").lower()
    if game_sample_method not in ("first", "random"):
        raise ValueError(f"Unknown game_sample_method: {game_sample_method}")
    if game_sample_method == "random" and max_games is None:
        raise ValueError("game_sample_method='random' requires max_games.")

    sampled_games = [] if game_sample_method == "random" else None

    # Decompress and open the file
    dctx = zstd.ZstdDecompressor()

    with open(pgn_file_path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            while True:
                if max_games and game_sample_method == "first" and eligible_games >= max_games:
                    break

                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                games_seen += 1

                result = game.headers.get('Result', '*')
                if result == '*':  # Incomplete game
                    continue

                termination = game.headers.get('Termination', '')
                term_lower = termination.lower()
                if skip_terms and any(term in term_lower for term in skip_terms):
                    continue

                # Extract metadata
                white_rating = safe_int_rating(game.headers.get('WhiteElo', '0'))
                black_rating = safe_int_rating(game.headers.get('BlackElo', '0'))
                rating_mean = (white_rating + black_rating) / 2.0
                rating_diff = white_rating - black_rating
                rating_abs_diff = abs(rating_diff)
                white_rating_diff = safe_int_rating(game.headers.get('WhiteRatingDiff', ''), default=0)
                black_rating_diff = safe_int_rating(game.headers.get('BlackRatingDiff', ''), default=0)
                time_control = game.headers.get('TimeControl', '')
                time_control_base, time_control_inc, time_control_class = parse_time_control(time_control)
                winner = pgn_result_to_winner(result)
                white_title = game.headers.get('WhiteTitle', '')
                black_title = game.headers.get('BlackTitle', '')
                white_is_bot = white_title.upper() == 'BOT'
                black_is_bot = black_title.upper() == 'BOT'
                eco = game.headers.get('ECO', '')
                opening = game.headers.get('Opening', '')
                variant = game.headers.get('Variant', '')
                event = game.headers.get('Event', '')
                rated = 'rated' in event.lower()

                # Count total moves first
                mainline_nodes = list(game.mainline())
                total_turns = max(0, len(mainline_nodes) - 1)
                num_moves = (total_turns + 1) // 2

                board_end = game.board()
                for node in mainline_nodes:
                    if node.move is None:
                        continue
                    board_end.push(node.move)
                board_is_mate = board_end.is_checkmate()
                board_is_stalemate = board_end.is_stalemate()
                victory_status = pgn_result_to_victory_status(
                    result,
                    termination,
                    board_is_mate=board_is_mate,
                    board_is_stalemate=board_is_stalemate,
                )

                # Process moves
                board = game.board()
                move_count = 0
                game_has_eval = False
                states_for_game = []

                for node in mainline_nodes:
                    if node.move is None:
                        continue
                    board.push(node.move)
                    move_count += 1

                    if move_count < start_skip:
                        continue

                    comment = node.comment or ""
                    eval_data = parse_eval_comment(comment)
                    if eval_data:
                        game_has_eval = True
                    if evals_only_states and eval_data is None:
                        continue

                    if eval_data is None:
                        eval_available = False
                        eval_cp = None
                        eval_mate = None
                        eval_type = None
                        eval_raw = None
                    else:
                        eval_available = eval_data["eval_available"]
                        eval_cp = eval_data["eval_cp"]
                        eval_mate = eval_data["eval_mate"]
                        eval_type = eval_data["eval_type"]
                        eval_raw = eval_data["eval_raw"]

                    clock_sec, clock_str = parse_clock_comment(comment)

                    state = [
                        None,  # gameid placeholder
                        move_count,
                        total_turns,
                        victory_status,
                        winner,
                        white_rating,
                        black_rating,
                        [board.has_queenside_castling_rights(chess.WHITE), board.has_kingside_castling_rights(chess.WHITE)],
                        [board.has_queenside_castling_rights(chess.BLACK), board.has_kingside_castling_rights(chess.BLACK)],
                        get_np_board(board),
                        result,
                        termination,
                        rating_mean,
                        rating_diff,
                        rating_abs_diff,
                        white_rating_diff,
                        black_rating_diff,
                        time_control,
                        time_control_base,
                        time_control_inc,
                        time_control_class,
                        total_turns,
                        num_moves,
                        white_title,
                        black_title,
                        white_is_bot,
                        black_is_bot,
                        eco,
                        opening,
                        variant,
                        rated,
                        eval_available,
                        eval_cp,
                        eval_mate,
                        eval_type,
                        eval_raw,
                        clock_sec,
                        clock_str,
                    ]
                    states_for_game.append(state)

                if require_eval and not game_has_eval:
                    continue

                if state_sample_method != "global_ply":
                    states_for_game = sample_state_rows(
                        states_for_game,
                        states_per_game,
                        state_sample_method,
                        rng,
                    )
                if len(states_for_game) == 0:
                    continue

                if state_sample_method == "global_ply":
                    start_ply = max(1, start_skip + 1)
                    ply_hist = update_ply_hist(ply_hist, total_turns, start_ply=start_ply)

                eligible_games += 1

                if game_sample_method == "random":
                    if len(sampled_games) < max_games:
                        sampled_games.append(states_for_game)
                    else:
                        replacement_index = rng.randint(0, eligible_games - 1)
                        if replacement_index < max_games:
                            sampled_games[replacement_index] = states_for_game
                else:
                    sog.append(len(states))
                    for row in states_for_game:
                        row[0] = gameid
                        states.append(row)
                    gameid += 1

                if eligible_games % 1000 == 0:
                    total_states = len(states) if game_sample_method == "first" else sum(
                        len(game_states) for game_states in sampled_games
                    )
                    print(
                        f'Processed {games_seen} games, kept {eligible_games}, {total_states} states so far'
                    )

    if game_sample_method == "random":
        rng.shuffle(sampled_games)
        for game_states in sampled_games:
            sog.append(len(states))
            for row in game_states:
                row[0] = gameid
                states.append(row)
            gameid += 1

    if state_sample_method == "global_ply":
        target_total = None
        if states_per_game and states_per_game > 0:
            target_total = states_per_game * len(sog)
        states = sample_states_global_ply(states, target_total, rng, ply_hist=ply_hist)
        states, sog = rebuild_gameids_and_sog(states)

    print(
        f'Finished processing {games_seen} games ({eligible_games} kept) from lichess PGN file'
    )
    return states, sog

def process_path(
    path,
    outpath,
    max_games=1000,
    start_skip=0,
    game_sample_method="first",
    states_per_game=None,
    state_sample_method="all",
    require_eval=True,
    evals_only_states=True,
    random_seed=None,
    skip_terminations=None,
):
    print(f"Path is: {path}")

    # Process all lichess PGN files in data folder
    print('Processing lichess PGN files...')
    data_path = Path(path)
    outpath = Path(outpath)
    output_prefix = outpath.name
    output_dir = outpath.parent if outpath.parent != Path('.') else Path('.')
    if "pgn_files_for_test" in data_path.parts and output_dir.name != "ds_test":
        output_dir = Path("ds_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    lichess_files = sorted(data_path.glob("*.zst"))

    if len(lichess_files) == 0:
        print('Warning: No .zst files found in data folder')
    else:
        print(f'Found {len(lichess_files)} .zst file(s) in data folder')
        
        gameid_offset = 0
        games_remaining = max_games if game_sample_method == "first" else None

        for file_idx, lichess_file in enumerate(lichess_files):
            print(f'Processing file {file_idx + 1}/{len(lichess_files)}: {lichess_file.name}')
            max_games_for_file = games_remaining if game_sample_method == "first" else max_games
            lichess_states, lichess_sog = process_lichess_pgn(
                str(lichess_file),
                start_gameid=gameid_offset,
                max_games=max_games_for_file,
                start_skip=start_skip,
                game_sample_method=game_sample_method,
                states_per_game=states_per_game,
                state_sample_method=state_sample_method,
                require_eval=require_eval,
                evals_only_states=evals_only_states,
                random_seed=random_seed,
                skip_terminations=skip_terminations,
            )
            
            game_count = len(lichess_sog)
            gameid_offset += game_count
            if game_sample_method == "first" and games_remaining is not None:
                games_remaining -= game_count
                if games_remaining <= 0:
                    print('Reached max games; stopping further file processing')
            
            print(f'Generated {len(lichess_states)} states from {lichess_file.name}')
        
            # Create DataFrame
            states_df = pd.DataFrame(lichess_states, columns=STATE_COLUMNS)

            pgn_base = lichess_file.name
            if pgn_base.endswith(".zst"):
                pgn_base = pgn_base[:-4]
            if pgn_base.endswith(".pgn"):
                pgn_base = pgn_base[:-4]
            output_name = f"{pgn_base}_{output_prefix}_{file_idx}.pkl"
            with open(output_dir / output_name, "wb") as f:
                pickle.dump(states_df, f)
            if game_sample_method == "first" and games_remaining is not None and games_remaining <= 0:
                break

def main():
    parser = argparse.ArgumentParser(
        description="Process lichess PGN files into state datasets."
    )
    parser.add_argument("path", help="Directory with .zst PGN files")
    parser.add_argument("outpath", help="Output prefix for .pkl files")
    parser.add_argument("games", nargs="?", type=int, default=None, help="Max games (positional)")
    parser.add_argument("skip", nargs="?", type=int, default=0, help="Ply to skip at start (positional)")
    parser.add_argument("--max-games", type=int, default=1000, help="Max games to keep")
    parser.add_argument("--start-skip", type=int, default=None, help="Ply to skip at start")
    parser.add_argument(
        "--game-sample-method",
        choices=["first", "random"],
        default="first",
        help="How to sample games",
    )
    parser.add_argument(
        "--states-per-game",
        type=int,
        default=None,
        help="Limit number of states per game",
    )
    parser.add_argument(
        "--state-sample-method",
        choices=["all", "first", "last", "uniform", "random", "global_ply"],
        default="all",
        help="How to sample states within a game",
    )
    parser.add_argument(
        "--allow-missing-evals",
        action="store_true",
        help="Allow games with no evals",
    )
    parser.add_argument(
        "--include-missing-eval-states",
        action="store_true",
        help="Include states without evals",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--skip-terminations",
        type=str,
        default="",
        help="Comma-separated termination substrings to skip",
    )

    args = parser.parse_args()
    max_games = args.games if args.games is not None else args.max_games
    if max_games is not None and max_games <= 0:
        max_games = None
    start_skip = args.start_skip if args.start_skip is not None else args.skip
    require_eval = not args.allow_missing_evals
    evals_only_states = not args.include_missing_eval_states
    skip_terms = [term.strip().lower() for term in args.skip_terminations.split(",") if term.strip()]

    process_path(
        args.path,
        args.outpath,
        max_games=max_games,
        start_skip=start_skip,
        game_sample_method=args.game_sample_method,
        states_per_game=args.states_per_game,
        state_sample_method=args.state_sample_method,
        require_eval=require_eval,
        evals_only_states=evals_only_states,
        random_seed=args.random_seed,
        skip_terminations=skip_terms,
    )

if __name__ == "__main__":
    main()



