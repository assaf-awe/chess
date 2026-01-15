import torch
from torch.utils.data import Dataset, DataLoader
import pickle 
import numpy as np 

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
    DS = train_ds, test_ds
    return DS