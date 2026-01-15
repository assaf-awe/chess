import chess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from chess_utils import matrix_to_board
import torch.nn.functional as F
import torch.nn as nn
from chess_models import *
import time
import gc
import sys
import runpy

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

def sprint(*args, **kwargs):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(*args, **kwargs)
    return buf.getvalue()

def criterion(output, targets, vec=None):
    loss = torch.nn.CrossEntropyLoss()
    loss2 = torch.nn.CrossEntropyLoss(reduction='none')
    if vec == None:
        return loss(output, targets+1)
    else:
        nc = (vec.max()+1).long().item()
        VEC = F.one_hot(vec.long(),num_classes=nc).float()
        ret = loss2(output, targets+1).unsqueeze(0) @ VEC
        ret = ret/VEC.sum(axis=0)
        return ret.mean()    

def accuracy(output, targets, vec=None):
    if vec == None:
        return (torch.round(torch.argmax(output,axis=1)-1) == targets).sum().item(), len(targets)
    else:
        VEC = F.one_hot(vec.long(),num_classes=3).float()
        ret = (torch.round(torch.argmax(output,axis=1)-1) == targets)*1. @ VEC
        return ret.cpu().numpy(), VEC.sum(axis=0).cpu().numpy()


def run_model(_datasets, _model, _epochs=10, _lr=1e-3, _optim='adam', _batch_size=256, _weight_decay = 1e-3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'tr_ds_size={len(_datasets[0])}, model={_model.__class__.__name__}, {_epochs=}, {_lr=}, {_optim=}, {_batch_size=}, {_weight_decay=}, Using device: {device}')
    lr = _lr
    epochs = _epochs

    model = _model.to(device)
    if _optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=_weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=_weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=(1/2)**(1/25))


    train_ds, test_ds = _datasets
    
    train_dataloader = DataLoader(train_ds, batch_size=_batch_size, num_workers=2, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=_batch_size, num_workers=8, shuffle=False)

    mloss = []
    for epoch in range(epochs):
        starttime = time.perf_counter()
        print(optimizer.param_groups[0]['lr'])

        model.train()
        p = 0
        P = 0 
        acc_sum = 0
        acc_sum_sum = 0
        loss_sum=np.array(0.)
        for features, targets in train_dataloader:
            P += 1
            if P%1000 == 0:
                print('#', end="",flush=True)
            features = features.to(device)
            targets = targets.to(device)

            vec3 = targets[:,0]+1 
            if  len(vec3.unique()) < 3 :
                # print(f'found states: {vec3.unique()}. Dropping batch')
                continue        

            output = model(features)
            # Normlized loss
            loss = criterion(output, torch.round(targets[:,0]).long(),vec3.to(device))
            # Non-normlized loss
            # loss = criterion(output, torch.round(targets[:,0]).long())

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            scheduler.step()

            loss_sum += np.array(loss.item())
            acc = accuracy(output, targets[:,0], vec3)
            acc_sum += acc[0]
            acc_sum_sum += acc[1]
            p += 1

        print(f'  epoch running time: {(time.perf_counter()-starttime)/60:.1f}m')
        print(f'{epoch}.       mLoss:{loss_sum/p:.4f} | mAcc:{acc_sum.sum()/acc_sum_sum.sum():.2%} | wAcc:{(acc_sum/acc_sum_sum).mean():.2%} | Acc[W/D/B]:',np.char.mod('%.2f%%', acc_sum/acc_sum_sum * 100), f' | DumpedBatch:{(1-p/P):.2%}')
        # mloss.append(loss_sum/p)


        model.eval()
        p = 0
        acc_sum = 0
        acc_sum_sum = 0
        loss_sum=np.array(0.)
        with torch.no_grad():
            for features, targets in test_dataloader:
                features = features.to(device)
                targets = targets.to(device)
                output = model(features)
                vec3 = targets[:,0]+1
                loss = criterion(output, torch.round(targets[:,0]).long())

                loss_sum += np.array(loss.item())
                acc = accuracy(output, targets[:,0], vec3)
                acc_sum += acc[0]
                acc_sum_sum += acc[1]
                p += 1
        print(f'TEST: {epoch}. mLoss:{loss_sum/p:.4f} | \033[1mmAcc:{acc_sum.sum()/acc_sum_sum.sum():.2%} | wAcc:{(acc_sum/acc_sum_sum).mean():.2%}\033[0m | Acc[W/D/B]:',np.char.mod('%.2f%%', acc_sum/acc_sum_sum * 100))
    print(' ')
    return model



############################################################################################################
def main():
    if len(sys.argv) > 1:
        settings = sys.argv[1]
    else:
        print('must specify settings file ')
        print(f'Usage {sys.argv[0]} settings.py')
        exit(1)
    ds_select = 'lichess500k' #'lichess500k' #'lichess100k' 'large'  #'fish40'    #'kaggle', 'fish', 'fish40'

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
    DS = train_ds, test_ds

###################################################################################################
    print (f'loading settings from {settings}')
    ret = runpy.run_path(settings, run_name="__main__")
    ret = ret['ret']

    for m in ret:   
        print(m['save_filename'])
        run_model(_datasets=DS, 
            _model=m['model'], 
            _epochs=m['epochs'], 
            _lr=m['lr'], 
            _optim=m['optim'], 
            _weight_decay=m['weight_decay']
            )
        torch.save(model.state_dict(), m['save_filename'])

        
if __name__ == "__main__":
    main()


