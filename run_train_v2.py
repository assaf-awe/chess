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


class ChessDataset(Dataset):
    def __init__(self, filename, castling=False):
        with open(filename, "rb") as f:
            self.states_df = pickle.load(f)
        self.castling = castling

    def __len__(self):
        return len(self.states_df)

    def __getitem__(self, idx):
        # display(self.states_df.iloc[idx])
        if self.states_df.iloc[idx].winner == 'black':
            winner = -1 
        elif self.states_df.iloc[idx].winner == 'white':
            winner = 1 
        else:
            winner = 0
        if self.states_df.iloc[idx].turn %2:   #white turn 
            brd_state =  self.states_df.iloc[idx].matrix
        else:
            brd_state =  np.flipud(-self.states_df.iloc[idx].matrix).copy() 
            winner = -winner
        n = torch.tensor(brd_state+7).unsqueeze(0).to(torch.int64)
        brd_state3d = torch.zeros(15, 8, 8, dtype=torch.float32)
        brd_state3d.scatter_(0, n, 1.0)
        brd_state3d = brd_state3d[[0,1,2,3,4,5,6,8,9,10,11,12,13,14],:,:]
        if self.castling:
            if self.states_df.iloc[idx].turn %2:   #white turn 
                brd_state3d[-1,7,0] = states_df.iloc[idx].white_castling[0]*1
                brd_state3d[-1,7,7] = states_df.iloc[idx].white_castling[1]*1
                brd_state3d[0,0,0] = states_df.iloc[idx].black_castling[0]*1
                brd_state3d[0,0,7] = states_df.iloc[idx].black_castling[1]*1
            else:
                brd_state3d[-1,7,0] = states_df.iloc[idx].black_castling[0]*1
                brd_state3d[-1,7,7] = states_df.iloc[idx].black_castling[1]*1
                brd_state3d[0,0,0] = states_df.iloc[idx].white_castling[0]*1
                brd_state3d[0,0,7] = states_df.iloc[idx].white_castling[1]*1            
        steps = self.states_df.iloc[idx].turns-self.states_df.iloc[idx].turn 
        result = torch.tensor((winner, steps), dtype=torch.float32)
        return brd_state3d, result
        # return torch.tensor(brd_state3d, dtype=torch.float32).unsqueeze(0), result


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
    # device = 'cpu'
    lr = _lr
    epochs = _epochs

    model = _model.to(device)
    if _optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=_weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=_weight_decay)

    train_ds, test_ds = _datasets
    
    train_dataloader = DataLoader(train_ds, batch_size=_batch_size, num_workers=8, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=_batch_size, num_workers=8, shuffle=False)

    mloss = []
    for epoch in range(epochs):
        if epoch == epochs//2:
            optimizer.param_groups[0]['lr'] /= 10
            print (f'updating lr to {optimizer.param_groups[0]['lr']}')

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

            loss_sum += np.array(loss.item())
            acc = accuracy(output, targets[:,0], vec3)
            acc_sum += acc[0]
            acc_sum_sum += acc[1]
            p += 1
        print(f'\n{epoch}.       mLoss:{loss_sum/p:.4f} | mAcc:{acc_sum.sum()/acc_sum_sum.sum():.2%} | wAcc:{(acc_sum/acc_sum_sum).mean():.2%} | Acc[W/D/B]:',np.char.mod('%.2f%%', acc_sum/acc_sum_sum * 100), f' | DumpedBatch:{(1-p/P):.2%}')
        mloss.append(loss_sum/p)


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

    return model


class MyChessNet_trnsfrm2_do2(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=4):
        super().__init__()

        self.embed = nn.Linear(14, d_model)

        # 2D positional embedding: 8x8 = 64 tokens
        self.pos_embed = nn.Parameter(torch.randn(64, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=0.2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cls = nn.Linear(d_model, 3)

    def forward(self, x):
        # x: (B, 14, 8, 8)
        B = x.size(0)

        x = x.permute(0, 2, 3, 1)   # (B, 8, 8, 14)
        x = x.reshape(B, 64, 14)    # (B, 64, 14)

        x = self.embed(x)           # (B, 64, d_model)
        x = x + self.pos_embed      # add positional info

        x = self.encoder(x)         # (B, 64, d_model)
        x = x.mean(dim=1)           # global average pooling

        return self.cls(x)          # (B, 3) logits		



############################################################################################################
def main():
    print(f'Runnig script: {__file__}')


    ds_select = 'fish40'    #'kaggle', 'fish', 'fish40'

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
        
    DS = train_ds, test_ds


###################################################################################################
    # model = MyChessNet_trnsfrm2()
    # epochs = 6 
    # lr = 1e-3
    # optim = 'adamW'
    # weight_decay =1e-3
    # run_model(_datasets=DS, 
    #     _model=model, 
    #     _epochs=epochs, 
    #     _lr=lr, 
    #     _optim=optim, 
    #     _weight_decay=weight_decay
    #     )

###################################################################################################
    # model = MyChessNet_trnsfrm2_do2()
    # epochs = 10 # 
    # lr = 1e-3
    # optim = 'adamW'
    # weight_decay =1e-3
    # run_model(_datasets=DS, 
    #     _model=model, 
    #     _epochs=epochs, 
    #     _lr=lr, 
    #     _optim=optim, 
    #     _weight_decay=weight_decay
    #     )
###################################################################################################
    # model = MyChessNet()
    # epochs = 6
    # lr = 1e-3
    # optim = 'adam'
    # weight_decay =1e-3
    # run_model(_datasets=DS, 
    #     _model=model, 
    #     _epochs=epochs, 
    #     _lr=lr, 
    #     _optim=optim, 
    #     _weight_decay=weight_decay
    #     )

###################################################################################################
    # train_ds_large = ChessDataset("ds/fish40_large_train_states_ds.pkl")
    # test_ds_large = ChessDataset("ds/fish40_large_test_states_ds.pkl")
    # DSL = train_ds_large, test_ds_large
    
    # model = MyChessNet_trnsfrm2()
    # epochs = 6 
    # lr = 1e-3
    # optim = 'adamW'
    # weight_decay =1e-3
    # run_model(_datasets=DSL, 
    #     _model=model, 
    #     _epochs=epochs, 
    #     _lr=lr, 
    #     _optim=optim, 
    #     _weight_decay=weight_decay
    #     )
###################################################################################################
    print ('with castling')
    model = MyChessNet_trnsfrm2()
    epochs = 6 
    lr = 1e-3
    optim = 'adamW'
    weight_decay =1e-3
    run_model(_datasets=DS, 
        _model=model, 
        _epochs=epochs, 
        _lr=lr, 
        _optim=optim, 
        _weight_decay=weight_decay
        )
###################################################################################################
    model = MyChessNet_trnsfrm2_do2()
    epochs = 6 
    lr = 1e-3
    optim = 'adamW'
    weight_decay =1e-3
    run_model(_datasets=DS, 
        _model=model, 
        _epochs=epochs, 
        _lr=lr, 
        _optim=optim, 
        _weight_decay=weight_decay
        )        
###################################################################################################


if __name__ == "__main__":
    main()
    