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
from datasets import get_dataset

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
    
    




###################################################################################################
    print (f'loading settings from {settings}')
    ret = runpy.run_path(settings, run_name="__main__")
    ret = ret['ret']

    for m in ret:   
        print(f"Model will be save to: {m['save_filename']}")
        print(f"running on {m['dataset']} dataset")
        DS = get_dataset(m['dataset'])
        run_model(
            _datasets=DS, 
            _model=m['model'], 
            _epochs=m['epochs'], 
            _lr=m['lr'], 
            _optim=m['optim'], 
            _weight_decay=m['weight_decay']
            )
        torch.save(model.state_dict(), m['save_filename'])

        
if __name__ == "__main__":
    main()


