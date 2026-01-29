import torch
import torch.nn as nn
from chess_models import *


if __name__ == '__main__':
    ret = [    
    {
        'model': MyChessNet_hist(),
        'epochs': 50 ,
        'lr': 1e-3,
        'optim': 'adam',
        'weight_decay': 1e-3,
        'save_filename': "models/hist_",
        'train_path': 'ds/train',
        'val_path': 'ds/val',
    },

    {
        'model': MyChessNet_trnsfrm2_do(d_model=128, nhead=16, num_layers=6, do_f=1),
        'epochs': 50 ,
        'lr': 1e-4,
        'optim': 'adamW',
        'weight_decay': 1e-3,
        'save_filename': "models/trnsfrm2_",
        'train_path': 'ds/train',
        'val_path': 'ds/val',
    },

    # {
    #     'model': MyChessNet(),
    #     'epochs': 20 ,
    #     'lr': 1e-4,
    #     'optim': 'adam',
    #     'weight_decay': 1e-3,
    #     'save_filename': "models/cnn_lichess498k",
    #     'train_path': 'ds/train',
    #     'val_path': 'ds/val',
    # },

    ]


