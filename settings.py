import torch
import torch.nn as nn
from chess_models import *

if __name__ == '__main__':

    ret = [    
    {
        'model': MyChessNet_trnsfrm2_do(d_model=64, nhead=8, num_layers=4, do_f=.5),
        'epochs': 4 ,
        'lr': 1e-4,
        'optim': 'adamW',
        'weight_decay': 1e-3,
        'save_filename': "models/trnsfrm2_do_lichess500k_4e.pth"
    },
    {
        'model': MyChessNet_hist(),
        'epochs': 4 ,
        'lr': 1e-3,
        'optim': 'adam',
        'weight_decay': 1e-3,
        'save_filename': "models/hist_lichess500k_4e.pth"
    },

    ]



#     model = MyChessNet_trnsfrm2_do(d_model=64, nhead=8, num_layers=4, do_f=.5)
#     epochs = 4 
#     lr = 1e-4
#     optim = 'adamW'
#     weight_decay =1e-3
#     run_model(_datasets=DS, 
#         _model=model, 
#         _epochs=epochs, 
#         _lr=lr, 
#         _optim=optim, 
#         _weight_decay=weight_decay
#         )
#     torch.save(model.state_dict(), "models/trnsfrm2_do_lichess500k_4e.pth")
# ###################################################################################################
#     model = MyChessNet_hist()
#     epochs = 4
#     lr = 1e-3
#     optim = 'adam'
#     weight_decay =1e-3
#     run_model(_datasets=DS, 
#         _model=model, 
#         _epochs=epochs, 
#         _lr=lr, 
#         _optim=optim, 
#         _weight_decay=weight_decay
#         )
#     torch.save(model.state_dict(), "models/hist_lichess500k_4e.pth")