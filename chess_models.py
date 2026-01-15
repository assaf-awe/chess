import chess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from chess_utils import matrix_to_board
import torch.nn.functional as F
import math

import torch.nn as nn

class MyChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        do = 0.0

        self.cnn_0 = nn.Conv2d(in_channels=14,  out_channels=64,  kernel_size=1, padding=0, stride=1) 

        self.fc_1 = nn.Linear(64*8*8,  4096)
        self.fc_2 = nn.Linear(4096,    4096)
        self.fc_3 = nn.Linear(4096,    4096) 
        self.fc_4 = nn.Linear(4096,    2048)
        self.fc_5 = nn.Linear(2048,    2048) #here comes pooling 4->2
        self.fc_6 = nn.Linear(2048,    1024)
        self.fc_7 = nn.Linear(1024,    512) #here comes pooling 2->1

        self.ln2d_1 = nn.LayerNorm([4096])
        self.ln2d_2 = nn.LayerNorm([4096])
        self.ln2d_3 = nn.LayerNorm([4096])
        self.ln2d_4 = nn.LayerNorm([2048])
        self.ln2d_5 = nn.LayerNorm([2048])
        self.ln2d_6 = nn.LayerNorm([1024])
        self.ln2d_7 = nn.LayerNorm([512])

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.output_layer = nn.Linear(512, 3)
        self.dropout = nn.Dropout(p=do)

    def forward(self, x):
        x = self.cnn_0(x)

        x = torch.flatten(x,1)

        x = torch.relu(self.fc_1(x))
        # x = self.ln2d_1(x)
        x = self.dropout(x)

        x = torch.relu(self.fc_2(x))
        # x = self.ln2d_2(x)
        x = self.dropout(x)

        # x = torch.relu(self.fc_3(x))
        # x = self.ln2d_3(x)
        # x = self.dropout(x)

        x = torch.relu(self.fc_4(x))
        # x = self.ln2d_4(x)
        x = self.dropout(x)

        # x = torch.relu(self.fc_5(x))
        # x = self.ln2d_5(x)
        # x = self.dropout(x)

        x = torch.relu(self.fc_6(x))
        # x = self.ln2d_6(x)
        x = self.dropout(x)

        x = torch.relu(self.fc_7(x))
        # x = self.ln2d_7(x)
        x = self.dropout(x)

        x = self.output_layer(x)
        # print(x.shape)

        return x
		
class MyChessNet_hist(nn.Module):
    def __init__(self):
        super().__init__()
                
        self.net = nn.Sequential(
            nn.Linear(14, 14),
            nn.ReLU(),
            nn.Linear(14, 14),
            nn.ReLU(),
            nn.Linear(14, 14),
            nn.ReLU(),
            nn.Linear(14, 14),
            nn.ReLU(),
            nn.Linear(14, 14),
            nn.ReLU(),
            nn.Linear(14, 3)  # output logits
        )

    def forward(self, x):
        x = torch.sum(x,axis=(2,3))
        # print(x)

        return self.net(x)


class MyChessNet_simple(nn.Module):
    def __init__(self, sharpness=10.0):
        super().__init__()
        self.vars = nn.Parameter(torch.tensor([0.0, 1.0]))  # v0, v1
        self.k = sharpness
        tmpvec = torch.tensor([0.,0,-9, -5, -3, -3, -1, 1, 3, 3, 5, 9, 0 , 0]).T
        self.register_buffer("vec", tmpvec)

    def forward(self, x):
        x = torch.sum(x,axis=(2,3))
        x = x@self.vec 
        
        v0, v1 = self.vars
        s0 = torch.sigmoid(self.k * (x - v0))   # ~1 if x > v0
        s1 = torch.sigmoid(self.k * (x - v1))   # ~1 if x > v1

        out1 = 1 - s0               # x < v0
        out2 = s0 * (1 - s1)        # v0 < x < v1
        out3 = s1                   # x > v1

        return torch.stack([out1, out2, out3], dim=-1)
		
		
class MyChessNet_trnsfrm(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()

        # embed 14 → 64 per spatial location
        self.embed = nn.Linear(14, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,   # ← this is dim_ff
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cls = nn.Linear(d_model, 3)  # logits

    def forward(self, x):
        # x: (B, 14, 8, 8)
        B = x.size(0)

        x = x.permute(0, 2, 3, 1)      # (B, 8, 8, 14)
        x = x.reshape(B, 64, 14)       # (B, 64 tokens, 14)

        x = self.embed(x)              # (B, 64, 64)
        x = self.encoder(x)            # (B, 64, 64)

        x = x.mean(dim=1)              # global pooling
        return self.cls(x)             # (B, 3) logits

class MyChessNet_trnsfrm2(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=4):
        super().__init__()

        self.embed = nn.Linear(14, d_model)

        # 2D positional embedding: 8x8 = 64 tokens
        self.pos_embed = nn.Parameter(torch.randn(64, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
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



def sinusoidal_2d_positional_encoding(h, w, d_model, device):
    assert d_model % 4 == 0, "d_model must be divisible by 4"

    pe = torch.zeros(h, w, d_model, device=device)

    d_half = d_model // 2
    div_term = torch.exp(
        torch.arange(0, d_half, 2, device=device)
        * (-math.log(10000.0) / d_half)
    )

    pos_w = torch.arange(w, device=device).unsqueeze(1)
    pos_h = torch.arange(h, device=device).unsqueeze(1)

    # width (x)
    pe[:, :, 0:d_half:2] = torch.sin(pos_w * div_term)
    pe[:, :, 1:d_half:2] = torch.cos(pos_w * div_term)

    # height (y)
    pe[:, :, d_half::2] = torch.sin(pos_h * div_term)
    pe[:, :, d_half+1::2] = torch.cos(pos_h * div_term)

    return pe.view(h * w, d_model)

class HeavyTransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # HEAVY hyperparameters
        d_model = 128
        nhead = 8
        num_layers = 6
        dim_feedforward = 512
        dropout = 0.1

        # embed per spatial token (14 → 128)
        self.embed = nn.Linear(14, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(d_model, 3)  # logits

    def forward(self, x):
        # x: (B, 14, 8, 8)
        B, _, H, W = x.shape
        device = x.device

        # tokens
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, 14)

        # embedding
        x = self.embed(x)  # (B, 64, 128)

        # positional encoding
        pos = sinusoidal_2d_positional_encoding(H, W, x.size(-1), device)
        x = x + pos.unsqueeze(0)

        # transformer
        x = self.encoder(x)

        # global pooling
        x = x.mean(dim=1)

        return self.head(x)  # (B, 3) logits
		
class HeavyTransformerClassifier_sdo(nn.Module):
    def __init__(self, do_factor=1):
        super().__init__()

        # HEAVY hyperparameters
        d_model = 128
        nhead = 8
        num_layers = 6
        dim_feedforward = 512
        dropout = 0.1 * do_factor
        ph_dropout = 0.3 * do_factor
        

        # embed per spatial token (14 → 128)
        self.embed = nn.Linear(14, d_model)
        self.embed_dropout = nn.Dropout(p=dropout)
        self.pos_dropout = nn.Dropout(p=dropout)
        self.pre_head_dropout = nn.Dropout(p=ph_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(d_model, 3)  # logits

    def forward(self, x):
        # x: (B, 14, 8, 8)
        B, _, H, W = x.shape
        device = x.device

        # tokens
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, 14)

        # embedding
        x = self.embed(x)  # (B, 64, 128)
        x = self.embed_dropout(x)

        # positional encoding
        pos = sinusoidal_2d_positional_encoding(H, W, x.size(-1), device)
        x = x + pos.unsqueeze(0)
        x = self.pos_dropout(x)


        # transformer
        x = self.encoder(x)

        # global pooling
        x = x.mean(dim=1)
        x = self.pre_head_dropout(x)

        return self.head(x)  # (B, 3) logits 


class MyChessNet_trnsfrm2_do(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=4, do_f=1 ):
        super().__init__()

        dropout = 0.1 * do_f
        ph_dropout = 0.2 * do_f
        self.embed = nn.Linear(14, d_model)

        # 2D positional embedding: 8x8 = 64 tokens
        self.pos_embed = nn.Parameter(torch.randn(64, d_model))
        self.embed_dropout = nn.Dropout(p=dropout)
        self.pre_head_dropout = nn.Dropout(p=ph_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
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
        x = self.embed_dropout(x)
        x = x + self.pos_embed      # add positional info

        x = self.encoder(x)         # (B, 64, d_model)
        x = x.mean(dim=1)           # global average pooling
        x = self.pre_head_dropout(x)

        return self.cls(x)          # (B, 3) logits		


# class MyChessNet_trnsfrm2_do(nn.Module):
#     def __init__(self, d_model=64, nhead=8, num_layers=4, do_f=1 ):
#         super().__init__()

#         self.embed = nn.Linear(14, d_model)

#         # 2D positional embedding: 8x8 = 64 tokens
#         self.pos_embed = nn.Parameter(torch.randn(64, d_model))

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dropout=0.1 * do_f,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers
#         )

#         self.cls = nn.Linear(d_model, 3)

#     def forward(self, x):
#         # x: (B, 14, 8, 8)
#         B = x.size(0)

#         x = x.permute(0, 2, 3, 1)   # (B, 8, 8, 14)
#         x = x.reshape(B, 64, 14)    # (B, 64, 14)

#         x = self.embed(x)           # (B, 64, d_model)
#         x = x + self.pos_embed      # add positional info

#         x = self.encoder(x)         # (B, 64, d_model)
#         x = x.mean(dim=1)           # global average pooling

#         return self.cls(x)          # (B, 3) logits		

