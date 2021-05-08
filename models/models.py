import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Decoder(nn.Module):
    def __init__(self,
                 dim,
                 dim_vocab,
                 seq_max,
                 mlp_ratio=4.,
                 drop=0.,
                 act_layer=nn.GELU,
                 norm=nn.LayerNorm):

        super().__init__()
        self.seq_max = seq_max
        self.dim_vocab = dim_vocab

        self.drop_path = nn.Identity()
        # FF over features
        self.mlp1 = Mlp(in_features=dim,
                        hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer,
                        drop=drop)
        self.norm1 = norm(dim)

        # FF over patches
        # self.mlp2 = Mlp(in_features=dim,
        #                 hidden_features=int(dim * mlp_ratio),
        #                 act_layer=act_layer,
        #                 drop=drop)
        # self.norm2 = norm(dim)

        self.fc = nn.Linear(dim, dim_vocab * seq_max)

    def forward(self, x, is_train=False):
        x = x + self.drop_path(self.mlp1(self.norm1(x)))
        #x = x.transpose(-2, -1)
        #x = x + self.drop_path(self.mlp2(self.norm2(x)))
        #x = x.transpose(-2, -1)
        x = self.fc(x)
        x = x.view(x.shape[0], self.dim_vocab, self.seq_max)

        if is_train:
            x = F.softmax(x, dim=-1)

        return x


class Encoder(nn.Module):
    def __init__(self,
                 dim,
                 n_tokens,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm=nn.LayerNorm):

        super().__init__()
        #self.drop_path = DropPath(
        #    drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

        # FF over features
        self.mlp1 = Mlp(in_features=dim,
                        hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer,
                        drop=drop)
        self.norm1 = norm(dim)

        # FF over patches
        # self.mlp2 = Mlp(in_features=n_tokens,
        #                 hidden_features=int(n_tokens * mlp_ratio),
        #                 act_layer=act_layer,
        #                 drop=drop)
        # self.norm2 = norm(n_tokens)

    def forward(self, x):
        x = x + self.drop_path(self.mlp1(self.norm1(x)))
        #x = x.transpose(-2, -1)
        #x = x + self.drop_path(self.mlp2(self.norm2(x)))
        #x = x.transpose(-2, -1)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x