import math
from typing import Optional
from logging import getLogger
logger = getLogger(__name__)

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Decoder(nn.Module):
    def __init__(
        self,
        seq_max=2048,
        dim_vocab=302,
        c_in=2048,
    ):

        super().__init__()
        self.seq_max = seq_max
        self.dim_vocab = dim_vocab

        size_k = (5, 3)
        stride = 1
        layers = []
        self.act = nn.GELU()

        # 1,8 : 128 -> 256
        i_max = 2
        for i in range(i_max):
            conv = nn.Conv2d(c_in * 2**i, c_in * 2**(i + 1), size_k, stride)
            norm = nn.BatchNorm2d(c_in * 2**(i + 1))
            act = self.act
            layers.extend([conv, norm, act])

        # 256 -> vocab
        conv = nn.Conv2d(c_in * 2**i_max, self.seq_max * dim_vocab, size_k,
                         stride)
        norm = nn.BatchNorm2d(self.seq_max * dim_vocab)
        layers.extend([conv, norm, self.act])

        self.convs = nn.ModuleList(layers)

        # FF over features
        # self.mlp1 = Mlp(in_features=dim,
        #                 hidden_features=int(dim * mlp_ratio),
        #                 act_layer=act_layer,
        #                 drop=drop)
        # self.norm1 = norm(dim)

        # FF over patches
        # self.mlp2 = Mlp(in_features=dim,
        #                 hidden_features=int(dim * 0.5),
        #                 act_layer=act_layer,
        #                 drop=drop)
        # self.norm2 = norm(dim)

        # self.norm3 = norm(dim)
        # self.fc1 = nn.Linear(dim, seq_max)
        # self.rect = nn.GELU()

        # self.norm4 = norm(seq_max)
        # self.fc2 = nn.Linear(seq_max, dim_vocab * seq_max)

    def forward(self, x):
        #x = x + self.mlp1(self.norm1(x))
        #x = x + self.mlp2(self.norm2(x))
        #x = self.fc1(self.norm3(self.rect(x)))
        #x = self.fc2(self.norm4(self.rect(x)))

        for i, l in enumerate(self.convs):
            x = l(x)
            logger.debug("x {}".format(x.shape))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.shape[0], self.dim_vocab, self.seq_max)

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
        #x = x + self.drop_path(self.mlp2(self.norm2(x)))
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