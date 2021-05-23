import math
from typing import Optional
from logging import getLogger

from numpy.core.fromnumeric import take
logger = getLogger(__name__)

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Discriminator(nn.Module):
    def __init__(
        self,
        dim_vocab,
        seq_max,
    ):

        super().__init__()
        self.seq_max = seq_max - 1
        self.dim_vocab = dim_vocab
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # dim_vis_emb=512 * 64 * 8

        c_in = 512
        c_mid = c_in * 8 * 1
        c_out = c_in * 2
        c_class = 1
        dim_embed = 128

        # image
        self.pool = nn.AdaptiveAvgPool3d((c_in, 8, 1))
        self.flat = nn.Flatten()

        self.norm1 = nn.LayerNorm(c_mid)
        self.fc1 = nn.Linear(c_mid, c_out)

        # tags
        self.embed = nn.Embedding(self.dim_vocab, dim_embed)
        self.norm2 = nn.LayerNorm(dim_embed * self.seq_max)
        self.fc2 = nn.Linear(dim_embed * self.seq_max, c_out)

        # classify
        self.norm3 = nn.LayerNorm(c_out * 2)
        self.fc3 = nn.Linear(c_out * 2, c_in)

        self.norm4 = nn.LayerNorm(c_in)
        self.fc4 = nn.Linear(c_in, c_class)

    def forward(self, x_im, x_tag):

        # im
        x_im = self.pool(x_im)
        x_im = self.flat(x_im)
        x_im = self.norm1(x_im)
        x_im = self.fc1(x_im)
        x_im = self.act(x_im)

        # tag forward
        x_tag = self.embed(x_tag)
        x_tag = self.flat(x_tag)
        x_tag = self.norm2(x_tag)
        x_tag = self.fc2(x_tag)
        x_tag = self.act(x_tag)

        # classify
        x = torch.cat([x_im, x_tag], axis=1)
        x = self.norm3(x)
        x = self.fc3(x)
        x_tag = self.act(x_tag)

        x = self.norm4(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


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
        self.mlp2 = Mlp(in_features=dim,
                        hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer,
                        drop=drop)
        self.norm2 = norm(dim)

        self.norm3 = norm(dim)
        self.fc = nn.Linear(dim, dim_vocab * seq_max)

    def forward(self, x, is_train=False):
        x = x + self.drop_path(self.mlp1(self.norm1(x)))
        #x = x.transpose(-2, -1)
        x = x + self.drop_path(self.mlp2(self.norm2(x)))
        #x = x.transpose(-2, -1)
        x = self.fc(self.norm3(x))
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
        self.mlp2 = Mlp(in_features=n_tokens,
                        hidden_features=int(n_tokens * mlp_ratio),
                        act_layer=act_layer,
                        drop=drop)
        self.norm2 = norm(n_tokens)

    def forward(self, x):
        x = x + self.drop_path(self.mlp1(self.norm1(x)))
        #x = x.transpose(-2, -1)
        x = x + self.drop_path(self.mlp2(self.norm2(x)))
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