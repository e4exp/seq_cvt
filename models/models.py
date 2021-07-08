import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class ImageTextLSTM(nn.Module):
    def __init__(self, args, dim_image, dim_embed, dim_hidden, dim_target):
        super(ImageTextLSTM, self).__init__()
        self.hidden_dim = dim_hidden

        self.word_embeddings = nn.Embedding(args.vocab_size, dim_embed)

        #d_in = dim_image + dim_embed
        d_in = dim_embed
        #self.lstm = nn.LSTM(d_in, dim_hidden, num_layers=1, batch_first=True)

        self.linear0 = nn.Linear(dim_embed * args.seq_len, dim_hidden)
        self.norm0 = nn.LayerNorm(dim_hidden)

        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.norm1 = nn.LayerNorm(dim_hidden)

        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

        d_in_concat = dim_image + dim_hidden
        self.linear3 = nn.Linear(d_in_concat, dim_hidden)
        self.norm3 = nn.LayerNorm(dim_hidden)

        self.linear4 = nn.Linear(dim_hidden, dim_target)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x_img, x_tag, hiddens_0=None):
        embeds = self.word_embeddings(x_tag)
        #x_img = x_img.unsqueeze(1).repeat(1, embeds.shape[1], 1)
        #embeds = torch.cat([x_img, embeds], axis=2)
        hiddens = self.linear0(embeds.view(embeds.shape[0], -1))
        hiddens = self.relu(hiddens)
        hiddens = self.norm0(hiddens)

        hiddens = self.linear1(hiddens)
        hiddens = self.relu(hiddens)
        hiddens = self.norm1(hiddens)

        hiddens = self.linear2(hiddens)
        hiddens = self.relu(hiddens)
        hiddens = self.norm2(hiddens)

        #_, hiddens = self.lstm(embeds, hiddens_0)
        #_, hiddens = self.lstm(embeds)
        #print(hiddens[0].shape)
        #out = self.linear1(hiddens[0].view(-1, self.hidden_dim))

        #print(hiddens[0].view(-1, self.hidden_dim).shape)
        #print(x_img.shape)
        #out = torch.cat([hiddens[0].view(-1, self.hidden_dim), x_img], axis=1)
        out = torch.cat([hiddens, x_img], axis=1)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.norm3(out)

        out = self.linear4(out)
        out = self.sigmoid(out)

        return out, hiddens

    def init_hidden(self, b, args):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, b, self.hidden_dim).to(args.device,
                                                      non_blocking=True),
                torch.zeros(1, b, self.hidden_dim).to(args.device,
                                                      non_blocking=True))


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