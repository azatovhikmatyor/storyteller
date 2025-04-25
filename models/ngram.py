import os
import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm

import tokenizer.tokenizer as tokenizer


@dataclass(frozen=True)
class NGramConfig:
    vocab_size: int
    n_embd: int
    block_size: int


class NGram(nn.Module):
    def __init__(self, config: NGramConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.fc = nn.Linear(config.n_embd*config.block_size, config.vocab_size)

    def forward(self, x: Tensor):
        x = self.tok_emb(x)
        B, T, C = x.shape
        x = x.view(B, T*C)
        x = self.fc(x)
        return x
    