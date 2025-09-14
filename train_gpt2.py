from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # key, query, value projection for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.n_embed * 4, config.n_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Block(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.attention = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = x + self.attention(self.ln1(x))
        outputs = outputs + self.mlp(self.ln1(outputs))
        return outputs


class GPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.Module(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.ln_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
