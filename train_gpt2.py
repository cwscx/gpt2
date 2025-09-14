from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384


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
