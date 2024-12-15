"""Module that implements Transformer.

Based on the code provided in:
  https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""
from typing import Union

import timm
import torch
from torch import nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            expanded_mask: torch.Tensor = mask.unsqueeze(dim=1).repeat(1, dots.size(dim=1), 1, 1)
            max_neg_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~expanded_mask, max_neg_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_attention:
            return self.to_out(out), attn

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., drop_path_prob: float = .0):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

        self.drop_path = timm.layers.DropPath(
            drop_prob=drop_path_prob) if drop_path_prob > .0 else nn.Identity()

    def forward(self, x, mask: torch.Tensor = None, return_attention: bool = False
                ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if return_attention:
            attn, attn_map = self.attn(x, mask=mask, return_attention=return_attention)
        else:
            attn = self.attn(x, mask=mask)
        x = self.drop_path(attn) + x
        x = self.drop_path(self.ff(x)) + x

        if return_attention:
            return x, attn_map
        return x
