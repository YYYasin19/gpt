import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(
        self,
        src_embed_dim: int,
        context_length: int,
        head_size: int = 16,
        dropout_p: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.head_size = head_size
        self.device = device
        self.query = nn.Linear(src_embed_dim, head_size, bias=False).to(self.device)  # what am I looking for? [b, c, h]
        self.key = nn.Linear(src_embed_dim, head_size, bias=False).to(self.device)  # what am I? [b, c, h]
        self.value = nn.Linear(src_embed_dim, head_size, bias=False).to(self.device)  # what can I tell you about me?
        self.dropout = nn.Dropout(dropout_p)

        # don't optimize the tril, that's only here for masking
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)).to(self.device))

    def forward(self, x):
        batch, context, embed = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        weights = q @ k.transpose(-2, -1)  # [b, c, h] @ [b, h, c] -> [b, c, c]
        weights = weights / embed ** (-0.5)  # preserve variance of weights
        weights = weights.masked_fill(self.tril[:context, :context] == 0, float("-inf"))  # only in decoder blocks
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        head_size: int,
        num_heads: int,
        src_embed_dim: int,
        context_length: int,
        dropout_p: float,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(
                    head_size=head_size,
                    src_embed_dim=src_embed_dim,
                    context_length=context_length,
                    device=self.device,
                )
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(src_embed_dim, src_embed_dim).to(self.device)
        self.dropout = nn.Dropout(dropout_p).to(self.device)

    def forward(self, x):
        """
        calculate multiple attention passes (in parallel) and concat the result
        returns: [batch_size, context, context]
        """
        res = torch.cat([h(x) for h in self.attention_heads], dim=-1)
        res = self.projection(res)
        res = self.dropout(res)
        return res


class RotaryEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, base: int = 10_000, device=torch.device("cpu")):
        """
        embed_dim: number of features
        base: constant used for the matrix values.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.base = base
        self.device = device

        self.theta = (self.base ** (torch.arange(0, embed_dim // 2))).to(self.device)  # TODO

    def forward(self, x):
        """
        x: shape [batch, seq, n_heads, embed_dim] -> [seq, batch, n_heads, embed_dim]
        """
        x = x.permute(1, 0, 2, 3)
        seq_len = x.shape[0]
        theta = (1.0 / self.base ** (torch.arange(0, self.embed_dim // 2))).to(self.device)
        seq = torch.arange(0, seq_len).to(self.device)
        # TODO: cache this
        idx_theta = seq * theta
        idx_theta = torch.cat([idx_theta, idx_theta], dim=0)
        cos = idx_theta.cos()[:, None, None, :]
        sin = idx_theta.sin()[:, None, None, :]
        x_rope = x  # TODO: we don't have to apply the transformation to all dimensions
        d_2 = self.embed_dim // 2
        x_rope_neg = torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
        x_rope = (x_rope * cos[:seq_len]) + (x_rope_neg * sin[:seq_len])
        x_rope = x_rope.permute(1, 0, 2, 3)
        return x_rope

    def __repr__(self):
        return f"RotaryEmbeddings(embed_dim={self.embed_dim}, base={self.base}, device={self.device})"


class AttentionRoPE(AttentionHead):
    def __init__(
        self,
        src_embed_dim: int,
        context_length: int,
        head_size: int = 16,
        dropout_p: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(src_embed_dim, context_length, head_size, dropout_p, device)
        self.query_rope = RotaryEmbeddings(src_embed_dim, device=device)
        self.key_rope = RotaryEmbeddings(src_embed_dim, device=device)

    def forward(self, x):
        """
        Modify regular forward pass of AttentionHead to use RotaryEmbeddings.
        """
        batch, context, embed = x.shape
        k, q, v = self.key_rope(x), self.query_rope(x), self.value(x)
        weights = q @ k.transpose(-2, -1)  # [b, c, h] @ [b, h, c] -> [b, c, c]
        weights = weights / embed ** (-0.5)  # preserve variance of weights
        weights = weights.masked_fill(self.tril[:context, :context] == 0, float("-inf"))  # only in decoder blocks
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ v
