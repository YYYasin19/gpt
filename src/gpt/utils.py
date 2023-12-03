import torch
from torch.functional import F


def weighted_aggergation(x: torch.Tensor):
    """
    performs simple weighted aggregation for a token given it's past tokens.
    Example: x has shape [batch, context, embed_dim]
    """
    assert len(x.shape) == 3, "Wrong dim for x"
    b, context, emb = x.shape
    tril = torch.tril(torch.ones(context, context))
    tril = tril / tril.sum(1, keepdim=True)
    return tril @ x


def weighted_aggergation_softmax(x: torch.Tensor):
    assert len(x.shape) == 3, "Wrong dim for x"
    b, context, emb = x.shape
    tril = torch.tril(torch.ones(context, context))
    weights = torch.zeros((context, context))
    weights = weights.masked_fill(tril == 0, float("-inf"))
    weights = F.softmax(weights, dim=1)

    return weights @ x


if __name__ == "__main__":
    x = torch.randn((1, 3, 2))  # batch, context, emd
    print(weighted_aggergation(x))
