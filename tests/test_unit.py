import torch
import pytest

from gpt.attention import MultiHeadAttention


@pytest.fixture(params=[True, False])
def multihead_attention(request):
    return MultiHeadAttention(
        head_size=64,
        num_heads=4,
        src_embed_dim=256,
        context_length=10,
        dropout_p=0.1,
        rope=request.param,
        device=torch.device("cpu"),
    )


def test_multihead_attention_init(multihead_attention):
    assert isinstance(multihead_attention, MultiHeadAttention)
    assert multihead_attention.device == torch.device("cpu")
    assert len(multihead_attention.attention_heads) == 4


def test_multihead_attention_forward(multihead_attention):
    x = torch.rand(1, 10, 256)  # batch_size=1, context_length=10, src_embed_dim=256
    output = multihead_attention(x)
    assert output.shape == (1, 10, 256)  # batch_size=1, context_length=10, src_embed_dim=256
