import torch
import pytest
from torch.utils.data import DataLoader
from gpt.data import TextDataset
from gpt.attention import MultiHeadAttention
from gpt.model import LM


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


def test_generation():
    text = open("data/tiny-shakespeare.txt").read()
    train_dataset = TextDataset(text, device=torch.device("cpu"), context_length=64, batch_size=4)

    model = LM(
        train_dataset.vocab_size,
        context_length=train_dataset.context_length,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        dropout_p=0.2,
        rope=False,
        device=torch.device("cpu"),
    )

    prompt = "ROMEO:"
    prompt_encoded = train_dataset.encode_batch([prompt])
    generated = model.generate(prompt_encoded, max_len=100)
    generated_text = train_dataset.decode_batch(generated)[0]
    assert len(generated_text) == 100
