import torch
from torch import nn
from gpt.data import TextDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, device="cpu"):
        super().__init__()
        self.fforward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ).to(device)

    def forward(self, x):
        self.fforward(x)


class TransformerBlock(nn.Module):
    def __init__(self, num_head: int, src_embed_dim: int, context_length: int, device="cpu"):
        super().__init__()
        head_size = src_embed_dim // num_head
        self.device = device
        self.attention = MultiHeadAttention(
            head_size=head_size, num_heads=num_head, src_embed_dim=src_embed_dim, context_length=context_length, device=device
        )
        self.fforward = FeedForward(src_embed_dim, device)

    def forward(self, x):
        # go through attention mechanism and, additionally, skip connections (better training though res/skip connections)
        x = x + self.attention(x)
        x = x + self.fforward(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, src_embed_dim: int, context_length: int, head_size: int = 16, device="cpu"):
        super().__init__()
        self.head_size = head_size
        self.device = device
        self.query = nn.Linear(src_embed_dim, head_size, bias=False).to(self.device)  # what am I looking for? [b, c, h]
        self.key = nn.Linear(src_embed_dim, head_size, bias=False).to(self.device)  # what am I? [b, c, h]
        self.value = nn.Linear(src_embed_dim, head_size, bias=False).to(self.device)  # what can I tell you about me?

        # don't optimize the tril, that's only here for masking
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)).to(self.device))

    def forward(self, x):
        batch, context, embed = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        weights = q @ k.transpose(-2, -1)  # [b, c, h] @ [b, h, c] -> [b, c, c]
        weights = weights / embed ** (-0.5)  # preserve variance of weights
        weights = weights.masked_fill(self.tril[:context, :context] == 0, float("-inf"))  # only in decoder blocks
        weights = F.softmax(weights, dim=-1)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int, num_heads: int, src_embed_dim: int, context_length: int, device="cpu"):
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
        self.projection = nn.Linear(src_embed_dim, src_embed_dim)

    def forward(self, x):
        """
        calculate multiple attention passes (in parallel) and concat the result
        returns: [batch_size, context, context]
        """
        res = torch.cat([h(x) for h in self.attention_heads], dim=-1)
        res = self.projection(res)
        return res


class LM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embed_dim: int = 32, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

        # creates an Embedding that takes vectors of dim [vocab_size] and outputs vectors of dim [vocab_size]
        # TODO: use custom Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.pos = nn.Embedding(context_length, embed_dim).to(device)  # positional embeddings -> gives each token a space where it is
        self.linear_head = nn.Linear(embed_dim, vocab_size).to(device)

        # self.attention = AttentionHead(embed_dim=embed_dim, context_length=context_length, head_size=embed_dim)
        self.attention = MultiHeadAttention(
            head_size=embed_dim // 4, num_heads=4, context_length=context_length, src_embed_dim=embed_dim, device=self.device
        )

        # allows for some computation between the attention output and the logit creation
        self.fforward = FeedForward(embed_dim, device)

        self.blocks = nn.Sequential(
            TransformerBlock(num_head=4, src_embed_dim=embed_dim, context_length=context_length, device=self.device),
            TransformerBlock(num_head=4, src_embed_dim=embed_dim, context_length=context_length, device=self.device),
            TransformerBlock(num_head=4, src_embed_dim=embed_dim, context_length=context_length, device=self.device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, context = x.shape
        x = x.to(self.device)
        tok_embeddings = self.embedding(x)  # (B, C, embed)
        pos_embeddings = self.pos(torch.arange(context, device=self.device))  # (C, embed)
        x = tok_embeddings + pos_embeddings  # (B, C, embed)
        x = self.blocks(x)
        logits = self.linear_head(x)  # (B, C, vocab_size)
        return logits

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch_size, context_length)
        y: tensor of shape (batch_size, context_length)
        """
        x, y = x.to(self.device), y.to(self.device)
        logits = self(x)
        batch, context, vocab = logits.shape
        loss = F.cross_entropy(logits.view(batch * context, vocab), y.view(batch * context))
        return loss

    def generate(self, input_ids, max_len: int = 100):
        """
        input_ids: tensor of shape (batch_size, context_length)
        return: tensor of shape (batch_size, max_len)
        """
        batch, context = input_ids.shape
        input_ids = input_ids.to(self.device)

        for _ in range(max_len):
            input_cropped = input_ids[:, -context:]
            logits = self(input_cropped)
            # ATTENTION: we only look at the last token (hence the name 'Bigram')
            logits_last = logits[:, -1, :]  # (batch_size, vocab_size)
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


def train(model: nn.Module, data: DataLoader, epochs: int = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

    for epoch in tqdm(range(epochs), desc="Training"):
        for x_batch, y_batch in data:
            optimizer.zero_grad()
            loss = model.loss(x_batch, y_batch)
            loss.backward()
            optimizer.step()

        tqdm.write(f"Epoch {epoch + 1}/{epochs} - loss: {loss.item()}")

    return model


if __name__ == "__main__":
    context_length = 16
    dataset = TextDataset(open("data/tiny-shakespeare.txt").read(), device=device, context_length=context_length)
    data = DataLoader(dataset, batch_size=2)
    model = LM(data.dataset.vocab_size, context_length=dataset.context_length, device=device)
    x_batch, y_batch = next(iter(data))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    res = model(x_batch)
    batch, context, vocab = res.shape
    loss = F.cross_entropy(res.view(batch * context, vocab), y_batch.view(batch * context))

    """
    Note: Internally, this applies the softmax function to the logits and then computes the cross entropy loss.
    The softmax gives us the probability distribution over the vocabulary for each token in the context.
    The target is a single token, so we can use the cross entropy loss to compare the predicted distribution.
    """
    res_softmax = F.softmax(res.view(batch * context, vocab), dim=-1)
    target_prob = torch.zeros_like(res_softmax).to(device)  # e.g. 65 zeros
    target_prob[:, y_batch] = 1.0  # the probability of the target token is 1.0

    loss2 = -torch.sum(target_prob * torch.log(res_softmax))  # something like this

    print(f"{data.dataset.decode_batch(x_batch)} -> {data.dataset.decode_batch(y_batch)}: {res.shape}")

    # generate new content
    start_text = "Before we proceed"
    generated = model.generate(data.dataset.encode_batch([start_text[:context]]), max_len=100)
    print(data.dataset.decode_batch(generated))

    # train the model and generate new content again
    model = torch.compile(model, fullgraph=True, dynamic=False)
    model = train(model, DataLoader(dataset, batch_size=32, num_workers=4), epochs=1)
    generated = model.generate(data.dataset.encode_batch([start_text[:context]]), max_len=1000)
    print(data.dataset.decode_batch(generated)[0])
