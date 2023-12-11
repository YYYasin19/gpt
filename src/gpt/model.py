import torch
from torch import nn
from gpt.data import TextDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from gpt.attention import MultiHeadAttention

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout_p: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.fforward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout_p),
        ).to(device)

    def forward(self, x):
        return self.fforward(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_head: int,
        src_embed_dim: int,
        context_length: int,
        dropout_p: float,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        head_size = src_embed_dim // num_head
        self.device = device
        self.attention = MultiHeadAttention(
            head_size=head_size,
            num_heads=num_head,
            src_embed_dim=src_embed_dim,
            context_length=context_length,
            dropout_p=dropout_p,
            device=device,
        )
        self.fforward = FeedForward(src_embed_dim, dropout_p=dropout_p, device=device)
        self.layer_norm_attn = nn.LayerNorm(src_embed_dim).to(self.device)
        self.layer_norm_ffwd = nn.LayerNorm(src_embed_dim).to(self.device)

    def forward(self, x):
        # go through attention mechanism and, additionally, skip connections (better training though res/skip connections)
        x = x + self.attention(self.layer_norm_attn(x))
        x = x + self.fforward(self.layer_norm_ffwd(x))
        return x


class LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_p: float,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__()
        self.device = device

        # creates an Embedding that takes vectors of dim [vocab_size] and outputs vectors of dim [vocab_size]
        # TODO: use custom Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.pos = nn.Embedding(context_length, embed_dim).to(device)  # positional embeddings -> gives each token a space where it is

        # self.attention = AttentionHead(embed_dim=embed_dim, context_length=context_length, head_size=embed_dim)
        self.attention = MultiHeadAttention(
            head_size=embed_dim // num_heads,
            num_heads=num_heads,
            context_length=context_length,
            src_embed_dim=embed_dim,
            dropout_p=dropout_p,
            device=self.device,
        )

        # allows for some computation between the attention output and the logit creation
        self.fforward = FeedForward(embed_dim, dropout_p=dropout_p, device=device)
        num_layers = 4
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    num_head=4, src_embed_dim=embed_dim, context_length=context_length, dropout_p=dropout_p, device=self.device
                )
                for _ in range(num_layers)
            ],
            nn.LayerNorm(embed_dim),
        ).to(self.device)

        self.linear_head = nn.Linear(embed_dim, vocab_size).to(device)

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


def train(model: nn.Module, train_data: TextDataset, iterations: int = 100):
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    for it in tqdm(range(iterations), desc="Training"):
        x_batch, y_batch = train_data.sample_batch()

        optimizer.zero_grad()
        loss = model.loss(x_batch, y_batch)
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            tqdm.write(f"{it}/{iterations} - loss: {loss.item():.4f}")

    return model


if __name__ == "__main__":
    context_length = 64
    dropout_p = 0.2
    embedding_size = 128
    batch_size = 64
    num_layers = 4
    num_heads = 4

    text = open("data/tiny-shakespeare.txt").read()
    divide = int(0.9 * len(text))
    train_dataset = TextDataset(text[:divide], device=device, context_length=context_length, batch_size=batch_size)
    val_dataset = TextDataset(text[divide:], device=device, context_length=context_length, batch_size=batch_size)

    data = DataLoader(train_dataset, batch_size=2)
    model = LM(
        train_dataset.vocab_size,
        context_length=train_dataset.context_length,
        embed_dim=embedding_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_p=dropout_p,
        device=device,
    )
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

    print(f"{train_dataset.decode_batch(x_batch)} -> {train_dataset.decode_batch(y_batch)}: {res.shape}")

    # generate new content
    start_text = "Before we proceed"
    generated = model.generate(train_dataset.encode_batch([start_text[:context]]), max_len=50)
    print(train_dataset.decode_batch(generated))

    # train the model and generate new content again
    # model = torch.compile(model, fullgraph=True, dynamic=True, mode="reduce-overhead")
    model = train(model, train_dataset, iterations=500)  # type: ignore
    print(f"Val. Loss: {torch.tensor([model.loss(*val_dataset.sample_batch()) for _ in range(50)]).mean():.4f}")
    generated = model.generate(train_dataset.encode_batch([start_text[:context]]), max_len=1000)
    print(train_dataset.decode_batch(generated)[0])
