import torch
from torch import nn
from gpt.data import TextDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int, device: torch.device):
        super().__init__()
        self.device = device

        # creates an Embedding that takes vectors of dim [vocab_size] and outputs vectors of dim [vocab_size]
        # TODO: use custom Embeddings
        self.embedding = nn.Embedding(vocab_size, vocab_size).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        logits = self.embedding(x)  # (batch_size, context_length, vocab_size)
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
        input_ids = input_ids.to(self.device)

        for _ in range(max_len):
            logits = self(input_ids)
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
    dataset = TextDataset(open("data/tiny-shakespeare.txt").read(), device=device)
    data = DataLoader(dataset, batch_size=2)
    model = BigramLM(data.dataset.vocab_size, device)
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
    generated = model.generate(data.dataset.encode_batch([start_text]), max_len=100)
    print(data.dataset.decode_batch(generated))

    # train the model and generate new content again
    model = train(model, DataLoader(dataset, batch_size=32), epochs=5)
    generated = model.generate(data.dataset.encode_batch([start_text]), max_len=100)
    print(data.dataset.decode_batch(generated)[0])
