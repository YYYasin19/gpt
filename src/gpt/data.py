import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, text: str, context_length: int = 8, device: torch.device = torch.device("cpu")):
        """
        context_length: the max amount of characters to be used as context
        """
        self.device = device

        # create encoding
        self.tokens = sorted(list(set(text)))
        self.encoding = {t: i for i, t in enumerate(self.tokens)}
        self.decoding = {i: t for i, t in enumerate(self.tokens)}

        # encode text
        self.original_text = text
        self.data = torch.tensor(self.encode(text), dtype=torch.long, device=self.device)

        # metadata
        self.vocab_size = len(self.tokens)
        self.data_len = len(self.data) - context_length
        self.context_length = context_length

    def encode(self, text: str) -> list[int]:
        return list(map(self.encoding.get, text))

    def decode(self, encoded: torch.Tensor) -> str:
        return "".join(list(map(self.decoding.get, encoded.tolist())))

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """
        convert a string to a list of ascii codes
        param text: batch of strings shape (batch_size, 1)
        return: tensor of shape (batch_size, seq_len)
        """
        return torch.tensor([self.encode(s) for s in texts], dtype=torch.long, device=self.device)

    def decode_batch(self, encoded: torch.Tensor) -> list[str]:
        """
        convert a list of ascii codes to a string
        param encoded: tensor of shape (batch_size, seq_len)
        return: batch of strings shape (batch_size, 1)
        """
        return [self.decode(e) for e in encoded]

    def __len__(self):
        """
        this is used by the DataLoader to figure out how many samples there are in the dataset
        """
        return len(self.data) - self.context_length

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return x, y where
            x is a tensor of shape (context_length,)
            y is a tensor of shape (1,) containing the next character
        """
        # Alternative: Randomly sample a starting index -> our DataLoader will shuffle the data
        # idx = torch.randint(self.data_len - self.context_length, (1,))
        # Example: idx = 5, x = data[5:13], y = data[6:14]
        return (
            self.data[idx : idx + self.context_length],
            self.data[idx + 1 : idx + self.context_length + 1],
        )


shakespear_dataset = TextDataset(open("data/tiny-shakespeare.txt").read(), device=device)

if __name__ == "__main__":
    shake = open("data/tiny-shakespeare.txt").read()
    dataset = TextDataset(shake, device=device)
    print(f"vocab_size: {dataset.vocab_size} data_len: {dataset.data_len:,}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for x, y in dataloader:
        print(f"x: {dataset.decode(x[0])} -> y: {dataset.decode(y[0])}")
        break
