from gpt.model import LM, train as train_lm
from gpt.data import Encoder, TextDataset
import torch
import click


config_dict = {
    "text": "data/tiny-shakespeare.txt",
    "store-weights": "model.pt",
    "load-weights": None,
    "device": torch.device("mps" if torch.cuda.is_available() else "cpu"),
}


def load_model(weights_path, config: dict):
    enc = Encoder()
    config["vocab_size"] = enc.encoding.max_token_value
    model = LM(**config)
    model.load_state_dict(torch.load(weights_path))
    return model


@click.group()
def main():
    pass


@main.command()
@click.option("--context_length", default=64, help="Context length for the model.")
@click.option("--iterations", default=100, help="Number of iterations for training.")
@click.option("--batch_size", default=32, help="Batch size for training.")
@click.option("--text", default="data/tiny-shakespeare.txt", help="Path to the text file for training.")
@click.option("--store_weights", default="model.pt", help="Path to store the trained model weights.")
@click.option("--embed_dim", default=128, help="Embedding dimension for the model.")
@click.option("--num_layers", default=6, help="Number of layers for the model.")
@click.option("--num_heads", default=6, help="Number of heads for the model.")
@click.option("--dropout_p", default=0.1, help="Dropout probability for the model.")
def train(context_length, iterations, batch_size, text, store_weights, embed_dim, num_layers, num_heads, dropout_p):
    config_dict.update(
        {
            "context_length": context_length,
            "iterations": iterations,
            "batch_size": batch_size,
            "text": text,
            "store-weights": store_weights,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout_p": dropout_p,
            "store_weights": "model.pt",
        }
    )
    dataset = TextDataset(
        open(text, "r").read(),
        context_length=config_dict["context_length"],
        batch_size=config_dict["batch_size"],
        device=config_dict["device"],
    )
    model = LM(dataset.vocab_size, **config_dict)
    model = train_lm(model, dataset, iterations=iterations)
    torch.save(model.state_dict(), store_weights)
    with open("config.txt", "w") as f:
        config_dict.pop("device")
        f.write(str(config_dict))


@main.command()
@click.argument("weight-path", type=click.Path(exists=True))
@click.argument("config-path", type=click.Path(exists=True))
@click.option("--device", default="cpu", help="Device to run the model on.")
@click.option("--gen-len", default=100, help="Number of tokens to generate")
def prompt(weight_path, config_path, device, gen_len):
    with open(config_path, "r") as f:
        config_dict = eval(f.read())

    config_dict["device"] = torch.device(device)

    model = load_model(weight_path, config_dict)

    enc = Encoder()
    while True:
        text = input(">> Enter some text: ")
        text = text.strip()
        if not text:
            break
        text = text[: config_dict["context_length"]]
        generated = model.generate(enc.encode_batch([text]), max_len=gen_len)
        print(enc.decode_batch(generated)[0])


if __name__ == "__main__":
    main()
