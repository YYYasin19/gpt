# Generative Pre-trained Transformer

This repository implements the Decoder part of the [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) in pure PyTorch.
The encoding is done using OpenAI's [tiktoken](https://github.com/openai/tiktoken) library.

## Installation

You'll need [pixi](https://github.com/prefix-dev/pixi) or, when you have all the dependencies installed, you can just use pip.

```bash
pixi install # super easy
pip install . --no-deps
```

## Usage

```bash
gpt train --iterations 5000 --text data/tiny-shakespeare.txt
```

this will create a `model.pt` and `config.txt` file.

You can use those to then generate text using

```bash
gpt prompt --model model.pt --config config.txt
```

which simply generates a fixed amount of tokens given the prompt.
