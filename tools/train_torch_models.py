#!/usr/bin/env python3
"""Trains the models that the MLR examples run.

Three models are trained, all of them small enough to be exported as raw
float32 files that the MLR compiler embeds into the program with `mlr.load`:

* an MLP that classifies MNIST digits (`examples/mlp`),
* a GRU that generates text byte by byte (`examples/gru`),
* a GPT (a small causal transformer) that does the same (`examples/gpt`).

The models are the ones of the PyTorch example library
(`ml/model/02-nn/torch`): the same widths, the same number of layers and the
same hyperparameters, only the corpus is read as bytes instead of characters
so that the runtime can print the generated text directly.

Usage:
    python3 tools/train_torch_models.py                 # train everything
    python3 tools/train_torch_models.py --models mlp    # only the MLP
    python3 tools/train_torch_models.py --epochs 300    # more training
"""

import argparse
import gzip
import math
import os
import struct
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "examples", "lm", "input.txt")

# The configuration of the language models, the same as in the PyTorch
# example library.
BLOCK = 16
WIDTH = 32
LAYERS = 2
HEADS = 2
HEAD_DIM = WIDTH // HEADS
TEMPERATURE = 0.8

# The configuration of the classifier.
HIDDEN = (128, 64)
CLASSES = 10
IMG = 28


# ───────── data ─────────

def download(url, path):
    if os.path.exists(path):
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def load_mnist():
    """The MNIST training and test sets, in the original format."""
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train-images-idx3-ubyte.gz": base + "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": base + "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": base + "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": base + "t10k-labels-idx1-ubyte.gz",
    }
    for name, url in files.items():
        download(url, os.path.join(HERE, name))
    with gzip.open(os.path.join(HERE, "train-images-idx3-ubyte.gz"), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        x_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    with gzip.open(os.path.join(HERE, "train-labels-idx1-ubyte.gz"), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        y_train = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(os.path.join(HERE, "t10k-images-idx3-ubyte.gz"), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        x_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    with gzip.open(os.path.join(HERE, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        y_test = np.frombuffer(f.read(), dtype=np.uint8)
    return (
        x_train.astype(np.float32) / 255.0,
        y_train.astype(np.int64),
        x_test.astype(np.float32) / 255.0,
        y_test.astype(np.int64),
    )


def load_corpus():
    """The corpus, read as bytes; the vocabulary is the set of bytes."""
    with open(CORPUS, "rb") as f:
        data = f.read()
    vocab = sorted(set(data))
    return data, vocab


def windows(data, vocab, block):
    """The training windows of the corpus: every line gives a few of them."""
    index = {byte: i for i, byte in enumerate(vocab)}
    sequences = []
    for line in data.split(b"\n"):
        if not line:
            continue
        tokens = [index[byte] for byte in line]
        if len(tokens) < block + 1:
            continue
        for start in range(0, len(tokens) - block + 1):
            sequences.append(tokens[start:start + block])
    return sequences


# ───────── models ─────────

class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN, classes=CLASSES):
        super().__init__()
        layers = []
        sizes = [IMG * IMG, *hidden, classes]
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(fan_in, fan_out))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class GRULanguageModel(nn.Module):
    """A character (here: byte) level GRU language model."""

    def __init__(self, vocab, width=WIDTH):
        super().__init__()
        self.wte = nn.Embedding(vocab, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def forward(self, tokens):
        hidden, _ = self.gru(self.wte(tokens))
        return self.lm_head(hidden)


class CausalSelfAttention(nn.Module):
    def __init__(self, width, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=2)
        q = q.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
        k = k.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
        v = v.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // self.n_head)
        future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        out = scores.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(batch, length, width)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, width, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, n_head)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    """A small causal transformer with pre layer normalization."""

    def __init__(self, vocab, block=BLOCK, width=WIDTH, n_layer=LAYERS, n_head=HEADS):
        super().__init__()
        self.block = block
        self.wte = nn.Embedding(vocab, width)
        self.wpe = nn.Embedding(block, width)
        self.blocks = nn.ModuleList(TransformerBlock(width, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        position = torch.arange(length, device=tokens.device)
        x = self.wte(tokens) + self.wpe(position)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


# ───────── training ─────────

def train_language_model(model, sequences, epochs, lr=0.01):
    data = torch.tensor(sequences, dtype=torch.long)
    x = data[:, :-1]
    y = data[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    total = epochs * len(data)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(0.05, 0.05 + 0.95 * (step + 1) / total) if step < total // 20 else
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - total // 20) / max(1, total - total // 20))),
    )
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(data), 32):
            batch_x = x[i:i + 32]
            batch_y = y[i:i + 32]
            logits = model(batch_x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch_x)
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {epoch + 1}/{epochs} loss {total_loss / len(data):.4f} "
                  f"({time.time() - start:.0f}s)")
    return model


def train_classifier(model, x_train, y_train, epochs, lr=0.002):
    data = torch.tensor(x_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        order = torch.randperm(len(data))
        total = 0.0
        for i in range(0, len(data), 64):
            index = order[i:i + 64]
            logits = model(data[index])
            loss = F.cross_entropy(logits, labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        print(f"  epoch {epoch + 1}/{epochs} loss {total / len(data):.4f}")
    return model


# ───────── export ─────────

def save(out_dir, name, array):
    os.makedirs(out_dir, exist_ok=True)
    array = np.ascontiguousarray(array, dtype=np.float32)
    path = os.path.join(out_dir, name)
    array.tofile(path)
    print(f"  {name}: {array.shape}")


def save_int(out_dir, name, array):
    os.makedirs(out_dir, exist_ok=True)
    array = np.ascontiguousarray(array, dtype=np.int32)
    array.tofile(os.path.join(out_dir, name))
    print(f"  {name}: {array.shape} (integers)")


def export_mlp(model, x_test, y_test):
    out = os.path.join(ROOT, "examples", "mlp")
    save(out, "w1.bin", model.net[0].weight.detach().numpy().T)
    save(out, "b1.bin", model.net[0].bias.detach().numpy())
    save(out, "w2.bin", model.net[2].weight.detach().numpy().T)
    save(out, "b2.bin", model.net[2].bias.detach().numpy())
    save(out, "w3.bin", model.net[4].weight.detach().numpy().T)
    save(out, "b3.bin", model.net[4].bias.detach().numpy())
    # A hundred test images and their labels.
    save(out, "images.bin", x_test[:100].reshape(100, IMG * IMG))
    save_int(out, "labels.bin", y_test[:100])


def export_gru(model, vocab):
    out = os.path.join(ROOT, "examples", "gru")
    save(out, "wte.bin", model.wte.weight.detach().numpy())
    # The table that maps a token index back to its byte, so that the
    # example can print the generated text.
    save_int(out, "vocab.bin", np.array(vocab, dtype=np.int32))
    w_ih = model.gru.weight_ih_l0.detach().numpy()
    w_hh = model.gru.weight_hh_l0.detach().numpy()
    b_ih = model.gru.bias_ih_l0.detach().numpy()
    b_hh = model.gru.bias_hh_l0.detach().numpy()
    # The gates of PyTorch are ordered reset, update, new.
    for index, name in enumerate(("r", "z", "n")):
        rows = slice(index * WIDTH, (index + 1) * WIDTH)
        save(out, f"w_{name}_ih.bin", w_ih[rows])
        save(out, f"w_{name}_hh.bin", w_hh[rows])
        save(out, f"b_{name}.bin", b_ih[rows] + b_hh[rows])
    save(out, "lm_head.bin", model.lm_head.weight.detach().numpy().T)


def export_gpt(model, vocab):
    out = os.path.join(ROOT, "examples", "gpt")
    save(out, "wte.bin", model.wte.weight.detach().numpy())
    save_int(out, "vocab.bin", np.array(vocab, dtype=np.int32))
    save(out, "wpe.bin", model.wpe.weight.detach().numpy())
    for layer, block in enumerate(model.blocks):
        prefix = f"layer{layer}"
        save(out, f"{prefix}_ln1_g.bin", block.ln1.weight.detach().numpy())
        save(out, f"{prefix}_ln1_b.bin", block.ln1.bias.detach().numpy())
        save(out, f"{prefix}_ln2_g.bin", block.ln2.weight.detach().numpy())
        save(out, f"{prefix}_ln2_b.bin", block.ln2.bias.detach().numpy())
        qkv = block.attn.qkv.weight.detach().numpy()
        proj = block.attn.proj.weight.detach().numpy()
        for head in range(HEADS):
            columns = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
            rows = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
            save(out, f"{prefix}_wq{head}.bin", qkv[0:WIDTH, columns])
            save(out, f"{prefix}_wk{head}.bin", qkv[WIDTH:2 * WIDTH, columns])
            save(out, f"{prefix}_wv{head}.bin", qkv[2 * WIDTH:3 * WIDTH, columns])
            save(out, f"{prefix}_wproj{head}.bin", proj[rows, :])
        save(out, f"{prefix}_fc1.bin", block.fc1.weight.detach().numpy().T)
        save(out, f"{prefix}_fc2.bin", block.fc2.weight.detach().numpy().T)
    save(out, "ln_f_g.bin", model.ln_f.weight.detach().numpy())
    save(out, "ln_f_b.bin", model.ln_f.bias.detach().numpy())
    save(out, "lm_head.bin", model.lm_head.weight.detach().numpy().T)


# ───────── main ─────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="mlp,gru,gpt")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wanted = [name.strip() for name in args.models.split(",") if name.strip()]

    if "mlp" in wanted:
        print("training the MLP on MNIST")
        x_train, y_train, x_test, y_test = load_mnist()
        model = MLP()
        epochs = args.epochs or 5
        train_classifier(model, x_train, y_train, epochs)
        with torch.no_grad():
            logits = model(torch.tensor(x_test[:1000], dtype=torch.float32))
            accuracy = (logits.argmax(1).numpy() == y_test[:1000]).mean()
        print(f"  test accuracy: {accuracy:.2%}")
        export_mlp(model, x_test, y_test)

    if "gru" in wanted or "gpt" in wanted:
        data, vocab = load_corpus()
        sequences = windows(data, vocab, BLOCK)
        print(f"corpus: {len(data)} bytes, {len(vocab)} different bytes, {len(sequences)} windows")
        epochs = args.epochs or 300

    if "gru" in wanted:
        print("training the GRU")
        model = GRULanguageModel(len(vocab))
        train_language_model(model, sequences, epochs)
        export_gru(model, vocab)
        # The prompt of the example: the first bytes of a line of the corpus.
        prompt = sequences[0][:4]
        save_int(os.path.join(ROOT, "examples", "gru"), "prompt.bin", prompt)
        print(f"  prompt: {bytes(vocab[i] for i in prompt)!r}")

    if "gpt" in wanted:
        print("training the GPT")
        model = GPT(len(vocab))
        train_language_model(model, sequences, epochs)
        export_gpt(model, vocab)
        # The prompt of the example: a whole window of the corpus, so that
        # the attention of the example has its full context at once.
        prompt = sequences[0]
        save_int(os.path.join(ROOT, "examples", "gpt"), "prompt.bin", prompt)
        print(f"  prompt: {bytes(vocab[i] for i in prompt)!r}")


if __name__ == "__main__":
    main()
