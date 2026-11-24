#!/usr/bin/env python3
"""Trains the small CNN that the MLR example runs.

The network is intentionally tiny so that it can be trained with numpy alone
(no deep learning framework) and so that the exported weights stay small:

    input   28x28x1
    conv1   4 filters, 5x5, padding 2, stride 1  ->  28x28x4
    relu
    maxpool 2x2                                     ->  14x14x4
    flatten                                        ->  784
    dense1  32 units, relu
    dense2  10 units
    argmax

The weights are written as raw little endian float32 files that the MLR
compiler embeds into the program with `mlr.load`.

Usage:
    python3 tools/train_mnist.py [--samples 8000] [--epochs 6]
"""

import argparse
import gzip
import os
import struct
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "examples", "mnist")

# Model configuration.
CONV_FILTERS = 4
CONV_KERNEL = 5
CONV_PAD = 2
POOL = 2
HIDDEN = 32
CLASSES = 10
IMG = 28


def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "not an image file"
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data.astype(np.float32) / 255.0


def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, "not a label file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def conv_forward(x, w, b, pad):
    """x: (N, F, H, W), w: (O, F, K, K) -> (N, O, H, W) with 'same' padding."""
    n, f, h, width = x.shape
    o, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    # im2col: (N, F*K*K, H*W)
    cols = np.zeros((n, f * k * k, h * width), dtype=np.float32)
    for c in range(f):
        for ky in range(k):
            for kx in range(k):
                patch = xp[:, c, ky : ky + h, kx : kx + width]
                cols[:, c * k * k + ky * k + kx, :] = patch.reshape(n, -1)
    wm = w.reshape(o, f * k * k)
    out = np.einsum("ok,nkp->nop", wm, cols)
    out += b.reshape(1, o, 1)
    return out.reshape(n, o, h, width), cols


def conv_backward(x, w, cols, dout, pad):
    """cols: (N, F*K*K, H*W) as produced by `conv_forward`."""
    n, f, h, width = x.shape
    o, _, k, _ = w.shape
    wm = w.reshape(o, f * k * k)
    dout = dout.reshape(n, o, h * width)
    dw = np.zeros_like(w)
    db = dout.sum(axis=(0, 2))
    dcols = np.zeros_like(cols)
    for c in range(f):
        for ky in range(k):
            for kx in range(k):
                idx = c * k * k + ky * k + kx
                dw[:, c, ky, kx] = (dout * cols[:, idx, :][:, None, :]).sum(axis=(0, 2))
                dcols[:, idx, :] = (wm[:, idx][None, :, None] * dout).sum(axis=1)
    xp = np.pad(np.zeros_like(x), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    for c in range(f):
        for ky in range(k):
            for kx in range(k):
                idx = c * k * k + ky * k + kx
                xp[:, c, ky : ky + h, kx : kx + width] += dcols[:, idx, :].reshape(n, h, width)
    dx = xp[:, :, pad : pad + h, pad : pad + width]
    return dx, dw, db


def pool_forward(x, k):
    n, f, h, width = x.shape
    ho, wo = h // k, width // k
    xr = x.reshape(n, f, ho, k, wo, k).transpose(0, 1, 2, 4, 3, 5).reshape(n, f, ho, wo, k * k)
    out = xr.max(axis=-1)
    arg = xr.argmax(axis=-1)
    return out, arg


def pool_backward(x, arg, dout, k):
    n, f, h, width = x.shape
    ho, wo = h // k, width // k
    dx = np.zeros((n, f, ho, wo, k * k), dtype=np.float32)
    flat = np.arange(dout.size).reshape(dout.shape)
    np.put_along_axis(dx, arg.reshape(n, f, ho, wo, 1), dout.reshape(n, f, ho, wo, 1), axis=4)
    dx = dx.reshape(n, f, ho, wo, k, k).transpose(0, 1, 2, 4, 3, 5).reshape(n, f, h, width)
    return dx


def forward(params, x, training):
    w1, b1, w2, b2, w3, b3 = params
    n = x.shape[0]
    conv, cols = conv_forward(x, w1, b1, CONV_PAD)
    relu1 = np.maximum(conv, 0.0)
    pooled, arg = pool_forward(relu1, POOL)
    flat = pooled.reshape(n, -1)
    h1 = flat @ w2 + b2
    relu2 = np.maximum(h1, 0.0)
    logits = relu2 @ w3 + b3
    return logits, (x, conv, cols, relu1, arg, pooled, flat, h1, relu2)


def backward(params, cache, dout):
    w1, b1, w2, b2, w3, b3 = params
    x, conv, cols, relu1, arg, pooled, flat, h1, relu2 = cache
    n = conv.shape[0]
    dw3 = relu2.T @ dout
    db3 = dout.sum(axis=0)
    dh = dout @ w3.T
    dh = dh * (h1 > 0)
    dw2 = flat.T @ dh
    db2 = dh.sum(axis=0)
    dflat = dh @ w2.T
    dpooled = dflat.reshape(pooled.shape)
    dpool = pool_backward(relu1, arg, dpooled, POOL)
    dconv = dpool * (conv > 0)
    dx, dw1, db1 = conv_backward(x, w1, cols, dconv, CONV_PAD)
    return [dw1, db1, dw2, db2, dw3, db3], dx


def softmax_cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), labels] + 1e-12).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), labels] -= 1.0
    dlogits /= n
    return loss, dlogits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--test", type=int, default=100)
    args = ap.parse_args()

    print("loading MNIST ...")
    xtr = load_idx_images(os.path.join(HERE, "train-images-idx3-ubyte.gz"))
    ytr = load_idx_labels(os.path.join(HERE, "train-labels-idx1-ubyte.gz"))
    xte = load_idx_labels(os.path.join(HERE, "t10k-labels-idx1-ubyte.gz"))
    xte = load_idx_images(os.path.join(HERE, "t10k-images-idx3-ubyte.gz"))
    yte = load_idx_labels(os.path.join(HERE, "t10k-labels-idx1-ubyte.gz"))

    rng = np.random.default_rng(1234)
    idx = rng.permutation(len(xtr))[: args.samples]
    xtr, ytr = xtr[idx], ytr[idx]
    xtr = xtr.reshape(len(xtr), 1, IMG, IMG)
    xte = xte[: args.test].reshape(args.test, 1, IMG, IMG)

    # Parameters.
    conv_out = CONV_FILTERS * IMG * IMG
    w1 = rng.normal(0, 0.1, (CONV_FILTERS, 1, CONV_KERNEL, CONV_KERNEL)).astype(np.float32)
    b1 = np.zeros(CONV_FILTERS, dtype=np.float32)
    w2 = rng.normal(0, 0.05, (conv_out // (POOL * POOL), HIDDEN)).astype(np.float32)
    b2 = np.zeros(HIDDEN, dtype=np.float32)
    w3 = rng.normal(0, 0.05, (HIDDEN, CLASSES)).astype(np.float32)
    b3 = np.zeros(CLASSES, dtype=np.float32)
    params = [w1, b1, w2, b2, w3, b3]
    velocity = [np.zeros_like(p) for p in params]

    for epoch in range(args.epochs):
        order = rng.permutation(len(xtr))
        total = 0.0
        start = time.time()
        for i in range(0, len(order) - args.batch + 1, args.batch):
            sel = order[i : i + args.batch]
            xb, yb = xtr[sel], ytr[sel]
            logits, cache = forward(params, xb, True)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            grads, _ = backward(params, cache, dlogits)
            for p, g, v in zip(params, grads, velocity):
                v *= 0.9
                v -= args.lr * g
                p += v
            total += loss
        batches = max(1, (len(order) - args.batch + 1) // args.batch)
        print(
            f"epoch {epoch + 1}/{args.epochs} loss {total / batches:.4f} "
            f"({time.time() - start:.1f}s)"
        )

    # Evaluate on the test set.
    logits, _ = forward(params, xte, False)
    pred = logits.argmax(axis=1)
    accuracy = (pred == yte[: args.test]).mean()
    print(f"test accuracy on {args.test} samples: {accuracy * 100:.1f}%")

    os.makedirs(OUT, exist_ok=True)

    def save(name, array):
        path = os.path.join(OUT, name)
        array.astype(np.float32).tofile(path)
        print(f"  {name}: {array.shape} ({array.size} floats)")

    save("w1.bin", w1)
    save("b1.bin", b1)
    save("w2.bin", w2)
    save("b2.bin", b2)
    save("w3.bin", w3)
    save("b3.bin", b3)
    save("images.bin", xte.reshape(args.test, IMG, IMG))
    # The labels are integers and are stored as such.
    yte[: args.test].astype(np.int32).tofile(os.path.join(OUT, "labels.bin"))
    print(f"  labels.bin: ({args.test},) (integers)")
    with open(os.path.join(OUT, "accuracy.txt"), "w") as f:
        f.write(f"{accuracy * 100:.1f}\n")
    print(f"weights written to {OUT}")


if __name__ == "__main__":
    main()
