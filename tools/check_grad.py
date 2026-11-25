#!/usr/bin/env python3
"""Compares the gradients of the MLR program with the ones of PyTorch.

The program `examples/train/train_mlp.mlr` trains an MLP; this script builds
the same model in PyTorch, runs the same forward pass on the same batch and
prints the loss and a few entries of every gradient. The MLR program
`examples/train/grad_check.mlr` prints the same numbers; the two outputs
must agree to f32 precision.

Usage:
    python3 tools/check_grad.py    # the PyTorch numbers
    scripts/run.sh examples/train/grad_check.mlr    # the MLP numbers of MLR
    scripts/run.sh examples/train/grad_check2.mlr   # the conv/pool/embed numbers
"""

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os_path = __file__.rsplit("/", 2)[0]

x = torch.from_numpy(
    np.fromfile(f"{ROOT}/examples/train/images.bin", dtype=np.float32).reshape(100, 784)
)
y = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/labels.bin", dtype=np.int32).astype(np.int64))
w1 = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/w1.bin", dtype=np.float32).reshape(784, 128))
b1 = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/b1.bin", dtype=np.float32))
w2 = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/w2.bin", dtype=np.float32).reshape(128, 64))
b2 = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/b2.bin", dtype=np.float32))
w3 = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/w3.bin", dtype=np.float32).reshape(64, 10))
b3 = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/b3.bin", dtype=np.float32))

for p in (w1, b1, w2, b2, w3, b3):
    p.requires_grad_()

h1 = F.relu(x @ w1 + b1)
h2 = F.relu(h1 @ w2 + b2)
loss = F.cross_entropy(h2 @ w3 + b3, y)
loss.backward()

print(f"loss        {loss.item():.6f}")
print(f"dw1[0,0]    {w1.grad[0, 0].item():.6f}")
print(f"dw1[783,127] {w1.grad[783, 127].item():.6f}")
print(f"db1[0]      {b1.grad[0].item():.6f}")
print(f"dw2[0,0]    {w2.grad[0, 0].item():.6f}")
print(f"dw3[0,0]    {w3.grad[0, 0].item():.6f}")
print(f"db3[9]      {b3.grad[9].item():.6f}")

# ───────── the convolution, the max pooling and the embedding ─────────

xc = torch.from_numpy(
    np.fromfile(f"{ROOT}/examples/train/gcx.bin", dtype=np.float32).reshape(8, 8).copy()
).requires_grad_()
cw1 = torch.from_numpy(
    np.fromfile(f"{ROOT}/examples/train/gccw1.bin", dtype=np.float32).reshape(4, 1, 3, 3)
)
cb1 = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/gccb1.bin", dtype=np.float32))
cw2 = torch.from_numpy(
    np.fromfile(f"{ROOT}/examples/train/gccw2.bin", dtype=np.float32).reshape(64, 1)
)
for p in (cw1, cb1, cw2):
    p.requires_grad_()
c1 = F.relu(F.conv2d(xc[None, None], cw1, cb1, padding=1))
p1 = F.max_pool2d(c1, 2)
loss_c = (p1.reshape(-1) @ cw2).sum()
loss_c.backward()
print(f"conv loss   {loss_c.item():.6f}")
print(f"dcw1[0,0,0,0] {cw1.grad[0,0,0,0].item():.6f}")
print(f"dcw1[3,0,2,2] {cw1.grad[3,0,2,2].item():.6f}")
print(f"dcb1[2]     {cb1.grad[2].item():.6f}")
print(f"dcw2[63,0]  {cw2.grad[63,0].item():.6f}")
print(f"dcx[7,7]    {xc.grad[7,7].item():.6f}")

wte = torch.from_numpy(
    np.fromfile(f"{ROOT}/examples/train/gewte.bin", dtype=np.float32).reshape(10, 8)
)
head = torch.from_numpy(
    np.fromfile(f"{ROOT}/examples/train/gehead.bin", dtype=np.float32).reshape(8, 10)
)
eb = torch.from_numpy(np.fromfile(f"{ROOT}/examples/train/geb.bin", dtype=np.float32))
ids = torch.from_numpy(
    np.fromfile(f"{ROOT}/examples/train/geids.bin", dtype=np.int32).astype(np.int64)
)
ey = torch.from_numpy(
    np.fromfile(f"{ROOT}/examples/train/gey.bin", dtype=np.int32).astype(np.int64)
)
for p in (wte, head, eb):
    p.requires_grad_()
e = F.embedding(ids, wte)
loss_e = F.cross_entropy(e @ head + eb, ey)
loss_e.backward()
print(f"emb loss    {loss_e.item():.6f}")
print(f"dwte[0,0]   {wte.grad[0,0].item():.6f}")
print(f"dwte[5,7]   {wte.grad[5,7].item():.6f}")
print(f"dwte[1,0]   {wte.grad[1,0].item():.6f}")
print(f"dhead[0,0]  {head.grad[0,0].item():.6f}")
print(f"deb[9]      {eb.grad[9].item():.6f}")
