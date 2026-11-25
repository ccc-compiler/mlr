#!/usr/bin/env python3
"""Converts a tensor of a model into the raw format of `mlr.load_runtime`:

    [8 bytes: magic "MLRRWT01"] [8: rank] [rank*8: dims] [count*4: values]

Usage:
    python3 tools/to_raw.py model.safetensors wte.weight model_wte.bin
    python3 tools/to_raw.py weights.npy model.bin
"""

import struct
import sys

import numpy as np


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    source, key, dest = sys.argv[1], sys.argv[2], sys.argv[3]

    if source.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(source, framework="np") as f:
            array = np.asarray(f.get_tensor(key))
    elif source.endswith(".npy"):
        array = np.load(source)
    else:
        print(f"cannot read `{source}`; use .safetensors or .npy")
        sys.exit(1)

    array = np.ascontiguousarray(array, dtype=np.float32)
    with open(dest, "wb") as f:
        f.write(b"MLRRWT01")
        f.write(struct.pack("<q", array.ndim))
        for d in array.shape:
            f.write(struct.pack("<q", d))
        array.tofile(f)
    print(f"{dest}: {array.shape} ({array.size} floats)")


if __name__ == "__main__":
    main()
