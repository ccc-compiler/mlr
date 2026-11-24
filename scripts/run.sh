#!/bin/sh
# Compile an MLR program, lower it to LLVM IR and run it.
#
#   scripts/run.sh examples/scalar.mlr
#
# Requires: mlir-opt, mlir-translate and clang.
set -e

src="$1"
if [ -z "$src" ]; then
    echo "usage: scripts/run.sh <file.mlr>" >&2
    exit 1
fi

root=$(cd "$(dirname "$0")/.." && pwd)
# The programs may load data files with a path relative to the repository.
cd "$root"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cargo run --quiet --manifest-path "$root/Cargo.toml" -- "$src" -o "$work/prog.mlir"

mlir-opt "$work/prog.mlir" \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-to-llvm \
    --reconcile-unrealized-casts \
    -o "$work/lowered.mlir"

mlir-translate --mlir-to-llvmir "$work/lowered.mlir" -o "$work/prog.ll"

sdk=$(xcrun --show-sdk-path 2>/dev/null || true)
if [ -n "$sdk" ]; then
    clang -isysroot "$sdk" -Wno-override-init "$work/prog.ll" "$root/runtime/mlr_runtime.c" -o "$work/prog"
else
    clang -Wno-override-init "$work/prog.ll" "$root/runtime/mlr_runtime.c" -o "$work/prog"
fi

"$work/prog"
