#!/bin/sh
# Run every example on both runtimes.
#
#   scripts/run_all.sh
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

examples="examples/scalar.mlr examples/tensor.mlr examples/features.mlr \
examples/mlp/mnist.mlr examples/mnist/mnist.mlr examples/gru/gru.mlr \
examples/gpt/gpt.mlr examples/train/train_mlp.mlr \
examples/train/train_cnn.mlr"

pass=0
fail=0

for src in $examples; do
    [ -f "$src" ] || continue
    for script in scripts/run.sh scripts/run_mps.sh; do
        printf '%-28s %-20s ' "$src" "$(basename "$script")"
        if out=$("$script" "$src" 2>/dev/null); then
            echo "ok ($(echo "$out" | head -1))"
            pass=$((pass + 1))
        else
            echo "FAILED"
            fail=$((fail + 1))
        fi
    done
done

echo "----------------------------------------"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
