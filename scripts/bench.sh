#!/bin/sh
# Time the fixed matrix-multiplication workload of examples/bench/bench.mlr.
#
#   scripts/bench.sh                 # one thread, the CPU runtime
#   MLR_THREADS=4 scripts/bench.sh   # four threads
#   scripts/bench.sh mps             # the Metal runtime (macOS)
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

runtime="${1:-cpu}"
case "$runtime" in
cpu) script="scripts/run.sh" ;;
mps) script="scripts/run_mps.sh" ;;
*) echo "usage: scripts/bench.sh [cpu|mps]" >&2; exit 1 ;;
esac

printf 'runtime=%s threads=${MLR_THREADS:-1}\n' "$runtime" >&2
"$script" examples/bench/bench.mlr
