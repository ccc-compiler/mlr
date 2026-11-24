#!/bin/sh
# Run the whole test suite.
#
#   scripts/test.sh
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

echo "== unit and integration tests =="
cargo test
