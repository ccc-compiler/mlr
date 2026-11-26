# AGENTS.md

## What this repo is

MLR is a zero-dependency Rust compiler for a high-level language that emits MLIR **text** (not via MLIR C++ libraries). It has its own IR (`src/ir.rs`) that mirrors MLIR's model; lowering is a structural region-to-region mapping.

## Commands

- Build: `cargo build --release` (no external crates; `Cargo.toml` has an empty `[dependencies]` by design — don't add crates casually).
- CI: `.github/workflows/ci.yml` (Linux CPU + macOS MPS; runs `cargo test --release` with `MLR_REQUIRE_E2E=1`, plus `scripts/run_all.sh` on macOS).
- Tests: `cargo test` (or `scripts/test.sh`). Integration tests in `tests/compile.rs` drive the `mlr` binary via `CARGO_BIN_EXE_mlr`; `tests/ir.rs` builds bad IR modules directly against the lib target to test the verifier.
- Run an example end-to-end: `scripts/run.sh examples/scalar.mlr` (or `scripts/run_mps.sh` for the Metal/GPU runtime). These need `mlir-opt`, `mlir-translate`, and `clang` — on macOS via Homebrew llvm (`/opt/homebrew/opt/llvm/bin`). E2e tests **skip when the tool chain is missing, but any actual run failure fails the test** (a run that fails while the tools are present panics); set `MLR_REQUIRE_E2E=1` to make a missing tool chain a hard failure (CI does). The MPS GPU test only runs on macOS.

## Gotchas

- `scripts/run.sh` / `run_mps.sh` `cd` to the repo root before running: example programs resolve data/weight files (`.bin`, `.pt`) **relative to the repo root**, not the example's directory.
- `mlr.load` and `mlr.save_pt` handle raw `.bin`, PyTorch `.pt`, safetensors, and `hf://` URLs; implementation is `src/pt.rs`.
- The runtime library has two interchangeable implementations of the same interface: `runtime/mlr_runtime.c` (CPU loops) and `runtime/mlr_runtime_mps.mm` (Metal Performance Shaders, macOS only). Compiled programs don't know which one they're linked against; keep the interfaces in sync when touching either.
- `--stage <1-5>` stops after a lowering stage; use `--verify` to check output with `mlir-opt` and `--ast` for the typed tree when debugging.

## Compiler pipeline (order matters)

`lexer.rs` → `parser.rs` → `check.rs` (types + tensor shapes) → `build.rs` (Stage 1: typed tree → high-level IR) → `passes.rs` (Stage 2 canonicalize, Stage 3 domain lowering: `mlr.map` → `linalg.generic`) → `passes/bufferize.rs` (Stage 4: tensors → memrefs) → `passes/runtime_ops.rs` (Stage 5: domain ops → `func.call @mlr_*` into the runtime).

Everything after `check.rs` operates on `src/ir.rs` types (`Value`, `Op`, `Region`, `Type`, `Attr`), not the AST. When changing semantics, changes usually span `check.rs` + `build.rs` + a pass, and the printed MLIR must stay verifiable by `mlir-opt`.

## Docs

- `README.md` is English and `README.tw.md` its Traditional Chinese mirror — keep both in sync (the TW file is not auto-generated).
- `doc/mlr.md` — language spec
- `doc/architecture.md` — compiler pipeline and IR design (read this first)
- `doc/struct.md`, `doc/train.md`, `doc/lib.md` — struct types, autodiff/training, stdlib design
- `doc/roadmap.md` — written in Chinese; v0.3 items are marked done, §5–§6 deliberately rejected (don't "fix" those omissions)
- `tools/*.py` — plain-numpy training scripts that generate the `.bin`/`.pt` weights the examples embed at compile time
