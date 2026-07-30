#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace_dir"

: "${CARGO:=cargo}"
"$CARGO" build --release --target wasm32-unknown-unknown
mkdir -p pkg
cp target/wasm32-unknown-unknown/release/rainbow_fmm.wasm pkg/rainbow_fmm.wasm

bytes="$(wc -c < pkg/rainbow_fmm.wasm)"
echo "Built pkg/rainbow_fmm.wasm (${bytes} bytes)"

