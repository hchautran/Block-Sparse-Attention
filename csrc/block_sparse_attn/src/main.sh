#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CUTLASS_DIR="${ROOT_DIR}/csrc/cutlass"
SRC_DIR="."
OUT_DIR="."

NVCC_BIN="${NVCC_BIN:-nvcc}"
SM_ARCH="${SM_ARCH:-80}"
SM_ARCH_UNDERSCORE="sm_${SM_ARCH}"

mkdir -p "${OUT_DIR}"

if ! command -v "${NVCC_BIN}" >/dev/null 2>&1; then
  echo "nvcc not found (set NVCC_BIN if needed)." >&2
  exit 1
fi

if [ ! -f "${CUTLASS_DIR}/include/cute/algorithm/copy.hpp" ]; then
  echo "CUTLASS headers not found at ${CUTLASS_DIR}/include." >&2
  echo "Check that the cutlass submodule is present or run this script with bash." >&2
  exit 1
fi

"${NVCC_BIN}" \
  -std=c++17 \
  -O3 \
  -arch=${SM_ARCH_UNDERSCORE} \
  -I"${SRC_DIR}" \
  -I"${ROOT_DIR}/csrc/block_sparse_attn" \
  -I"${CUTLASS_DIR}/include" \
  -o "main_sm${SM_ARCH}" \
  "main.cu"

./main_sm${SM_ARCH}
