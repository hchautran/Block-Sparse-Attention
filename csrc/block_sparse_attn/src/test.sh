#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
DEBUG="${DEBUG:-0}"

clear
echo "Compiling test.cu with ${NVCC} (ARCH=${ARCH})..."
"${NVCC}" -std=c++17 -O2 -lineinfo -arch="${ARCH}" \
  -I "${SCRIPT_DIR}/../../cutlass/include" \
  "${SCRIPT_DIR}/test.cu" -lcublas -o "${SCRIPT_DIR}/test_cute"

echo "Running ${SCRIPT_DIR}/test_cute"
"${SCRIPT_DIR}/test_cute"
