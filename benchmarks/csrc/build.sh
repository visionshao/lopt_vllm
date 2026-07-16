#!/bin/bash
# LoPT C++ extension — independent build script.
#
# Builds the lopt_cpp extension as a standalone .so that can be placed
# where vllm can import it (vllm/vllm/ or the site-packages equivalent).
#
# This is faster than building through vllm's full CMake (which downloads
# CUTLASS and compiles all CUDA kernels). It also isolates any pybind11 /
# Python include issues to just this one file.
#
# Usage:
#   cd benchmarks/csrc && bash build.sh
#   # The .so is placed at ../../vllm/vllm/lopt_cpp*.so
#
# Environment variables:
#   LOPT_INSTALL_DIR  — target directory for the .so (default: ../../vllm/vllm)
#   CXX               — C++ compiler (default: g++)
#   CXXFLAGS          — extra compiler flags

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${LOPT_INSTALL_DIR:-${SCRIPT_DIR}/../../vllm/vllm}"
CXX="${CXX:-g++}"

# Resolve install dir to absolute
INSTALL_DIR="$(cd "$INSTALL_DIR" 2>/dev/null && pwd || echo "$INSTALL_DIR")"

echo "==> lopt_cpp build script"
echo "    Source:     ${SCRIPT_DIR}/match_merge.cpp"
echo "    Install:    ${INSTALL_DIR}/"
echo "    Compiler:   ${CXX}"

# 1. Get Python include path for Python.h
PYTHON_INCLUDES="$(python3-config --includes 2>/dev/null)"
if [ -z "$PYTHON_INCLUDES" ]; then
    echo "ERROR: python3-config not found or no includes"
    exit 1
fi
echo "    Python:     ${PYTHON_INCLUDES}"

# 2. Get pybind11 include path via torch (torch bundles pybind11 headers)
PYBIND11_INCLUDE="$(python3 -c "
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'include'))
" 2>/dev/null)"
if [ -z "$PYBIND11_INCLUDE" ]; then
    echo "ERROR: Could not find pybind11 include path via torch"
    exit 1
fi
echo "    Pybind11:   ${PYBIND11_INCLUDE}"

# 3. Get torch include dir (for any torch headers that pybind11 might need)
TORCH_INCLUDES=()
while IFS= read -r line; do
    TORCH_INCLUDES+=("$line")
done < <(python3 -c "
import torch.utils.cpp_extension as e
for p in e.include_paths():
    print(p)
" 2>/dev/null | grep .)

# 4. Get extension suffix (.cpython-312-x86_64-linux-gnu.so)
EXT_SUFFIX="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"
if [ -z "$EXT_SUFFIX" ]; then
    EXT_SUFFIX=".so"  # fallback
fi

OUTPUT_NAME="lopt_cpp${EXT_SUFFIX}"
OUTPUT_PATH="${INSTALL_DIR}/${OUTPUT_NAME}"

echo "    Output:     ${OUTPUT_PATH}"
echo ""

# Create output directory
mkdir -p "${INSTALL_DIR}"

# Build torch -I flags
TORCH_I_FLAGS=()
for d in "${TORCH_INCLUDES[@]}"; do
    TORCH_I_FLAGS+=("-I${d}")
done

# Compile
set -x
"${CXX}" -O3 -Wall -shared -std=c++17 -fPIC \
    ${PYTHON_INCLUDES} \
    -I"${PYBIND11_INCLUDE}" \
    "${TORCH_I_FLAGS[@]}" \
    "${SCRIPT_DIR}/match_merge.cpp" \
    -o "${OUTPUT_PATH}"
set +x

echo ""
echo "=== BUILD SUCCESS ==="
echo "    ${OUTPUT_PATH}"
ls -lh "${OUTPUT_PATH}"

# Verify import
echo ""
echo "==> Verifying import..."
python3 -c "
import sys
sys.path.insert(0, '${INSTALL_DIR}/..')  # parent of vllm package
from vllm import lopt_cpp
print('    Functions:', [x for x in dir(lopt_cpp) if not x.startswith('_')])
print('    Import OK!')
"
