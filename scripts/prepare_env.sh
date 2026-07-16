curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

conda create -n vllm0140 python=3.12
conda activate vllm0140

# git clone https://github.com/vllm-project/vllm.git
# cd vllm
# git checkout v0.14.0

export MAX_JOBS=32
uv pip install -v --editable . --torch-backend=cu128
pip install vllm[bench]

# Install pybind11 (needed for lopt_cpp C++ extension headers)
pip install pybind11

# Build lopt_cpp C++ extension (独立编译，不通过 vLLM 的 CMake 系统)
bash ../benchmarks/csrc/build.sh