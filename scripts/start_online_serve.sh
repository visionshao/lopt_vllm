
seq_len1=8192
seq_len2=65536
seq_len3=131072
gpu_num=1
server_name=qwen3-8b

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Build lopt_cpp extension if not already present
cd "$(dirname "$0")/../benchmarks/csrc"
if ! python -c "from vllm import lopt_cpp" 2>/dev/null; then
    echo "Building lopt_cpp extension..."
    bash build.sh
fi
cd - > /dev/null

# vllm serve ../models/Qwen3-8B \
#  --served-model-name ${server_name} \
#  --port 8008 \
#  --dtype bfloat16 \
#  --max-model-len ${seq_len3} \
#  --max-num-batched-tokens ${seq_len3} \
#  --max-num-seqs 1 \
#  --tensor-parallel-size ${gpu_num} \
#  --data-parallel-size 1


vllm serve ../models/Qwen3-8B \
 --served-model-name ${server_name} \
 --port 8008 \
 --dtype bfloat16 \
 --max-model-len ${seq_len3} \
 --max-num-batched-tokens ${seq_len3} \
 --max-num-seqs 1 \
 --tensor-parallel-size ${gpu_num} \
 --data-parallel-size 1 \
 --enable-lopt \
 --lopt-pool-size 8 \
 --lopt-chunk-size 2048
