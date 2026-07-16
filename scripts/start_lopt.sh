# Build lopt_cpp extension if not already present
cd "$(dirname "$0")/../benchmarks/csrc"
if ! python -c "from vllm import lopt_cpp" 2>/dev/null; then
    echo "Building lopt_cpp extension..."
    bash build.sh
fi
cd ../../scripts

cd ../benchmarks
# "LongBenchV2,LEval,ClongEval"
for model in Qwen3-8B
do
    for n_proc in 32
    do
        for chunk_size in -1
        do
            for seq_len in -1
            do
                python run_lopt.py \
                    --n_proc $n_proc \
                    --model $model \
                    --seq_len $seq_len \
                    --save_dir "./results" \
                    --dataset "LEval"
            done
        done
    done
done