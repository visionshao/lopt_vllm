

seq_len1=8192
seq_len2=16384
seq_len3=32768
seq_len4=65536
seq_len5=132000
gpu_num=1

# run benchmarking script

vllm bench serve --port 8008 --save-result --save-detailed \
  --backend openai \
  --host localhost \
  --model qwen3-8b \
  --tokenizer ../models/Qwen3-8B \
  --endpoint /v1/completions \
  --dataset-name custom \
  --dataset-path ../data/vllm_service/LongBenchV2_${seq_len2}.jsonl \
  --custom-output-len 32 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --temperature=0.3 \
  --top-p=0.75 \
  --result-dir "./log/"