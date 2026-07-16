# LoPT vLLM — Project Context

## Project Overview
LoPT (Lossless Parallel Tokenizer) integrated into vLLM v0.14.0.
Speeds up long-text tokenization by chunking + multi-process parallel tokenize + C++ match/merge.

## Key Files

### vLLM Integration (already merged)
- `vllm/vllm/tokenizers/lopt_wrapper.py` — Parallel tokenizer (sync, mp.Pool)
- `vllm/vllm/tokenizers/lopt_utils.py` — Helper functions (chunks, pairs, flatten, match, merge)
- `vllm/vllm/tokenizers/__init__.py` — Exports `maybe_get_lopt_tokenizer`

### Benchmark / Dev
- `benchmarks/run_lopt.py` — Standalone benchmark comparing LoPT vs HF tokenizer
- `benchmarks/utils.py` — Benchmark utilities (data loading)
- `benchmarks/csrc/` — C++ extension source (`match_merge.cpp`, compiled by CMake)
- `benchmarks/results/` — Benchmark result logs

### Scripts
- `scripts/start_lopt.sh` — Run LoPT benchmark
- `scripts/start_online_serve.sh` — Start vLLM serve with `--enable-lopt`
- `scripts/benchmark.sh` — vllm bench serve benchmarking
- `scripts/prepare_env.sh` — Environment setup (conda, uv, pip install)

### Docs
- `docs/integration_plan.md` — Original integration plan (5 files to modify)
- `docs/performance_optimization.md` — Thread-pool nesting fix analysis
- `docs/development_notes.md` — Dev notes (mp.Pool lifecycle verification)

## Datasets
- `data/raw/` — 测试 **LoPT 方法本身**效果与准确度的原始评测数据（LongBenchV2 / LEval / ClongEval）
- `data/vllm_service/` — 测试 **vLLM 真实服务**场景的预处理数据集（LongBenchV2 多长度版本）
- `data/prepare_datasets.py` — 预处理脚本，从 `raw/` 读取，处理后写入 `vllm_service/`

## Architecture
```
Input → LLMEngine → InputPreprocessor._tokenize_prompt()
  → [long text?] → LoptParallelTokenizer.__call__()
    → chunks() → mp.Pool.map() tokenize → C++ match() → C++ merge()
  → [short text] → standard HF tokenizer
```

## Build
```bash
# C++ extension is built automatically during pip install:
cd vllm && pip install -e .
```

## Usage
```bash
# Enable LoPT:
vllm serve <model> --enable-lopt --lopt-pool-size 8 --lopt-chunk-size 2048

# Benchmark:
cd benchmarks && python run_lopt.py --model Qwen2.5-7B --dataset LEval
```
