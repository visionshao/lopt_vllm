# LoPT — Lossless Parallel Tokenizer

**LoPT** 通过将长文本分块、多进程并行 tokenize、C++ match/merge 精确合并重叠区域，
在保证 token 对齐完全无损的前提下，大幅加速长文本 tokenization。

已集成到 **vLLM v0.14.0**，兼容 **CUDA 12.8** 及 **Python 3.12**。

---

## 架构

```
输入文本
  → LoptParallelTokenizer.__call__()
    → [短文本 < 2×chunk_size] → 标准 HF tokenizer（串行快速路径）
    → [长文本]
        ├─ 1. chunks(text, chunk_size, overlap)          → 分块
        ├─ 2. mp.Pool.map(_tokenize_chunk, chunks)       → 并行 tokenize
        ├─ 3. 提取 offset_mapping 起始位置
        ├─ 4. C++ lopt_cpp.match(chunk_a, chunk_b)       → 查找重叠边界
        ├─ 5. C++ lopt_cpp.merge(shards, indices)        → 无损合并
        └─ → BatchEncoding（与串行 HF tokenizer 结果完全一致）
```

## 特性

- **Lossless** — 输出 token ID 与 HuggingFace 串行结果 100% 一致（benchmark 已验证）
- **并行加速** — 通过 `multiprocessing.Pool` 多进程并行 tokenize
- **C++ 核心** — 重叠匹配和合并通过 pybind11 C++17 扩展实现，性能极佳
- **透明回退** — 短文本自动走标准 HF tokenizer，无需额外配置

---

## 目录结构

| 路径 | 用途 |
|------|------|
| `vllm/` | 已合入 LoPT 的 vLLM v0.14.0 完整代码 |
| `benchmarks/csrc/` | C++ pybind11 扩展源码（`match_merge.cpp`）+ 独立编译脚本 |
| `benchmarks/results/` | 基准测试结果日志 |
| `benchmarks/run_lopt.py` | LoPT vs HF 独立 benchmark 脚本 |
| `benchmarks/utils.py` | Benchmark 数据加载工具 |
| `data/raw/` | 测试 **LoPT 方法准确度**的原始评测数据集（LongBenchV2 / LEval / ClongEval） |
| `data/vllm_service/` | 测试 **vLLM 服务性能**的预处理数据集（LongBenchV2 多长度版本） |
| `data/prepare_datasets.py` | 数据集预处理脚本：读 `raw/`，处理后写入 `vllm_service/` |
| `data/README.md` | 数据集说明（目录结构、下载方式、预处理方法） |
| `models/` | 本地模型权重（如 `Qwen3-8B/`），供 benchmark 和 serve 脚本使用 |
| `scripts/` | 运行脚本（benchmark、serve、环境配置、smoke test） |

### 数据集说明

- **`data/raw/`** — 原始数据集，用于验证 LoPT 并行 tokenize 结果是否与串行一致（LongBenchV2、LEval、CLongEval）
- **`data/vllm_service/`** — 对原始数据做截断/填充预处理后生成的多长度版本（8K/16K/32K/64K/128K），用于测试启用 LoPT 时 vLLM 服务的吞吐和延迟

---

## 环境构建

### 前置要求

- Python 3.12, CUDA 12.8
- Conda + [uv](https://docs.astral.sh/uv/) 包管理器

### 完整安装步骤

```bash
# 1. 创建 conda 环境
conda create -n vllm-lopt python=3.12 -y
conda activate vllm-lopt

# 2. 安装 LoPT 增强的 vLLM（editable 模式 + CUDA 12.8）
#    ⚠️ 此步骤会编译所有 CUDA kernel，耗时 10-30+ 分钟
#    ⚠️ 建议设置 MAX_JOBS 避免 OOM
export MAX_JOBS=24
uv pip install -v --editable vllm/ --torch-backend=cu128

# 3. 安装 benchmark 依赖
pip install vllm[bench]

# 4. 编译 C++ 扩展（独立编译，~5 秒）
cd benchmarks/csrc && bash build.sh && cd ../..
```

> **编译调优：** 默认 Ninja 并行数等于 CPU 核心数（224 核环境中可达 67+ 个并行编译进程），
> 在 cgroup 内存限制（如 250 GiB）下容易触发 OOM kill。
> 设置 `MAX_JOBS=24` 可将编译池限制为 24 个并行任务，内存稳定在 55–84 GiB。

### C++ 扩展独立编译

C++ 扩展可以不经过 vLLM CMake 系统独立编译：

```bash
cd benchmarks/csrc
bash build.sh
# 输出: vllm/vllm/lopt_cpp.cpython-312-x86_64-linux-gnu.so
```

环境变量：`LOPT_INSTALL_DIR`（目标目录，默认 `../../vllm/vllm`）、`CXX`（编译器）、`CXXFLAGS`。

---

## 使用

### vLLM Serve 启用 LoPT

```bash
vllm serve Qwen3-8B \
    --enable-lopt \
    --lopt-pool-size 8 \
    --lopt-chunk-size 2048 \
    --max-model-len 131072 \
    --max-num-batched-tokens 131072 \
    --port 8008
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-lopt` | `False` | 开启并行 tokenization |
| `--lopt-pool-size` | `16` | 并行进程数 |
| `--lopt-chunk-size` | `4096` | 分块大小（字符数） |

### API 测试

```bash
curl http://localhost:8008/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-8b",
        "prompt": "San Francisco is a",
        "max_tokens": 32,
        "temperature": 0
    }'
```

### 基准测试（LoPT 方法性能）

在 `benchmarks/` 下运行独立 benchmark，对比 LoPT 与标准 HF tokenizer 的速度和准确度：

```bash
cd benchmarks
python run_lopt.py --model Qwen3-8B --dataset LEval --n_proc 32
```

### 服务压测（vLLM 在线服务性能）

```bash
# 先启动服务
bash scripts/start_online_serve.sh

# 服务就绪后，另开终端压测
bash scripts/benchmark.sh
```

---

## Benchmark 结果

| 指标 | 数据 |
|------|------|
| **模型** | Qwen3-8B |
| **数据集** | LEval（537 个样本） |
| **并行进程数** | 32 |
| **HF 串行平均** | 0.04 ms |
| **LoPT 并行平均** | 0.01 ms |
| **加速比** | **4.4×** |
| **准确度** | **100.00%**（所有 token 完全匹配） |

结果文件：`benchmarks/results/nproc=32_seqlen=-1_chunksize=-1_model=Qwen3-8B_dataset=LEval.txt`

---

## C++ 扩展

两个 pybind11 函数用于合并阶段的精确拼接：

| 函数 | 说明 |
|------|------|
| `match(chunks0, chunks1, chunk_size, mismatch_thres)` | 查找相邻 chunk 之间最长的重叠后缀/前缀 |
| `merge(chunks, matches)` | 根据 match 结果拼接各 chunk，去除重叠 |

源码位于 `benchmarks/csrc/match_merge.cpp`。
编译后通过 `from vllm import lopt_cpp` 在 Python 中调用。

---

## vLLM 集成修改点

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `vllm/tokenizers/lopt_utils.py` | **新增** | 核心工具函数（`chunks`、`pairs`、`flatten` 等） |
| `vllm/tokenizers/lopt_wrapper.py` | **新增** | `LoptParallelTokenizer` + `maybe_get_lopt_tokenizer` 工厂函数 |
| `vllm/tokenizers/__init__.py` | **修改** | 导出 `maybe_get_lopt_tokenizer` |
| `vllm/config/model.py` | **修改** | `ModelConfig` 新增 `enable_lopt` / `lopt_pool_size` / `lopt_chunk_size` 字段 |
| `vllm/engine/arg_utils.py` | **修改** | 添加 CLI 参数 `--enable-lopt` / `--lopt-pool-size` / `--lopt-chunk-size` |
| `vllm/entrypoints/openai/serving_completion.py` | **修改** | 根据 `enable_lopt` 决定是否调用 `LoptParallelTokenizer` |
| `vllm/CMakeLists.txt` | **修改** | 新增 `lopt_cpp` CMake 编译目标 |
| `vllm/setup.py` | **修改** | 新增 `_build_lopt_cpp()` 方法，作为 C++ 扩展的独立编译回退方案 |

---

## 已知问题

- **集成范围**：LoPT 当前仅在 OpenAI-compatible serving 层（`serving_completion.py`，`/v1/completions` 端点）生效。engine 级的 `preprocess.py` 中的集成代码已注释，因此 `/v1/chat/completions` 等端点暂未启用 LoPT。
- **数组维度修复**：C++ `merge()` 函数要求 1D numpy 数组（`unchecked<1>()`）。`_parallel_encode` 方法中使用 `flatten()` 将 batch 维度 `(1, seq_len)` 展平为 `(seq_len,)` 后再传入 C++ 扩展。
