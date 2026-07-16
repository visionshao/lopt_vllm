# 数据集

## 目录结构

```
data/
├── raw/                          # 原始数据集（不上传 GitHub，需自行下载）
│   ├── LongBenchV2/
│   │   └── data.json             # LongBenchV2 评测数据
│   ├── LEval/
│   │   ├── Closed-ended-tasks/   # 7 个 JSONL 文件
│   │   └── Open-eneded-tasks/    # 13 个 JSONL 文件
│   └── ClongEval/
│       ├── 1-1_long_story_qa/
│       ├── 1-2_long_conversation_memory/
│       ├── 2-1_long_story_summarization/
│       ├── 3-1_stacked_news_labeling/
│       ├── 3-2_stacked_typo_detection/
│       ├── 4-1_key_passage_retrieval/
│       └── 4-2_table_querying/
├── vllm_service/                 # 预处理后的服务压测数据集（不上传 GitHub）
│   ├── LongBenchV2.jsonl         # 完整版（444 MB）
│   ├── LongBenchV2_8192.jsonl    # 8K tokens
│   ├── LongBenchV2_16384.jsonl   # 16K
│   ├── LongBenchV2_32768.jsonl   # 32K
│   ├── LongBenchV2_65536.jsonl   # 64K
│   └── LongBenchV2_131072.jsonl  # 128K
└── prepare_datasets.py           # 预处理脚本：读 raw/ → 写入 vllm_service/
```

## 用途

- **`data/raw/`** — 原始评测数据，用于验证 LoPT 并行 tokenize 结果是否与串行 HF tokenizer 一致
- **`data/vllm_service/`** — 对原始 LongBenchV2 做截断/填充预处理后的多长度版本，用于 vLLM 在线服务压测

## 数据准备

如果从 GitHub clone 后需要生成数据：

### 1. 准备原始数据集

将 LongBenchV2 / LEval / CLongEval 数据集放在 `data/raw/` 对应目录下。

### 2. 预处理服务压测数据

```bash
python data/prepare_datasets.py
```

这会读取 `data/raw/LongBenchV2/data.json`，生成多个长度版本的 JSONL 到 `data/vllm_service/`。
