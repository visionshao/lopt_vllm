"""
LoPT (Lossless Parallel Tokenizer) — Benchmark Script.

Compares parallel tokenization (LoPT) against HuggingFace tokenizer
on long-text datasets, measuring performance and verifying correctness.
"""
import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from contextlib import contextmanager
from functools import reduce

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, BatchEncoding
from tqdm import tqdm

from utils import chunks, flatten, pairs, get_data

try:
    from vllm import lopt_cpp
    CPP_AVAILABLE = True
except ImportError:
    # Try to auto-build lopt_cpp
    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
    build_script = os.path.join(csrc_dir, "build.sh")
    if os.path.exists(build_script):
        import subprocess
        print("lopt_cpp not found. Attempting auto-build...")
        ret = subprocess.call(["bash", build_script], cwd=csrc_dir)
        if ret == 0:
            try:
                from vllm import lopt_cpp
                CPP_AVAILABLE = True
                print("lopt_cpp auto-build succeeded.")
            except ImportError:
                CPP_AVAILABLE = False
        else:
            CPP_AVAILABLE = False
    else:
        CPP_AVAILABLE = False
    if not CPP_AVAILABLE:
        print("Warning: lopt_cpp module not found. Parallel matching will fail.")

os.environ["TOKENIZATION_PARALLELISM"] = "false"


@contextmanager
def timer(name: str = "", verbose: bool = False):
    """Context manager: measure code block execution time."""
    start = time.perf_counter()
    result = {}
    yield result
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"[Timer] {name}: {elapsed:.4f}s")
    result['elapsed'] = elapsed


@dataclass
class TokenizationMetrics:
    """Performance metrics for a single tokenization run."""
    hf_time: float = 0.0
    lopt_time: float = 0.0
    chunk_num: int = 0
    pt_time: float = 0.0
    mp_time: float = 0.0
    match_time: float = 0.0
    merge_time: float = 0.0
    convert_time: float = 0.0
    match_attempts: int = 0
    equal: int = 0

    def to_log_line(self) -> str:
        return (f"{self.hf_time:.6f} {self.lopt_time:.6f} {self.chunk_num} "
                f"{self.pt_time:.6f} {self.mp_time:.6f} {self.match_time:.6f} "
                f"{self.merge_time:.6f} {self.convert_time:.6f} {self.match_attempts} {self.equal}\n")


class ParallelTokenizer:
    """Parallel long-text tokenizer for benchmarking."""

    def __init__(self, model_path: str, pool_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ match/merge extension is required but not found.")
        self.pool_size = pool_size
        self.pool = mp.Pool(pool_size, initializer=self._init_worker, initargs=(model_path,))

    @staticmethod
    def _init_worker(model_path):
        global _tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    @staticmethod
    def _tokenize_chunk(text):
        return _tokenizer(text, return_tensors="np", return_offsets_mapping=True, add_special_tokens=False)


    def _cpp_match_wrapper(self, tokens_a: np.ndarray, tokens_b: np.ndarray,
                           chunk_size: int) -> Tuple[int, int]:
        """Safe wrapper around C++ match function. Returns (match_position, error_code)."""
        a = np.ascontiguousarray(tokens_a, dtype=np.int64)
        b = np.ascontiguousarray(tokens_b, dtype=np.int64)
        res = lopt_cpp.match(a, b, chunk_size, 2)  # mode=2 for suffix matching
        if res[0] < 0:
            raise RuntimeError("C++ match returned error")
        return res

    def check_overlap(self, prompt: str, overlap: int, chunk_size: int) -> Union[
        Tuple[BatchEncoding, Dict[str, float]], str
    ]:
        """
        Chunk, parallel-tokenize, match, and merge overlapping regions.

        Returns (merged_encoding, timings_dict) on success,
        or "match_failed" on failure.
        """
        timings = {}
        with timer() as t_total:
            with timer() as t_chunk:
                text_chunks = list(chunks(prompt, chunk_size, overlap))
            timings["chunk"] = t_chunk["elapsed"]

            with timer() as t_mp:
                shards = self.pool.map(self._tokenize_chunk, text_chunks)
            timings["mp"] = t_mp["elapsed"]

            if len(shards) == 1:
                result = shards[0]
                result.pop("offset_mapping", None)
                for k in result:
                    result[k] = torch.tensor(result[k]).unsqueeze(0)
                timings["total"] = t_total["elapsed"]
                return result, timings

            with timer() as t_slice:
                tokens_shards = [
                    flatten(shard["offset_mapping"])[::2] for shard in shards
                ]
            timings["slice"] = t_slice["elapsed"]

            with timer() as t_match:
                try:
                    matches = [self._cpp_match_wrapper(_[0], _[1], chunk_size) for _ in pairs(tokens_shards)]
                except Exception as e:
                    print(e)
                    return "match failed"
                matches = [len(tokens_shards[0])] + list(reduce(lambda x, y: x + y, matches)) + [0]
            timings["match"] = t_match["elapsed"]

            with timer() as t_merge:
                merged = shards[0].__class__()
                merged.pop("offset_mapping", None)
                for key in shards[0].keys():
                    if key != "offset_mapping":
                        merged[key] = lopt_cpp.merge(
                            [flatten(shard[key]) for shard in shards], matches
                        )[np.newaxis, :]
            timings["merge"] = t_merge["elapsed"]

            with timer() as t_convert:
                for k in merged:
                    merged[k] = torch.tensor(merged[k])
            timings["convert"] = t_convert["elapsed"]

        timings["total"] = t_total["elapsed"]
        return merged, timings

    def query_llm(self, prompt: str, args) -> TokenizationMetrics:
        """Run tokenization on a single prompt and collect performance metrics."""
        metrics = TokenizationMetrics()
        prompt_len = len(prompt)

        # Determine initial chunk size
        if args.chunk_size < 0:
            chunk_size = max(prompt_len // args.n_proc, 2048)
        else:
            chunk_size = min(args.chunk_size, prompt_len)
        overlap = chunk_size // 8

        # Truncate or pad to target sequence length
        processed_prompt = prompt
        if args.seq_len > 0:
            max_seq_len = args.seq_len
            input_ids = self.tokenizer.encode(prompt)
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len // 2] + input_ids[-max_seq_len // 2:]
                processed_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            else:
                repeat_times = max_seq_len // len(input_ids) + 1
                processed_prompt = (prompt + " ") * repeat_times

        # HuggingFace baseline
        with timer() as t_hf:
            hf_encoding = self.tokenizer(processed_prompt, return_tensors="pt", add_special_tokens=False)
        metrics.hf_time = t_hf["elapsed"]

        # Parallel tokenization (may retry with larger chunk size)
        attempt = 0
        lopt_start = time.time()
        result_encoding = None
        timings = {}

        while chunk_size <= len(processed_prompt):
            attempt += 1
            res, tinfo = self.check_overlap(processed_prompt, overlap, chunk_size)
            if res != "match_failed":
                result_encoding = res
                timings = tinfo
                break
            else:
                chunk_size = min(chunk_size * 2, len(processed_prompt))

        if result_encoding is None:
            result_encoding = hf_encoding
            metrics.equal = 0
            timings = {}
        else:
            metrics.lopt_time = time.time() - lopt_start
            metrics.chunk_num = len(result_encoding["input_ids"]) // chunk_size + 1
            metrics.mp_time = timings.get("mp", 0.0)
            metrics.match_time = timings.get("match", 0.0)
            metrics.merge_time = timings.get("merge", 0.0)
            metrics.convert_time = timings.get("convert", 0.0)
            metrics.pt_time = timings.get("total", 0.0)
            metrics.match_attempts = attempt
            try:
                metrics.equal = int(
                    hf_encoding["input_ids"].equal(result_encoding["input_ids"].long())
                )
            except Exception:
                metrics.equal = 0

        return metrics


def profile_dataset(data: List[str], fout, args):
    """Profile a dataset and write per-sample metrics to output file."""
    model_path = f"../models/{args.model}"
    parallel_tokenizer = ParallelTokenizer(model_path, pool_size=args.n_proc)

    total_hf = 0.0
    total_lopt = 0.0
    total_acc = 0
    count = 0

    parallel_tokenizer.query_llm(data[0], args)

    for item in tqdm(data, desc="Profiling"):
        try:
            metrics = parallel_tokenizer.query_llm(item, args)
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
        total_hf += metrics.hf_time
        total_lopt += metrics.lopt_time
        total_acc += metrics.equal
        count += 1

        fout.write(metrics.to_log_line())
        fout.flush()

    if count > 0:
        avg_hf = total_hf / count * 1000
        avg_lopt = total_lopt / count * 1000
        avg_acc = total_acc / count
        print(f"Average HF time: {avg_hf:.4f}ms, LOPT time: {avg_lopt:.4f}ms, Accuracy: {avg_acc:.2%}")
    else:
        print("No valid samples processed.")


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    datasets = args.dataset.split(",")
    prefix = f"nproc={args.n_proc}_seqlen={args.seq_len}_chunksize={args.chunk_size}_model={args.model}"

    for ds in datasets:
        data = get_data(ds)
        out_file = os.path.join(args.save_dir, f"{prefix}_dataset={ds}.txt")
        with open(out_file, "w", encoding="utf-8") as fout:
            fout.write("hf_time lopt_time chunk_num pt_time mp_time match_time merge_time convert_time match_attempts equal\n")
            profile_dataset(data, fout, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel long-text tokenization benchmark")
    parser.add_argument("--save_dir", "-s", type=str, default="./results")
    parser.add_argument("--model", "-m", type=str, default="qwen3-8b")
    parser.add_argument("--dataset", type=str, default="LongBenchV2")
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="Chunk size for splitting; -1 for auto")
    parser.add_argument("--seq_len", type=int, default=-1,
                        help="Target sequence length; -1 to skip truncation/padding")
    parser.add_argument("--n_proc", "-n", type=int, default=16,
                        help="Number of parallel processes")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    main(args)