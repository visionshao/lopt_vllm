try:
    from vllm import lopt_cpp
    from .lopt_utils import chunks, pairs, flatten
    LOPT_AVAILABLE = True
except ImportError:
    LOPT_AVAILABLE = False

import time
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from transformers import BatchEncoding
import numpy as np
from functools import reduce
from typing import Optional
from vllm.logger import init_logger

logger = init_logger(__name__)

class LoptParallelTokenizer:
    """LOPT parallel tokenizer wrapper for vLLM integration.
    
    This class provides parallel tokenization for long texts by:
    1. Splitting text into overlapping chunks
    2. Processing chunks in parallel using multiple processes
    3. Matching and merging overlapping regions using C++ extension
    4. Returning results compatible with HF tokenizer format
    """
    
    def __init__(
        self,
        model_path: str,
        pool_size: int = 8,
        chunk_size: int = 2048,
        overlap_ratio: float = 0.125,  # 1/8 overlap
    ):
        self.model_path = model_path
        self.pool_size = pool_size
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)
        
        # Initialize main tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=True
        )
        
        # Initialize multiprocessing pool
        self.pool = mp.Pool(
            pool_size, 
            initializer=self._init_worker, 
            initargs=(model_path,)
        )
        print("The pool initialized.")
    
    @staticmethod
    def _init_worker(model_path: str):
        """Initialize worker process with tokenizer."""
        global _worker_tokenizer
        _worker_tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=True
        )
    
    @staticmethod
    def _tokenize_chunk(text: str):
        """Tokenize a single chunk (called by worker processes)."""
        global _worker_tokenizer
        return _worker_tokenizer(
            text, 
            return_tensors="np", 
            return_offsets_mapping=True, 
            add_special_tokens=False
        )
    
    def _cpp_match_wrapper(
        self, 
        tokens_a: np.ndarray, 
        tokens_b: np.ndarray, 
        chunk_size: int
    ) -> tuple[int, int]:
        """C++ match function wrapper with safety checks."""
        # Ensure contiguous memory and correct dtype
        a = np.ascontiguousarray(tokens_a, dtype=np.int64)
        b = np.ascontiguousarray(tokens_b, dtype=np.int64)
        
        # Call C++ match function (mode=2 for suffix matching)
        res = lopt_cpp.match(a, b, chunk_size, 2)
        
        if res[0] < 0:
            raise RuntimeError(f"C++ match returned error: {res}")
        
        return res
    
    def __call__(
        self, 
        text: str, 
        add_special_tokens: bool = False
    ) -> "BatchEncoding":
        """Main entry point: parallel encode text.
        
        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        # For short texts, use standard tokenizer
        t0 = time.perf_counter()                                                                                                                                                
        if len(text) < self.chunk_size * 2:                                                                                                                                     
            result = self.tokenizer(text, add_special_tokens)                                                                                                                                 
            print(f"Short path: {(time.perf_counter()-t0)*1000:.1f}ms")                                                                                                         
            return result                                                                                                                                                       
        else:                                                                                                                                                                   
            result = self._parallel_encode(text)                                                                                                                                
            print(f"Parallel path: {(time.perf_counter()-t0)*1000:.1f}ms")                                                                                                      
            return result 

    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self.__call__(text, add_special_tokens).input_ids
        
        
    def _parallel_encode(self, text: str):
        """Internal parallel encoding implementation."""
        # Step 1: Split text into overlapping chunks
        text_chunks = list(chunks(text, self.chunk_size, self.overlap))
        
        # Step 2: Parallel tokenization using process pool
        shards = self.pool.map(self._tokenize_chunk, text_chunks)
        
        # Single chunk: return directly
        if len(shards) == 1:
            result = shards[0]
            result.pop("offset_mapping", None)
            for k in result:
                result[k] = result[k][np.newaxis, :]
            return result
        
        # Step 3: Extract token sequences for matching
        # Use offset_mapping start positions for matching
        tokens_shards = [
                flatten(shard["offset_mapping"])[::2] for shard in shards
            ]
        
        # Step 4: Match overlapping regions between adjacent chunks
        try:
            matches = [self._cpp_match_wrapper(_[0], _[1], self.chunk_size) for _ in pairs(tokens_shards)]
            # 构造合并用的 matches 列表（与原始逻辑一致）
            matches = [len(tokens_shards[0])] + list(reduce(lambda x, y: x + y, matches)) + [0]
        except RuntimeError as e:
            # Fall back to standard tokenizer on match failure
            logger.warning("Fall back to standard tokenizer on match failure")
            return self.tokenizer(text, return_tensors="np")
        
        # Step 5: Merge shards using C++ extension
        merged = shards[0].__class__()
        merged.pop("offset_mapping", None)
        
        for key in shards[0].keys():
            if key != "offset_mapping":
                merged[key] = lopt_cpp.merge(
                    [flatten(shard[key]) for shard in shards], matches
                )[np.newaxis, :].tolist()
        
        return merged


def maybe_get_lopt_tokenizer(
    model_path: str,
    enable_lopt: bool = False,
    lopt_pool_size: int = 8,
    lopt_chunk_size: int = 2048,
) -> Optional[LoptParallelTokenizer]:
    """Factory function to create LOPT tokenizer if enabled and available.
    
    Args:
        model_path: Path to the model
        enable_lopt: Whether to enable LOPT parallel tokenization
        lopt_pool_size: Number of processes for parallel tokenization
        lopt_chunk_size: Size of each chunk for splitting text
        
    Returns:
        LoptParallelTokenizer instance if enabled and available, None otherwise
    """
    if not enable_lopt:
        return None
    
    if not LOPT_AVAILABLE:
        logger.warning(
            "LOPT was requested but lopt_cpp module is not available. "
            "Falling back to standard tokenization."
        )
        return None
    
    try:
        return LoptParallelTokenizer(
            model_path=model_path,
            pool_size=lopt_pool_size,
            chunk_size=lopt_chunk_size,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize LOPT tokenizer: {e}")
        return None