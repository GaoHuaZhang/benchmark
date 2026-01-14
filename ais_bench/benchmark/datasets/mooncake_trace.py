import json
import os
import random
import hashlib
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
from datasets import Dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.utils.file.load_tokenizer import load_tokenizer
from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator

# ============================================================================
# 1. RNG 管理系统
# ============================================================================

class RNGManager:
    """随机数生成器管理器"""

    def __init__(self, root_seed: int | None):
        self._root_seed = root_seed
        if root_seed is not None:
            # 设置全局随机种子（防御性措施）
            random.seed(root_seed)
            np_seed = (root_seed ^ (root_seed >> 32)) & 0xFFFFFFFF
            np.random.seed(np_seed)

    def derive(self, identifier: str) -> random.Random:
        """从标识符派生子 RNG"""
        if self._root_seed is not None:
            # 确定性派生：使用 SHA-256 哈希
            seed_string = f"{self._root_seed}:{identifier}"
            hash_bytes = hashlib.sha256(seed_string.encode("utf-8")).digest()
            child_seed = int.from_bytes(hash_bytes[:8], byteorder="big")
            return random.Random(child_seed)
        else:
            # 非确定性：使用系统随机数
            return random.Random()


_rng_manager: RNGManager | None = None


def init_rng(seed: int | None):
    """初始化全局 RNG 管理器"""
    global _rng_manager
    _rng_manager = RNGManager(seed)


def derive_rng(identifier: str) -> random.Random:
    """派生子 RNG"""
    if _rng_manager is None:
        raise RuntimeError("RNG manager not initialized. Call init_rng() first.")
    return _rng_manager.derive(identifier)


# ============================================================================
# 2. 语料库加载
# ============================================================================

DEFAULT_CORPUS_FILE = "assets/shakespeare.txt"
MAX_CHARS_PER_CHUNK = 10_000


def initialize_corpus(tokenizer, corpus_path: Path) -> list[int]:
    """
    加载并 tokenize 语料库

    使用基于字符数的分块策略，确保在不同机器上产生相同的分块边界。
    """
    with open(corpus_path, encoding="utf-8") as f:
        lines = f.readlines()

    # 预处理：过滤空行
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    def tokenize_chunk(chunk: list[str]) -> list[int]:
        """Tokenize 一个文本块"""
        text = " ".join(chunk)
        tokens = tokenizer.encode(text, add_special_tokens=False)  # 返回 token ID 列表
        return tokens

    # 基于字符数的分块（确定性分块）
    chunks = []
    buffer = []
    char_count = 0

    for line in non_empty_lines:
        buffer.append(line)
        char_count += len(line)

        if char_count >= MAX_CHARS_PER_CHUNK:
            chunks.append(buffer)
            buffer = []
            char_count = 0

    # 添加剩余行作为最后一个块
    if buffer:
        chunks.append(buffer)

    # 多线程 tokenize（线程数不影响可复现性，因为分块是确定性的）
    num_threads = min(os.cpu_count() or 4, 8)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        tokenized_chunks = list(executor.map(tokenize_chunk, chunks))

    # 展平所有 token
    tokenized_corpus = [
        token for chunk in tokenized_chunks for token in chunk
    ]

    return tokenized_corpus


# ============================================================================
# 3. PromptGenerator
# ============================================================================

class PromptGenerator:
    """Prompt 生成器"""

    def __init__(self, tokenizer, tokenized_corpus: list[int], root_seed: int | None = None, block_size: int = 512):
        self.tokenizer = tokenizer
        self._tokenized_corpus = tokenized_corpus
        self._corpus_size = len(tokenized_corpus)

        # 初始化 RNG（用于语料库采样）
        self._corpus_rng = derive_rng("dataset.prompt.corpus")

        # Hash ID 缓存：hash_id -> token 列表
        self._cache: dict[int, list[int]] = {}

        # 块大小（默认 512 tokens）
        self.block_size = block_size

    def generate(
        self,
        mean: int | None = None,
        stddev: int | None = None,
        hash_ids: list[int] | None = None,
    ) -> str:
        """
        生成 prompt 的主入口

        Args:
            mean: 目标 token 数量（如果使用 hash_ids，这是总 token 数）
            stddev: 标准差（对于 hash_ids 模式，通常为 0）
            hash_ids: Hash ID 列表，用于缓存复用

        Returns:
            生成的 prompt 文本
        """
        if hash_ids:
            return self._generate_cached_prompt(
                mean, hash_ids, self.block_size
            )

        # 无 hash_ids：使用正态分布采样 token 数量
        num_tokens = self._sample_num_tokens(mean, stddev)
        return self.generate_prompt(num_tokens)

    def generate_prompt(self, num_tokens: int) -> str:
        """生成指定 token 数量的 prompt"""
        tokens = self._sample_tokens(num_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _sample_tokens(self, num_tokens: int) -> list[int]:
        """
        从语料库采样指定数量的 tokens

        使用循环采样：如果超出语料库末尾，从开头继续。
        """
        if num_tokens > self._corpus_size:
            # 如果请求的 token 数超过语料库大小，返回整个语料库
            return self._tokenized_corpus.copy()

        # 随机选择起始位置
        start_idx = self._corpus_rng.randrange(self._corpus_size)

        end_idx = start_idx + num_tokens
        prompt_tokens = self._tokenized_corpus[start_idx:end_idx]

        # 如果超出语料库末尾，从开头继续
        if end_idx > self._corpus_size:
            prompt_tokens += self._tokenized_corpus[: end_idx - self._corpus_size]

        return prompt_tokens

    def _sample_num_tokens(self, mean: int | None, stddev: int | None) -> int:
        """从正态分布采样 token 数量"""
        if mean is None:
            raise ValueError("mean must be provided")

        if stddev is None or stddev == 0:
            return mean

        # 使用正态分布采样（确保返回正整数）
        length_rng = derive_rng("dataset.prompt.length")
        while True:
            value = int(length_rng.gauss(mean, stddev))
            if value > 0:
                return value

    def _generate_cached_prompt(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> str:
        """
        基于 hash_ids 生成 prompt（使用缓存机制）

        每个 hash_id 对应一个 token 块。如果 hash_id 在缓存中，复用缓存的 tokens；
        否则生成新的 tokens 并缓存。

        Args:
            num_tokens: 总 token 数量
            hash_ids: Hash ID 列表
            block_size: 每个 hash 块的 token 数量（默认 512）

        Returns:
            生成的 prompt 文本
        """
        final_prompt: list[int] = []
        current_block_size = block_size

        # 计算最后一个块的大小
        final_block_size = num_tokens - ((len(hash_ids) - 1) * block_size)

        # 验证参数
        if final_block_size <= 0 or block_size < final_block_size:
            raise ValueError(
                f"Input length: {num_tokens}, Hash IDs: {hash_ids}, Block size: {block_size} "
                f"are not compatible. Final block size: {final_block_size} must be > 0 and <= {block_size}."
            )

        # 处理每个 hash_id
        for index, hash_id in enumerate(hash_ids):
            # 最后一个 hash_id 使用剩余 tokens
            if index == len(hash_ids) - 1:
                current_block_size = final_block_size

            # 如果 hash_id 不在缓存中，生成并缓存
            if hash_id not in self._cache:
                prompt_tokens: list[int] = []

                # 如果 tokenizer 支持块分隔符，插入 BOS/EOS token
                # 这确保不同块之间不会合并
                block_separation_token_id = getattr(
                    self.tokenizer, 'block_separation_token_id', None
                )

                if block_separation_token_id is not None:
                    prompt_tokens.append(block_separation_token_id)
                    prompt_tokens += self._sample_tokens(current_block_size - 1)
                else:
                    prompt_tokens += self._sample_tokens(current_block_size)

                # 缓存 token 列表
                self._cache[hash_id] = prompt_tokens

            # 复用缓存的 tokens
            final_prompt.extend(self._cache[hash_id])

        # 解码为文本（不跳过特殊 tokens，保留块分隔符）
        return self.tokenizer.decode(final_prompt, skip_special_tokens=False)


# ============================================================================
# 4. Mooncake Trace 数据模型
# ============================================================================

class MooncakeTrace:
    """Mooncake trace 数据模型"""

    def __init__(self, data: dict[str, Any]):
        # 验证：input_length 必须存在
        if "input_length" not in data or data["input_length"] is None:
            raise ValueError("'input_length' must be provided")

        self.input_length = data["input_length"]
        self.output_length = data.get("output_length")
        self.hash_ids = data.get("hash_ids")
        self.timestamp = data.get("timestamp")


def load_mooncake_trace(filename: str) -> list[MooncakeTrace]:
    """
    从 JSONL 文件加载 Mooncake trace 数据

    Returns:
        trace数据列表
    """
    traces = []

    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            traces.append(MooncakeTrace(json.loads(line)))

    return traces


# ============================================================================
# 5. MooncakeTraceDataset
# ============================================================================

@LOAD_DATASET.register_module()
class MooncakeTraceDataset(BaseDataset):
    def load(self, path, model_path, random_seed=None, generated_prompts_path=""):
        """
        加载 Mooncake trace 数据集

        Args:
            path: 原始包含hashid和trace数据的JSONL文件路径
            model_path: 模型路径，用于加载tokenizer
            random_seed: 随机数种子
            generated_prompts_path: 生成prompt数据的文件路径，如果存在则复用

        Returns:
            Dataset: 包含prompt、timestamp、max_out_len三个字段的数据集
        """
        path = get_data_path(path)
        self.logger.info(f"Loading mooncake trace dataset from: {path}")

        # 处理generated_prompts_path
        if not generated_prompts_path:
            # 生成默认缓存文件路径：在原文件名后加上_generated_prompts
            dir_name = os.path.dirname(path)
            base_name = os.path.basename(path)
            name_without_ext, ext = os.path.splitext(base_name)
            generated_prompts_path = os.path.join(
                dir_name, f"{name_without_ext}_generated_prompts{ext}"
            )
        else:
            generated_prompts_path = get_data_path(generated_prompts_path)

        self.logger.info(f"Generated prompts cache path: {generated_prompts_path}")

        # 检查缓存文件是否存在，如果存在则直接加载
        if os.path.exists(generated_prompts_path):
            self.logger.info(f"Found existing generated prompts file, loading from: {generated_prompts_path}")
            dataset_list = []
            with open(generated_prompts_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        dataset_list.append(json.loads(line.strip()))
            self.logger.info(f"Successfully loaded {len(dataset_list)} items from cache file")
            return Dataset.from_list(dataset_list)

        # 如果文件不存在，需要生成dataset
        self.logger.info(f"Cache file not found, generating prompts from source file")

        # 1. 初始化 RNG 系统
        init_rng(random_seed)

        # 2. 加载 tokenizer
        self.logger.info(f"Loading tokenizer from: {model_path}")
        tokenizer = load_tokenizer(model_path)
        self.logger.info(f"Tokenizer loaded successfully, vocab_size: {tokenizer.vocab_size}")

        # 3. 加载并 tokenize 语料库
        # 尝试从多个可能的位置查找语料库文件
        corpus_path = None
        possible_paths = [
            Path(__file__).parent / DEFAULT_CORPUS_FILE,
            Path(__file__).parent.parent / DEFAULT_CORPUS_FILE,
            Path(__file__).parent.parent.parent / DEFAULT_CORPUS_FILE,
        ]

        for p in possible_paths:
            if p.exists():
                corpus_path = p
                break

        if corpus_path is None:
            # 如果找不到，尝试从aiperf复制或使用绝对路径
            # 这里我们使用一个fallback：如果找不到文件，创建一个简单的提示
            raise FileNotFoundError(
                f"Corpus file not found. Please ensure {DEFAULT_CORPUS_FILE} exists in one of: "
                f"{[str(p) for p in possible_paths]}"
            )

        self.logger.info(f"Loading corpus from: {corpus_path}")
        tokenized_corpus = initialize_corpus(tokenizer, corpus_path)
        self.logger.info(f"Corpus loaded successfully, {len(tokenized_corpus)} tokens")

        # 4. 创建 PromptGenerator
        prompt_generator = PromptGenerator(tokenizer, tokenized_corpus, root_seed=random_seed)

        # 5. 加载 Mooncake trace 数据
        trace_data = load_mooncake_trace(path)
        self.logger.info(f"Loaded {len(trace_data)} traces from source file")

        # 6. 转换为 prompts
        prompts = []
        for trace in trace_data:
            # 使用 input_length 生成 prompt
            prompt = prompt_generator.generate(
                mean=trace.input_length,
                stddev=0,  # Mooncake trace 通常不使用标准差
                hash_ids=trace.hash_ids or [],
            )

            item = {
                "prompt": prompt,
                "max_out_len": trace.output_length or 0,
                "answer": "mock_answer",
            }
            # 如果有timestamp，添加到结果中
            if trace.timestamp is not None:
                item["timestamp"] = trace.timestamp
            else:
                item["timestamp"] = 0  # 默认值

            prompts.append(item)
        self.logger.info(f"Generated {len(prompts)} prompts, saving to cache file: {generated_prompts_path}")
        # 保存到缓存文件
        with open(generated_prompts_path, "w", encoding="utf-8") as f:
            for item in prompts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.logger.info(f"Successfully saved generated prompts to: {generated_prompts_path}")

        return Dataset.from_list(prompts)


class MooncakeTraceEvaluator(BaseEvaluator):
    pass
