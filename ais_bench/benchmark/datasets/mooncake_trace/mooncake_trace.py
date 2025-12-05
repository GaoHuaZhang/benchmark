import json
import random

from datasets import Dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.utils.file.load_tokenizer import load_tokenizer
from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator

@LOAD_DATASET.register_module()
class MooncakeTraceDataset(BaseDataset):

    def prompts_generator(self, tokenizer, token_length, hash_ids, block_size=512, hash_id_cache=None):
        """Generate a prompt based on hash_ids and token_length.

        The generated prompt will have exactly token_length tokens when encoded by the tokenizer.

        Args:
            tokenizer: The tokenizer to use for encoding/decoding
            token_length: Total number of tokens required
            hash_ids: List of hash IDs, each representing a block of tokens
            block_size: Number of tokens per hash block (default 512)
            hash_id_cache: Dictionary to cache prompts by hash_id

        Returns:
            str: Generated prompt text that encodes to exactly token_length tokens
        """
        if hash_id_cache is None:
            hash_id_cache = {}

        if not hash_ids:
            # If no hash_ids, generate a simple prompt with token_length tokens
            # Use iterative approach to ensure exact token count
            return self._generate_prompt_with_exact_length(tokenizer, token_length)

        final_prompt_tokens = []
        current_block_size = block_size

        # Calculate the final block size
        final_block_size = token_length - ((len(hash_ids) - 1) * block_size)
        if final_block_size <= 0 or block_size < final_block_size:
            raise ValueError(
                f"Input length: {token_length}, Hash IDs: {hash_ids}, Block size: {block_size} "
                f"are not compatible. The final hash block size: {final_block_size} must be "
                f"greater than 0 and less than or equal to {block_size}."
            )

        # First, generate all token IDs for each hash_id block
        all_token_ids = []
        for index, hash_id in enumerate(hash_ids):
            # For the last hash ID, use the remaining tokens as the block size
            if index == len(hash_ids) - 1:
                current_block_size = final_block_size

            if hash_id not in hash_id_cache:
                # Generate new tokens for this hash_id
                # Use a deterministic seed based on hash_id for reproducibility
                random.seed(hash_id)
                tokens = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(current_block_size)]
                hash_id_cache[hash_id] = tokens
                random.seed()  # Reset seed

            all_token_ids.extend(hash_id_cache[hash_id])

        # Decode all tokens to text
        combined_prompt = tokenizer.decode(all_token_ids, skip_special_tokens=True)

        # Adjust to ensure exact token length (accounting for potential token merging at boundaries)
        return self._adjust_prompt_length(tokenizer, combined_prompt, token_length)

    def _generate_prompt_with_exact_length(self, tokenizer, token_length):
        """Generate a prompt that encodes to exactly token_length tokens."""
        # Start with a simple pattern
        base_text = "A " * max(1, token_length // 2)
        return self._adjust_prompt_length(tokenizer, base_text, token_length)

    def _adjust_prompt_length(self, tokenizer, prompt, target_token_length):
        """Adjust prompt text to ensure it encodes to exactly target_token_length tokens.

        Args:
            tokenizer: The tokenizer to use
            prompt: Initial prompt text
            target_token_length: Desired number of tokens

        Returns:
            str: Adjusted prompt text that encodes to exactly target_token_length tokens
        """
        # Encode to check current token count
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        current_length = len(encoded)

        if current_length == target_token_length:
            return prompt

        # If we need more tokens, append characters
        if current_length < target_token_length:
            # Find a character that adds tokens when appended
            padding_chars = [" ", "A", "B", "C", "D", "E"]
            padding_char = " "

            for char in padding_chars:
                test_text = prompt + char
                test_encoded = tokenizer.encode(test_text, add_special_tokens=False)
                if len(test_encoded) > current_length:
                    padding_char = char
                    break

            # Add padding until we reach target length
            max_iterations = target_token_length * 2  # Safety limit
            iteration = 0
            while current_length < target_token_length and iteration < max_iterations:
                prompt += padding_char
                encoded = tokenizer.encode(prompt, add_special_tokens=False)
                new_length = len(encoded)

                # If adding a character doesn't increase token count, try a different approach
                if new_length == current_length:
                    # Try adding a word instead
                    prompt += " word"
                    encoded = tokenizer.encode(prompt, add_special_tokens=False)
                    new_length = len(encoded)

                current_length = new_length
                iteration += 1

                if current_length >= target_token_length:
                    break

        # If we have too many tokens, use binary search to find the right length
        if current_length > target_token_length:
            # Binary search for the right text length
            left, right = 0, len(prompt)
            best_prompt = prompt
            best_length_diff = abs(current_length - target_token_length)

            # Limit binary search iterations
            max_binary_iterations = 50
            binary_iteration = 0

            while left < right and binary_iteration < max_binary_iterations:
                mid = (left + right) // 2
                if mid == 0:
                    break

                test_prompt = prompt[:mid]
                test_encoded = tokenizer.encode(test_prompt, add_special_tokens=False)
                test_length = len(test_encoded)

                if test_length == target_token_length:
                    return test_prompt
                elif test_length < target_token_length:
                    length_diff = target_token_length - test_length
                    if length_diff < best_length_diff:
                        best_prompt = test_prompt
                        best_length_diff = length_diff
                    left = mid + 1
                else:
                    right = mid

                binary_iteration += 1

            # Fine-tune from the best result
            prompt = best_prompt
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            current_length = len(encoded)

            # Fine-tune by adding/removing characters one by one
            max_fine_tune = 100  # Safety limit
            fine_tune_iter = 0

            while current_length != target_token_length and fine_tune_iter < max_fine_tune:
                if current_length < target_token_length:
                    # Try adding different characters
                    for char in [" ", "A", "B", "C"]:
                        test_prompt = prompt + char
                        test_encoded = tokenizer.encode(test_prompt, add_special_tokens=False)
                        if len(test_encoded) == target_token_length:
                            return test_prompt
                        elif len(test_encoded) > target_token_length:
                            break
                    # If no single char works, add a space and continue
                    prompt += " "
                else:
                    # Remove one character from end
                    if len(prompt) > 0:
                        prompt = prompt[:-1]
                    else:
                        break

                encoded = tokenizer.encode(prompt, add_special_tokens=False)
                current_length = len(encoded)
                fine_tune_iter += 1

        return prompt

    def load(self, path, model_path, prefix_ratio=None):
        hash_id_cache = dict()
        tokenizer = load_tokenizer(model_path)
        path = get_data_path(path)
        json_data = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                json_data.append(data)
        json_data.sort(key=lambda x: x.get('timestamp', 0))
        dataset = []
        for data in json_data:
            input_length = data.get('input_length', 0)
            output_length = data.get('output_length', 0)
            hash_ids = data.get('hash_ids', [])

            # Generate prompt using hash_ids
            prompt = self.prompts_generator(
                tokenizer=tokenizer,
                token_length=input_length,
                hash_ids=hash_ids,
                block_size=512,
                hash_id_cache=hash_id_cache
            )

            dataset.append({
                "question": prompt,
                "answer": "",  # Empty answer as placeholder
                "max_out_len": output_length
            })

        return Dataset.from_list(dataset)





class MooncakeTraceEvaluator(BaseEvaluator):
    pass