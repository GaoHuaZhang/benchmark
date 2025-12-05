from typing import Dict, List, Any, Tuple, Set, Iterable
import json
import os
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer

from datasets import Dataset
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.datasets.base import BaseDataset


@dataclass
class NumberRange:
    lower: tuple[int, float] = None
    upper: tuple[int, float] = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True


def _check_keys_equal(got_keys, true_keys, comment):
    for key in got_keys:
        if key not in true_keys:
            raise ValueError(f"{key} is not a valid key for {comment}.")
    if got_keys != true_keys:
        raise ValueError(f"Expect keys {true_keys} for {comment}, but got keys {set(got_keys)}.")

def _ensure_keys_present(check_keys: Iterable, required_keys:Set, comment:str):
    if not required_keys.issubset(set(check_keys)):
            raise ValueError(f"Missing required key(s): {{{required_keys - set(check_keys)}}} for {comment}.")

def check_type(name: str, value: Any, types: Tuple):
    if not isinstance(value, types):
        raise ValueError(f"Parameter {name} should have type {types} for SyntheticConfig,",
                         f"but got {type(value).__name__}.")

def check_range(name: str, value: Any, param: NumberRange):
    lower, upper = param.lower, param.upper
    lower_inclusive, upper_inclusive = param.lower_inclusive, param.upper_inclusive
    # 构造区间的字符串表示
    lower_bound = '[' if lower_inclusive else '('
    upper_bound = ']' if upper_inclusive else ')'
    lower_str = str(lower) if lower is not None else '-inf'
    upper_str = str(upper) if upper is not None else '+inf'

    # 构造完整区间表示字符串
    interval_str = f"{lower_bound}{lower_str}, {upper_str}{upper_bound}"

    # 检查下限
    if lower is not None:
        lt = (lower_inclusive and value < lower)
        le = (not lower_inclusive and value <= lower)
        if le or lt:
            raise ValueError(f"Parameter {name} is {value}, not within the required range {interval_str}")
    # 检查上限
    if upper is not None:
        gt = (upper_inclusive and value > upper)
        ge = (not upper_inclusive and value >= upper)
        if gt or ge:
            raise ValueError(f"Parameter {name} is {value}, not within the required range {interval_str}")

def normalize_file_path(file_path:str) -> str:
    return os.path.abspath(os.path.expanduser(file_path))

@LOAD_DATASET.register_module()
class SyntheticDataset(BaseDataset):

    @staticmethod
    def check_synthetic_string_config(synthetic_config: Dict):
        input_str = "Input"
        output_str = "Output"
        _ensure_keys_present(synthetic_config.keys(), {input_str, output_str}, "SyntheticConfig")

        for key in (input_str, output_str):
            conf = synthetic_config.get(key)
            _check_keys_equal(conf.keys(), {"Method", "Params"}, 'SyntheticConfig["{key}"]')
            method = conf.get("Method")
            params = conf.get("Params")
            uniform_str = "uniform"
            gaussian_str = "gaussian"
            zipf_str = "zipf"
            min_value_str = "MinValue"
            max_value_str = "MaxValue"
            mean_str = "Mean"
            var_str = "Var"
            alpha_str = "Alpha"

            if method == uniform_str:
                _check_keys_equal(params.keys(), {min_value_str, max_value_str}, uniform_str)
            elif method == gaussian_str:
                _check_keys_equal(params.keys(), {mean_str, var_str, min_value_str, max_value_str}, gaussian_str)
            elif method == zipf_str:
                _check_keys_equal(params.keys(), {alpha_str, min_value_str, max_value_str}, zipf_str)
            else:
                raise ValueError(f'Method should be one of {{{uniform_str, gaussian_str, zipf_str}}}, '
                                 f'but got {method}.')
            min_float32_value = -3.0e38
            max_float32_value = 3.0e38
            for param_name, param_value in params.items():
                desc_name = key + " " + param_name
                if param_name in (min_value_str, max_value_str):
                    check_type(desc_name, param_value, types=(int,))
                    if key == input_str:
                        check_range(desc_name, param_value, NumberRange(1, 2**20))    # 2**20 = 1M
                    elif key == output_str:
                        check_range(desc_name, param_value, NumberRange(1, 2**20))
                elif param_name == mean_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(min_float32_value, max_float32_value))
                elif param_name == var_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(0, max_float32_value))
                elif param_name == alpha_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(1.0, 10.0, lower_inclusive=False))
            min_value = params.get(min_value_str)
            max_value = params.get(max_value_str)
            if min_value > max_value:
                raise ValueError(f'MinValue should less than MaxValue, '
                                 f'but got MinValue is {min_value}, and MaxValue is {max_value}.')

    @staticmethod
    def check_synthetic_tokenid_config(synthetic_config: Dict):
        """Check tokenid config, supporting both RequestSize and Input/Output distribution configs."""
        input_str = "Input"
        output_str = "Output"
        request_size_key = "RequestSize"
        prefix_len_key = "PrefixLen"

        # Check if using distribution config (Input/Output) or fixed RequestSize
        has_distribution = input_str in synthetic_config or output_str in synthetic_config
        has_request_size = request_size_key in synthetic_config

        if not has_distribution and not has_request_size:
            raise ValueError(f"TokenIdConfig must have either {request_size_key} or {input_str}/{output_str} configuration.")

        # Validate RequestSize if present
        if has_request_size:
            request_size_value = synthetic_config.get(request_size_key)
            check_type(request_size_key, request_size_value, types=(int, ))
            check_range(request_size_key, request_size_value, NumberRange(1, 2**20))

        # Validate Input/Output distribution configs if present
        if has_distribution:
            if input_str in synthetic_config:
                input_conf = synthetic_config.get(input_str)
                _check_keys_equal(input_conf.keys(), {"Method", "Params"}, f'TokenIdConfig["{input_str}"]')
                method = input_conf.get("Method")
                params = input_conf.get("Params")
                uniform_str = "uniform"
                gaussian_str = "gaussian"
                zipf_str = "zipf"
                min_value_str = "MinValue"
                max_value_str = "MaxValue"
                mean_str = "Mean"
                var_str = "Var"
                alpha_str = "Alpha"

                if method == uniform_str:
                    _check_keys_equal(params.keys(), {min_value_str, max_value_str}, uniform_str)
                elif method == gaussian_str:
                    _check_keys_equal(params.keys(), {mean_str, var_str, min_value_str, max_value_str}, gaussian_str)
                elif method == zipf_str:
                    _check_keys_equal(params.keys(), {alpha_str, min_value_str, max_value_str}, zipf_str)
                else:
                    raise ValueError(f'Method should be one of {{{uniform_str, gaussian_str, zipf_str}}}, '
                                     f'but got {method}.')

                for param_name, param_value in params.items():
                    desc_name = input_str + " " + param_name
                    if param_name in (min_value_str, max_value_str):
                        check_type(desc_name, param_value, types=(int,))
                        check_range(desc_name, param_value, NumberRange(1, 2**20))
                    elif param_name == mean_str:
                        check_type(desc_name, param_value, types=(int, float))
                        check_range(desc_name, param_value, NumberRange(-3.0e38, 3.0e38))
                    elif param_name == var_str:
                        check_type(desc_name, param_value, types=(int, float))
                        check_range(desc_name, param_value, NumberRange(0, 3.0e38))
                    elif param_name == alpha_str:
                        check_type(desc_name, param_value, types=(int, float))
                        check_range(desc_name, param_value, NumberRange(1.0, 10.0, lower_inclusive=False))

                min_value = params.get(min_value_str)
                max_value = params.get(max_value_str)
                if min_value > max_value:
                    raise ValueError(f'MinValue should less than MaxValue, '
                                     f'but got MinValue is {min_value}, and MaxValue is {max_value}.')

            if output_str in synthetic_config:
                output_conf = synthetic_config.get(output_str)
                _check_keys_equal(output_conf.keys(), {"Method", "Params"}, f'TokenIdConfig["{output_str}"]')
                method = output_conf.get("Method")
                params = output_conf.get("Params")
                uniform_str = "uniform"
                gaussian_str = "gaussian"
                zipf_str = "zipf"
                min_value_str = "MinValue"
                max_value_str = "MaxValue"
                mean_str = "Mean"
                var_str = "Var"
                alpha_str = "Alpha"

                if method == uniform_str:
                    _check_keys_equal(params.keys(), {min_value_str, max_value_str}, uniform_str)
                elif method == gaussian_str:
                    _check_keys_equal(params.keys(), {mean_str, var_str, min_value_str, max_value_str}, gaussian_str)
                elif method == zipf_str:
                    _check_keys_equal(params.keys(), {alpha_str, min_value_str, max_value_str}, zipf_str)
                else:
                    raise ValueError(f'Method should be one of {{{uniform_str, gaussian_str, zipf_str}}}, '
                                     f'but got {method}.')

                for param_name, param_value in params.items():
                    desc_name = output_str + " " + param_name
                    if param_name in (min_value_str, max_value_str):
                        check_type(desc_name, param_value, types=(int,))
                        check_range(desc_name, param_value, NumberRange(1, 2**20))
                    elif param_name == mean_str:
                        check_type(desc_name, param_value, types=(int, float))
                        check_range(desc_name, param_value, NumberRange(-3.0e38, 3.0e38))
                    elif param_name == var_str:
                        check_type(desc_name, param_value, types=(int, float))
                        check_range(desc_name, param_value, NumberRange(0, 3.0e38))
                    elif param_name == alpha_str:
                        check_type(desc_name, param_value, types=(int, float))
                        check_range(desc_name, param_value, NumberRange(1.0, 10.0, lower_inclusive=False))

                min_value = params.get(min_value_str)
                max_value = params.get(max_value_str)
                if min_value > max_value:
                    raise ValueError(f'MinValue should less than MaxValue, '
                                     f'but got MinValue is {min_value}, and MaxValue is {max_value}.')

        # Validate PrefixLen if present (can be int or float)
        if prefix_len_key in synthetic_config:
            prefix_len_value = synthetic_config.get(prefix_len_key)
            check_type(prefix_len_key, prefix_len_value, types=(int, float))
            if isinstance(prefix_len_value, float):
                check_range(prefix_len_key, prefix_len_value, NumberRange(0.0, 1.0))
            else:
                check_range(prefix_len_key, prefix_len_value, NumberRange(0, 2**20))

    @staticmethod
    def _check_synthetic_config(synthetic_config: Dict):
        config_type_key = "Type"
        request_count_key = "RequestCount"

        required_keys = {config_type_key, request_count_key}
        _ensure_keys_present(synthetic_config.keys(), required_keys, "SyntheticConfig")

        config_type_value = synthetic_config.get(config_type_key)
        check_type(config_type_key, config_type_value, types=(str, ))
        config_type_value = config_type_value.lower()

        request_count_value = synthetic_config.get(request_count_key)
        check_type(request_count_key, request_count_value, types=(int, ))
        check_range(request_count_key, request_count_value, NumberRange(1, 2**20))

        check_map = {
            "string": SyntheticDataset.check_synthetic_string_config,
            "tokenid": SyntheticDataset.check_synthetic_tokenid_config
        }

        check_func = check_map.get(config_type_value)
        if not check_func:
            raise ValueError(f"Expect type should from {check_map.keys()}(case-insensitive) for SyntheticConfig,",
                             f"but got {config_type_value}.")

        type_config = {}
        type_config_key = "StringConfig" if config_type_value == "string" else "TokenIdConfig"
        _ensure_keys_present(synthetic_config.keys(), {type_config_key}, "SyntheticConfig")
        type_config = synthetic_config.get(type_config_key)

        check_func(type_config)

    @staticmethod
    def sample_one_value(method: str, params: dict) -> int:
        # Sample one value, the args have been checked before
        min_value = params["MinValue"]
        max_value = params["MaxValue"]
        if method == "uniform":
            value = np.random.uniform(min_value, max_value)
        elif method == "gaussian":
            mean = params["Mean"]
            stddev = np.sqrt(params["Var"])
            value = np.random.normal(mean, stddev)
            value = np.clip(value, min_value, max_value)
        elif method == "zipf":
            alpha = params["Alpha"]
            value = np.random.zipf(alpha)
            value = np.clip(value, min_value, max_value)
        else:
            raise ValueError(f"Unknown method: {method}, method should be one of {{uniform, gaussian, zipf}}.")
        return int(value)

    @staticmethod
    def read_line(self, line: List[int]) -> Dict:
        """Get a data dict according to line.

        Args:
            line (List[int]): Input line should be a list with 2 integral elements, it represents the number of input
                token and output token respectively.

        Returns:
            data (str): Constructed input with 'A'.
            num_expect_generated_tokens (int): max_tokens.
        """
        if not hasattr(line, '__len__') or len(line) != 2:
            raise ValueError("Input line should be a list with 2 integral elements.")
        default_str = "A"
        num_input_token, num_expect_generated_tokens = line
        data = " ".join([default_str] * num_input_token)
        return data, num_expect_generated_tokens

    @staticmethod
    def find_first_file_path(search_path: str, search_file: str) -> str:
        """
        Recursively find the first search_file file path in a directory tree (Linux compatible).

        Args:
            search_path (str): Path to the root directory for searching
            search_file (str): Full name of the expected file for searching

        Returns:
            str: Full path to the first found search_file file

        Raises:
            ValueError: If input path is invalid or not a directory
            FileNotFoundError: If no search_file file is found in the directory tree

        Note:
            Uses breadth-first search strategy for finding files
            Follows Linux filesystem case sensitivity rules
        """
        # Normalize path (expand ~ and resolve relative paths)
        normalized_path = normalize_file_path(search_path)

        # Validate input path
        if not os.path.exists(normalized_path):
            raise ValueError(f"Path does not exist: {normalized_path}")
        if not os.path.isdir(normalized_path):
            raise ValueError(f"Not a directory: {normalized_path}")

        # Implement BFS search using os.walk
        for root, dirs, files in os.walk(normalized_path):
            # Check current directory first before descending into subdirectories
            if search_file in files:
                # Return immediately when first match is found
                return os.path.join(root, search_file)

        # Handle case where no config file is found
        raise FileNotFoundError(f"No {search_file} found in directory tree: {normalized_path}")

    @staticmethod
    def generate_valid_random_ids(valid_indices, request_size: int) -> torch.Tensor:
        """
        Generates random integers in [0, vocab_size) excluding special IDs.

        Args:
            valid_indices: valid indices in tokenizer file
            request_size: Number of random values to generate

        Returns:
            Tensor of shape (request_size,) with dtype torch.int64
        """

        # Randomly select from valid indices
        rand_indices = torch.randint(0, len(valid_indices), (request_size,))

        return valid_indices[rand_indices].to(torch.int64)

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

                # Check if we've reached the target exactly
                if current_length == target_token_length:
                    return prompt
                elif current_length > target_token_length:
                    # If we overshot, we'll handle it in the next section
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

            # If we already have the exact length, return
            if current_length == target_token_length:
                return prompt

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

    def load(self, config, **kwargs):
        self.logger = AISLogger()
        dataset = []
        model_path_key = "ModelPath"
        config[model_path_key] = kwargs.get("model_path", None)
        self._check_synthetic_config(config)
        request_count = config.get("RequestCount")
        config_type = config.get("Type").lower()
        trust_remote_code = config.get("TrustRemoteCode")
        if config_type == "string":
            string_config = config.get("StringConfig")
            input_method = string_config["Input"]["Method"]
            input_params = string_config["Input"]["Params"]
            output_method = string_config["Output"]["Method"]
            output_params = string_config["Output"]["Params"]
            num_input_output_tokens = [[self.sample_one_value(input_method, input_params),
                                            self.sample_one_value(output_method, output_params)]
                                        for _ in range(request_count)]

            for num_input_output_token in tqdm(num_input_output_tokens, desc="Constructing synthetic string datasets ..."):
                data, max_tokens = self.read_line(self, num_input_output_token)
                dataset.append({"question": data, "answer": "aaa", "max_out_len": max_tokens})

        elif config_type == "tokenid":
            tokenid_config = config.get("TokenIdConfig")
            model_path_value = config.get("ModelPath", None)

            check_type(model_path_key, model_path_value, types=(str, ))
            if not os.path.exists(normalize_file_path(model_path_value)):
                raise ValueError(f"ModelPath does not exist: {str(model_path_value)}")

            model_path_value = normalize_file_path(model_path_value)
            tokenizer_file_path = self.find_first_file_path(model_path_value, "tokenizer_config.json")

            tokenizer_model = AutoTokenizer.from_pretrained(
                os.path.dirname(tokenizer_file_path),
                trust_remote_code=trust_remote_code
            )

            vocab_size = tokenizer_model.vocab_size
            vocab_size = tokenid_config.get("VocabSize", None) if not vocab_size else vocab_size # The vocab_size defined in the model has higher priority
            if not vocab_size:
                raise ValueError(f"The configuration vocab_size was not found in the dataset config file {model_path_value}",
                                 f"or tokenizer config file {tokenizer_file_path}")

            all_special_ids = tokenizer_model.all_special_ids
            self.logger.info(f"Current tokenizer model: {tokenizer_model.__class__.__name__}")
            self.logger.debug(f"Token id range: (0, {vocab_size}) excluding the values {all_special_ids}")

            # Create mask of valid IDs
            valid_ids = torch.ones(vocab_size, dtype=torch.bool)
            original_array = np.array(all_special_ids)
            filtered_array = original_array[original_array < vocab_size]
            valid_ids[filtered_array.tolist()] = False

            # Generate random indices for valid IDs
            valid_indices = torch.where(valid_ids)[0]

            # Check if using distribution config or fixed RequestSize
            input_config = tokenid_config.get("Input", None)
            output_config = tokenid_config.get("Output", None)
            request_size = tokenid_config.get("RequestSize", None)
            prefix_len = tokenid_config.get("PrefixLen", 0)

            # Initialize prefix pool if Input.MaxValue is specified
            prefix_pool = None
            prefix_pool_size = 0
            if input_config is not None:
                input_max_value = input_config.get("Params", {}).get("MaxValue", None)
                if input_max_value is not None:
                    # Initialize prefix pool with size = Input.MaxValue
                    prefix_pool_size = input_max_value
                    # Use fixed seed for reproducibility
                    torch.manual_seed(42)
                    prefix_pool = self.generate_valid_random_ids(valid_indices, prefix_pool_size)
                    self.logger.info(f"Initialized prefix pool with size: {prefix_pool_size}")

            # Determine if using distribution or fixed size
            use_distribution = input_config is not None

            if use_distribution:
                # Use distribution-based sampling
                input_method = input_config["Method"]
                input_params = input_config["Params"]

                # Sample input/output lengths for each request
                num_input_output_tokens = []
                for _ in range(request_count):
                    input_length = self.sample_one_value(input_method, input_params)
                    if output_config is not None:
                        output_method = output_config["Method"]
                        output_params = output_config["Params"]
                        output_length = self.sample_one_value(output_method, output_params)
                    else:
                        output_length = None
                    num_input_output_tokens.append((input_length, output_length))
            else:
                # Use fixed RequestSize (backward compatibility)
                if request_size is None:
                    raise ValueError("RequestSize must be specified when Input/Output distribution is not used.")
                # Calculate actual prefix length for fixed size mode
                if isinstance(prefix_len, float):
                    # Fractional prefix: calculate based on request_size
                    fixed_prefix_len = int(request_size * prefix_len)
                else:
                    fixed_prefix_len = prefix_len
                remaining_size = request_size - fixed_prefix_len
                if remaining_size < 0:
                    raise ValueError(f"PrefixLen ({fixed_prefix_len}) cannot be greater than RequestSize ({request_size})")
                num_input_output_tokens = [(request_size, None) for _ in range(request_count)]

            # Generate datasets
            for idx, (input_length, output_length) in enumerate(tqdm(num_input_output_tokens, desc="Constructing synthetic tokenid datasets ...")):
                # Calculate actual prefix length for this request
                if use_distribution:
                    # In distribution mode, calculate based on input_length
                    if isinstance(prefix_len, float):
                        actual_prefix_len = int(input_length * prefix_len)
                    else:
                        actual_prefix_len = prefix_len
                else:
                    # In fixed size mode, use pre-calculated fixed_prefix_len
                    actual_prefix_len = fixed_prefix_len

                # Select prefix tokenids
                if actual_prefix_len > 0:
                    if prefix_pool is not None and prefix_pool_size > 0:
                        # Select from pre-initialized pool
                        if actual_prefix_len > prefix_pool_size:
                            # If needed length exceeds pool, repeat or extend
                            repeat_times = (actual_prefix_len // prefix_pool_size) + 1
                            extended_pool = torch.cat([prefix_pool] * repeat_times)
                            prefix_tokenids = extended_pool[:actual_prefix_len]
                        else:
                            # Select from pool - always start from beginning for same-length prefixes
                            # This ensures prefixes of the same length are reused
                            start_idx = 0
                            # Handle wrap-around case if prefix length exceeds pool
                            if actual_prefix_len <= prefix_pool_size:
                                # Simple case: can take contiguous slice from start
                                prefix_tokenids = prefix_pool[start_idx:start_idx + actual_prefix_len]
                            else:
                                # Wrap-around case: need to concatenate from end and beginning
                                # Repeat the pool to cover the needed length
                                repeat_times = (actual_prefix_len // prefix_pool_size) + 1
                                extended_pool = torch.cat([prefix_pool] * repeat_times)
                                prefix_tokenids = extended_pool[:actual_prefix_len]
                    else:
                        # Generate new prefix tokenids
                        prefix_tokenids = self.generate_valid_random_ids(valid_indices, actual_prefix_len)
                else:
                    prefix_tokenids = torch.tensor([], dtype=torch.int64)

                # Generate remaining tokenids
                remaining_length = input_length - actual_prefix_len
                if remaining_length > 0:
                    remaining_tokenids = self.generate_valid_random_ids(valid_indices, remaining_length)
                    # Combine prefix and remaining tokenids
                    all_tokenids = torch.cat([prefix_tokenids, remaining_tokenids])
                else:
                    all_tokenids = prefix_tokenids

                # Decode to prompt
                prompt = tokenizer_model.decode(all_tokenids.tolist(), skip_special_tokens=True)

                # Adjust to ensure exact token length
                prompt = self._adjust_prompt_length(tokenizer_model, prompt, input_length)

                # Create dataset entry
                entry = {"question": prompt, "answer": "aaa"}
                if output_length is not None:
                    entry["max_out_len"] = output_length
                dataset.append(entry)

        else:
            raise ValueError(f"Invalid type:{config_type}. Should choose one from {{string, tokenid}}")

        return Dataset.from_list(dataset)