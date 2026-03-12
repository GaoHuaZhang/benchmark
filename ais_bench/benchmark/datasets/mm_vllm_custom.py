"""vLLM-style multimodal custom dataset (prompt + images + audios jsonl)."""
import json
import base64
import mimetypes
import os

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.utils.prompt import (
    AIS_CONTENT_TAG,
    AIS_TEXT_START,
    AIS_IMAGE_START,
    AIS_AUDIO_START,
)

from .base import BaseDataset

logger = AISLogger()


def _is_url(path_or_url):
    lower = path_or_url.lower()
    return lower.startswith("http://") or lower.startswith("https://")


def _resolve_local_path(path, base_path):
    if os.path.isabs(path):
        return path
    if base_path:
        return os.path.abspath(os.path.join(base_path, path))
    return os.path.abspath(path)


def _read_file_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def _detect_mime(path, default):
    mime, _ = mimetypes.guess_type(path)
    return mime or default


def _load_audio_from_txt(path_or_url, base_path):
    """Read base64 audio content from a .txt file path."""
    if _is_url(path_or_url):
        raise ValueError("audios only support .txt file paths, not URL")
    local_path = _resolve_local_path(path_or_url, base_path)
    if not os.path.exists(local_path):
        raise FileNotFoundError("Audio txt file not found: %s" % local_path)
    if not local_path.lower().endswith(".txt"):
        raise ValueError("Audio input must be .txt path, got: %s" % local_path)
    with open(local_path, "r", encoding="utf-8") as f:
        return f.read().strip()


@LOAD_DATASET.register_module()
class MMVllmCustomDataset(BaseDataset):
    """Dataset for vLLM-style jsonl: prompt, images[], audios[] (per debug_vllm_multimodal_guide)."""

    @staticmethod
    def load(path, mm_type="path", base_path=None):
        """
        Load jsonl where each line has:
          - prompt (str, required)
          - images (list of str, optional, paths or URLs)
          - audios (list of str, optional, .txt paths whose content is base64 audio)
          - answer (optional, for future accuracy eval)

        mm_type: "path" = image URL/path as-is; "base64" = image as data URL.
        base_path: prefix for relative image/audio paths.
        """
        path = get_data_path(path, local_mode=True)
        if not os.path.isfile(path):
            raise AISBenchConfigError(
                UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                "Dataset file not found: %s" % path,
            )

        dataset = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise AISBenchConfigError(
                        UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                        "Invalid JSON at line %d: %s" % (lineno, e),
                    ) from e
                if not isinstance(obj, dict):
                    raise AISBenchConfigError(
                        UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                        "Line %d: expected JSON object" % lineno,
                    )

                prompt = obj.get("prompt")
                if prompt is None:
                    raise AISBenchConfigError(
                        UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                        "Line %d: missing required field 'prompt'" % lineno,
                    )
                if not isinstance(prompt, str):
                    raise AISBenchConfigError(
                        UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                        "Line %d: 'prompt' must be string" % lineno,
                    )

                images = obj.get("images")
                if images is None:
                    images = []
                if not isinstance(images, list):
                    raise AISBenchConfigError(
                        UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                        "Line %d: 'images' must be list" % lineno,
                    )

                audios = obj.get("audios")
                if audios is None:
                    audios = []
                if not isinstance(audios, list):
                    raise AISBenchConfigError(
                        UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                        "Line %d: 'audios' must be list" % lineno,
                    )

                # Build image values for content segments (path or data URL)
                image_values = []
                for img in images:
                    if not img:
                        continue
                    if _is_url(img):
                        image_values.append(img)
                        continue
                    local_path = _resolve_local_path(img, base_path)
                    if not os.path.exists(local_path):
                        raise FileNotFoundError(
                            "Line %d: image file not found: %s" % (lineno, local_path)
                        )
                    if mm_type == "base64":
                        b64 = _read_file_base64(local_path)
                        mime = _detect_mime(local_path, "image/png")
                        image_values.append("data:%s;base64,%s" % (mime, b64))
                    else:
                        image_values.append(local_path)

                # Build audio values (base64 from .txt)
                audio_values = []
                for aud in audios:
                    if not aud:
                        continue
                    b64_content = _load_audio_from_txt(aud, base_path)
                    audio_values.append("data:audio/wav;base64,%s" % b64_content)

                # content string for PromptList.format_mm: AIS_* segments
                content_parts = [AIS_TEXT_START + prompt + AIS_CONTENT_TAG]
                for v in image_values:
                    content_parts.append(AIS_IMAGE_START + v + AIS_CONTENT_TAG)
                for v in audio_values:
                    content_parts.append(AIS_AUDIO_START + v + AIS_CONTENT_TAG)
                content = "".join(content_parts)

                row = {
                    "prompt": prompt,
                    "question": prompt,
                    "content": content,
                    "images": image_values,
                    "audios": audio_values,
                    "answer": obj.get("answer", ""),
                }
                dataset.append(row)

        return Dataset.from_list(dataset)


class MMVllmCustomEvaluator(BaseEvaluator):
    """Placeholder evaluator for mm_vllm_custom (e.g. perf-only)."""

    def score(self, predictions, references):
        return {"accuracy": 1}
