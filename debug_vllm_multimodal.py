import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
'''
单条请求样例（文本+多图，图片 base64）：

python debug_vllm_multimodal.py --base-url http://127.0.0.1:8080 --model your-model-name --prompt "..." --images /path/a.png /path/b.jpg --image-format base64

批量 jsonl 推理样例（文本+多图+多音频，图片 base64，音频为 txt 文件中的 base64 内容）：
python debug_vllm_multimodal.py --base-url http://127.0.0.1:8080 --model your-model-name --jsonl-path /path/to/samples.jsonl --base-path /data/root --image-format base64 --output-jsonl /path/to/results.jsonl --verbose
'''

def is_url(path_or_url: str) -> bool:
    lower = path_or_url.lower()
    return lower.startswith("http://") or lower.startswith("https://")


def detect_mime_type(path: str, default: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or default


def read_file_base64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def resolve_local_path(path: str, base_path: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    if base_path:
        return os.path.abspath(os.path.join(base_path, path))
    return os.path.abspath(path)


def load_media(
    path_or_url: str,
    mode: str,
    media_type: str,
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    解析单个图片/音频资源。

    返回结构：
    - kind = "url": {"kind": "url", "value": "<url_or_path>"}
    - kind = "base64": {"kind": "base64", "value": "<b64>", "mime": "<mime>", "ext": "<ext_without_dot>"}
    """
    if is_url(path_or_url):
        return {"kind": "url", "value": path_or_url}

    local_path = resolve_local_path(path_or_url, base_path)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"{media_type} 文件不存在: {local_path}")

    if mode == "url":
        # 直接返回本地绝对路径，假设服务端能访问该路径或自行做映射
        return {"kind": "url", "value": local_path}

    ext = os.path.splitext(local_path)[1].lstrip(".").lower()
    if media_type == "image":
        default_mime = "image/png"
    else:
        default_mime = "audio/wav"
    mime = detect_mime_type(local_path, default_mime)
    b64 = read_file_base64(local_path)
    return {"kind": "base64", "value": b64, "mime": mime, "ext": ext or None}


def load_audio_from_txt_base64(
    path_or_b64: str,
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从 txt 文件加载 base64 编码的音频内容。

    输入为 .txt 文件路径，文件内容为 base64 编码的音频数据。
    返回 {"data": "<base64>", "format": "wav"}。
    """
    if is_url(path_or_b64):
        raise ValueError("音频仅支持 txt 文件路径，不支持 URL")

    local_path = resolve_local_path(path_or_b64, base_path)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"音频 txt 文件不存在: {local_path}")

    if not local_path.lower().endswith(".txt"):
        raise ValueError(f"音频输入须为 .txt 文件路径，当前: {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        b64_content = f.read().strip()

    return {"data": b64_content, "format": "wav"}


def build_image_contents(
    images: Iterable[str],
    image_format: str,
    base_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    for img in images:
        if not img:
            continue
        media = load_media(img, image_format, "image", base_path)
        if media["kind"] == "url":
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "file://" + media["value"],
                    },
                }
            )
        else:
            data_url = f"data:{media['mime']};base64,{media['value']}"
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    },
                }
            )
    return contents


def build_audio_contents(
    audios: Iterable[str],
    base_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    构造音频 content 块。音频仅支持 txt 文件路径，文件内容为 base64 编码的音频数据。

    采用类似 OpenAI input_audio 的约定：
    {"type": "input_audio", "audio": {"data": "<base64>", "format": "wav"}}
    """
    contents: List[Dict[str, Any]] = []
    for audio in audios:
        if not audio:
            continue
        loaded = load_audio_from_txt_base64(audio, base_path)
        contents.append(
            {
                "type": "input_audio",
                "audio": {
                    "data": loaded["data"],
                    "format": loaded["format"],
                },
            }
        )
    return contents


def build_message_content(
    prompt: str,
    images: Iterable[str],
    audios: Iterable[str],
    image_format: str,
    base_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    content.extend(build_image_contents(images, image_format, base_path))
    content.extend(build_audio_contents(audios, base_path))
    return content


def call_vllm_chat_completion(
    base_url: str,
    api_key: Optional[str],
    model: str,
    content: List[Dict[str, Any]],
    timeout: float,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    if extra_params:
        payload.update(extra_params)

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(
            f"请求失败: status={resp.status_code}, body={resp.text[:500]}"
        )
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"响应 JSON 解析失败: {exc}, body={resp.text[:500]}"
        ) from exc


def extract_reply_text(resp: Dict[str, Any]) -> Optional[str]:
    try:
        choices = resp.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        return message.get("content")
    except Exception:  # noqa: BLE001
        return None


def iter_jsonl(
    path: str,
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] 第 {idx} 行 JSON 解析失败: {exc}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(obj, dict):
                print(
                    f"[WARN] 第 {idx} 行不是 JSON 对象，已跳过",
                    file=sys.stderr,
                )
                continue
            yield idx, obj


def run_single_sample(
    args: argparse.Namespace,
) -> int:
    images = args.images or []
    audios = args.audios or []
    content = build_message_content(
        prompt=args.prompt or "",
        images=images,
        audios=audios,
        image_format=args.image_format,
        base_path=args.base_path,
    )
    if not content:
        print("没有有效的 prompt / image / audio 输入", file=sys.stderr)
        return 1

    extra_params = {}
    if args.temperature is not None:
        extra_params["temperature"] = args.temperature
    if args.max_tokens is not None:
        extra_params["max_tokens"] = args.max_tokens

    try:
        resp = call_vllm_chat_completion(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            content=content,
            timeout=args.timeout,
            extra_params=extra_params or None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 请求失败: {exc}", file=sys.stderr)
        return 1

    reply = extract_reply_text(resp)
    print("=== 模型回复 ===")
    if reply is None:
        print("[WARN] 响应中未找到 message.content 字段")
        print(json.dumps(resp, ensure_ascii=False, indent=2))
    else:
        print(reply)
    return 0


def run_jsonl_batch(
    args: argparse.Namespace,
) -> int:
    output_fp = None
    if args.output_jsonl:
        output_fp = open(args.output_jsonl, "w", encoding="utf-8")

    processed = 0
    failed = 0
    extra_params = {}
    if args.temperature is not None:
        extra_params["temperature"] = args.temperature
    if args.max_tokens is not None:
        extra_params["max_tokens"] = args.max_tokens

    for lineno, sample in iter_jsonl(args.jsonl_path):
        if args.max_samples is not None and processed >= args.max_samples:
            break

        prompt = sample.get("prompt", "")
        images = sample.get("images") or []
        audios = sample.get("audios") or []
        if not isinstance(images, list) or not isinstance(audios, list):
            print(
                f"[WARN] 第 {lineno} 行的 images/audios 字段不是列表，已跳过",
                file=sys.stderr,
            )
            failed += 1
            continue

        content = build_message_content(
            prompt=prompt or "",
            images=images,
            audios=audios,
            image_format=args.image_format,
            base_path=args.base_path,
        )
        if not content:
            print(
                f"[WARN] 第 {lineno} 行没有有效的 prompt/images/audios，已跳过",
                file=sys.stderr,
            )
            failed += 1
            continue

        if args.verbose:
            print(
                f"==> 行 {lineno}: prompt={repr(prompt)[:80]}, "
                f"images={len(images)}, audios={len(audios)}"
            )

        try:
            resp = call_vllm_chat_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                content=content,
                timeout=args.timeout,
                extra_params=extra_params or None,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"[ERROR] 第 {lineno} 行请求失败: {exc}"
            if args.fail_on_error:
                print(msg, file=sys.stderr)
                if output_fp:
                    output_fp.close()
                return 1
            print(msg, file=sys.stderr)
            failed += 1
            continue

        reply = extract_reply_text(resp)
        if args.verbose:
            print(f"<== 行 {lineno} 回复: {repr(reply)[:120]}")

        result_record: Dict[str, Any] = {
            "lineno": lineno,
            "prompt": prompt,
            "images": images,
            "audios": audios,
            "response": reply,
            "raw_response": resp if args.save_raw_response else None,
        }
        if not args.save_raw_response:
            result_record.pop("raw_response", None)

        if output_fp:
            output_fp.write(json.dumps(result_record, ensure_ascii=False) + "\n")

        processed += 1
        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)

    if output_fp:
        output_fp.close()

    print(
        f"完成处理 jsonl: 成功 {processed - failed} 条, 失败 {failed} 条, 总计 {processed} 条"
    )
    return 0 if failed == 0 else 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "本地 vLLM OpenAI 兼容多模态调试脚本，"
            "支持 prompt + 多图 + 多音频，以及 jsonl 批量推理。"
        )
    )

    # 服务与模型配置
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="vLLM OpenAI 兼容服务地址，例如 http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="可选，服务需要鉴权时传入，对应 Authorization: Bearer <api-key>",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称，对应服务端可用的 model 字段",
    )

    # 调用参数
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="可选，采样温度，将透传到请求体中",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="可选，最大生成 token 数，将透传到请求体中",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP 请求超时时间（秒），默认 60",
    )

    # 单样本模式
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="单条请求的文本 prompt（与 --jsonl-path 互斥）",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        default=None,
        help="单条请求的图片路径或 URL，可传多个",
    )
    parser.add_argument(
        "--audios",
        type=str,
        nargs="*",
        default=None,
        help="单条请求的音频 txt 文件路径（文件内容为 base64 编码），可传多个",
    )

    # 批量 jsonl 模式
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        help="jsonl 文件路径，每行包含 prompt/images/audios 字段",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="从 jsonl 中最多读取多少条样本，默认读取全部",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="可选，将推理结果写入该 jsonl 文件",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="在输出 jsonl 中包含完整原始响应 JSON",
    )

    # 资源编码方式
    parser.add_argument(
        "--image-format",
        type=str,
        choices=["url", "base64"],
        default="url",
        help="图片在请求中的编码方式：url 或 base64，默认 url",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="当 jsonl 中是相对路径时，用作拼接根目录（可选）",
    )

    # 其他
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="批量模式中两次请求之间的睡眠时间（毫秒），默认 0",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每条样本的简要输入和输出摘要",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="批量模式下遇到任意一条失败时立即退出并返回非 0",
    )

    args = parser.parse_args(argv)

    if args.jsonl_path and (args.prompt or args.images or args.audios):
        parser.error("不能同时指定 --jsonl-path 和 单条样本参数 (--prompt/--images/--audios)")
    if not args.jsonl_path and not (args.prompt or args.images or args.audios):
        parser.error("必须提供单条样本参数 (--prompt/--images/--audios) 或 --jsonl-path 之一")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.jsonl_path:
        return run_jsonl_batch(args)
    return run_single_sample(args)


if __name__ == "__main__":
    raise SystemExit(main())

