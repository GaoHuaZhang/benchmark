
# vLLM 多模态调试脚本设计与使用说明

## 功能概述

在本仓库中新增调试脚本 `debug_vllm_multimodal.py`，用于直接访问本地 **vLLM OpenAI 兼容服务**，发送包含 **文本 prompt、多张图片、多段音频** 的多模态请求，支持：

- 单条请求调试：命令行直接指定 `prompt`、`images`、`audios`；
- 批量推理调试：从 jsonl 文件读取多条样本（按行包含 `prompt`、`images`、`audios` 字段），顺序调用服务并可选保存推理结果；
- 图片/音频两种编码方式：在请求体中以 **URL** 或 **base64 data URL / base64 数据字段** 的形式携带，便于适配不同服务端实现。

脚本仅依赖标准库与 `requests`，可作为 vLLM 服务联调/排错的轻量级工具，不依赖 AISBench 内部 Runner/Task 体系。

## 请求结构与接口约定

脚本假定本地 vLLM 提供 **OpenAI 兼容** Chat Completions 接口：

- HTTP 方法：`POST`
- 路径：`{base_url}/v1/chat/completions`
- 关键字段：
  - `model`: 目标模型名称；
  - `messages`: 列表，至少包含一个 `{"role": "user", "content": [...]}`；
  - `content`: 由多种类型的块构成的列表，当前脚本构造：
    - 文本块：`{"type": "text", "text": "<prompt 文本>"}`；
    - 图片块（URL 模式）：`{"type": "image_url", "image_url": {"url": "<图片 URL 或本地绝对路径>"}}`；
    - 图片块（base64 模式）：`{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<base64 字符串>"}}`；
    - 音频块（URL 模式，约定）：`{"type": "input_audio", "audio_url": {"url": "<音频 URL 或本地绝对路径>"}}`；
    - 音频块（base64 模式，约定）：`{"type": "input_audio", "audio": {"data": "<base64 字符串>", "format": "<wav/mp3 等>"}}`。

如需与具体服务端实现完全对齐，可在脚本中集中调整上述结构（图片在 `build_image_contents`，音频在 `build_audio_contents`），保持调用入口不变。

## 脚本参数说明

脚本入口为仓库根目录下的：

```bash
python debug_vllm_multimodal.py ...
```

核心参数如下（部分参数可选）：

- 服务配置：
  - `--base-url`: vLLM OpenAI 兼容服务地址，例如 `http://127.0.0.1:8080`；
  - `--api-key`: 可选，若服务端需要鉴权，则作为 `Authorization: Bearer <api-key>` 透传；
  - `--model`: 模型名称，对应服务端 `model` 字段。
- 生成控制：
  - `--temperature`: 采样温度，若指定会直接透传到请求体；
  - `--max-tokens`: 最大生成 token 数，若指定会直接透传到请求体；
  - `--timeout`: HTTP 请求超时时间（秒），默认 60。
- 单条样本模式：
  - `--prompt`: 单条请求的文本 prompt；
  - `--images`: 多个图片路径或 URL，`nargs="*"`，可一次传多张；
  - `--audios`: 多个音频路径或 URL，`nargs="*"`。
- 批量 jsonl 模式：
  - `--jsonl-path`: jsonl 文件路径，启用后按行读取样本；
  - `--max-samples`: 最多处理多少条样本，默认读取全部；
  - `--output-jsonl`: 可选，将每条样本的输入与推理结果写入新的 jsonl 文件；
  - `--save-raw-response`: 在输出 jsonl 中保留完整原始响应 JSON。
- 资源编码控制：
  - `--image-format`: `url` 或 `base64`，控制图片在请求体中以 URL 还是 base64 data URL 传输；
  - `--audio-format`: `url` 或 `base64`，控制音频在请求体中以 URL 还是 base64 数据字段传输；
  - `--base-path`: 当 jsonl 中提供的是相对路径时，用作拼接根目录。
- 其他：
  - `--sleep-ms`: 批量模式中，两次请求之间的睡眠时间（毫秒），用于限速或观察服务行为；
  - `--verbose`: 打印每条样本的输入摘要（行号、图片/音频数）和回复截断内容；
  - `--fail-on-error`: 批量模式下，只要有一条请求失败就立即返回非 0 退出码。

注意：`--jsonl-path` 与单样本参数 (`--prompt`/`--images`/`--audios`) 互斥，脚本会在参数解析阶段做校验。

## jsonl 格式约定

批量模式下，jsonl 文件每一行是一个 JSON 对象，推荐结构为：

```json
{"prompt": "描述一幅夏日海滩的画面", "images": ["/abs/path/to/img1.png", "/abs/path/to/img2.jpg"], "audios": ["https://example.com/audio1.wav"]}
{"prompt": "请根据图片内容生成一段故事", "images": ["rel/path/to/img3.png"], "audios": []}
```

字段说明：

- `prompt`：字符串，必填；
- `images`：字符串列表，可为空或省略，元素为绝对路径、相对路径或 URL；
- `audios`：字符串列表，可为空或省略，元素为绝对路径、相对路径或 URL。

脚本行为：

- 对于 URL（以 `http://` 或 `https://` 开头）直接按 URL 发送；
- 对于本地路径：
  - 若是相对路径且指定了 `--base-path`，则拼接为 `os.path.join(base_path, path)` 后取绝对路径；
  - 在 `--image-format base64` / `--audio-format base64` 时，会读取本地文件并转为 base64；
  - 在 `url` 模式下会直接使用本地绝对路径字符串，由服务端自行解析（如需要可自行扩展为 `file://` URL 或其他映射逻辑）。

## 使用示例

### 单条请求（仅文本）

```bash
python debug_vllm_multimodal.py \
  --base-url http://127.0.0.1:8080 \
  --model your-model-name \
  --prompt "请用中文解释什么是大语言模型"
```

### 单条请求（文本 + 多张本地图片，图片以 base64 传输）

```bash
python debug_vllm_multimodal.py \
  --base-url http://127.0.0.1:8080 \
  --model your-model-name \
  --prompt "请描述这些图片的共同点" \
  --images /data/imgs/a.png /data/imgs/b.jpg \
  --image-format base64
```

### 单条请求（文本 + 远程音频，音频以 URL 传输）

```bash
python debug_vllm_multimodal.py \
  --base-url http://127.0.0.1:8080 \
  --model your-model-name \
  --prompt "请根据音频内容做转写" \
  --audios https://example.com/audio1.wav https://example.com/audio2.wav \
  --audio-format url
```

### jsonl 批量推理（图片 base64，音频 URL，保存结果）

```bash
python debug_vllm_multimodal.py \
  --base-url http://127.0.0.1:8080 \
  --model your-model-name \
  --jsonl-path /path/to/samples.jsonl \
  --base-path /data/multimodal_root \
  --image-format base64 \
  --audio-format url \
  --output-jsonl /path/to/results.jsonl \
  --verbose
```

## 异常处理与调试建议

- **参数错误**：未指定 `--base-url` / `--model`，或同时指定 `--jsonl-path` 及单样本参数时，脚本会在启动时直接报错并退出；
- **文件不存在**：当图片/音频为本地路径但文件不存在时，会抛出 `FileNotFoundError` 并打印错误信息；在批量模式下可结合 `--fail-on-error` 控制是否立即退出；
- **JSON 解析错误**：jsonl 中某一行不是合法 JSON 时，该行会被跳过并打印警告，不影响其他行；
- **服务端错误**：当 HTTP 状态码非 2xx 时，会打印状态码和响应体前 500 字符，便于排查服务端问题；
- **响应格式异常**：当响应中不存在 `choices[0].message.content` 时，脚本会打印整段 JSON 响应，便于调整服务端字段映射；
- **性能调试**：通过 `--sleep-ms` 控制请求节奏，避免在本地调试时因瞬时高 QPS 影响 vLLM 服务稳定性。
