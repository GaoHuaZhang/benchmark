# MM VLLM Custom

vLLM 风格多模态自定义数据集：支持从 jsonl 加载 `prompt`、`images`、`audios` 字段，并按 OpenAI Chat Completions 多模态结构发送请求，用于性能测评。

## 数据集格式

jsonl 每行一个 JSON 对象，与 `debug_vllm_multimodal_guide.md` 约定一致：

- **prompt**（必填）：字符串，文本提示。
- **images**（可选）：字符串数组，图片路径（绝对/相对）或 URL；可省略或空数组。
- **audios**（可选）：字符串数组，**仅支持 .txt 文件路径**，txt 内容为 base64 编码的音频；可省略或空数组。
- **answer**（可选）：标准答案，当前仅占位，便于后续做精度评估。

示例：

```json
{"prompt": "描述一幅夏日海滩的画面", "images": ["/abs/path/to/img1.png", "/abs/path/to/img2.jpg"], "audios": []}
{"prompt": "请根据图片内容生成一段故事", "images": ["rel/path/to/img3.png"], "audios": ["rel/path/to/audio.txt"]}
```

相对路径可与配置中的 `base_path` 拼接为绝对路径。

## 与 mm_custom 的差异

| 项目     | mm_custom           | mm_vllm_custom        |
|----------|---------------------|------------------------|
| 每行字段 | type, path, question, answer | prompt, images, audios, answer |
| 多模态   | 每行单一类型 (image/video/audio) | 每行可同时多图、多音频 |
| 音频     | 直接 wav 等文件路径 | 仅 .txt 路径，内容为 base64 |

## 部署与使用

1. 将 jsonl 放到 `ais_bench/datasets/mm_vllm_custom/mm_vllm_custom.jsonl`（或修改配置中的 `path`）。
2. 需要时在数据集配置中设置 `base_path`，用于解析相对路径。
3. 性能测评示例：

```bash
ais_bench --models vllm_api_stream_chat --datasets mm_vllm_custom_gen --mode perf
```

## 可用任务

| 任务名称            | 说明                     | 评估指标 | 配置文件 |
|---------------------|--------------------------|----------|----------|
| mm_vllm_custom_gen   | vLLM 多模态自定义生成任务 | accuracy | [mm_vllm_custom_gen.py](mm_vllm_custom_gen.py) |
