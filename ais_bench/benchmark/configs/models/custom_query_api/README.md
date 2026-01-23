# CustomQueryAPI 使用说明

## 概述

`CustomQueryAPI` 是一个自定义的 API 模型包装器，支持自定义的请求/响应格式。该接口使用 `query` 字段作为请求参数，响应格式为 `{res: <generated_text>}`，并且 URL 直接使用用户指定的完整地址，不进行路径拼接。

## 接口规范

### 请求格式

**请求体格式**:
```json
{
    "query": "<prompt_string>",
    "stream": false,
    "<generation_kwargs>": "<value>",
    "<additional_args>": "<value>"
}
```

**说明**:
- `query`: 必需字段，包含输入的提示文本
- `stream`: 布尔值，表示是否启用流式输出（当前仅支持非流式，即 `false`）
- `generation_kwargs`: 可选的生成参数，如 `temperature`、`max_tokens` 等
- `additional_args`: 其他自定义参数

### 响应格式

**非流式响应格式**:
```json
{
    "res": "<generated_text>"
}
```

**说明**:
- `res`: 必需字段，包含模型生成的文本内容

### URL 配置

- 用户指定的 `url` 参数即为完整的访问 URL
- 不会自动拼接 `v1/completions` 等路径
- 如果未指定 `url`，将使用 `host_ip` 和 `host_port` 构建基础 URL

## 配置示例

### 基本配置

```python
from ais_bench.benchmark.models import CustomQueryAPI

models = [
    dict(
        attr="service",
        type=CustomQueryAPI,
        abbr="custom-query-api-general",
        path="",
        stream=False,
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="http://localhost:8080/api/infer",  # 完整的 API URL
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
        ),
    )
]
```

### 使用完整 URL 配置

```python
models = [
    dict(
        attr="service",
        type=CustomQueryAPI,
        abbr="custom-query-api",
        url="https://api.example.com/v1/custom/inference",  # 直接使用完整 URL
        stream=False,
        api_key="your-api-key-here",
        max_out_len=1024,
        generation_kwargs=dict(
            temperature=0.7,
            top_p=0.9,
        ),
    )
]
```

### 使用 host_ip 和 host_port 配置

```python
models = [
    dict(
        attr="service",
        type=CustomQueryAPI,
        abbr="custom-query-api",
        host_ip="192.168.1.100",
        host_port=9000,
        stream=False,
        max_out_len=512,
        generation_kwargs=dict(
            temperature=0.01,
        ),
    )
]
```

## 参数说明

### 必需参数

- `type`: 模型类型，设置为 `CustomQueryAPI`
- `abbr`: 模型的唯一标识符
- `attr`: 设置为 `"service"` 表示服务化 API 模型

### 可选参数

- `url` (str): 完整的 API URL 地址。如果指定，将直接使用此 URL，`host_ip` 和 `host_port` 将被忽略
- `host_ip` (str): API 服务的主机 IP 地址，默认为 `"localhost"`
- `host_port` (int): API 服务的端口号，默认为 `8080`
- `stream` (bool): 是否启用流式输出，默认为 `False`（当前仅支持非流式）
- `max_out_len` (int): 最大输出长度，控制生成文本的最大 token 数，默认为 `4096`
- `retry` (int): 请求失败时的重试次数，默认为 `2`
- `api_key` (str): API 服务的密钥，如果提供，将在请求头中添加 `Authorization: Bearer <api_key>`
- `generation_kwargs` (dict): 生成参数配置，这些参数会被包含在请求体中传递给 API 服务
- `batch_size` (int): 请求发送的最大并发数，默认为 `1`
- `request_rate` (float): 请求发送频率，每 `1/request_rate` 秒发送 1 个请求，小于 0.1 则一次性发送所有请求，默认为 `0`
- `trust_remote_code` (bool): 是否信任远程代码，默认为 `False`
- `enable_ssl` (bool): 是否启用 SSL 连接，默认为 `False`
- `verbose` (bool): 是否启用详细日志输出，默认为 `False`

## 使用示例

### 命令行使用

```bash
# 使用自定义查询 API 进行评测
ais_bench --models custom-query-api-general --datasets demo_gsm8k_gen_4_shot_cot_str
```

### 配置文件使用

创建配置文件 `my_custom_config.py`:

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import CustomQueryAPI

with read_base():
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets
    from ais_bench.benchmark.configs.models.custom_query_api.custom_query_api_general import models

# 自定义模型配置
models[0]['url'] = 'http://your-api-server:8080/infer'
models[0]['api_key'] = 'your-api-key'
models[0]['max_out_len'] = 1024
```

运行配置:

```bash
ais_bench my_custom_config.py
```

## 注意事项

1. **URL 配置**: 如果指定了 `url` 参数，系统将直接使用该 URL，不会进行任何路径拼接。确保 URL 是完整的、可访问的地址。

2. **请求格式**: 请求体使用 `query` 字段而不是 `prompt` 字段，这与标准的 vLLM API 不同。

3. **响应格式**: 响应必须包含 `res` 字段，系统将从该字段提取生成的文本。

4. **流式支持**: 当前实现仅支持非流式接口（`stream=False`）。如果需要流式支持，需要额外实现 `parse_stream_response` 方法。

5. **错误处理**: 如果 API 返回非 200 状态码，错误信息将记录在输出的 `error_info` 字段中。

## 故障排查

### 常见问题

1. **连接失败**: 检查 `url`、`host_ip` 和 `host_port` 配置是否正确，确保 API 服务正在运行。

2. **认证失败**: 如果 API 需要认证，确保 `api_key` 配置正确。

3. **响应解析失败**: 确保 API 返回的 JSON 格式包含 `res` 字段。

4. **请求格式错误**: 检查 `generation_kwargs` 中的参数是否符合 API 服务的要求。

## 相关文件

- 模型实现: `ais_bench/benchmark/models/api_models/custom_query_api.py`
- 配置文件: `ais_bench/benchmark/configs/models/custom_query_api/custom_query_api_general.py`
