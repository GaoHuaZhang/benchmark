# 多Pod基准测试使用指南

## 概述

多Pod基准测试用于在多个Pod（节点）上并行运行性能评测任务，并将各Pod的测试结果合并为统一的报告。该方案适用于需要在不同配置或不同节点上同时进行性能测试的场景。

## 使用场景

- 需要在多个Pod上并行运行不同的模型-数据集组合
- 需要将多个Pod的测试结果合并为统一报告
- 需要对比不同配置下的性能表现

## 环境准备


## 准备工作

### 1. 配置文件准备

创建性能评测配置文件（如 `demo_infer_vllm_api_perf_dev.py`），配置多个模型和数据集组合。

#### 配置文件示例

```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as vllm_api_stream_chat,
    )
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )

# 定义数据集配置1
synthetic_config_test1 = {
    "Type":"string",
    "RequestCount": 10,
    "TrustRemoteCode": False,
    "StringConfig" : {
        "Input" : {
            "Method": "uniform",
            "Params": {"MinValue": 100, "MaxValue": 100}
        },
        "Output" : {
            "Method": "uniform",
            "Params": {"MinValue": 100, "MaxValue": 200}
        }
    },
}

# 定义数据集配置2
synthetic_config_test2 = {
    "Type":"string",
    "RequestCount": 20,
    "TrustRemoteCode": False,
    "StringConfig" : {
        "Input" : {
            "Method": "uniform",
            "Params": {"MinValue": 200, "MaxValue": 200}
        },
        "Output" : {
            "Method": "uniform",
            "Params": {"MinValue": 200, "MaxValue": 300}
        }
    },
}

# 创建数据集实例
synthetic_datasets_test1 = [synthetic_datasets[0].deepcopy()]
synthetic_datasets_test1[0]['abbr'] = 'synthetic-test1' 
synthetic_datasets_test1[0]['config'] = synthetic_config_test1

synthetic_datasets_test2 = [synthetic_datasets[0].deepcopy()]
synthetic_datasets_test2[0]['abbr'] = 'synthetic-test2'
synthetic_datasets_test2[0]['config'] = synthetic_config_test2

datasets = synthetic_datasets_test1 + synthetic_datasets_test2

# 创建模型实例
models_test1 = [vllm_api_stream_chat[0].deepcopy()]
models_test1[0]['abbr'] = 'demo-vllm-api-general-chat-test1'
models_test1[0]['path'] = '/home/theone/weight/DeepSeek-R1'
models_test1[0]['model'] = 'qwen'
models_test1[0]['batch_size'] = 100

models_test2 = [vllm_api_stream_chat[0].deepcopy()]
models_test2[0]['abbr'] = 'demo-vllm-api-general-chat-test2'
models_test2[0]['path'] = '/home/theone/weight/DeepSeek-R1'
models_test2[0]['model'] = 'qwen'
models_test2[0]['batch_size'] = 100

models = models_test1 + models_test2

# 关键：定义模型-数据集组合，每个组合对应一个Pod的任务
model_dataset_combinations = [
    dict(models=models_test1, datasets=synthetic_datasets_test1),  # Pod 1 的任务
    dict(models=models_test2, datasets=synthetic_datasets_test2),  # Pod 2 的任务
]
```

#### 配置要点

1. **使用 `model_dataset_combinations`**：通过该参数指定每个Pod需要执行的模型-数据集组合，避免默认的笛卡尔积组合
2. **唯一标识 `abbr`**：确保每个模型和数据集的 `abbr` 参数唯一，用于区分不同的测试任务
3. **模型配置**：根据实际需求配置模型的路径、名称、batch_size等参数

### 2. 合并配置准备

创建合并配置文件（如 `merge_config.py`），用于合并多个Pod的结果：

```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as vllm_api_stream_chat,
    )
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )

datasets = synthetic_datasets
models = vllm_api_stream_chat

models[0]['abbr'] = "merge_results"
models[0]['path'] = "/home/theone/weight/DeepSeek-R1"
datasets[0]['abbr'] = "merged"
```

## 运行步骤

### 1. 执行基准测试

在每个Pod上运行基准测试脚本：

```bash
bash multi_pod_benchmark.sh
```

或者手动执行：

```bash
# 1. 运行性能评测
ais_bench demo_infer_vllm_api_perf_dev.py \
  -m perf \
  --max-num-workers 2 \
  --pressure \
  --pressure-time 10

# 2. 合并 performances 目录下的结果
python merge_performances.py

# 3. 合并多个pod的性能结果（可选）
ais_bench merge_config.py -m perf --reuse
```

### 2. 脚本参数说明

- `-m perf`：运行模式，指定为性能评测模式
- `--max-num-workers 2`：最大并行任务数，范围 [1, CPU 核数]，默认 1
- `--pressure`：是否启用压力测试，默认 False
- `--pressure-time 10`：压力测试时间（秒），建议为平均 E2EL 的三倍
- `--reuse`：复用已有结果，用于合并阶段

### 3. 结果合并流程

#### 步骤1：自动合并 performances

`merge_performances.py` 脚本会自动：

1. 查找 `outputs/default` 下最新的文件夹
2. 合并所有子文件夹中的 `db_data` 到 `merge_results/db_data`
3. 合并所有 `*_details.jsonl` 文件为 `merged_details.jsonl`

合并后的结果位于：

```text
outputs/default/{最新时间戳}/performances/merge_results/
├── db_data/          # 合并后的数据库数据
└── merged_details.jsonl  # 合并后的详细结果
```

#### 步骤2：使用 AISBench 合并结果（可选）

如果需要生成统一的性能报告，运行：

```bash
ais_bench merge_config.py -m perf --reuse
```

该命令会：

- 读取合并后的结果
- 生成统一的性能报告
- 使用 `--reuse` 参数避免重复运行测试

## 输出结果说明

### 目录结构

```text
outputs/default/{时间戳}/
├── performances/
│   ├── {模型1}-{数据集1}/     # Pod 1 的结果
│   │   ├── db_data/
│   │   └── *_details.jsonl
│   ├── {模型2}-{数据集2}/     # Pod 2 的结果
│   │   ├── db_data/
│   │   └── *_details.jsonl
│   └── merge_results/         # 合并后的结果
│       ├── db_data/
│       └── merged_details.jsonl
```

### 性能结果说明

性能结果保存在{worker_dir}/performances/{model_abbr},合并结果默认路径为{worker_dir}/performances/merge_results，每个目录下的文件说明：
db_data: 数据库数据,保存了每个chunk返回的时间戳
*_details.jsonl: 详细结果,保存了每个请求的详细信息
*.csv: 性能结果,保存了case力度的性能结果，例如TTFT,TPOT,E2EL等
*.json: 性能结果,保存了全局力度的性能结果，例如Throughput，QPS等
详情可参考社区资料：https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/base_tutorials/results_intro/performance_metric.html#id2

## 注意事项

1. **唯一标识 `abbr`**：
   - 确保每个模型和数据集的 `abbr` 唯一
   - 相同 `abbr` 的组合会被认为是相同任务，可能被跳过

2. **配置文件路径**：
   - 确保 `multi_pod_benchmark.sh` 中的配置文件路径正确
   - 如果配置文件不在当前目录，需要修改脚本中的路径

3. **结果目录**：
   - `merge_performances.py` 会自动查找最新的输出目录
   - 确保在同一个 `outputs/default` 目录下运行所有Pod的测试

4. **压力测试参数**：
   - `--pressure-time` 建议设置为平均 E2EL 的三倍
   - 根据实际测试需求调整 `--max-num-workers` 参数

5. **合并顺序**：
   - 先运行所有Pod的基准测试
   - 再运行 `merge_performances.py` 合并结果
   - 最后运行 `ais_bench merge_config.py` 生成报告（可选）

## 故障排查

### 问题1：找不到输出目录

**错误信息**：`目录不存在: outputs/default`

**解决方案**：

- 确保已运行基准测试并生成了输出目录
- 检查当前工作目录是否正确

### 问题2：合并时跳过某些任务

**原因**：模型或数据集的 `abbr` 重复

**解决方案**：

- 检查配置文件中的 `abbr` 参数
- 确保每个模型和数据集的 `abbr` 唯一

### 问题3：压力测试时间过短

**现象**：测试结果不准确

**解决方案**：

- 增加 `--pressure-time` 参数值
- 建议设置为平均 E2EL 的三倍以上

## 参考文档

- [自定义配置文件运行AISBench](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/advanced_tutorials/run_custom_config.html)
- [AISBench 完整文档](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/index.html)

### 问题4： 支持的数据集类型,如何使用自定义数据集

- 示例中使用的是合成数据集，内容全为"A "，如果要使用自定义数据集，可以参考:https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/advanced_tutorials/custom_dataset.html#id4 准备自定义数据集，并使用ais_bench/benchmark/configs/datasets/custom/custom_qa_gen.py 或 ais_bench/benchmark/configs/datasets/custom/custom_qa_mcq.py 配置文件。