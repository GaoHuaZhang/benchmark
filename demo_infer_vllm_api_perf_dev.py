from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as vllm_api_stream_chat,
    )
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )
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
synthetic_datasets_test1 = [synthetic_datasets[0].deepcopy()]
synthetic_datasets_test1[0]['abbr'] = 'synthetic-test1' 
synthetic_datasets_test1[0]['config'] = synthetic_config_test1
synthetic_datasets_test2 = [synthetic_datasets[0].deepcopy()]
synthetic_datasets_test2[0]['abbr'] = 'synthetic-test2'
synthetic_datasets_test2[0]['config'] = synthetic_config_test2


datasets = synthetic_datasets_test1 + synthetic_datasets_test2  # 指定数据集列表


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
model_dataset_combinations = [
    dict(models=models_test1, datasets=synthetic_datasets_test1),
    dict(models=models_test2, datasets=synthetic_datasets_test2),
]
