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