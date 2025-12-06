from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="/home/theone/weight/DeepSeek-R1",
        model="qwen",
        stream=True,
        request_rate=100,
        traffic_cfg=dict(
        burstiness=0.5, # 突发性因子，默认为0，表示不突发，取值范围为0到1，1表示突发性最大，0表示不突发
        ramp_up_strategy="linear", # 爬升策略，默认为"linear"，表示线性爬升，可选值为"linear"和"exponential"
        ramp_up_start_rps=10, # 爬升起始速率，默认为0，表示不爬升
        ramp_up_end_rps=200, # 爬升终止速率，默认为0，表示不爬升
        ),
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=1000,
        trust_remote_code=False,
        generation_kwargs=dict(
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
