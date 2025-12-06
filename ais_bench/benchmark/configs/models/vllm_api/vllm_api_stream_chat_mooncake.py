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
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=500,
        trust_remote_code=False,
        generation_kwargs=dict(
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
