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
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
        ),
    )
]
