from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MooncakeTraceDataset, MooncakeTraceEvaluator
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator


mooncake_trace_reader_cfg = dict[str, list[str] | str](
    input_columns=["prompt", "timestamp","max_out_len"],
    output_column="answer"
)


mooncake_trace_infer_cfg = dict(
        prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mooncake_trace_eval_cfg = dict(
    evaluator=dict(type=MooncakeTraceEvaluator)
)

mooncake_trace_datasets = [
    dict(
        abbr='mooncake-trace',
        type=MooncakeTraceDataset,
        path='/home/theone/code/dev/benchmark/synthetic_trace_13.jsonl', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        prefix_ratio=0.4,
        generated_prompts_path='', # 生成的prompt缓存路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=mooncake_trace_reader_cfg,
        infer_cfg=mooncake_trace_infer_cfg,
        eval_cfg=mooncake_trace_eval_cfg
    )
]