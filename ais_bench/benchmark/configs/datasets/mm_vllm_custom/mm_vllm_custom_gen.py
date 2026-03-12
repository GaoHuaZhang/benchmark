"""vLLM-style multimodal custom dataset gen config (prompt + images + audios)."""
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMVllmCustomDataset, MMVllmCustomEvaluator


mm_vllm_custom_reader_cfg = dict(
    input_columns=['prompt', 'images', 'audios'],
    output_column='answer'
)


mm_vllm_custom_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "{image}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "{audio}"}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mm_vllm_custom_eval_cfg = dict(
    evaluator=dict(type=MMVllmCustomEvaluator)
)

mm_vllm_custom_datasets = [
    dict(
        abbr='mm_vllm_custom',
        type=MMVllmCustomDataset,
        path='/home/zhanggaohua/code/dev/benchmark/ais_bench/datasets/mm_vllm_custom/mm_vllm_custom.jsonl',
        mm_type='path',
        base_path=None,
        reader_cfg=mm_vllm_custom_reader_cfg,
        infer_cfg=mm_vllm_custom_infer_cfg,
        eval_cfg=mm_vllm_custom_eval_cfg
    )
]
