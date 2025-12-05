

echo "Starting benchmark"
ais_bench  ./demo_infer_vllm_api_perf_dev.py \
-m perf \
--max-num-workers 2
# --pressure
# --pressure-time 10
echo "Merging performances"
python merge_performances.py
echo "Merging results"
ais_bench merge_config.py  -m perf --reuse # 合并多个pod的性能结果，如果不需要合并可注释
echo "Benchmark completed"