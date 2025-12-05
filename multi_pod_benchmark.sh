

echo "Starting benchmark"
ais_bench  ../demo_infer_vllm_api_perf_dev.py \ # 性能评测配置文件参考
-m perf \ # 运行模式，不要改
--max-num-workers 2 \ # 最大并行任务数，范围 [1, CPU 核数]，默认 1。
--pressure \ # 是否启用压力测试，默认 False。
--pressure-time 10 \ # 压力测试时间, 建议为平均e2el的三倍
echo "Merging performances"
python merge_performances.py
echo "Merging results"
ais_bench merge_config.py  -m perf --reuse # 合并多个pod的性能结果，如果不需要合并可注释
echo "Benchmark completed"