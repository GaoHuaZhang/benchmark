## 🌟 亮点

1. **架构重构**：对cli、models、inferencer、tasks组件进行了重构，支持快速接入新的测试基准。
2. **任务管理界面**：新的任务UI管理界面，支持同时监控每个任务的详细执行状态，包括任务名称、进度、时间成本、状态、日志路径、扩展参数等。
3. **并行执行**：扩展了多任务并行功能，支持多个性能或精度测评任务并行执行。
4. **新增测评基准**：[docvqa](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/docvqa/README.md)、[infovqa](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/infovqa/README.md)、[ocrbench_v2](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/ocrbench_v2/README.md)、[omnidocbench](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/omnidocbench/README.md)、[mmmu](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/mmmu/README.md)、[mmmu_pro](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/mmmu_pro/README.md)、[mmstar](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/mmstar/README.md)、[mm_custom](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/advanced_tutorials/custom_dataset.html#id3)、[videomme](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/videomme/README.md)、[FewCLUE_bustm](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_bustm/README.md)、[FewCLUE_chid](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_chid/README.md)、[FewCLUE_cluewsc](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_cluewsc/README.md)、[FewCLUE_csl](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_csl/README.md)、[FewCLUE_eprstmt](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_eprstmt/README.md)、[FewCLUE_tnews](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_tnews/README.md)、[dapo_math](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/dapo_math/README.md)、[leval](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/leval/README.md)
5. **新增模型**：新增vllm/vllm ascend VL 离线推理模型

---

## 🚀 新特性

### 数据集

- 数据集：新增OcrBench-v2。([#35](https://github.com/AISBench/benchmark/pull/35))
- 数据集：新增video-mme。([#34](https://github.com/AISBench/benchmark/pull/34))
- 数据集：新增MMStar。([#31](https://github.com/AISBench/benchmark/pull/31))
- 数据集：新增DAPO-math-17k。([#13](https://github.com/AISBench/benchmark/pull/13))
- 数据集：新增InfoVQA和DocVQA。([Gitee #299](https://gitee.com/aisbench/benchmark/pulls/299))
- 数据集：新增MMMU。([Gitee #291](https://gitee.com/aisbench/benchmark/pulls/291))
- 数据集：新增leval，支持精度测评及性能测评。([Gitee #284](https://gitee.com/aisbench/benchmark/pulls/284))([Gitee #283](https://gitee.com/aisbench/benchmark/pulls/283))([Gitee #282](https://gitee.com/aisbench/benchmark/pulls/282))([Gitee #281](https://gitee.com/aisbench/benchmark/pulls/281))([Gitee #280](https://gitee.com/aisbench/benchmark/pulls/280))
- 数据集：新增OmniDocBench。([Gitee #209](https://gitee.com/aisbench/benchmark/pulls/209))

### 模型

- 模型：新增vllm/vllm ascend VL 离线推理模型。([#26](https://github.com/AISBench/benchmark/pull/26))

### 功能

- 功能：`--num-prompts`参数适配精度场景，支持精度模式下指定前n条数据进行推理。([#25](https://github.com/AISBench/benchmark/pull/25))
- 功能：新增模型配置参数，包括流式推理开关`stream`、自定义URL路径`url`、自定义API key`api_key`。([#4](https://github.com/AISBench/benchmark/pull/4))
- 功能：api模型推理新增warmup功能。([Gitee #195](https://gitee.com/aisbench/benchmark/pulls/195))
- 功能：支持自定义多模态数据集性能测评。([Gitee #279](https://gitee.com/aisbench/benchmark/pulls/279))
- 功能：部分数据集支持服务化PPL（混淆度）测评。([Gitee #275](https://gitee.com/aisbench/benchmark/pulls/275))
- 功能：新增readthedocs文档支持。([Gitee #179](https://gitee.com/aisbench/benchmark/pulls/179))
- 功能：新增任务管理器对任务执行状态进行监控。([Gitee #165](https://gitee.com/aisbench/benchmark/pulls/165))

---

## 🐛 问题修复

- 修复：合并数据集推理模式（--merge-ds）下，think内容无法通过`extract_non_reasoning_content`去除的问题。([Gitee #161](https://gitee.com/aisbench/benchmark/pulls/161))
- 修复：livecodebench由于多进程嵌套导致可能出现的死锁问题。([Gitee #144](https://gitee.com/aisbench/benchmark/pulls/144))

---

## ⚙️ 优化与重构

- 重构：BFCL V3测评，支持保存更多的推理过程信息。([Gitee #287](https://gitee.com/aisbench/benchmark/pulls/287))
- 重构：合并数据集推理（--merge-ds），合并模式下保存结果包含各个子数据集的推理结果。([Gitee #198](https://gitee.com/aisbench/benchmark/pulls/198))
- 重构：性能详情数据的保存格式为.db格式，实现性能测评结果实时落盘，保障任务中断数据完整性。([Gitee #197](https://gitee.com/aisbench/benchmark/pulls/197))
- 重构：多轮对话推理任务，多轮对话与模型实现解耦。([Gitee #196](https://gitee.com/aisbench/benchmark/pulls/196))
- 重构：性能测评calculator实现，过滤并发波动导致的稳态被判定为退出的逻辑。([Gitee #186](https://gitee.com/aisbench/benchmark/pulls/186))
- 重构：prompttemplate实现，支持自定义扩展。([Gitee #178](https://gitee.com/aisbench/benchmark/pulls/178))
- 重构：合成数据集和自定义数据集，统一数据集配置方式，支持同时指定多个不同配置的合成数据集和自定义数据集任务。([Gitee #175](https://gitee.com/aisbench/benchmark/pulls/175))
- 重构：outputhandler，支持根据inferencer需要自定义扩展。([Gitee #172](https://gitee.com/aisbench/benchmark/pulls/172))
- 重构：删除models的冗余clients组件依赖。([Gitee #158](https://gitee.com/aisbench/benchmark/pulls/158))

---

## 🏗️ 基础设施重构

- 基础设施：重构local models组件，定义`batch_inference`执行推理业务，提高可拓展性。([Gitee #207](https://gitee.com/aisbench/benchmark/pulls/207))
- 基础设施：重构api models组件，流式和非流式实现归一，通过`stream`参数指定推理模式，同时抽象公共接口，方便快速接入新模型后端。([Gitee #171](https://gitee.com/aisbench/benchmark/pulls/171))
- 基础设施：重构inferencer组件，根据调用的models类比(api_models和local_models)不同，采用不同的推理方式。采用多进程+协程的调用方式，提高并发能力。测试结果数据格式`json` -> `jsonl`降低IO压力，提高数据保存效率。([Gitee #170](https://gitee.com/aisbench/benchmark/pulls/170))
- 基础设施：重构infer task组件，将多进程并发能力从inferencer迁移到task层级，将请求流控和进度监控采用独立的模块进行实现。([Gitee #169](https://gitee.com/aisbench/benchmark/pulls/169))
- 基础设施：重构命令行和工作流执行控制管道。([Gitee #167](https://gitee.com/aisbench/benchmark/pulls/167))
- 基础设施：统一api_runner和local_runner实现。([Gitee #157](https://gitee.com/aisbench/benchmark/pulls/157))
- 基础设施：采用错误码对错误信息进行统一管理，通过url快速查看解决方案。([Gitee #150](https://gitee.com/aisbench/benchmark/pulls/150))

---

## 🔄 CI/CD 优化

- CI/CD：MR自动化执行UT用例。([Gitee #301](https://gitee.com/aisbench/benchmark/pulls/301))
