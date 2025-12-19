## 🌟 Highlights

1. **Architecture Refactoring**: Refactored cli, models, inferencer, and tasks components to support rapid integration of new test benchmarks.
2. **Task Management Interface**: New task UI management interface that supports simultaneous monitoring of detailed execution status for each task, including task name, progress, time cost, status, log path, extended parameters, etc.
3. **Parallel Execution**: Extended multi-task parallel functionality, supporting parallel execution of multiple performance or accuracy evaluation tasks.
4. **New Evaluation Benchmarks**: [docvqa](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/docvqa/README.md), [infovqa](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/infovqa/README.md), [ocrbench_v2](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/ocrbench_v2/README.md), [omnidocbench](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/omnidocbench/README.md), [mmmu](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/mmmu/README.md), [mmmu_pro](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/mmmu_pro/README.md), [mmstar](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/mmstar/README.md), [mm_custom](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/advanced_tutorials/custom_dataset.html#id3), [videomme](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/videomme/README.md), [FewCLUE_bustm](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_bustm/README.md), [FewCLUE_chid](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_chid/README.md), [FewCLUE_cluewsc](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_cluewsc/README.md), [FewCLUE_csl](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_csl/README.md), [FewCLUE_eprstmt](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_eprstmt/README.md), [FewCLUE_tnews](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_tnews/README.md), [dapo_math](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/dapo_math/README.md), [leval](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/leval/README.md)
5. **New Models**: Added vllm/vllm ascend VL offline inference model

---

## 🚀 New Features

### Datasets

- Dataset: Added OcrBench-v2. ([#35](https://github.com/AISBench/benchmark/pull/35))
- Dataset: Added video-mme. ([#34](https://github.com/AISBench/benchmark/pull/34))
- Dataset: Added MMStar. ([#31](https://github.com/AISBench/benchmark/pull/31))
- Dataset: Added DAPO-math-17k. ([#13](https://github.com/AISBench/benchmark/pull/13))
- Dataset: Added InfoVQA and DocVQA. ([Gitee #299](https://gitee.com/aisbench/benchmark/pulls/299))
- Dataset: Added MMMU. ([Gitee #291](https://gitee.com/aisbench/benchmark/pulls/291))
- Dataset: Added leval, supporting accuracy evaluation and performance evaluation. ([Gitee #284](https://gitee.com/aisbench/benchmark/pulls/284))([Gitee #283](https://gitee.com/aisbench/benchmark/pulls/283))([Gitee #282](https://gitee.com/aisbench/benchmark/pulls/282))([Gitee #281](https://gitee.com/aisbench/benchmark/pulls/281))([Gitee #280](https://gitee.com/aisbench/benchmark/pulls/280))
- Dataset: Added OmniDocBench. ([Gitee #209](https://gitee.com/aisbench/benchmark/pulls/209))

### Models

- Model: Added vllm/vllm ascend VL offline inference model. ([#26](https://github.com/AISBench/benchmark/pull/26))

### Features

- Feature: `--num-prompts` parameter adapted for accuracy scenarios, supporting inference on the first n data items in accuracy mode. ([#25](https://github.com/AISBench/benchmark/pull/25))
- Feature: Added model configuration parameters, including streaming inference switch `stream`, custom URL path `url`, and custom API key `api_key`. ([#4](https://github.com/AISBench/benchmark/pull/4))
- Feature: Added warmup functionality for API model inference. ([Gitee #195](https://gitee.com/aisbench/benchmark/pulls/195))
- Feature: Support for custom multimodal dataset performance evaluation. ([Gitee #279](https://gitee.com/aisbench/benchmark/pulls/279))
- Feature: Some datasets support service-based PPL (perplexity) evaluation. ([Gitee #275](https://gitee.com/aisbench/benchmark/pulls/275))
- Feature: Added readthedocs documentation support. ([Gitee #179](https://gitee.com/aisbench/benchmark/pulls/179))
- Feature: Added task manager to monitor task execution status. ([Gitee #165](https://gitee.com/aisbench/benchmark/pulls/165))

---

## 🐛 Bug Fixes

- Fix: Fixed issue where think content could not be removed via `extract_non_reasoning_content` in merged dataset inference mode (--merge-ds). ([Gitee #161](https://gitee.com/aisbench/benchmark/pulls/161))
- Fix: Fixed potential deadlock issue in livecodebench caused by nested multi-process operations. ([Gitee #144](https://gitee.com/aisbench/benchmark/pulls/144))

---

## ⚙️ Optimizations and Refactoring

- Refactor: BFCL V3 evaluation, supporting saving more inference process information. ([Gitee #287](https://gitee.com/aisbench/benchmark/pulls/287))
- Refactor: Merged dataset inference (--merge-ds), merged mode saves results including inference results from each sub-dataset. ([Gitee #198](https://gitee.com/aisbench/benchmark/pulls/198))
- Refactor: Performance detail data saved in `.db` format, enabling real-time persistence of performance evaluation results, ensuring data integrity when tasks are interrupted. ([Gitee #197](https://gitee.com/aisbench/benchmark/pulls/197))
- Refactor: Multi-turn dialogue inference tasks, decoupling multi-turn dialogue from model implementation. ([Gitee #196](https://gitee.com/aisbench/benchmark/pulls/196))
- Refactor: Performance evaluation calculator implementation, filtering logic where steady state caused by concurrency fluctuations is incorrectly judged as exit. ([Gitee #186](https://gitee.com/aisbench/benchmark/pulls/186))
- Refactor: PromptTemplate implementation, supporting custom extensions. ([Gitee #178](https://gitee.com/aisbench/benchmark/pulls/178))
- Refactor: Synthetic datasets and custom datasets, unified dataset configuration approach, supporting simultaneous specification of multiple synthetic and custom dataset tasks with different configurations. ([Gitee #175](https://gitee.com/aisbench/benchmark/pulls/175))
- Refactor: OutputHandler, supporting custom extensions based on inferencer needs. ([Gitee #172](https://gitee.com/aisbench/benchmark/pulls/172))
- Refactor: Removed redundant clients component dependency from models. ([Gitee #158](https://gitee.com/aisbench/benchmark/pulls/158))

---

## 🏗️ Infrastructure Refactoring

- Infrastructure: Refactored local models component, defined `batch_inference` to execute inference business, improving extensibility. ([Gitee #207](https://gitee.com/aisbench/benchmark/pulls/207))
- Infrastructure: Refactored API models component, unified streaming and non-streaming implementations, specified inference mode through `stream` parameter, and abstracted common interfaces for rapid integration of new model backends. ([Gitee #171](https://gitee.com/aisbench/benchmark/pulls/171))
- Infrastructure: Refactored inferencer component, adopting different inference methods based on the model type called (api_models and local_models). Adopted multi-process + coroutine calling approach to improve concurrency. Changed test result data format from `json` to `jsonl` to reduce IO pressure and improve data saving efficiency. ([Gitee #170](https://gitee.com/aisbench/benchmark/pulls/170))
- Infrastructure: Refactored infer task component, migrated multi-process concurrency capability from inferencer to task level, implemented request flow control and progress monitoring as independent modules. ([Gitee #169](https://gitee.com/aisbench/benchmark/pulls/169))
- Infrastructure: Refactored command-line and workflow execution control pipeline. ([Gitee #167](https://gitee.com/aisbench/benchmark/pulls/167))
- Infrastructure: Unified api_runner and local_runner implementations. ([Gitee #157](https://gitee.com/aisbench/benchmark/pulls/157))
- Infrastructure: Adopted error codes for unified error information management, enabling quick access to solutions via URL. ([Gitee #150](https://gitee.com/aisbench/benchmark/pulls/150))

---

## 🔄 CI/CD Optimizations

- CI/CD: Automated UT case execution for MR. ([Gitee #301](https://gitee.com/aisbench/benchmark/pulls/301))

