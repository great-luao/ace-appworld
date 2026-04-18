# Classifier Method NoteBook

## Project Overview

这一部分方法的目标是：在任务已经跑完、并且已经留下 `lm_calls.jsonl`、`environment_io.md`、`predicted_environment_io.md` 的前提下，额外引入一个 posthoc classifier agent，对每一次 code execution 的“预测环境输出 vs 实际环境输出”的差异做结构化判别。

它不是在线干预生成的 agent，也不是直接写错题本的 agent，而是一个中间分析模块。它的职责是：

- 将一次完整 trajectory 拆成逐 interaction 的判别单元；
- 对每个 interaction 标注 `primary_board` 和 `diff_category`；
- 保存分类结果，供后续统计分析和 playbook 生成使用；
- 在 task 级别聚合这些分类结果，分析它们与做题成败的关系。

当前设计上，classifier 只保留 posthoc 模式，不再与 evaluation/generation 做同步联动。
并且当前流程已经拆成两步：

- `appworld run PredictionDiffClassification`
  - 只生成每个 task 的 `prediction_diff_classification.jsonl`
- `python experiments/code/ace/prediction_diff_analysis.py <CONFIG_NAME> <DATASET_NAME...>`
  - 只读取已有 classification 结果并生成 `analysis/` 聚合统计

## Environment Setting

这部分逻辑完全基于已有实验输出文件工作，不重新执行环境，也不重新跑原始 agent。主要依赖：

- `experiments/outputs/<EXPERIMENT_NAME>/tasks/<TASK_ID>/logs/lm_calls.jsonl`
- `experiments/outputs/<EXPERIMENT_NAME>/tasks/<TASK_ID>/logs/environment_io.md`
- `experiments/outputs/<EXPERIMENT_NAME>/tasks/<TASK_ID>/logs/predicted_environment_io.md`
- `experiments/outputs/<EXPERIMENT_NAME>/tasks/<TASK_ID>/evaluation/report.md`

因此它对 GPU 没有依赖，主要消耗是 classifier 自身的 API 调用成本。

## Key Commands

- 独立运行 classifier:
  - `APPWORLD_PROJECT_PATH=/public/home/luao/LLM/ace/ace-appworld /public/home/luao/LLM/ace/.venv/bin/appworld run PredictionDiffClassification`
- 基于已有 classifier 输出做 analysis:
  - `/public/home/luao/LLM/ace/.venv/bin/python experiments/code/ace/prediction_diff_analysis.py PredictionDiffClassification train_difficulty_3`
  - `/public/home/luao/LLM/ace/.venv/bin/python experiments/code/ace/prediction_diff_analysis.py PredictionDiffClassification train_difficulty_3 train_difficulty_2`

## Method Logic

### Trajectory Reconstruction

classifier 不直接读取“最终答案”，而是先重建 interaction 级轨迹。

当前做法：

- 用 `environment_io.md` 作为 interaction 主轴，确定每个 task 一共有多少次环境交互；
- 用 `predicted_environment_io.md` 读取每步对应的预测输出；
- 用 `lm_calls.jsonl` 恢复每步生成当前代码前的 reasoning，以及 prediction call 的原始输出；
- 用 `classifier_lm_calls.jsonl` 保留 classifier 自己的完整 prompt 与响应，作为需要时的复盘入口；
- 对每一个 interaction，仅保留它之前的历史，不向 classifier 暴露未来信息。

换句话说，classifier 输入的是：

- prior trajectory
- current reasoning text
- current code
- predicted output
- actual output

而不是整个 task 的完整后验信息。

### Prompt Design

classifier prompt 的设计目标不是“解释整道题哪里错了”，而是“解释当前 interaction 的预测误差属于什么类型”。

### Statistical Analysis

analysis 单独执行后，会在 source experiment 下生成 `analysis/` 文件：

- `task_level_prediction_diff_summary.jsonl`
- `prediction_diff_stats.json`
- `prediction_diff_stats.md`

当前分析逻辑主要做：

- interaction 级分类结果聚合到 task 级；
- 统计每个 task 中各类 `primary_board` / `diff_category` 的计数；
- 将这些计数与 task 的 success/failure 对齐；
- 输出 failure rate、odds ratio、以及 count-vs-failure 的简单相关性指标。

这一部分的目标是先证明：
“某类 prediction diff 出现得越多，task 更容易失败”

之后再考虑把这类模式转写成 playbook。

注意：
- analysis 现在不再在 classifier run 结束后自动触发；
- analysis 的输入是 source experiment 下已经存在的 per-task classification 输出。

## Non-Obvious Patterns

- classifier 只在 `prediction-diff-classification` 独立 run_type 下工作；
- 当前版本不会自动接入原始 evaluation/generation loop；
- analysis 通过独立脚本执行，不修改原始 `appworld evaluate` 行为；
- analysis 脚本支持一次传入多个 dataset，会按顺序合并并去重 task ids 后生成一份综合统计；
- 如果限制了 `max_interactions_per_task`，输出文件会带 `.partial_*` 后缀，避免覆盖完整分类结果；
- `skip_existing_outputs` 不只是看文件是否存在，而是会检查 classifier 输出记录数是否与 `environment_io.md` 的 interaction 数一致；
- 当 classifier 自身 JSON 输出不规范时，代码里仍然有一层轻量 fallback，用启发式规则补出 `primary_board` / `diff_category`，避免整条 interaction 丢失。

## Current Constraints

- 目前 `predicted_output` 只向 classifier 暴露单字段，不再区分 raw/clipped 两种输入视图；
- `other` 类别已经加入，但 few-shot 对 `other` 的覆盖仍然较弱；
- 当前统计分析仍然是较轻量的相关性分析，还没有进入更系统的因果或分层分析；
- classifier 目前主要服务于方法研究，不直接回流到 generator prompt。

## Key Files

- `experiments/code/ace/prediction_diff_classifier.py`
  - classifier 主逻辑：轨迹重建、interaction 分类、jsonl 落盘、task-level aggregation、stats 生成。
- `experiments/prompts/prediction_diff_classifier_prompt.txt`
  - classifier prompt：taxonomy、tie-break rules、few-shot、输入格式、输出格式。
- `experiments/code/ace/run.py`
  - `prediction-diff-classification` run_type 的注册与分发入口。
- `experiments/code/ace/prediction_diff_analysis.py`
  - 独立 analysis 入口：读取 config 和 dataset，聚合已有 classification 结果并写回 `analysis/`。
- `experiments/configs/PredictionDiffClassification.jsonnet`
  - 当前 classifier 的主配置入口。
