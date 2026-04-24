# AGENTS.md - Execution NoteBook for ACE + AppWorld

## Project Overview

这是一个 Python 3.12 的 AppWorld + ACE 实验仓库：实验由 `experiments/configs/*.jsonnet` 驱动，运行时会动态加载 `experiments/code/*/run.py`，并把每个 task 的执行状态、环境反馈与模型调用日志落盘到 `experiments/outputs/`，其“非标准点”在于 agent loop 与环境执行器（`AppWorld.execute`）是松耦合的可插拔流水线。我们会在这个框架上实现新论文相关的实验代码。

此外，仓库根目录下维护了 `METHODS/` 文件夹，用来持续记录我们论文方法设计的不同板块。它不是代码入口，而是方法学笔记区。在进行跟科研相关的代码写作前，请先阅读 `METHODS/AGENTS.md`，了解项目结构、代码风格、测试规则和边界条件等关键信息。

## Environment Setting

这部分代码在学校的集群上，有时候会登陆带 GPU 的节点，有时候在共享节点。当前任务主要是 API 调用与代码执行链路实验，不依赖 GPU；共享节点也能跑。

## Key Commands

- 激活环境: `source /public/home/luao/LLM/ace/.venv/bin/activate.fish`
- 执行脚本: 用 `python` 而不是 `uv run`
- 列出可运行配置: `appworld run options`
- 运行实验: `appworld run <CONFIG_NAME>`
- 评测结果: `appworld evaluate <CONFIG_NAME> <DATASET_NAME>`
- 单任务调试: `appworld run <CONFIG_NAME> --task-id <TASK_ID>`

## Project Structure

- `src/appworld/`: AppWorld 环境核心（任务加载、执行器、评测器、CLI）。
- `experiments/code/ace/`: 目前主要实验框架（ReAct/ACE agent、LLM 封装、run 入口）。
- `experiments/configs/`: jsonnet 实验配置（模型参数、prompt 路径、dataset、run_type）。
- `experiments/prompts/`: 主 prompt 与辅助注入 prompt。
- `data/datasets/`: 数据集 task id 列表（可自定义子集）。
- `experiments/outputs/`: 每次实验输出（logs、dbs、evaluation 报告）。
- `experiments/skillbanks/`: 运用我们方法收集到的skillbank json文件。
- `METHODS/`: 方法设计文档区；按主题拆分记录论文方法、算法设定、prompt 方案、分析维度与后续待验证假设。

## Code Style

代表性模式（每步先让模型产出代码，再由 `AppWorld` 执行并把输出回灌历史）：

```python
output = self.language_model.generate(messages=messages)
code, fixed_output_content = self.extract_code_and_fix_content(output["content"])
self.messages.append({"role": "assistant", "content": fixed_output_content + "\n\n"})
execution_outputs = [
    ExecutionIO(content=world.execute(execution_input.content), metadata=execution_input.metadata)
    for execution_input in execution_inputs
]
```

- 简单明了，多利用python的特性。
- 除非事先声明，否则不需要为一个功能设计backup。
- 有必要的话，可以修改原始库中的核心代码。
- 新增实验逻辑优先放在 `experiments/code/ace`，尽量不改 `src/appworld` 的通用能力。
- 代码和功能不是越多越好！尽量维持一个轻量级的代码库，少写各类fallback以及异常处理，发生异常让代码自然报错即可，我们之后再针对报错进行修复和完善。
- 如果需要的话，可以引入新的库，但需要提醒用户或自行安装。

## Testing Rules

- 参考原始库的READEME.md

## Boundaries

### ✅ Allowed without asking
- Read files, list directory contents
- Run lint, typecheck, single test files

### ⚠️ Ask first
- Install or remove packages
- Delete files
- Push to git or open PRs

### 🚫 Never
- Commit secrets, `.env` files, or credentials

### Special Cases
- 因为我们的实验以调用api为主，api可能不稳定，若发现输出显示api在不断尝试重连的类似信息，则取消实验并向用户汇报。

## Key Files

- `README.md`: 环境安装与官方实验运行说明。
- `src/appworld/cli.py`: `appworld run/evaluate` 入口。
- `src/appworld/environment.py`: 代码执行与 `environment_io.md` 写入逻辑。
- `experiments/code/ace/run.py`: run_type 到 agent 的分发入口。
- `experiments/code/ace/base_agent.py`: non-ACE 任务循环与预测日志落盘。
- `experiments/code/ace/base_react.py`: ReAct 主循环与 output prediction injection 逻辑。
