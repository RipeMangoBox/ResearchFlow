---
name: code-context-paper-retrieval
description: Retrieves code-context-relevant papers from the local knowledge base in two modes (brief/deep), with fixed order paperCollection -> paperAnalysis -> optional paperPDFs. Environment detection prioritizes codebase environment files; asks user if none found. Trigger timing: BEFORE code modification, not after.
---

# Code Context Paper Retrieval

## What this skill does

面向"代码实现 / 改造 / 规划"场景，从本地论文知识库中给出与当前代码上下文最相关的论文证据。

固定检索顺序：

1. `paperCollection/`（索引层，先召回）
2. `paperAnalysis/`（分析层，补全方法与证据）
3. `paperPDFs/`（仅在必要时建议进一步阅读）

输出两档：

- `brief`：必看论文 3-5 篇 + 对应 analysis note + 是否建议读 PDF
- `deep`：更完整候选 + 对比理由 + 落地建议

## Environment Detection（执行前必须完成）

在运行任何脚本之前，必须先确定 Python 环境。按以下优先级查找：

1. **目标 codebase 内的环境文件**（优先级从高到低）：
   - `environment.yml` / `environment.yaml`（conda）
   - `requirements.txt` / `requirements*.txt`
   - `pyproject.toml`（查看 `[project.dependencies]` 或 `[tool.poetry.dependencies]`）
   - `setup.py` / `setup.cfg`
2. **从环境文件推断 conda env 名称**：读取 `environment.yml` 的 `name:` 字段，检查该 conda env 是否已存在（`conda env list`）。
3. **如果以上均未找到或无法确认**：**必须主动询问用户**，说明未找到环境文件，请用户提供：
   - conda 环境名称，或
   - Python 解释器路径，或
   - 确认使用系统默认 Python

**禁止行为**：不得在未确认环境的情况下自行猜测或随意使用 `python3` / `python` 执行脚本。

## Triggering

### 核心原则：改代码之前检索，不是改完之后

当 agent 收到涉及代码修改的任务时（如"实现 X 模块"、"重构 Y 的 loss 函数"、"加入 Z 机制"），应在动手写代码之前判断是否需要检索论文支撑。

### 建议式触发（推荐）

agent 在以下场景中，**执行代码修改前**主动询问用户：

> "检测到本次任务涉及 [motion diffusion / attention mechanism / ...]，本地知识库中可能有相关论文支撑。是否先检索？（brief / deep / 跳过）"

触发条件（满足任一即可）：
- 用户任务描述中包含与 `paperCollection/by_task/` 或 `by_technique/` 匹配的关键词
- 用户要修改的文件路径涉及模型核心模块（如 `model/`、`network/`、`loss/`、`train/` 等）
- 用户明确提到某个方法名或论文名

不触发条件（避免打扰）：
- 纯 bug fix、格式调整、注释修改
- 配置文件 / 脚本参数修改
- 用户已在本轮对话中跳过过一次检索建议

### 显式触发

- 用户直接调用：`/code-context-paper-retrieval`
- 或在对话中说"帮我查一下相关论文"、"有没有相关的 paper"等

### 脚本辅助（可选）

```bash
# 使用已确认的 conda 环境
conda run -n <env_name> python ".claude/skills/code-context-paper-retrieval/scripts/code_context_paper_retrieval/query_code_context_papers.py" --mode brief --query "motion diffusion training pipeline"
```

## Inputs

- `--mode brief|deep`
- `--query "..."`（可选，代码任务描述）
- `--changed-file <path>`（可多次）
- `--top-k`（deep 候选上限）

## Minimal workflow

1. **环境检测**（见上方 Environment Detection 节）。
2. 从用户任务描述 / 目标文件提取关键词。
3. 在 `paperCollection/_AllPapers.md`、`by_task/*.md`、`by_technique/*.md` 第一轮召回。
4. 读取命中论文对应 `paperAnalysis/**/*.md` 的 `core_operator`、`primary_logic`、TL;DR 摘要。
5. 输出 brief/deep 结果，并在证据不足时建议进一步阅读 PDF。
6. **用户确认后，再开始代码修改。**

## Output contract

### brief

- 3-5 篇必看论文
- 每篇包含：analysis 路径、方法线索、是否建议读 PDF（含一句理由）

### deep

- 更完整候选列表（可由 `--top-k` 调整）
- 每篇包含：analysis 路径、core_operator/primary_logic 线索、证据摘要、PDF 建议
- 结尾给出简要对比与落地建议

## Non-goals

- 不在代码修改完成后才触发（那时已经晚了）
- 不对非核心改动（bug fix、格式、配置）触发检索建议
- 不在用户已跳过的同一轮对话中重复建议
