---
name: research-question-bank
description: Given a research direction and rough task description, generates a structured question/challenge list grounded in local KB and web search, covering domain open problems, top-venue reviewer concerns, and actionable research cuts. Saves output to QuestionBank/ as a reusable Markdown file. Use when the user wants to map out the question landscape for a research area before committing to a specific idea.
---

# Research Question Bank

## Purpose

用户给出一个研究方向和粗略任务描述，agent 通过检索本地论文知识库（`paperCollection` + `paperAnalysis`）和 web 搜索，生成该领域的结构化问题/挑战清单，覆盖：

1. 领域核心开放问题
2. 顶会审稿人关注视角
3. 可执行的研究切口

产出保存到 `QuestionBank/` 目录，作为长期可复用的问题资产。

## Positioning

- 这是一个**问题发现**工具，不是 idea 生成工具（那是 `research-brainstorm-from-kb`）
- 不是审稿模拟（那是 `reviewer-stress-test`）
- 定位：idea 之前的上游——先搞清楚"这个方向有哪些值得问的问题"，再决定做哪个

## Input

用户提供：
- **研究方向**：如"human motion generation"、"multi-modal interaction"、"video diffusion evaluation"
- **粗略任务描述**（可选）：如"想做 text-to-motion 的评测改进"、"想探索 interaction-aware generation"
- **目标 venue**（可选）：如 CVPR、ICLR、SIGGRAPH（影响审稿人视角的侧重）

## Knowledge Protocol

1. **KB-first**：通过 `papers-query-knowledge-base` 检索本地知识库
   - 按 `by_task/` 定位该方向已有论文
   - 按 `by_technique/` 识别主流技术路线
   - 读取相关 `paperAnalysis/` 的 `core_operator`、`primary_logic` 提取方法共性与差异
2. **Web supplement**：搜索该方向最新进展
   - 近 1-2 年顶会 accepted papers 趋势
   - 知名团队的最新工作方向
   - 公开的 reviewer comments / OpenReview discussions（如有）
3. **综合生成**：基于 KB 证据 + web 信息，生成问题清单

## Output Structure

生成的 Markdown 文件遵循以下结构：

```markdown
---
created: {{ISO_DATE}}
updated: {{ISO_DATE}}
direction: {{研究方向}}
tags:
  - question-bank
  - {{方向标签}}
---

# Question Bank: {{研究方向}}

> 基于 paperCollection + paperAnalysis 本地知识库检索与 web 搜索生成。

## 一、领域核心开放问题

按子领域/子任务分组，每个问题附：
- 问题陈述（1-2 句）
- 为什么重要（1 句）
- 当前进展概况（引用 KB 中的论文或 web 来源）

## 二、顶会审稿人关注视角

按审稿维度组织（参考目标 venue 的评审标准）：

### 2.1 Novelty & Non-triviality
- 审稿人会问的典型问题
- 该方向常见的"看似新实则不新"陷阱

### 2.2 Technical Soundness
- 该方向常见的技术漏洞
- 容易被质疑的假设

### 2.3 Experimental Rigor
- 该方向的 baseline 共识与争议
- 评测指标的已知缺陷
- 常被要求的 ablation 类型

### 2.4 Significance & Impact
- 审稿人对该方向的"疲劳点"（什么类型的工作已经太多）
- 什么样的贡献更容易被认可

## 三、可执行研究切口

基于上述问题，给出 3-5 个具体的研究切口建议：
- 切口描述（1-2 句）
- 对应的核心问题（引用上方编号）
- 预估难度与所需资源
- 最小验证方案

## 四、问题模板（可复用）

提供 5-7 个通用问题模板，用户未来遇到新想法时可直接套用。
```

## File Naming & Storage

- 目录：`QuestionBank/`（位于仓库根目录）
- 文件名：`YYYY-MM-DD_<topic>.md`
  - `YYYY-MM-DD`：创建日期，如 `2026-04-08`
  - `<topic>`：英文小写 + 下划线，如 `motion_generation`、`video_diffusion_evaluation`
  - 完整示例：`2026-04-08_motion_generation.md`
- 如果同日同 topic 文件已存在：追加新的分节，不覆盖已有内容

## Workflow

1. 接收用户的方向和任务描述
2. 通过 `papers-query-knowledge-base` 检索本地 KB
3. Web 搜索补充最新进展和审稿人视角
4. 按 Output Structure 生成问题清单
5. 写入 `QuestionBank/` 目录
6. 向用户汇报：生成了多少问题、覆盖了哪些维度、建议优先关注哪几个

## Boundaries

- 不生成具体的 idea 或方案（那是 `research-brainstorm-from-kb` 和 `idea-focus-coach` 的职责）
- 不做审稿模拟打分（那是 `reviewer-stress-test` 的职责）
- 不依赖特定项目代码或实验结果
- 问题清单应保持方向级别的通用性，不绑定到用户的某个具体项目
