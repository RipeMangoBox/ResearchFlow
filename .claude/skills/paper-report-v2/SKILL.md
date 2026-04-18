---
name: paper-report-v2
follows: rf-obsidian-markdown
description: Generates deep structured reports for single papers with formula derivation, pipeline module decomposition, experiment analysis, and recursive related work. Use when the user asks for a "deep report", "V2 report", "detailed paper analysis", or "公式推导" for a specific paper. Requires a PDF path or paper title as input.
---

# PaperReportV2 — Deep Structured Paper Report

Generate a comprehensive, structured deep report for a single paper. This goes beyond the standard analysis note (`papers-analyze-pdf`) by requiring:

1. **Metadata table** with acceptance status, team info, code repos
2. **Core insight** as a single sentence: "因为X → 在Y上改了Z → 取得W效果"
3. **Motivation figure** or constructed example
4. **Pipeline decomposition** — each module mapped to the architecture figure
5. **Formula step-by-step derivation** — aligned with baseline formula
6. **Experiment analysis** — not just numbers, but why and failure cases
7. **Recursive related work** — build knowledge graph edges
8. **Knowledge base positioning** — where this paper fits

---

## Input

- A PDF path (local file), or
- A paper title + metadata (if already in the knowledge base)

## Output

- A Markdown report written to `paperAnalysis/<Category>/<Venue_Year>/<Year>_<Title>_report_v2.md`
- Suffix `_report_v2` distinguishes from the standard analysis note

---

## Report Template

The report MUST follow this exact structure. Every section is required.

```markdown
# {论文标题}

## 1. 元数据

| 属性 | 值 |
|------|-----|
| 中文题名 | {title_zh — translate if needed} |
| 英文题名 | {title} |
| 发表状态 | {accepted / under_review / preprint} |
| 会议/期刊 | {venue} ({acceptance_type}: oral/poster/spotlight, if known) |
| arXiv | [{arxiv_id}](https://arxiv.org/abs/{arxiv_id}) |
| DOI | {doi} |
| OpenReview | {openreview_link, if available} |
| 官方代码 | [{code_url}]({code_url}) ⭐{stars, if known} |
| 非官方代码 | {if any} |
| 作者 | {Author1 (Affiliation), Author2 (Affiliation), ...} |
| 团队/实验室 | {team_name_zh} ({institution}) |
| 主要任务 | {primary tasks from taxonomy} |
| 主要 baseline | {baseline methods} |

## 2. 核心观点

> 因为「{具体问题描述}」，作者在「{baseline method(s)}」的基础上
> 改进了「{具体改进的机制/模块}」，从而在「{任务/数据集}」上
> 取得了「{具体效果: +X.X% on benchmark}」。

## 3. 动机 / Demo / Figure 1

{如果论文有 motivation figure / demo / 效果对比图 (通常是 Figure 1):}
![Figure 1: {caption}]({figure_url_or_local_path})

**问题可视化**: {解释图中每个元素、对应的问题是什么}

**为什么现有方法不够**: {用图中的对比说明}

{如果论文没有动机图:}
**具体例子**: {构造一个具体场景来说明问题}
{例如: "给定一段10分钟的长视频和问题'谁在第3分钟打开了窗户？', 现有方法X因为Y而无法回答，因为..."}

## 4. Pipeline 架构

![Pipeline]({pipeline_figure_url_or_path})

{pipeline 图的整体描述}

### 模块 A: {名称} (对应 Pipeline 图中 {位置标注})
- **输入**: {具体输入格式和来源}
- **输出**: {具体输出格式}
- **作用**: {这个模块做了什么}
- **对应 baseline 的哪一部分**: {baseline 中对应的模块}
- **改动点**: {相比 baseline 改了什么}
- **和最终效果的关系**: {去掉这个模块会怎样 — 参考 ablation}

### 模块 B: {名称}
{同上格式}

## 5. 核心公式推导

### Formula 1: {公式名称, e.g., "改进的 Group Reward"}

**直觉**: {为什么要这样设计? 一句话}

**Baseline 公式** ({baseline_name}):
$$L_{baseline} = \mathbb{E}_{...} [...]$$
其中:
- $\pi_\theta$: {含义}
- $A_t$: {含义}
{只解释关键符号}

**从 baseline 到本文的变化**:
{关键 insight: 为什么原来的公式有问题? 改了什么假设?}

**推导过程**:

$$\text{Step 1: } {公式} \quad \text{({做了什么变化})}$$
{解释: 为什么加这一项 / 改这个权重}

$$\text{Step 2: } {公式} \quad \text{({进一步变化})}$$
{解释}

$$\text{最终: } L_{final} = {完整公式}$$

**参数总结**:
| 符号 | 含义 | 取值/范围 | 来源 |
|------|------|----------|------|
| $\alpha$ | {含义} | {range} | {learned/fixed/ablated} |

**在 Pipeline 中的位置**: {对应模块 X 的 training objective}
**对应 ablation**: {Table N, 去掉该项后性能变化 ΔX%}

### Formula 2: {如有第二个核心公式}
{同上格式}

## 6. 实验效果分析

### 主实验
| Method | {Benchmark1} | {Benchmark2} | {Benchmark3} |
|--------|-------------|-------------|-------------|
| {Baseline A} | {score} | {score} | {score} |
| {Baseline B} | {score} | {score} | {score} |
| **{本文方法}** | **{score}** | **{score}** | **{score}** |

### 关键分析
- **哪些 benchmark 支持核心 claim**: {具体说明}
- **哪些结果只是边际提升**: {诚实标注}
- **Ablation 是否证明了关键模块**: {Table X 显示去掉 Y 后下降 Z%}
- **失败案例**: {论文是否展示了失败case? 什么条件下不work?}
- **成本分析**: {训练时间/GPU数量/推理速度/参数量, 如论文提及}

## 7. Related Work / 引用网络

### 直接 Baseline
- [[P__{BaselinePaper1}]] — {一句话说明本文在它基础上改了什么}
- [[P__{BaselinePaper2}]] — {关系}

### 核心引用 (需递归分析)
- [[P__{CitedPaper1}]] — {本文引用了它的哪个组件/思想}
- [[P__{CitedPaper2}]] — {本文在它的基础上改进了什么}
{如果引用的论文不在知识库中，标注: ⚠️ 需要入库分析}

### 被引用 / 后续工作
- [[P__{FollowUp1}]] — {后续工作如何在本文基础上继续改进}
{可通过 Semantic Scholar citations API 获取}

### 相关 Awesome 仓库
- [awesome-{topic}]({url}) — {子领域综合仓库}

## 8. 在知识库中的位置

- **所属任务**: [[T__{task}]]
- **所属方法族**: [[M__{method}]]
- **改进的 slot**: {reward_function / visual_encoder / ...}
- **形成的新 concept**: [[C__{concept}]] (如果有)
- **是否应该成为 baseline**: {判断: 如果 downstream ≥ 3 则是}
- **Facets**:
  - modality: {video / image / ...}
  - paradigm: {rl / sft / ...}
  - scenario: {long_video / streaming / ...}
  - mechanism: {reward_design / kv_cache / ...}
  - constraint: {context_length / memory / ...}
- **后续应检索的关键词**: {用于 discovery 扩展}
```

---

## Formula Derivation Rules (CRITICAL)

The existing `papers-analyze-pdf` skill explicitly says "do not derive formulas". This skill does the **opposite** — formula derivation is required.

**Key principle**: Always align with the baseline formula.

```
baseline_formula
  ↓ changed_assumption (为什么原来的不行)
  ↓ new_term_added (加了什么)
  ↓ new_weight_or_normalization (改了什么权重)
  ↓ final_objective
```

Do NOT just show the final formula. Show the derivation path from baseline.

**Rules**:
1. Every formula must have a "Baseline 公式" section showing what existed before
2. Every step must explain WHY this change was made, not just WHAT changed
3. Every parameter must have a table with symbol/meaning/range/source
4. Every formula must be linked to its Pipeline module and Ablation experiment
5. If the paper has ≥3 core formulas, prioritize the 2 most important

---

## Recursive Related Work Rules

When analyzing a paper's references:

1. **GROBID-parsed references**: Use structured ref data from L2 parse
2. **Check knowledge base**: If [[P__RefTitle]] exists → add wikilink
3. **If not in KB**: Check importance:
   - Is it a direct baseline? → Mark as ⚠️ 需要入库
   - Is it same mechanism_family? → Mark as ⚠️ 需要入库
   - Is it highly cited (>100)? → Mark as ⚠️ 需要入库
   - Otherwise → just mention, no ingest
4. **Citations (被引用)**: If MCP tools available, call `discover_related_papers`
5. **Awesome repos**: Search GitHub for `awesome-{topic}` repos

---

## Wikilink Budget

Paper reports should have 8-15 wikilinks:
- 1-2 task links: [[T__xxx]]
- 1-2 method links: [[M__xxx]]
- 1-3 mechanism links: [[C__xxx]]
- 1-3 dataset links: [[D__xxx]]
- 2-5 paper links: [[P__xxx]] (baselines + follow-ups)

Do NOT create wikilinks for every mentioned paper. Only link papers that are in or should be in the knowledge base.

---

## Quality Checklist

Before finalizing the report, verify:

- [ ] 元数据表完整 (venue, team, code url)
- [ ] 核心观点是一句完整的因果句
- [ ] 动机图/示例清楚解释了问题
- [ ] Pipeline 每个模块都有 输入/输出/改动点
- [ ] 公式从 baseline 逐步推导，不是直接贴最终结果
- [ ] 公式参数有表格解释
- [ ] 实验分析不只是贴数字，有失败案例分析
- [ ] Related work 区分了 baseline / 引用 / 被引用
- [ ] 知识库定位包含 facets 和 method evolution 判断
