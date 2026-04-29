# 用户 README（Skill 使用说明）

[English](User_README.md) | [中文](User_README_CN.md)

本文件用于**快速选 skill**。
仅作导航，不参与执行。

## 1. 快速路由

- 不知道从哪里开始：`research-workflow`
- 从 Zotero 或本地 PDF 文件夹导入论文：`papers-sync-from-zotero`
- 从网页或 GitHub 收集论文：`papers-collect-from-web` / `papers-collect-from-github-awesome`
- 批量下载并修复 PDF：`papers-download-from-list`
- 把 PDF 分析进知识库：`papers-analyze-pdf`
- 重建索引或检索知识库：`papers-build-collection-index` / `papers-query-knowledge-base`
- 生成论文文字对比：`papers-query-knowledge-base`
- 从 KB 证据 + 领域瓶颈诊断 + web 论文 + 明确决策规则生成 idea：`idea-emerge`
- 方向仍然开放、需要发散 idea：`research-brainstorm-from-kb`
- 已经有方向，需要收敛成可执行方案：`idea-focus-coach`
- 想做严格的审稿式压测，而不是共创收敛：`reviewer-stress-test`
- 诊断反复出现的 skill 失配：`skill-fit-guard`
- 写或更新每日研究日志：`write-daily-log`
- 将 ResearchFlow 迁移到其他专业领域：`domain-fork`

## 2. 分类与 skill 简述

### 2.1 知识库构建

- `papers-sync-from-zotero`
  - 何时使用：你想从 Zotero 或本地 PDF 文件夹导入论文到 ResearchFlow。
  - 输入：Zotero library ID + API key（API 模式），或 PDF 文件夹路径（fallback 模式）。
  - 输出：PDF 复制到 `paperPDFs/`，富 metadata 写入 `manifest.jsonl`，轻量行追加到 `analysis_log.csv`。
  - 支持增量同步。

- `papers-collect-from-web`
  - 何时使用：你已经有一批网页列表，想批量筛论文。
  - 输入：URL + 关键词 + venue 约束。
  - 输出：后续可继续处理的 triage 列表。

- `papers-collect-from-github-awesome`
  - 何时使用：来源是 GitHub 仓库（awesome list、survey 配套 repo、lab paper list、conference accepted paper repo 等）。
  - 输入：GitHub 仓库 URL。
  - 输出：与本地分析流程对齐的候选列表。

- `papers-download-from-list`
  - 何时使用：你已经手工筛过候选列表，想落地本地 PDF。
  - 输入：如 `paperAnalysis/*.txt` 的候选列表。
  - 输出：下载或修复后的本地 PDF 与状态结果。

- `papers-analyze-pdf`
  - 何时使用：你想对单篇或批量 PDF 做结构化分析。
  - 输入：本地 PDF 路径。
  - 输出：`paperAnalysis/` 下的结构化 Markdown。
  - 语言说明：默认输出语言跟随 `AGENTS.md` 里的 `analysis_language`（默认 `zh`，改成 `en` 可输出英文笔记）；当前请求中如果用户显式指定语言，会覆盖这一次默认值。

- `papers-audit-metadata-consistency`
  - 何时使用：你怀疑日志和分析笔记存在不一致。
  - 输入：当前 `paperAnalysis/`。
  - 输出：一致性审计报告与质量问题列表。

- `papers-build-collection-index`
  - 何时使用：分析笔记变更后，需要刷新索引。
  - 输入：`paperAnalysis/` 中的 frontmatter。
  - 输出：写入或刷新 `paperCollection/index.jsonl`（agent 索引）和 `paperCollection/` Obsidian 导航页。

### 2.2 知识库检索与代码前论文检索

- `papers-query-knowledge-base`
  - 何时使用：你需要从本地知识库中找论文、取证据，或做跨论文的文字型综合总结；也可用于代码修改前的论文检索（指定 `mode: code-context`）。
  - 输入：研究问题，可附带方向和约束；code-context 模式下提供当前代码上下文或目标模块。
  - 输出：基于本地知识库证据的回答；code-context 模式返回 brief / deep 两种检索结果，含环境检测。

### 2.3 Paper idea 与研究规划

- `idea-emerge`
  - 何时使用：你想从证据和明确决策规则中生成 research idea，而不只是总结论文。
  - 输入：ResearchFlow KB 证据、领域瓶颈诊断、任务核心 web 论文、跨领域 operator、显式决策规则、实现痕迹、baseline、metric 和任务约束。
  - 输出：`paperIDEAs/` 下的 idea-emergence 笔记，包含领域瓶颈诊断、evidence ledger、knowledge/evidence cards、decision-rule cards、operator/gap cards、候选评分拆解、`S3` hard gates、被拒绝或暂存的 ideas 和迭代日志。
  - 路由说明：当任务起点是“证据和约束如何长出 idea”时先用这个；当已经有明确研究方向、需要发散变体时再用 `research-brainstorm-from-kb`。
- `research-brainstorm-from-kb`
  - 何时使用：你需要发散候选 idea。
  - 输入：问题陈述或方向草案。
  - 输出：带相关工作支撑的结构化 idea 候选。
  - 路由说明：当 idea 仍然开放、你要的是候选方向而不是 scope cut 时，用这个。

- `idea-focus-coach`
  - 何时使用：idea 太宽，需要逐步收敛。
  - 输入：初始想法与目标偏好。
  - 输出：聚焦后的目标、非目标、优先假设与 MVP 实验。
  - 独立使用：不依赖 brainstorm 或 reviewer 的输出。
  - 路由说明：当你已经有了一个真实方向，需要 scope cut、假设排序或 MVP 规划时，用这个。

- `reviewer-stress-test`
  - 何时使用：你想用 ICLR / CVPR / SIGGRAPH 风格对一个 idea 做压力测试。
  - 输入：idea、roadmap 或完整论文。
  - 输出：major / minor 风险与对应修复动作。
  - 独立使用：不依赖 focus 输出。
  - 路由说明：当你想先暴露 challenge 和 rejection risk，而不是继续共创时，用这个。

### 2.4 Pipeline 编排与过程保护

- `research-workflow`
  - 何时使用：你不确定自己当前处于哪个阶段。
  - 输入：当前任务描述。
  - 输出：阶段判断与推荐的下一步 skill。

- `notes-export-share-version`
  - 何时使用：内部笔记需要对外分享。
  - 输入：待导出的笔记。
  - 输出：去除内部痕迹后的可分享 Markdown。

- `skill-fit-guard`
  - 何时使用：某个 skill 输出明显失配，而且这种失配可能重复发生。
  - 输入：失配现象。
  - 输出：可能原因、修订选项，以及是否立即修订的询问。

- `write-daily-log`
  - 何时使用：你想写或更新每日研究日志，总结当天进展。
  - 输入：可选的目标文件路径；默认 `DailySummary/<today>.md`。
  - 输出：结构化日报，包含今日进展、核心结论、问题与思考、明日任务。
  - 证据来源：会话上下文 + workspace 下所有 git 仓库的 diff + 文件系统 artifact 扫描。
  - 核心逻辑：只收集"改变未来判断"的证据，不写流水账。每条信息必须通过筛选门控（改变决策、影响复现、产生明确后续动作、或是当天最重要的定量结果）。写完后自动执行一致性校验。
  - 禁止写入：活动流水账、未过滤的命令输出、无证据的推测性结论、混合评测环境的数字（除非显式标注）。

### 2.5 领域迁移

- `domain-fork`
  - 何时使用：你想把 ResearchFlow 的架构迁移到另一个专业领域，例如前端工程、会计、新闻等。
  - 输入：目标领域名称。
  - 输出：在交互式确认后，生成一套适配新领域的完整仓库，包括重命名后的 skills、目录结构与 README。
  - 触发方式：仅显式调用。

## 3. 触发策略

所有 skill 都通过 description 匹配或显式调用触发，不存在基于文件变更的自动触发。下表给出推荐触发方式。

| Skill | 触发方式 | 典型时机 |
|-------|----------|----------|
| `papers-sync-from-zotero` | 显式 | 用户想从 Zotero 或本地 PDF 文件夹导入论文时 |
| `papers-collect-from-web` | 显式 | 用户给出 URL 和主题约束时 |
| `papers-collect-from-github-awesome` | 显式 | 用户给出 GitHub 仓库 URL 时 |
| `papers-download-from-list` | 显式 / 建议式 | 收集完成后建议执行 |
| `papers-analyze-pdf` | 显式 / 建议式 | 下载完成后建议执行 |
| `papers-audit-metadata-consistency` | 建议式 | 批量分析完成后建议执行 |
| `papers-build-collection-index` | 建议式 | 分析完成后，如需刷新索引时 |
| `papers-query-knowledge-base` | 显式 / 静默 | 用户问答时显式；作为内部依赖时静默调用；代码修改前建议 code-context 模式 |
| `idea-emerge` | 显式 | 需要从 KB 证据、领域瓶颈诊断、任务核心 web 论文、跨领域 operator、明确决策规则中生成 idea 候选时 |
| `research-brainstorm-from-kb` | 显式 | 用户要求开放式候选方向时 |
| `idea-focus-coach` | 显式 | 用户已有真实方向，想做 scope cut 或 MVP 规划时 |
| `reviewer-stress-test` | 显式 | 用户已有较成型 idea，想先接受压测时 |
| `research-workflow` | 显式 / 建议式 | 用户不确定下一步做什么时 |
| `notes-export-share-version` | 显式 | 用户想对外分享笔记时 |
| `skill-fit-guard` | 建议式 | agent 检测到明显且反复的 skill 失配时 |
| `write-daily-log` | 显式 / 建议式 | 用户说"写日志"/"日报"，或长研究会话结束时 |
| `domain-fork` | 显式 | 用户明确要求把 ResearchFlow 迁移到其他领域时 |

触发方式说明：

- 显式：由用户直接调用，或由 description 匹配触发
- 建议式：agent 在上下文中建议运行，等待用户确认后执行
- 静默：作为其他 skill 的内部依赖被调用

## 4. 调用方式

- 直接描述目标（推荐）
- 直接点名 skill，例如“使用 papers-download-from-list”
- 使用 slash 形式，例如 `/papers-analyze-pdf`

## 5. 调用安全说明

- 实际路由只依赖 `.claude/skills-config.json` 与各 skill 的 `SKILL.md`
- `.claude/skills` 是唯一维护源。如需 `.codex/skills` 的兼容路径，请运行 `python3 scripts/setup_shared_skills.py` 或 `py -3 scripts\setup_shared_skills.py`
- `paperCollection/index.jsonl` 会在首次运行 `papers-build-collection-index` 后生成；在此之前直接从 `paperAnalysis/` 检索即可
- 本 `User_README_CN.md` 仅作导航，不影响执行
- `User_README.md` 同样仅作导航，不在注册表中
