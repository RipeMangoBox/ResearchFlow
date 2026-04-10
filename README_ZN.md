<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow logo" width="280"/>
</p>

<h1 align="center">ResearchFlow</h1>

<p align="center"><strong>本地科研知识库 + AI research assistant。<br/>把论文、分析、idea 沉淀成可复用的 research memory，让 Claude Code、Codex CLI 与其他 AI agent 共享同一份证据层。</strong></p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_ZN.md">中文</a>
</p>

<p align="center">
  <img alt="Semi-automated" src="https://img.shields.io/badge/Semi--automated-Research%20Workflow-1f6feb?style=flat-square"/>
  <img alt="Knowledge base" src="https://img.shields.io/badge/Local-Knowledge%20Base-0f766e?style=flat-square"/>
  <img alt="Claude Code compatible" src="https://img.shields.io/badge/Claude%20Code-compatible-d97706?style=flat-square"/>
  <img alt="Codex CLI compatible" src="https://img.shields.io/badge/Codex%20CLI-compatible-7c3aed?style=flat-square"/>
  <img alt="Obsidian optional" src="https://img.shields.io/badge/Obsidian-optional-475569?style=flat-square"/>
  <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-111827?style=flat-square"/>
</p>

---

> 🧠 **知识先行，而非执行先行。** 大多数 AI 科研工具关注“帮你跑实验、写论文”。ResearchFlow 更关注上游问题：**你的 agent 在做决策时，手里有没有足够的、结构化的、可检索的论文证据？** 没有这一层，idea 容易空泛，代码修改缺乏依据，reviewer 质疑也难以回应。
>
> 🪶 **纯 Markdown，零依赖，零锁定。** 整个系统就是本地文件夹 + Markdown + CSV。没有数据库，没有 Docker，没有后端服务。每个 skill 都是一个 `SKILL.md`，Claude Code、Codex CLI、Cursor 或你自己的 agent 都能读。
>
> 💡 _ResearchFlow 是一套方法论，不是一个平台。重要的是你积累下来的 research memory。_

---

## 🎯 不只是 prompt，而是完整的知识管线

给 ResearchFlow 一个研究方向，它可以帮你把知识库逐步建起来：

```text
collect → download → analyze → build index → research assist
```

你可以按四种模式使用它：

**构建模式**：从网页、会议页或 GitHub awesome 仓库收集候选论文。

```text
/papers-collect-from-github-awesome
从 https://github.com/Foruck/Awesome-Human-Motion 收集 motion generation 相关论文，
只保留 diffusion、controllability、real-time 相关条目。
```

**查询模式**：知识库建好后，按 task、method、标签或 paper title 检索。

```text
/papers-query-knowledge-base
哪些论文用了 diffusion + spatial control？它们的 core_operator 分别是什么？
```

**决策模式**：在改方法、选 baseline、写 related work 前做结构化对比。

```text
/papers-compare-table
对比 DART、OmniControl、MoMask 的表征设计，重点分析能否支持长时序生成。
```

**Idea 模式**：从本地知识库中提出有文献支撑的候选方向。

```text
/research-brainstorm-from-kb
我想研究 text-driven reactive motion generation，请基于本地知识库提出 3 个候选方向。
```

> 已经有 PDF？直接跳到 analyze。已经有 idea？直接跳到 focus 或 reviewer 压测。不确定当前该用哪个 skill？先用 `research-workflow`。

---

## ✨ 核心能力

| 能力 | 做什么 | 对应 skill |
|---|---|---|
| 🧭 流程入口 | 识别当前处于 collect / download / analyze / build / query / compare / ideate / focus / review 哪个阶段，并推荐下一步 | `research-workflow` |
| 📥 论文收集 | 从网页 / GitHub awesome 仓库批量提取候选论文，统一写入 `analysis_log.csv` | `papers-collect-from-web` · `papers-collect-from-github-awesome` |
| ⬇️ PDF 下载 | 按 triage 列表批量下载、去重、修复 PDF | `papers-download-from-list` |
| 🔬 结构化分析 | 把 PDF 解析为带 frontmatter 的 Markdown 笔记 | `papers-analyze-pdf` |
| 📊 索引构建 | 从分析笔记自动生成按 task / technique / venue 分类的导航页 | `papers-build-collection-index` |
| 🔍 知识检索 | 按标题、任务、技术标签、venue 检索分析笔记，返回文字摘要与证据引用 | `papers-query-knowledge-base` |
| ⚖️ 论文对比 | 生成结构化对比表，用于设计决策、Related Work 写作、baseline 选择 | `papers-compare-table` |
| 💡 Idea 生成 | 基于知识库输出有文献支撑的候选研究方向 | `research-brainstorm-from-kb` |
| 🎯 方案收敛 | 把宽泛 idea 逐步收窄为可执行方案，包含 scope cut 与 MVP 规划 | `idea-focus-coach` |
| 🔥 审稿压测 | 以审稿人视角做严格质询，输出 major-risk 诊断与修复路径 | `reviewer-stress-test` |
| 🔗 代码联动 | 改 model / loss / training 代码前，先检索论文依据 | `code-context-paper-retrieval` |
| 🏥 元数据审计 | 检查 title / venue / year / link 一致性，输出质量报告 | `papers-audit-metadata-consistency` |
| 📤 笔记导出 | 把内部笔记转成可分享版本，去除知识库内链 | `notes-export-share-version` |
| 🌐 领域迁移 | 把整套架构迁移到其他专业领域 | `domain-fork` |

> 完整 skill 列表与触发方式见 [`.claude/skills/User_README_ZN.md`](.claude/skills/User_README_ZN.md)。

---

## 🏗️ 三层架构

```text
┌─────────────────────────────────────────────────────────┐
│  输出层   paperIDEAs/                                   │
│           idea、研究方案、review 记录、联动产出            │
├─────────────────────────────────────────────────────────┤
│  统计层   paperCollection/（可选）                       │
│           按 task / technique / venue 的导航与统计页      │
├─────────────────────────────────────────────────────────┤
│  检索层   paperAnalysis/ + paperPDFs/                   │
│           结构化分析笔记 + 原始 PDF，所有 skill 的主数据源  │
└─────────────────────────────────────────────────────────┘
```

- **`paperAnalysis/`** 是整个系统的核心。每篇论文一个 `.md`，包含 `core_operator`、`primary_logic`、`dataset`、`metrics`、`venue` 等结构化字段，所有检索、对比、idea 和 code-context skill 都优先从这里取证。
- **`paperCollection/`** 是可选的统计导航层，主要服务 Obsidian 浏览、总览页和 backlink 探索。即使不 build，查询和 brainstorming 仍然可以正常工作。
- **`paperPDFs/`** 保留原始文献，适合在 evidence 不够时回溯细读。
- **`paperIDEAs/`** 存放 brainstorm 结果、收敛方案、review 压测记录等下游产出。

一个实用判断规则：

- agent 首先查 `paperAnalysis/`
- 需要总览或统计时再看 `paperCollection/`
- 证据不足时再回读 `paperPDFs/`
- 最终输出写入 `paperIDEAs/` 或你的联动代码仓库

---

## 🤖 Claude Code、Codex CLI 与其他 Agent

ResearchFlow 支持三类接入方式：

- **Claude Code / Cursor**：仓库自带 `.claude/skills`，可直接使用 skill 路由。
- **Codex CLI**：通过 `scripts/setup_shared_skills.py` 生成 `.codex/skills` 兼容入口，指向同一份 `.claude/skills`。
- **其他 AI agent**：即使不支持 skill 机制，也可以直接把这里当作本地知识库使用，按 `paperAnalysis/ → paperPDFs/ → paperIDEAs/` 的顺序协作。

这意味着同一份 research memory 可以被多个 agent 共享，而不用为不同工具重复整理文献和笔记。

---

## 🚀 快速开始

### 前置条件

- Claude Code、Codex CLI，或任何能读 `SKILL.md` 的 LLM agent
- Python 3.10+（PDF 下载、分析、索引脚本需要）
- Obsidian（可选，用于可视化浏览知识库）

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/ResearchFlow.git
cd ResearchFlow
```

### 2. 准备 skill 入口

- Claude Code / Cursor：可直接使用现成的 `.claude/skills`
- Codex CLI：需要先生成 `.codex/skills` 兼容入口，见 [Codex CLI 兼容](#codex-cli-compat)

### 3. 开始使用

你可以直接描述任务，或者显式点名 skill。

最通用的入口是：

```text
/research-workflow
我想从零构建一个 controllable motion generation 的知识库
```

也可以直接下具体任务：

```text
帮我从 https://github.com/Foruck/Awesome-Human-Motion 收集 motion generation 论文
```

### 4. 可选：Obsidian 可视化

如果你希望在 Obsidian 中可视化浏览知识库，见 [Obsidian 配置](#obsidian-config)。

---

<a id="usage-examples"></a>
## 📖 用法示例

先给几条更实用的经验规则：

- 尽量一条 prompt 只做一件事；分阶段任务拆成多条通常更稳。
- 如果你要在一条 prompt 里组合多个 skill，把主 skill 写在最前面，后续 skill 视为 follow-up。
- 示例里的路径、venue 和研究方向都只是模板，照着替换成你的实际目录即可。
- prompt 更长不一定更有效，优先写清楚目标、范围、输入路径和期望输出。
- 对于不需要额外输入的 skill，直接写 skill 本身通常就够了，例如 `/papers-build-collection-index`。

### A. 知识库构建与维护

<details>
<summary>1. 从零构建一个方向知识库</summary>

推荐先用统一入口，让 agent 判断当前阶段：

```text
/research-workflow
我想从零构建一个 controllable motion generation 的知识库。
请优先从 awesome 仓库和 ICLR 2026 accepted list 开始，
并告诉我当前最合适的下一步是 collect、download、analyze 还是 build。
```

如果你更希望自己掌控每一步，可以手动拆解：

可选预步骤，不显式点 skill：

```text
我希望构建 controllable human motion generation 的 knowledge base。
请先帮我搜索相关的 GitHub awesome 仓库或 curated paper list，推荐 2 个最合适的选项。
```

Prompt 1：收集候选论文

```text
/papers-collect-from-github-awesome
从这个 GitHub awesome 仓库收集 controllable human motion generation 相关论文：<URL>
只保留与 diffusion、controllability、real-time generation、long-form motion 相关的条目。
将结果整理成适合后续下载流程使用的候选清单，并标记建议保留与建议跳过的项。
```

Prompt 2：下载本地 PDF

```text
/papers-download-from-list
把当前候选清单中仍为 Wait 状态的论文下载到本地。
如果遇到链接失效或下载失败，请标记 Missing 并给出失败原因摘要。
下载完成后，返回成功下载、失败和跳过的统计结果。
```

Prompt 3：分析 PDF，写入 `paperAnalysis/`

```text
/papers-analyze-pdf
分析 paperPDFs/ 中新下载、但还没有对应 analysis note 的 PDF。
将分析结果写入 paperAnalysis/，并确保 frontmatter 中包含 title、venue、year、tags、core_operator、primary_logic。
完成后，告诉我本轮新入库了哪些论文。
```

Prompt 4：重建索引

```text
/papers-build-collection-index
```

</details>

<details>
<summary>2. 新 PDF 到来后增量刷新知识库</summary>

最简写法：

```text
/papers-analyze-pdf
分析 paperPDFs/Human_Motion_Generation/CVPR_2026/ 下新增的 PDF
```

```text
/papers-build-collection-index
```

如果你希望 agent 顺手总结新增内容，可以写得更明确一些：

```text
/papers-analyze-pdf
处理 paperAnalysis/analysis_log.csv 中仍需分析的项目。
把新 analysis note 写入 paperAnalysis/，并说明本轮新增了哪些论文。
```

`/papers-build-collection-index` 通常单独执行即可。如果你关心本轮变化，可以在它完成后继续追问新增了哪些方法标签、技术标签，以及哪些 venue / task 页面发生了变化。

</details>

<details>
<summary>3. 做一次知识库健康检查</summary>

```text
/papers-audit-metadata-consistency
```

这个步骤很适合在批量收集、批量分析之后运行一次。默认会扫描当前 `paperAnalysis/` 并输出质量报告。

</details>

### B. 知识驱动的研究决策

<details>
<summary>4. 为设计决策做论文对比</summary>

当你需要在多个设计方案之间做选择，或者要写 Related Work 表格、选 baseline、给导师或合作者做方法概览时：

```text
/papers-compare-table
请对比 DART、OmniControl、MoMask、ReactDance，重点分析它们的表征设计。
根据每种设计的核心思维、适用场景与能力特点，思考能否支持 <XXX>。
把对比结果保存到 paperIDEAs/compare_motion_designs.md。
```

如果你只需要文字型综合总结而不是结构化表格，用 `papers-query-knowledge-base` 即可。

</details>

<details>
<summary>5. 从知识库里做定向检索</summary>

```text
/papers-query-knowledge-base
请帮我找出本地知识库里所有和 reactive motion generation、diffusion、spatial control 相关的论文。
优先返回与 controllability 和 long-horizon consistency 相关的结果，
并总结它们各自的 core_operator、primary_logic 和局限性。
```

这是最适合“先摸清已有证据，再决定要不要 compare / brainstorm”的入口。

</details>

<details>
<summary>6. 从知识库中生成候选 idea</summary>

简洁写法：

```text
/research-brainstorm-from-kb
我想研究 text-driven reactive motion generation，
请基于本地知识库和必要的前沿检索提出 3 个候选方向。
```

如果你想让结果更可执行，可以把要求写细一些：

```text
/research-brainstorm-from-kb
我希望研究长音频驱动的 reactive motion generation。
请先提炼这个任务的第一性挑战，以本地知识库为主，必要时补充外部搜索，最后提出 3 个可行的候选方向。
每个方向都要包含：
- 问题切入点
- 为什么可行
- 哪些现有论文提供技术支撑
- 最小可行实验
把结果写入 paperIDEAs/。
```

</details>

<details>
<summary>7. 把一个宽泛 idea 收敛成可执行方案</summary>

如果你只是想现场一起收窄问题：

```text
/idea-focus-coach
我的 idea 是“用 diffusion model 做 reactive motion generation”，
但我不确定 scope 应该多大、第一个实验应该做什么。
请帮我逐步收窄到一个可执行的 MVP。
```

如果你已经有现成的 idea note，可以直接让它在原文件上继续工作：

```text
/idea-focus-coach
我在 @paperIDEAs/2026-10-08_new_motion_representation_raw.md 里设计了一种新的动作表征，
目标是同时支持高生成质量、高指令跟随能力和长时序一致性。
请把它收敛成一个聚焦方案，至少包含：
- scope cut
- 优先级最高的假设
- 本周可做的 1-2 个 MVP 实验
把结果写回原 markdown 文件。
```

</details>

<details>
<summary>8. 以审稿人视角压测一个 idea</summary>

简洁写法：

```text
/reviewer-stress-test
请以 ICLR reviewer 的视角审查我的 idea：
[粘贴你的 idea 描述或指向 paperIDEAs/ 中的文件]
重点关注 novelty、实验设计、与 SOTA 的差异化。
```

如果你已经有成型方案，可以写得更尖锐一些：

```text
/reviewer-stress-test
我在 @paperIDEAs/2026-10-10_new_motion_representation.md 里设计了一种新的动作表征，
目标是同时支持高生成质量、高指令跟随能力和长时序一致性。
请站在顶会审稿人角度，指出当前方案的主要漏洞、与相关工作的重叠、核心评测标准，以及最可能导致拒稿的风险。
把结果写回原 markdown 文件。
```

</details>

### C. 与代码和多 Agent 系统联动

<details>
<summary>9. 改代码前先检索论文依据</summary>

简洁写法：

```text
/code-context-paper-retrieval
我准备修改 motion decoder 的 attention 机制，
请先检索知识库中相关的 attention 设计方案和实验结论。
```

如果你希望结果直接服务实现规划，可以写得更具体：

```text
/code-context-paper-retrieval
在修改 @paperIDEAs/2026-10-12_new_motion_representation_roadmap.md 对应实现前，先从本地知识库检索相关论文。
请总结：
- 可迁移的 operator
- 关键实现约束
- 与当前代码结构可能冲突的点
- 哪些设计最符合当前 roadmap
如果知识库证据不足，也请明确指出。
```

完成这一步后，再用单独的 coding prompt 去改代码，通常会更稳。

</details>

<details>
<summary>10. 把 ResearchFlow 作为自动化系统的共享知识层</summary>

ResearchFlow 的知识库是纯文件系统，任何能读文件的 agent 都可以接入：

```text
ResearchFlow/          ← 共享知识层
├── paperAnalysis/     ← 所有 agent 的证据源
├── paperPDFs/         ← 原始文献
└── paperIDEAs/        ← 各 agent 写入的研究产出

Agent A (Claude Code)  ── 读取 paperAnalysis/ ──→ 生成 idea
Agent B (Codex CLI)    ── 读取 paperAnalysis/ ──→ 辅助代码修改
Agent C (自定义)        ── 读取 paperAnalysis/ ──→ 自动化实验设计
```

如果你想明确给 agent 约束，也可以直接贴下面这段：

```text
把 ResearchFlow/ 当作这个项目中多个 agent 共享的 research memory 与 evidence layer。
工作时遵循以下约定：
- 先从 paperAnalysis/ 做 agent 检索
- 如果需要统计视图、总览页面或 Obsidian 跳转，再使用 paperCollection/
- 新的研究想法写入 paperIDEAs/
- 如果本地知识库已经覆盖相关论文，不要重新从零复述整条文献链
- 在提出新 idea 或代码修改前，先检查本地知识库里是否已有可复用的 operator、baseline 或 reviewer concern
- 回传给自动化系统时，优先输出简洁、可追溯、基于证据的总结，而不是泛泛 brainstorming
```

</details>

---

## 📁 仓库结构

```text
ResearchFlow/
├── .claude/skills/             # skill 定义（唯一维护源）
├── .claude/skills-config.json  # skill 注册描述与路由元数据
├── .codex/
│   └── skills/                 # 由 scripts/setup_shared_skills.py 生成的兼容入口
├── paperAnalysis/              # 结构化分析笔记（核心数据源）
│   └── analysis_log.csv        # 论文候选列表与状态追踪
├── paperCollection/            # 可选：自动生成的导航索引
├── paperPDFs/                  # 原始 PDF 文件
├── paperIDEAs/                 # 下游产出：idea、方案、压测记录
├── scripts/                    # 辅助脚本（兼容入口、下载、分析、索引构建等）
└── .obsidian/                  # 可选：Obsidian 可视化配置
```

- `paperAnalysis/` 是 agent 面向的主知识层与证据层。
- `paperCollection/` 是生成式的统计 / 导航层，不是主检索层。
- `paperIDEAs/` 是下游研究产物层。
- `.claude/skills` 是唯一维护源；`.codex/skills` 只是兼容入口。
- 想快速浏览 skill 路线图，可直接看 `.claude/skills/User_README_ZN.md`；英文版对应 `.claude/skills/User_README.md`。

---

<a id="advanced-config"></a>
## 🔧 进阶配置

<a id="codex-cli-compat"></a>
<details>
<summary>Codex CLI 兼容</summary>

Claude Code / Cursor 不需要这一步。只有 Codex CLI 需要生成 `.codex/skills` 兼容入口。

macOS / Linux：

```bash
python3 scripts/setup_shared_skills.py
```

Windows：

```powershell
py -3 scripts\setup_shared_skills.py
```

如果你只是想检查兼容入口是否完整，也可以运行：

```bash
python3 scripts/setup_shared_skills.py --check
```

如果 `.codex/skills` 丢失，重新运行即可。

</details>

<a id="obsidian-config"></a>
<details>
<summary>Obsidian 配置</summary>

- Obsidian 只是可视化层，不是必需组件。
- 如果你有共享的配置包，请解压到仓库根目录并重命名为 `.obsidian`。
- 如果解压后文件夹名是 `obsidian setting`，请先改名为 `.obsidian`。

</details>

<details>
<summary>领域迁移（Domain Fork）</summary>

ResearchFlow 的架构不限于学术研究。通过 `domain-fork` skill，你可以把整套结构迁移到其他专业领域：

```text
/domain-fork
我想把 ResearchFlow 迁移到前端开发领域，
用于追踪框架演进、设计模式、性能优化方案。
```

如果你想保留方法论但弱化研究语义，也可以把它视为一个通用的本地知识工作流模板。

</details>

---

## 📝 说明

- `analysis_log.csv` 的论文状态主线是 `Wait → Downloaded → checked`，旁支状态为 `Missing` 与 `Skip`。
- 如果 analysis note 有新增或修改，而你又希望刷新 `paperCollection/` 页面，就应先 rebuild index。
- Obsidian 是可选的；即使只把仓库当普通本地目录给 agent 使用，也完全可行。
- `.claude/skills` 是唯一维护源。如需 `.codex/skills` 的兼容路径，运行 `scripts/setup_shared_skills.py`。
- 对 agent 而言，`build index` 不是每次都必须；需要总览和统计时再做即可。

## License

MIT
