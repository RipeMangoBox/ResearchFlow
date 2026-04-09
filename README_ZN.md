# ResearchFlow

**半自动化本地科研知识库与 research assistant，覆盖 paper analysis、idea generation、coding、experiments、writing 与 publication workflows。**

[English](README.md) | [中文](README_ZN.md)

---

ResearchFlow 支持从零构建研究知识库，把论文清单、PDF 与结构化分析笔记沉淀成 Claude Code、Codex CLI 与其他 AI 科研工具可以共享的 research memory，并配套一系列 skill 辅助从 idea 到 publication 的全科研流程。

```text
collect paper list -> paper analysis -> build index -> research assist
```

它可以同时服务三类使用方式：

- Claude Code / Cursor：通过 `.claude/skills`
- Codex CLI：通过 `AGENTS.md` 加上生成出来的 `.agents/skills` / `.codex/skills` 兼容入口
- 其他 AI 科研工具：直接读取本地 Markdown、PDF 和索引

它的重点不只是“管理论文”，而是把本地文献积累变成一个可检索、可比较、可生成 idea、可与本地代码仓库联动，同时又保持 ResearchFlow 自身作为工作区的 agent knowledge base。

## Domain Fork

ResearchFlow 的架构并不只适用于学术科研。通过 `domain-fork` skill，可以把同一套“本地知识积累 + 结构化索引 + workflow skill + 下游执行”的框架迁移到其他专业领域。

例如前端开发、会计、新闻采编、政策分析、法律工作流等，只要需要共享本地知识库和半自动化辅助，就可以用这套架构做领域化 fork。

如果你想先用一个中性的命名开始，也欢迎直接尝试 `YourAnyFlow`。

```text
/domain-fork
帮我 fork 一个前端工程版本的 ResearchFlow。
保留本地 knowledge base + workflow 架构，但把 papers、venues、analysis notes 映射成前端资料、实现模式、bug case 与框架对比。
生成适配后的仓库结构、重命名后的 skills 和 README。
欢迎尝试 YourAnyFlow!
```

## 它能做什么

- 从会议页面、博客、GitHub awesome 仓库中收集某个方向的论文候选
- 将本地 PDF 转成结构化论文分析笔记，沉淀 `core_operator`、`primary_logic`、tags、venue、year 等可复用字段
- 自动重建 `paperCollection/` 页面，用于统计、Obsidian 跳转和双链友好的浏览
- 以 `paperAnalysis/` 作为 agent 的主检索与证据层，用于对比、问题清单、idea 生成、审稿式质询，以及代码实现前的论文检索
- 让多个 agent、多个科研项目、多个代码仓库复用同一个本地知识库

## 用法示例

下面提供了若干提示词示例。

- 每个框的第一行都会显式写出推荐使用的 skill。
- 对于建库流程，建议按阶段逐条执行，而不是把 collect / download / analyze / build 合并成一条大 prompt。

### 1. 从零构建一个方向知识库

如果你还没有合适的论文来源，可以先让 agent 帮你找一个更适合 intake 的 awesome 仓库或 curated list。

```text
我希望构建 controllable human motion generation 的 knowledge base。
请先帮我搜索相关的 GitHub awesome 仓库或 curated paper list，推荐 2 个最合适的选项。
```

选定来源后，按阶段执行下面四条 prompt。

(1) 构建论文清单

```text
/papers-collect-from-github-awesome
从这个 GitHub awesome 仓库收集 controllable human motion generation 相关论文：<URL>
只保留与 diffusion、controllability、real-time generation、long-form motion 相关的条目。
将结果整理成适合后续下载流程使用的候选清单，并标记建议保留与建议跳过的项。
```

(2) 下载论文

```text
/papers-download-from-list
把当前候选清单中仍为 `Wait` 状态的论文下载到本地。
如果遇到链接失效或下载失败，请标记 `Missing` 并给出失败原因摘要。
下载完成后，返回成功下载、失败和跳过的统计结果。
```

(3) 论文分析

由于每次会话上下文有限，建议每次处理不超过 8 篇论文，待上下文接近饱和时开新的会话继续进行。

```text
/papers-analyze-pdf
分析 `paperPDFs/` 中新下载、但还没有对应 analysis note 的 PDF。
将分析结果写入 `paperAnalysis/`，并确保 frontmatter 中包含 title、venue、year、tags、core_operator、primary_logic。
完成后，告诉我本轮新入库了哪些论文。
```

(4) 索引构建

每篇论文分析后会带有 tag，索引构建通过整理所有论文的 tag，便于后续 research assist 环节快速检索相关论文。

```text
/papers-build-collection-index
基于最新的 `paperAnalysis/` 重建 `paperCollection/`。
完成后，简要总结目前知识库按 task、technique、venue 的覆盖情况。
```

### 2. 新 PDF 到来后增量刷新知识库

```text
/papers-analyze-pdf, /papers-build-collection-index
处理 `paperAnalysis/analysis_log.csv` 中需要处理的项目。
分析完成后重建 `paperCollection/`。
最后总结输出：
- 这次新加入了哪些论文
- 新增了哪些方法标签和技术标签
```

### 3. 为设计决策做论文对比

```text
/papers-compare-table
请对比 DART、OmniControl、MoMask、ReactDance，重点分析它们的表征设计。
根据每种设计的核心思维、适用场景与能力特点，思考能否支持 <XXX>。
分析结果保存到 `paperIDEAs/`。
```

### 4. 用知识库做 idea 生成，而不是泛泛 brainstorming

(1) 有大致方向，由 agent 生成初步 idea

```text
/research-brainstorm-from-kb, /reviewer-stress-test
我希望研究 `长音频时序定位` 任务。
请先定位这个任务的第一性挑战，再结合本地知识库与必要的外部搜索，借鉴视频理解领域的最新发展，组织作者视角与 CVPR/ICLR 审稿人视角的多轮讨论，最后提出 3 个可行 idea。
每个 idea 都要包含：
- 问题切入点是什么
- 它为什么可行
- 哪些现有论文提供了技术支撑与验证
- 作为技术支撑的论文，其代码仓库是否开源、可信度如何、与当前 idea 的关联强度如何
- 最小可行实验是什么
- 最可能遭遇的 reviewer objection 是什么
把最终结果写入 `paperIDEAs/`。
```

(2) 确认了初步 idea，需要与 agent 进一步完善

```text
/idea-focus-coach
我设计了一种新的动作表征，@paperIDEAs/2026-10-08_new_motion_representation_raw.md，目标是同时支持高生成质量、高指令跟随能力和长时序一致性。现在还需进一步确定以下环节：
1. XXXX
2. XXXXX
3. ...
结果写回 md。
```

(3) 有了较为完善的 idea，由 agent 扮演审稿人“事前验尸”

```text
/reviewer-stress-test /research-question-bank
我设计了一种新的动作表征，@paperIDEAs/2026-10-10_new_motion_representation.md，目标是同时支持高生成质量、高指令跟随能力和长时序一致性。
请站在顶会审稿人角度，指出当前方案的主要漏洞、相关工作相似性、审稿核心考察点与风险点等，结果更新到对应 md。
```

### 5. 将代码仓库链接进知识库工作区

把目标代码仓库链接到 ResearchFlow 下的 `linkedCodebases/`。这样可以保持 ResearchFlow 自身就是工作区，从而正常使用 Claude Code 和 Codex 的 skills，同时让 agent 在改代码前先做论文检索，更忠实于 idea / roadmap。

```text
/code-context-paper-retrieval
请按照 @paperIDEAs/2026-10-12_new_motion_representation_roadmap.md 完成代码修改，修改前如有需要请检索 KB 相关资料。
总结：
- 可迁移的 operator
- 关键实现约束
- 与当前代码结构可能冲突的点
- 哪些设计最符合我们当前的 idea / roadmap
然后给出代码改动方案，并开始实现。
如果现有知识库证据不足，请先明确指出再继续。
输出处理总结，如有推进请更新 @paperIDEAs/2026-10-12_new_motion_representation_roadmap.md。
```

### 6. 将知识库与自动化工具联动

ResearchFlow 可以作为高质量本地知识库，与现有自动科研工具结合，例如 [Deep Researcher Agent](https://github.com/Xiangyue-Zhang/auto-deep-researcher-24x7)。

```text
/papers-query-knowledge-base
把 `ResearchFlow/` 当作这个项目中多个 agent 共享的 research memory 与 evidence layer。
工作时遵循以下约定：
- 先从 `paperAnalysis/` 做 agent 检索
- 如果需要统计视图、总览页面、Obsidian 跳转或双链辅助，再使用 `paperCollection/`
- 新的问题清单写入 `QuestionBank/`
- 新的研究想法写入 `paperIDEAs/`
- 如果本地知识库已经覆盖相关论文，不要重新从零复述整条文献链
- 在提出新 idea 或代码修改前，先检查本地知识库里是否已有可复用的 operator、baseline 或 reviewer concern
- 回传给自动化系统时，优先输出简洁、可追溯、基于证据的总结，而不是泛泛 brainstorming
```

## 核心工作流


| 阶段                 | 目标                       | 主要产物                                             |
| ------------------ | ------------------------ | ------------------------------------------------ |
| Collect paper list | 从网页或 GitHub 列表收集候选论文     | 与本地流程对齐的候选清单                                     |
| Paper analysis     | 下载 PDF 并转成结构化分析笔记        | `paperPDFs/`、`paperAnalysis/`、`analysis_log.csv` |
| Build index        | 从 analysis notes 重建统计 / 导航页面 | 按 task、technique、venue 组织的 `paperCollection/`    |
| Research assist    | 查询、对比、生成 idea、审稿压测、与代码联动 | 回答、表格、`QuestionBank/`、`paperIDEAs/`、实现方案         |


`Download` 属于 collect 与 analysis 之间的 intake 环节。对 agent 而言，`paperAnalysis/` 才是主检索来源；`build index` 主要用于刷新 `paperCollection/` 的统计页和 Obsidian 导航页。

## `paperAnalysis` 与 `paperCollection` 如何配合

当 `paperAnalysis/` 累积起结构化分析笔记后，ResearchFlow 就不再只是“论文归档目录”。此时可以进一步重建 `paperCollection/`，作为统计和 Obsidian 侧导航的辅助层。


| 层级  | 主要目录                                             | 作用                            |
| --- | ------------------------------------------------ | ----------------------------- |
| Agent 检索 / 证据层 | `paperAnalysis/`                                 | agent 的主检索与证据层，提供结构化信息，如 operator、逻辑、链接、标签 |
| 统计 / 导航层 | `paperCollection/`                               | 从 `paperAnalysis/` 生成的总览页，用于统计、Obsidian 跳转和双链辅助 |
| 输出层 | `QuestionBank/`、`paperIDEAs/`、`linkedCodebases/` | 将知识转成问题清单、idea、实现方案与代码决策      |


基于这三层联动，可以自然扩展出几类科研 assist：

- `KB -> 对比决策`：判断该借哪种 operator、控制机制、训练策略
- `KB -> 问题清单`：梳理 open problems、评测缺口、基线缺口、审稿风险
- `KB -> idea 生成`：让 idea 建立在已有文献网络之上，而不是空泛发散
- `KB -> reviewer 模拟`：在立项或写作前先做一次严格质询
- `KB -> codebase 联动`：在改 model、loss、network、control、training 代码前先检索论文依据
- `KB -> 多 agent 协作`：Claude Code、Codex CLI 与其他 agent 共享同一个本地 research memory

## Claude Code、Codex CLI 与其他 Agent

### Claude Code / Cursor

ResearchFlow 自带 `.claude/skills`，因此在支持该约定的工具里可以直接使用 skill 路由。

### Codex CLI

ResearchFlow 现在也支持 Codex CLI，方式分两层：

- `AGENTS.md` 是 Codex 在仓库内的稳定入口，用来描述工作流和 source-of-truth 目录
- `scripts/setup_shared_skills.py` 会生成 `.agents/skills`、`.agents/skills-config.json`、`.codex/skills` 和 `.codex/skills-config.json`，统一指向同一份 `.claude/skills`
- 在 macOS/Linux 上使用软连接，在 Windows 上使用目录 junction 和硬链接配置文件

这样可以保持一份唯一维护的 skill 库，同时又给 Claude Code 和 Codex 提供各自熟悉的兼容路径。

### 其他 AI 科研工具

即便不支持 skills，这个仓库依然可以直接作为本地知识库使用：

- 先从 `paperAnalysis/` 做 agent 检索与证据读取
- 如果需要总览页、统计视图或 Obsidian 跳转，再使用 `paperCollection/`
- 需要更深验证时再打开 `paperPDFs/`
- 最终把输出写入 `QuestionBank/`、`paperIDEAs/` 或 `linkedCodebases/` 中联动的代码仓库

## 快速开始

如果你只想先把最小可用流程跑起来，先完成第 1 步和第 2 步即可。第 3 步和第 4 步都是可选增强项。

### 1. 克隆仓库

先把 ResearchFlow 拉到本地，并进入工作区根目录：

```bash
git clone https://github.com/<your-username>/ResearchFlow.git
cd ResearchFlow
```

### 2. 初始化 Claude Code / Codex 的共享 skill 入口

这一步会保持 `.claude/skills` 作为唯一维护入口，并自动生成 Codex 兼容路径。

macOS / Linux：

```bash
python3 scripts/setup_shared_skills.py
```

Windows：

```powershell
py -3 scripts\setup_shared_skills.py
```

### 3. 可选：加入共享的 Obsidian 配置

如果你想直接使用推荐的 Obsidian 工作区配置，可以继续执行这一步。

1. 从 [Google Drive](https://drive.google.com/file/d/1tSEfV6kVI5dViojqZjDU42AYY8viZ0M4/view?usp=sharing) 下载共享的 Obsidian 压缩包。
2. 在本地解压。
3. 将解压后的文件夹重命名为 `.obsidian`。
4. 把这个 `.obsidian` 文件夹放到仓库根目录。

### 4. 可选：链接外部代码仓库

如果你希望保持 ResearchFlow 自身作为工作区，同时又能在其中访问一个或多个本地代码仓库，就执行这一步。

推荐自动化方式：

macOS / Linux：

```bash
python3 scripts/link_codebase.py /path/to/your-codebase
```

Windows：

```powershell
py -3 scripts\link_codebase.py C:\path\to\your-codebase
```

如果希望在 `linkedCodebases/` 里使用自定义名字，可以写：

```bash
python3 scripts/link_codebase.py /path/to/your-codebase --name your-codebase
```

手动方式作为兜底：

```bash
ln -s /path/to/your-codebase linkedCodebases/your-codebase
```

Windows PowerShell 兜底方式：

```powershell
New-Item -ItemType Junction -Path .\linkedCodebases\your-codebase -Target C:\path\to\your-codebase
```

一个 ResearchFlow 可以同时链接多个代码仓库，同时保持自己作为 Claude/Codex 的工作区。

## 仓库结构

```text
ResearchFlow/
├── AGENTS.md
├── LOGO.png
├── .claude/skills/
├── .claude/skills-config.json
├── .agents/
│   └── skills/                  # 由 scripts/setup_shared_skills.py 生成
├── .codex/
│   └── skills/                  # 由 scripts/setup_shared_skills.py 生成
├── paperAnalysis/
│   └── analysis_log.csv
├── paperCollection/
├── paperPDFs/
├── QuestionBank/
├── paperIDEAs/
├── linkedCodebases/
├── scripts/
└── obsidian setting/
```

- `paperCollection/` 是生成式的统计 / 导航层
- `paperAnalysis/` 是 agent 面向的主知识层与检索层
- `QuestionBank/` 与 `paperIDEAs/` 是下游科研产物层
- `linkedCodebases/` 用于放本地代码仓库的软连接或 junction
- `scripts/` 存放维护和自动化脚本
- 想快速浏览 skill 路线图，可直接看 `.claude/skills/User_README_ZN.md`，英文版对应 `.claude/skills/User_README.md`

## 说明

- 需要使用知识库辅助科研前，如果 analysis note 有新增或修改，就应先 rebuild index，防止 KB 索引滞后
- `analysis_log.csv` 的论文状态主线是 `Wait -> Downloaded -> checked`，旁支状态为 `Missing` 与 `Skip`
- Obsidian 是可选的；即使只把仓库当普通本地目录给 agent 使用，也完全可行
- 如果 `.agents/skills` 或 `.codex/skills` 丢失了，重新运行 `scripts/setup_shared_skills.py` 即可
- 共享的 Obsidian 配置包是可公开分享的模板，已经去除了私有 workspace 状态和分享 token
- 如果解压后文件夹名是 `obsidian setting`，请先改名为 `.obsidian`，再放到仓库根目录

## Skill 使用细节

### 如何调用 skill

- 直接描述任务目标
- 直接点名 skill，例如 `papers-download-from-list`
- 使用 slash 方式，例如 `/papers-analyze-pdf`

### 推荐入口

- 如果不确定该从哪里开始，先用 `research-workflow`
- 对于知识库构建，通常按 `papers-collect-* -> papers-download-from-list -> papers-analyze-pdf -> papers-build-collection-index` 执行
- 对于检索与设计分析，可从 `papers-query-knowledge-base` 或 `papers-compare-table` 开始
- 对于 idea 生成与方案打磨，可使用 `research-brainstorm-from-kb`、`research-question-bank`、`idea-focus-coach`、`reviewer-stress-test`
- 对于代码联动，建议在修改模型或方法代码前先运行 `code-context-paper-retrieval`
- 对于整套仓库架构的跨领域迁移，使用 `domain-fork`

### 详细参考

- `.claude/skills/User_README_ZN.md`：中文版快速选 skill 指南
- `.claude/skills/User_README.md`：英文版快速选 skill 指南
- `.claude/skills/README.md`：按 workflow 家族整理的 skill 总览
- `.claude/skills-config.json`：skill 注册描述与路由元数据

## License

MIT
