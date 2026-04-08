---
name: research-brainstorm-from-kb
description: Structures and refines research ideas into `paperIDEAs/` notes using the local paper knowledge base and frontier techniques. Use when the user provides research questions or ideas and wants decomposed, scenario-driven optimization, related-work analysis based on `paperCollection` + `paperAnalysis`, and cross-domain support from image/video generation, MLLM, Agent, or RL, automatically saved as dated idea notes under `paperIDEAs/`.
---

# Research Idea Brainstorming

## 1. 使用场景（When to use）

在以下场景使用本 skill：

- **科研问题 / idea 输入**：用户用自然语言描述科研问题、研究方向或零散想法，希望系统性脑暴与收敛。
- **需要 本地论文库支撑**：需要基于 `paperCollection` + `paperAnalysis` 给出「相关工作支持 + 研究空间」。
- **希望生成可复用的 idea 笔记**：希望结果自动写成结构化 Markdown，保存在 local 的 `paperIDEAs/` 目录中，以便后续写作、实验和项目管理。

## 2. 依赖与路径（Dependencies & paths）

本 skill 依赖本地论文知识库和 `papers-query-knowledge-base` skill：

- **论文索引与分析**：见 `papers-query-knowledge-base` skill（`paperCollection/` + `paperAnalysis/`）
- **Idea 笔记目录**：`paperIDEAs/`

路径约定：

- 当当前 workspace 就是包含这些目录的仓库时，优先使用相对路径。
- 当 workspace 不包含该知识库时，使用你本机对应的绝对路径。

## 3. 文件命名与存储规则

生成脑暴结果时，**总是写入 / 更新 `paperIDEAs/` 目录下的 Markdown 文件**，命名规则：

- **文件名模式**：`YYYY-MM-DD_<核心缩写>.md`
  - `YYYY-MM-DD`：使用用户本地日期（如 `2025-03-09`）
  - `<核心缩写>`：用英文单词或拼音压缩的主题短语，使用小写和 `-` 连接（例如：`motion-llm-ideas`、`reactive-agent-motion`）
- **示例**：`2025-03-09_motion-llm-ideas.md`

行为约定：

- 如果当日同一核心缩写文件已存在，则在该文件中**追加新的脑暴小节**，而不是创建新文件。
- 如果用户明确指定要写入的 `paperIDEAs` 文件，则遵从用户指定的文件名。

## 4. 脑暴流程与输出结构

整体目标：基于用户原始想法，结合 本地论文库与前沿技术，完成**从发散到收敛**的一次系统脑暴，并留下可直接用于写作/选题的笔记结构。

### 4.1 想法拆解与联想

步骤：

1. **提炼问题核心**：用 1–3 句话重述用户问题，标出「任务 / 场景 / 关键瓶颈」。
2. **多维拆解**：从任务（task）、数据（modality / source）、模型（architecture）、约束（物理 / 语义 / 交互）、评估等维度拆分成若干子问题。
3. **横向联想**：类比 本地知识库中已有的相关主题（如 motion generation、HSI、HOI、Agent 结合等），指出可能的迁移或变体。

在生成笔记时，对应一个小节：

- **章节标题**：`## 一、想法拆解与联想`

### 4.2 真实场景与需求痛点

步骤：

1. **锁定真实场景**：假设或根据用户描述，明确典型应用场景（如 VR 交互、康复训练、动画制作、机器人操作等）。
2. **枚举需求与痛点**：
   - 当前实践 / 系统如何解决？
   - 哪些环节仍依赖人工、效率低或体验差？
   - 安全性、公平性、鲁棒性等隐性需求？
3. **将痛点映射回 idea**：指出原始想法在哪些环节直接缓解或放大价值。

在笔记中生成：

- **章节标题**：`## 二、真实场景与需求痛点`

### 4.3 相关工作支持与研究空间（基于 papers-query-knowledge-base）

始终通过 `papers-query-knowledge-base` skill 使用 本地论文库：

1. **定位相关任务 / 技术**：
   - 根据 idea 的任务属性，在 `paperCollection/by_task/<Task>.md` 中查找。
   - 根据关键技术词，在 `paperCollection/by_technique/_Index.md` 与对应 `<tag>.md` 中查找。
   - 必要时按 venue 在 `paperCollection/by_venue/` 中补充。
2. **选取代表性论文**：
   - 选 3–8 篇与想法高度相关的论文。
   - 对每篇：读取其 `paperAnalysis` 笔记的 frontmatter 中的 `core_operator`、`primary_logic` 以及 TL;DR。
   - 用 1–2 句总结「这篇工作做了什么」以及与当前想法的关系。
3. **总结支持与不足**：
   - **支持**：这些工作在哪些方面证明了 idea 的合理性或潜在价值？
   - **不足 / 空缺**：在哪些维度（场景、数据、约束、评估、系统化程度等）仍然留下研究空间？

在笔记中生成：

- **章节标题**：`## 三、相关工作支持与研究空间`
- 内容结构建议：
  - 「相关工作概览」小节，按任务或技术聚类列出论文及一行摘要。
  - 「支持点」小节，列出对当前想法有力的证据。
  - 「研究空间」小节，列出可投稿 CCF-A / top-tier 的潜在创新切入点。

### 4.4 前沿交叉技术与验证（图像 / 视频 / MLLM / Agent / RL）

目标：将当前 idea 接到更宽广的前沿技术生态，给出可行的交叉研究路径，并附上可进一步阅读的链接。

步骤：

1. **识别适合的前沿方向**：
   - 图像 / 视频生成（diffusion、video diffusion、3D / 4D 表达等）
   - 多模态大模型（MLLM）、多模态 Agent
   - 强化学习（RL）、基于奖励或偏好建模的生成
   - 热门领域的技术或者技术本身可借鉴的核心思想
   - 其它与当前 idea 高度契合的热门技术（如 3DGS、Neural ODE、Conformal prediction 等）
2. **检索与筛选**：
   - 优先利用 本地知识库中已有的相关分析笔记（如有）。
   - 对尚未覆盖的最新方法或应用，使用 Web 搜索获取代表性论文、技术博客或官方文档。
3. **给出交叉路径与验证思路**：
   - 说明该技术如何接入当前 idea 的 pipeline（作为模型 backbone、模块、评估工具、Agent 的子 skill 等）。
   - 给出简单的验证方案或原型实验设想。

在笔记中生成：

- **章节标题**：`## 四、前沿交叉技术与验证思路`
- 内容中应包含一个小表格或列表，格式示例：
  - 技术名称 / 方向
  - 简要说明（1–2 句）
  - 相关链接（论文 / 项目 / 博客）：使用 Markdown 链接，例如 `[Paper Title](https://...)`，避免裸露 URL。

### 4.5 总结与下一步行动

最后，对本次脑暴进行收敛式总结：

1. **总结核心 idea 及其研究位置**：用 3–5 句话概括「要解决什么、用什么关键思想、相对现有工作的新意在哪里」。
2. **列出可执行的下一步**：
   - 数据与场景：需要构建或整理什么样的数据 / 场景？
   - baseline 与实验：可以基于哪些现有方法快速搭 baseline？
   - 指标与评估：如何量化 idea 的优势？
3. **可选：投稿 venue 预判**：如合适，可给出 1–2 个适合的会议 / 期刊及理由。

在笔记中生成：

- **章节标题**：`## 五、总结与下一步`

## 5. 推荐的笔记模板

生成 `paperIDEAs` 笔记时，可以按以下模板组织内容（可根据具体场景略微调整）：

```markdown
---
created: {{ISO_DATETIME_NOW}}
updated: {{ISO_DATETIME_NOW}}
---

# {{YYYY-MM-DD}} {{问题核心简要描述}}

> 基于 paperCollection + paperAnalysis 的系统检索与脑暴；结合前沿技术（图像 / 视频 / MLLM / Agent / RL）给出交叉支持与研究空间。

---

## 一、想法拆解与联想
- 问题重述：
- 关键要素：
- 多维拆解：

## 二、真实场景与需求痛点
- 典型场景：
- 核心需求：
- 现有解决方案及痛点：

## 三、相关工作支持与研究空间
### 3.1 相关工作概览
- [Paper A]：一行摘要 + 与本想法的关系
- [Paper B]：一行摘要 + 与本想法的关系

### 3.2 支持点
- ...

### 3.3 研究空间
- ...

## 四、前沿交叉技术与验证思路
- 技术方向 A：简要说明 + [链接](https://...)
- 技术方向 B：简要说明 + [链接](https://...)

## 五、总结与下一步
- 核心 idea 概括：
- 近期可执行步骤：
- 潜在投稿 venue：
```

## 6. 实施要点

- **始终优先使用 `papers-query-knowledge-base` skill** 来定位 本地知识库内已有分析，然后再补充 Web 搜索的前沿内容。
- **保证输出是结构化 Markdown**，遵循上面的章节结构，方便后续检索与重构。
- 在引用外部资料时，**总是使用 Markdown 链接**，不直接粘贴裸 URL。
- 当用户已经打开某个 `paperIDEAs` 文件并明确要求在当前文件中脑暴时，应在该文件中追加对应结构的小节，而非创建新文件。
