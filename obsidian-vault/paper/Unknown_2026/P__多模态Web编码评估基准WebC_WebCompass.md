---
title: 'WebCompass: Towards Multimodal Web Coding Evaluation for Code Language Models'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.18224
aliases:
- 多模态Web编码评估基准WebCompass
- WebCompass
- WebCompass的核心直觉是：Web编码能力不是单一维度
method: WebCompass
---

# WebCompass: Towards Multimodal Web Coding Evaluation for Code Language Models

[Paper](https://arxiv.org/abs/2604.18224)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Code_Generation]] | **Method**: [[M__WebCompass]]

> [!tip] 核心洞察
> WebCompass的核心直觉是：Web编码能力不是单一维度，而是生成、编辑、修复三种本质不同能力的组合，且每种能力需要不同的评估范式。生成任务的开放性要求执行驱动的智能体评估，编辑/修复任务的局部约束性使清单式LLM评判足够可靠。通过将任务类型与评估协议解耦匹配，WebCompass揭示了现有基准无法观测到的能力边界——尤其是视觉质量和上下文感知修改能力，这两者是当前模型（特别是开源模型）最薄弱的环节。

| 属性 | 内容 |
|------|------|
| 中文题名 | 多模态Web编码评估基准WebCompass |
| 英文题名 | WebCompass: Towards Multimodal Web Coding Evaluation for Code Language Models |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18224) · [Code] · [Project] |
| 主要任务 | Web前端代码的生成、编辑、修复三类任务，支持文本/图像/视频三种输入模态 |
| 主要 baseline | 现有单任务基准（如WebArena、VisualWebArena、SWE-bench等仅覆盖部分任务/模态的基准） |

> [!abstract] 因为「现有Web编码基准仅覆盖文本生成任务、缺乏视频模态、且依赖静态指标无法评估视觉与交互质量」，作者「构建了1526实例的多模态数据集，并针对生成/编辑/修复任务分别设计了Agent-as-a-Judge与LLM-as-a-Judge两种评估范式」，在「10个代表性模型（5闭源+5开源）」上取得「揭示闭源-开源差距超26分、发现视觉质量为全模型瓶颈」的关键发现。

- **闭源-开源差距**：Claude-Opus-4.5 Overall 67.40 vs. 最佳开源 Qwen3-VL-235B-A22B-Instruct 41.14，差距 26.26 分
- **视觉质量瓶颈**：开源模型 DSQ 最低 20.80，顶级闭源模型未突破 65 分
- **编辑任务短板**：开源模型 ITG/FTI/STC 三维度仅 18-28 分，闭源模型 45-72 分

## 背景与动机

Web前端开发是一个多模态、多阶段的复杂工作流。开发者不仅要从零编写代码（生成），还要在现有代码库上做局部修改（编辑），以及定位并修复缺陷（修复）。输入模态也日益多元：自然语言需求文档、UI设计稿静态图、甚至产品演示视频。然而，现有评估基础设施严重滞后于这一现实。

具体而言，当前主流基准存在三重割裂。**任务覆盖上**，WebArena、VisualWebArena 等仅评估从零生成，SWE-bench 聚焦通用软件修复，没有基准同时系统覆盖生成/编辑/修复三类任务。**模态覆盖上**，几乎所有基准停留在文本或静态图像，视频作为动态视觉输入完全缺失——而视频在UI原型演示、交互录屏等场景中普遍存在。**评估维度上**，传统指标如 pass@k、单元测试通过率、截图相似度只能捕捉算法正确性，无法衡量视觉保真度（布局还原、美学一致性）、交互行为（状态转换、响应式设计）和可访问性等Web特有的质量维度。

这一评估空白的后果是严重的：一个模型可能通过DOM结构检查，但其生成页面的视觉质量和交互完整性可能仍然很差；模型在编辑任务中可能破坏原有代码库的隐含约束；修复任务中可能仅消除表面症状而未定位根因。现有静态检查、截图对比、DOM启发式规则等弱代理指标，以及依赖严格属性约定的脚本化测试，均难以泛化到开放式生成任务。因此，领域亟需一个统一覆盖三类任务、三种模态、并采用执行驱动评估的综合基准，以真实反映Web编码智能体的全生命周期能力。

本文提出 WebCompass，通过任务-评估协议解耦匹配的设计，首次实现了对Web编码能力的多维精细刻画。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/500305c7-516b-43c9-9334-012ea9060e20/figures/Figure_3.png)
*Figure 3: Figure 3: Overview of WebCompass. The benchmark supports three input modalities (text, image, video)and three task types (generation, editing, repair), resulting in seven complementary task categories*



## 核心创新

核心洞察：Web编码能力不是单一维度，而是生成、编辑、修复三种本质不同能力的组合，且每种能力的任务特性决定了其最优评估范式——生成任务的开放性要求执行驱动的智能体评估，编辑/修复任务的局部约束性使清单式LLM评判足够可靠。这一解耦匹配使 WebCompass 能够揭示现有基准无法观测到的能力边界，尤其是视觉质量和上下文感知修改能力。

| 维度 | Baseline（现有基准） | 本文（WebCompass） |
|------|----------------------|-------------------|
| 任务覆盖 | 仅生成 或 仅修复，割裂评估 | 生成/编辑/修复三类任务统一覆盖（326/900/300实例） |
| 输入模态 | 文本 或 静态图像 | 文本/图像/视频三种模态全支持 |
| 评估范式 | 单一静态指标（pass@k/截图相似度/DOM规则） | 任务自适应：生成→Agent-as-a-Judge；编辑/修复→LLM-as-a-Judge |
| 评估维度 | 算法正确性单一维度 | 九维度分组设计（RUN/SPI/DSQ/ITG/FTI/STC/RCT/ITI/RFF） |
| 数据构造 | 人工编写或简单爬取 | 多阶段人机协作：原型过滤→人工精选→逆向缺陷注入→难度分级 |

## 整体框架


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/500305c7-516b-43c9-9334-012ea9060e20/figures/Figure_4.png)
*Figure 4: Figure 4: Data construction pipeline for WebCompass. Top: prototypes are collected through multi-stagefiltering, manual selection, and page-level expansion. Bottom: each prototype is converted into ed*



WebCompass 的整体框架由**数据构建层**和**评估协议层**两大支柱构成，形成从任务定义到能力刻画的完整闭环。

**数据构建层**（对应 Figure 4）：输入为原始Web原型资源，经过多阶段过滤（自动质量筛选 + 人工精选）→ 原型分类标注（15个生成领域/16种编辑操作/11种缺陷类型）→ 任务实例化（按 Easy/Medium/Hard 三级标注难度）→ 输出 1526 个结构化任务实例。生成任务 326 个（文本引导/视觉引导/视频引导），编辑任务 900 个（文本引导/视觉引导，16种操作类型如添加组件、修改样式、重构布局），修复任务 300 个（诊断修复/视觉诊断修复，逆向工程注入缺陷）。

**评估协议层**针对任务特性差异化设计：
- **编辑/修复任务** → LLM-as-a-Judge（Figure 5）：输入为前后代码 diff + 静态截图对比，输出为清单式评分（ITG/FTI/STC 或 RCT/ITI/RFF）。因输出空间受约束，静态检查可靠。
- **生成任务** → Agent-as-a-Judge（Figure 6）：输入为生成代码，Claude Code（v2.0.67）作为评估编排器，通过 Chrome DevTools MCP Server（v0.19.0）在无头Chromium中启动网站，自动探索交互行为，迭代合成测试用例，输出三维度评分（RUN/SPI/DSQ）。

**评分聚合层**：九维度按任务类型分组，Overall 为算术平均，支持能力剖面独立分析。

```
原始资源 → [多阶段过滤] → 原型库 → [任务实例化] → 1526任务实例
                                              ↓
                    ┌─────────────────────────┼─────────────────────────┐
                    ↓                         ↓                         ↓
              生成任务(326)              编辑任务(900)               修复任务(300)
              文本/图像/视频输入         文本/图像输入                缺陷代码输入
                    ↓                         ↓                         ↓
           [Agent-as-a-Judge]         [LLM-as-a-Judge]            [LLM-as-a-Judge]
           Claude Code + MCP           清单式diff检查               清单式diff检查
           RUN/SPI/DSQ评分             ITG/FTI/STC评分              RCT/ITI/RFF评分
                    └─────────────────────────┬─────────────────────────┘
                                              ↓
                                    [九维度算术平均] → Overall
```

## 核心模块与公式推导

### 模块 1: 任务自适应评估协议选择（对应框架图 评估协议层）

**直觉**: 不同任务的本质特性（输出空间约束程度）决定了评估方法的上界——开放式任务必须执行验证，约束式任务可静态检查。

**Baseline 公式**（现有统一评估范式）: 对所有任务采用单一指标
$$S_{uniform} = \text{pass@}k \quad \text{或} \quad S_{uniform} = \text{SSIM}(\text{screenshot}_{pred}, \text{screenshot}_{ref})$$
符号: $k$ = 采样次数; SSIM = 结构相似性指数; 截图对比仅捕捉像素级相似。

**变化点**: 统一指标对生成任务严重不足——pass@k 无法评估视觉质量，SSIM 对布局偏移敏感但对交互行为无知；对编辑/修复任务又过度复杂——diff 空间的局部修改无需昂贵执行验证。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{T} \in \{\text{Generation}, \text{Editing}, \text{Repair}\} \quad \text{任务类型判别}$$
$$\text{Step 2}: \quad \text{Evaluator}(\mathcal{T}) = \begin{cases} \text{Agent-as-a-Judge} & \text{if } \mathcal{T} = \text{Generation} \\ \text{LLM-as-a-Judge} & \text{if } \mathcal{T} \in \{\text{Editing}, \text{Repair}\} \end{cases} \quad \text{协议路由}$$
$$\text{最终}: \quad S_{final}(\mathcal{T}) = \text{Evaluator}(\mathcal{T})\left(\text{output}, \text{ground truth}, \text{context}\right)$$

**对应消融**: 若对生成任务强制使用 LLM-as-a-Judge（静态检查），RUN/SPI/DSQ 维度将系统性低估，因交互行为无法观测。

### 模块 2: Agent-as-a-Judge 执行驱动评估（对应框架图 Figure 6）

**直觉**: 生成任务的正确性往往依赖多步运行时行为，必须将代码部署为可运行服务，通过主动探索验证功能完备性。

**Baseline 公式**（静态评估）: 
$$S_{static} = f_{LLM}\left(\text{code}_{pred}, \text{prompt}\right) \in [0, 100]$$
符号: $f_{LLM}$ = 大语言模型评判函数; 仅基于代码文本和提示词，无运行时信息。

**变化点**: 静态评估无法捕捉（1）JavaScript 动态执行错误、（2）响应式布局在不同视口的表现、（3）用户交互触发的状态转换。需要引入执行环境和主动探索机制。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{Env} = \text{Launch}\left(\text{code}_{pred}, \text{Chromium headless}\right) \quad \text{通过 MCP Server 启动无头浏览器}$$
$$\text{Step 2}: \quad \text{Traces} = \text{Explore}\left(\text{Env}, \text{Claude Code}, \{\text{click}, \text{scroll}, \text{input}, \text{resize}\}\right) \quad \text{主动交互探索}$$
$$\text{Step 3}: \quad \text{Tests}_{synth} = \text{Synthesize}\left(\text{Traces}, \text{spec}\right) \quad \text{迭代合成针对性测试用例}$$
$$\text{最终}: \quad S_{gen} = \frac{1}{3}\left(\text{RUN} + \text{SPI} + \text{DSQ}\right)$$
其中:
- RUN (Runnability) = $\mathbb{1}[\text{no runtime errors}] \times 100$，可运行性
- SPI (Specification Implementation) = $f_{LLM}(\text{observed behavior}, \text{spec}) \in [0, 100]$，规格实现度
- DSQ (Design Quality) = $f_{LLM}(\text{screenshot}, \text{design reference}) \in [0, 100]$，设计质量

**对应消融**: Figure 8 显示 Agent-based automatic evaluation 与 Human evaluation 在三个任务上的排名高度一致，验证了执行驱动评估的人类对齐度。

### 模块 3: 九维度评分聚合与能力剖面（对应框架图 评分聚合层）

**直觉**: 单一综合分数会掩盖任务间差异，需要分组维度使不同能力独立可解释。

**Baseline 公式**（传统综合评分）: 
$$S_{overall}^{baseline} = \text{mean}\left(\{\text{pass@}1, \text{BLEU}, \text{ROUGE}\}\right)$$
符号: 混合代码正确性与文本相似度指标，无Web特定维度。

**变化点**: 传统指标（1）不区分任务类型、（2）忽略视觉与交互维度、（3）无法定位能力短板。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{s}_{gen} = [\text{RUN}, \text{SPI}, \text{DSQ}] \quad \text{生成任务三维向量}$$
$$\text{Step 2}: \quad \mathbf{s}_{edit} = [\text{ITG}, \text{FTI}, \text{STC}] \quad \text{编辑任务三维向量}$$
$$\text{其中}: \text{ITG} = \text{Instruction Target Grounding}, \text{FTI} = \text{Functional Target Implementation}, \text{STC} = \text{Style and Theme Consistency}$$
$$\text{Step 3}: \quad \mathbf{s}_{repair} = [\text{RCT}, \text{ITI}, \text{RFF}] \quad \text{修复任务三维向量}$$
$$\text{其中}: \text{RCT} = \text{Root Cause Targeting}, \text{ITI} = \text{Interaction Target Integrity}, \text{RFF} = \text{Repair Functional Fidelity}$$
$$\text{最终}: \quad S_{overall} = \frac{1}{9}\left(\sum_{i=1}^{3} s_{gen,i} + \sum_{j=1}^{3} s_{edit,j} + \sum_{k=1}^{3} s_{repair,k}\right)$$

**对应消融**: Figure 10 显示编辑任务 16 种操作类型的得分分布，STC（样式一致性）为普遍短板；Figure 15 显示按难度分层的性能衰减模式，Hard 任务 DSQ 维度下降最剧烈。

## 实验与分析

实验评估了 10 个代表性模型：5 个闭源（Claude-Opus-4.5、GPT-4o 等）和 5 个开源 Qwen3-VL 系列（235B-A22B-Instruct/Thinking、30B、8B 等）。

| 模型类别 | 代表模型 | Overall | 生成 DSQ | 编辑 STC | 修复 RCT |
|---------|---------|---------|---------|---------|---------|
| 最佳闭源 | Claude-Opus-4.5 | 67.40 | ~65 | ~72 | 
| 最佳开源 | Qwen3-VL-235B-A22B-Instruct | 41.14 | 20.80 | 18-28 | 
| 差距 | — | **26.26** | **>44分** | **>44分** | — |
| 30B级开源 | Qwen3-VL-30B-A3B-Instruct | <33.5 | 


![Figure 15](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/500305c7-516b-43c9-9334-012ea9060e20/figures/Figure_15.png)
*Figure 15: Figure 11: Performance comparison across Generation, Editing, and Repair tasks by difficulty level.*



**核心发现分析**:

（1）**闭源-开源鸿沟具有统计显著性**：26.26 分的 Overall 差距远超随机波动，30B 量级开源模型得分不足顶级闭源模型的一半。这一差距在编辑任务最极端：开源模型 ITG/FTI/STC 三维度仅 18-28 分，闭源模型 45-72 分，揭示**上下文感知代码修改能力**是开源模型的重大缺口——编辑任务要求保留原有代码库功能的前提下精确执行局部修改，对代码理解和变更影响分析要求极高。

（2）**视觉质量为全模型瓶颈**：DSQ（生成）和 STC（编辑）是所有模型的最低得分轴，开源模型 DSQ 最低仅 20.80，即使 Claude-Opus-4.5 也未突破 65 分。这表明**视觉前端设计**是当前所有模型的普遍短板，无论规模大小。
![Figure 10](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/500305c7-516b-43c9-9334-012ea9060e20/figures/Figure_10.png)
*Figure 10: Figure 9: Overall score breakdown for editing tasks across 16 operation types. Scores are computed as theharmonic mean of Instruction Targeting, Feature Integrity, and Style Conformance per subtask, a*



（3）**修复任务的独特模式**：ITI（交互完整性）远高于 RFF 和 RCT，原因是 11 种缺陷类型中 9 种为视觉/语义类，交互层几乎不受影响；而 **RCT（根因定位）是所有模型的最低分维度**，说明"诊断能力"（定位缺陷根因）远难于"修复能力"（消除表面症状）。

（4）**难度分层衰减**（Figure 15）：Hard 任务在所有维度显著低于 Easy/Medium，DSQ 维度衰减最剧烈，验证视觉质量对复杂度最敏感。

**公平性检查与局限**:
- **Baseline 强度**：未包含 Llama-3 系列、DeepSeek-Coder-V3 等主流开源模型，可能低估开源上限
- **计算成本**：Agent-as-a-Judge 依赖 Claude Code + Chromium + MCP Server，单次评估成本显著高于静态检查
- **版本依赖风险**：Chrome DevTools MCP Server v0.19.0 和 Claude Code v2.0.67 的特定版本可能引入可复现性问题
- **Thinking 模式反直觉**：Qwen3-VL-235B-Thinking Overall 38.78 不优于 Instruct 模式 41.14，但仅基于单一模型系列，泛化性存疑
- **数据集规模**：1526 实例相比部分单任务基准无数量级优势，但多任务覆盖密度更高

## 方法谱系与知识库定位

**方法家族**：代码生成评估基准 → 多模态扩展 → 任务自适应评估

**父方法/直接继承**：WebArena（文本生成）、VisualWebArena（视觉生成）、SWE-bench（代码修复）——WebCompass 将这三条割裂的线统一为多模态、多任务的评估框架。

**改变的插槽**：
| 插槽 | 父方法做法 | WebCompass 改变 |
|------|----------|----------------|
| 数据 curation | 单任务人工编写/爬取 | 多阶段人机协作 + 逆向缺陷注入 + 难度分级 |
| 评估 recipe | 统一静态指标 | 任务类型→评估协议解耦路由 |
| 评估 mechanism | 单元测试/截图对比/LLM打分孤立使用 | Agent执行驱动 + LLM清单检查协同 |
| 评估维度 | 算法正确性单一维度 | 九维度分组，视觉/交互质量显式量化 |

**直接 Baselines 与差异**：
- **WebArena/VisualWebArena**：仅覆盖生成任务，无编辑/修复，无视频模态，评估依赖人工标注或简单规则
- **SWE-bench**：仅覆盖修复任务，文本模态，评估依赖单元测试通过率，无视觉维度
- **HumanEval/MBPP**：仅算法代码生成，无Web前端特性，无多模态输入

**后续方向**：
1. **规模扩展**：将 1526 实例扩展至万级，覆盖更多模型家族（Llama-3、DeepSeek-Coder 等）以缩小开源评估空白
2. **评估协议泛化**：将 Agent-as-a-Judge 的 MCP 桥接机制推广至其他需要运行时验证的开放式生成任务（如移动应用、桌面软件）
3. **诊断能力强化**：针对 RCT（根因定位）这一全模型最低分维度，设计专门的诊断能力训练目标和评估协议

**知识库标签**：
- 模态 facet: 多模态（文本+图像+视频）
- 范式 facet: 评估基准 / 任务自适应评估
- 场景 facet: Web前端开发 / 代码智能体
- 机制 facet: Agent-as-a-Judge / LLM-as-a-Judge / MCP桥接执行验证
- 约束 facet: 开放式生成 vs. 约束式编辑的评估方法选择

