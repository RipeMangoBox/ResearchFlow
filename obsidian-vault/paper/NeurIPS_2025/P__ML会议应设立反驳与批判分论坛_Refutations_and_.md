---
title: 'Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- ML会议应设立"反驳与批判"分论坛
- Refutations and
- Refutations and Critiques (R&C) Track
- ML conferences should establish a d
acceptance: Oral
cited_by: 3
method: Refutations and Critiques (R&C) Track
modalities:
- Text
---

# Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__Refutations_and_Critiques_(R&C)_Track]]

> [!tip] 核心洞察
> ML conferences should establish a dedicated "Refutations and Critiques" (R&C) Track to provide a high-profile, reputable platform for critical re-examination of prior research and foster a self-correcting research ecosystem.

| 中文题名 | ML会议应设立"反驳与批判"分论坛 |
| 英文题名 | Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track |
| 会议/期刊 | NeurIPS 2025 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.19882) |
| 主要任务 | Benchmark / Evaluation, Scientific Self-Correction in ML |
| 主要 baseline | 无（Position paper，非技术方法论文） |

> [!abstract] 因为「ML会议缺乏官方机制来批判和纠正有问题的已发表研究，导致错误、缺陷乃至潜在欺诈性研究长期存在于科学记录中」，作者「在现有会议组织结构」基础上提出「设立专门的 Refutations and Critiques (R&C) Track」，并以「对一篇ICLR 2025 Oral论文的示范性批判提交」展示该机制如何运作。

- 本文是 NeurIPS 2025 Oral position paper，无传统实验指标
- 核心贡献：提出 R&C Track 的正式结构设计，包含审稿原则、潜在陷阱分析及缓解策略
- 提供一份示范性（illustrative example）R&C 提交，针对近期 ICLR 2025 Oral 论文进行批判

## 背景与动机

机器学习领域正以惊人速度产出论文，但科学自我修正机制却严重滞后。一个具体例子是：Athalye et al. (2018) 的 "Obfuscated gradients give a false sense of security" 花了大量精力才打破当时多个对抗防御论文的虚假安全声明，而这类高价值的批判工作往往缺乏正式的发表渠道和学术认可。类似地，Carlini & Wagner (2017) 对 MagNet 防御的系统性反驳，以及 Carlini et al. (2022) 对 "privacy for free" 声明的驳斥，都属于对领域健康发展至关重要的工作，却只能在主会轨道外艰难寻找生存空间。

现有机制如何处理这一问题？第一，**Workshop 和 blog post**：如 Carlini (2020) 的 InstaHide 批判博客，但缺乏同行评审的正式性和持久性；第二，**Twitter/X threads 等社交媒体**：传播快但易逝、无质量把控；第三，**ReScience C 等复现期刊**：专注计算复现而非直接批判；第四，**依赖审稿过程改革**：Beygelzimer et al. (2021) 的 NeurIPS 一致性实验已表明，单纯改进审稿流程因"审稿人激励不足"和"机构对同行评审的舒适依赖"而难以奏效。

这些渠道的根本缺陷在于：**没有为高影响力批判提供声誉激励和正式学术认可**。Agarwal et al. (2021) 对深度强化学习统计实践的批判、Alemohammad et al. (2023) 对自消耗生成模型的警告，都说明领域需要系统性纠错，但批判者反而面临职业风险——耗时费力、易被视作"挑事"、难以获得引用和晋升积分。本文的核心动机正是填补这一结构性空白：建立一个官方、高质量、有声誉保障的批判发表平台。

## 核心创新

核心洞察：ML 会议需要一个**内嵌的、有声誉回报的纠错机制**，因为当前领域的高速论文产出与几乎为零的正式纠错渠道之间存在结构性失衡，从而使科学记录的自我净化成为可能。

与现有机制的差异在于：

| 维度 | 现有机制（Workshop/Blog/社交媒体） | 本文 R&C Track |
|:---|:---|:---|
| **正式性与持久性** | 非正式、易逝、无 DOI | 会议正式轨道，同行评审，永久存档 |
| **声誉激励** | 批判者承担职业风险，获认可困难 | 提供高可见度平台，批判作为可引用学术贡献 |
| **原论文作者参与** | 通常缺乏结构化互动，易演变为冲突 | 设计原论文作者参与审稿的机制，促进建设性对话 |
| **投稿类型覆盖** | 多为外部第三方批判 | 同时支持外部批判（external critiques）和自我批判（self-critiques） |
| **质量把控** | 参差不齐 | 专门的审稿原则，防范恶意攻击和 gaming |

## 整体框架

R&C Track 的框架可概括为四个核心组件，形成从投稿到发表的完整流程：

**输入**：已发表于 ML 会议/期刊的论文，或已公开的研究成果（包括预印本）。

**组件 A — 双重投稿类型（Dual Submission Types）**：
- **外部批判（External Critiques）**：第三方研究者对已有工作提出反驳，指出错误、缺陷、不可复现性或夸大声明
- **自我批判（Self-Critiques）**：原论文作者主动承认并纠正自己工作的局限或错误

**组件 B — 审稿流程设计（Review Process Design）**：
- 专门的审稿原则，确保批判基于技术事实而非人身攻击
- **原论文作者参与机制**：邀请被批判论文的作者参与审稿过程，提供回应和澄清机会

**组件 C — 质量保障与陷阱防范（Quality Assurance & Pitfall Mitigation）**：
- 识别并缓解潜在滥用：如竞争对手恶意攻击、选择性批判、琐碎化批判等
- 设计激励相容机制，确保审稿人和投稿者都有动机维护高标准

**组件 D — 输出与认可（Publication & Recognition）**：
- 接收的 R&C 论文作为正式会议出版物，获得与主会论文同等的学术认可
- 建立与原始论文的关联机制，使科学记录形成可追溯的修正链条

```
已发表论文 ──→ [投稿类型选择] ──┬──→ 外部批判 ──→ [专门审稿] ──→ [原作者回应] ──→ [质量把控] ──→ 正式发表
                                └──→ 自我批判 ──→ [专门审稿] ───────────────────────────→ [质量把控] ──→ 正式发表
```

**示范性实例**：作者提供了一份完整的 R&C 提交示例，针对一篇近期 ICLR 2025 Oral 论文进行系统性批判，展示从问题识别、证据呈现到结论撰写的完整规范。

## 核心模块与公式推导

本文作为 position paper，不涉及传统技术方法的数学公式或优化目标。以下对三个核心设计模块进行结构化分析，以"设计约束 → 机制选择 → 预期效果"的逻辑展开：

### 模块 1: 双重投稿类型（Dual Submission Types）

**直觉**：纠错动力可能来自外部发现，也可能来自内部反思，两种来源需要不同的激励结构。

**Baseline（现有机制）**：仅有外部纠错渠道（如博客、社交媒体），缺乏对原作者主动纠错的激励。

**设计约束**：
- 外部批判易引发对抗性冲突，原作者可能抵触
- 自我批判在现有体系中几乎无回报，作者缺乏动力公开承认错误

**本文机制**：
- **外部批判**：设立正式投稿通道，要求基于技术证据而非主观否定；审稿标准侧重"是否实质性影响原论文结论"
- **自我批判**：同等认可自我纠错的价值，将其视为研究成熟度和诚信的标志
- **关键设计**：两种类型共享同一审稿质量门槛，避免自我批判沦为"免责声明"式轻量内容

**对应讨论**：作者指出自我批判在医学等领域已有先例，ML 领域需建立类似文化。

---

### 模块 2: 原论文作者参与审稿（Original Author Involvement）

**直觉**：被批判者最了解自身工作的细节，其参与能提升审稿质量，但需防范利益冲突。

**Baseline（传统同行评审）**：审稿人匿名，作者无直接参与机会；对于批判性工作，传统模式易导致双方信息不对等。

**设计约束**：
- 完全排除原作者：可能遗漏关键上下文，批判基于误解
- 完全由原作者主导：利益冲突导致批判被压制

**本文机制（结构化参与）**：
- **Step 1**：R&C 投稿进入审稿流程后，系统自动通知被批判论文的通讯作者
- **Step 2**：原作者可选择提交"回应文档"（response document），对批判中的事实陈述进行确认或反驳
- **Step 3**：审稿人评估 R&C 投稿时，必须同时考虑原作者回应，但在回应存在争议时，以可验证的技术证据为最终依据
- **Step 4**：最终发表的 R&C 论文可附带原作者回应（或注明"原作者选择不回应"），形成完整对话记录

**关键保障**：原作者回应不构成"否决权"，仅作为审稿参考信息，防止机制被捕获。

---

### 模块 3: 陷阱识别与缓解（Pitfall Mitigation）

**直觉**：任何正式机制都可能被 gaming，需预判滥用模式并嵌入防御。

**Baseline（无专门机制）**：批判的质量和动机无系统审查，导致噪音甚至恶意攻击泛滥。

**识别的核心陷阱**：

| 陷阱类型 | 具体表现 | 缓解策略 |
|:---|:---|:---|
| **选择性批判（Cherry-picking）** | 专挑易攻击的薄弱工作，回避真正高影响力的错误 | 审稿标准强调"被批判工作的影响力"与"批判的实质性"并重 |
| **琐碎化批判（Trivial Critiques）** | 对排版错误、表述不清等次要问题大做文章 | 明确排除标准：不影响核心结论的问题不接收 |
| **恶意竞争攻击** | 竞争对手系统性打压对方工作 | 双盲审稿、利益冲突声明、对同一作者的频繁批判触发额外审查 |
| **钓鱼式批判（Fishing Expeditions）** | 无明确证据先发起批判，迫使原作者自证清白 | 投稿需包含初步技术证据，不能仅为质疑性陈述 |

**本文机制**：上述缓解策略嵌入审稿指南（reviewer guidelines），而非仅依赖审稿人自由裁量。

## 实验与分析

本文作为 position paper，不提供传统意义上的实验验证。但作者通过以下方式支撑其论点：

**示范性实例（Illustrative Example）**：作者撰写了一份完整的 R&C 提交，针对一篇近期 ICLR 2025 Oral 论文进行批判。该示例展示了：
- 如何结构化地识别原论文的核心声明
- 如何设计对照实验或重新分析来检验这些声明
- 如何撰写既尖锐又建设性的批判文本
- 如何回应可能的原作者辩护

这一示例的功能类似于"概念验证"（proof of concept），证明 R&C Track 的投稿格式和流程是可行的。然而，作者明确承认这是**单一案例、非系统性评估**。

**文献支撑强度**：作者引用了大量历史案例证明批判工作的价值和现有渠道的不足，包括：
- Athalye et al. (2018) 打破对抗防御虚假安全声明的经典工作
- Carlini & Wagner (2017) 对 MagNet 的系统性反驳
- Carlini et al. (2022) 对 "privacy for free" 声明的驳斥
- Beygelzimer et al. (2021) 的 NeurIPS 一致性实验揭示审稿系统固有局限

**公平性评估与局限**：
- **无实证验证**：作者未进行任何用户研究、试点项目或模拟实验来验证 R&C Track 的实际效果
- **无定量证据**：未提供数据说明现有机制的失败率，或 R&C Track 可能提升的纠错效率
- **激励假设未检验**：假设"正式发表渠道 + 声誉回报"能有效激励批判行为，但这一因果链条缺乏证据
- **机构采纳障碍**：未深入分析会议组织者的实际激励——为何要在有限议程中分配资源给可能引发争议的 R&C Track
- **潜在的滥用风险**：虽识别了多种陷阱，但缓解策略的有效性同样未经检验

总体而言，本文的证据强度较低（overall evidence strength: 0.2），其说服力主要依赖逻辑论证和历史案例的类比，而非系统评估。

## 方法谱系与知识库定位

**方法家族**：科学元研究（Meta-Research）/ 学术基础设施设计（Academic Infrastructure Design）

**直接相关先行工作**：
- **Beygelzimer et al. (2021)** — NeurIPS 一致性实验：揭示审稿系统的不一致性，本文据此论证"单纯改革审稿流程不足够"，需增设专门轨道
- **Agarwal et al. (2021)** — 深度强化学习统计实践批判： exemplar 式的高价值批判工作，说明此类贡献需要更好发表渠道
- **Carlini & Wagner (2017); Athalye et al. (2018); Carlini et al. (2022)** — 对抗安全与隐私领域的系列反驳工作：本文的核心"正例"，证明批判对领域健康发展的关键作用
- **ReScience C 等复现期刊**：功能相近但定位不同——ReScience C 专注计算复现，R&C Track 专注直接批判与错误纠正

**与 baseline 的差异**：本文无技术 baseline，其"创新"在于**会议组织结构设计**而非算法或模型。改变的 slot 为：
- **training_recipe**: 不适用（非技术方法）
- **data_curation**: 不适用
- **architecture / objective / inference**: 不适用
- **conference_organization**: 提出全新轨道结构、审稿原则、投稿类型分类

**后续可能方向**：
1. **试点评估**：在小型会议（如 COLT、AISTATS）或 workshop 中试行 R&C Track，收集定量数据验证效果
2. **激励模型形式化**：用博弈论或机制设计理论分析 R&C Track 的激励相容性，预测均衡行为
3. **跨领域比较**：调研医学（如 PubPeer）、物理学（如 arXiv comment）等领域的纠错机制，提炼可迁移设计

**知识库标签**：
- **modality**: text / academic_publishing
- **paradigm**: position_paper / policy_proposal
- **scenario**: conference_organizational_design
- **mechanism**: peer_review_reform, scientific_self-correction
- **constraint**: no_empirical_validation, incentive_design_challenge
