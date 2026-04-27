---
title: 'SWE-chat: Coding Agent Interactions From Real Users in the Wild'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.20779
aliases:
- 真实场景下AI编码代理交互数据集SWE-chat
- SWE-chat
method: SWE-chat
---

# SWE-chat: Coding Agent Interactions From Real Users in the Wild

[Paper](https://arxiv.org/abs/2604.20779)

**Topics**: [[T__Agent]], [[T__Code_Generation]] | **Method**: [[M__SWE-chat]]

| 中文题名 | 真实场景下AI编码代理交互数据集SWE-chat |
| 英文题名 | SWE-chat: Coding Agent Interactions From Real Users in the Wild |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20779) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 构建真实用户与AI编码代理交互的大规模数据集；分析真实使用模式、失败模式、代码采纳率与人机协作动态 |
| 主要 baseline | SWE-bench, SWE-smith-trajectories, CoderForge-Preview |

> [!abstract] 因为「现有评估依赖人工策划的孤立基准，缺乏真实开发工作流中的迭代交互数据」，作者在「GitHub公开仓库的Claude Code使用数据」基础上构建了「SWE-chat数据集」，在「真实场景分析」上取得「覆盖8,000+仓库、10,000+会话的系统性实证发现」

- **规模**: 10,000+ 真实用户-编码代理会话，覆盖 8,000+ 公开GitHub仓库
- **核心发现**: 仅 35.7% 的代理生成代码在后续被修改或删除（即64.3%存活），远低于基准测试暗示的效能
- **模式识别**: 识别出三种编码模式——human-only (0%代理代码)、collaborative (混合)、agent-dominated (高比例代理代码)

## 背景与动机

AI编码代理（如Claude Code、GitHub Copilot Chat、Cursor）正被开发者大规模采用，但我们对这些工具在真实工作环境中的实际表现知之甚少。想象一位开发者面对一个复杂的遗留代码库：她不会一次性给出完美提示，而是与代理进行多轮对话，不断修正方向、审查输出、决定采纳或丢弃代码。这种动态的人机协作过程——充满模糊性、迭代性和即兴判断——与当前主流评估范式形成鲜明对比。

现有方法如何处理这一问题？**SWE-bench** 及其变体通过精心策划的GitHub issue-PR对，在隔离环境中测试代理端到端解决能力，但完全排除了真实用户的参与和反馈循环。**SWE-smith-trajectories** 记录了代理工具调用轨迹和代码差异，却缺乏真实用户提示，无法还原"谁说了什么、为什么这样说"的交互上下文。**CoderForge-Preview** 同样提供代理执行轨迹，但缺少关键的代码归因信息——无法区分最终代码中哪些是人类编写、哪些是代理生成。

这些方法的共同短板在于：**它们都剥离了真实开发工作流的核心要素——人的能动性**。具体而言，现有数据无法回答：开发者实际上如何提示和引导代理？代理生成的代码有多少被采纳、多少被丢弃？代理在哪些环节失败？用户如何识别和应对这些失败？在代理自主性快速提升的背景下（如Claude Code可自主执行多步工具调用），这种认知缺口尤为危险——我们正在大规模部署一项我们尚未充分理解其实际效果的技术。

本文通过挖掘GitHub公开仓库中Claude Code的真实使用日志，构建首个大规模真实人机编码交互数据集SWE-chat，系统填补上述空白。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/975f7a52-9ca6-46d2-bdd1-5587b2c54360/figures/Figure_1.png)
*Figure 1: Figure 1: We present SWE-chat, a continually growing dataset of real human-coding agentinteractions collected from public GitHub repositories. Developers opt in via installingEntire.io, an open-source*



## 核心创新

核心洞察：真实开发工作流中的"代码归因"是理解代理有效性的关键钥匙，因为现有数据集将人机贡献混为一谈导致系统性高估代理效能，从而使基于真实交互痕迹的细粒度人机协作分析成为可能。

| 维度 | Baseline (SWE-bench/SWE-smith) | 本文 (SWE-chat) |
|:---|:---|:---|
| 数据来源 | 人工策划的issue-PR对或代理自主执行轨迹 | 真实用户主动发起的Claude Code会话日志 |
| 用户参与 | 无（完全自主代理） | 完整保留：用户提示、中断、修正、决策 |
| 代码归因 | 无（仅最终diff，不区分人机） | 逐行归因：人类编写 vs 代理生成，含后续修改/删除追踪 |
| 评估视角 | 任务完成率（二进制：解决/未解决） | 代码生存率、会话成功率分布、turn-level oversight模式 |
| 时间维度 | 单次执行快照 | 完整会话历史，含多轮迭代和长期代码演化 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/975f7a52-9ca6-46d2-bdd1-5587b2c54360/figures/Figure_3.png)
*Figure 3: Figure 3: Structure of a coding agent sessionin SWE-chat. Each session consists of alternat-ing user prompts and agent responses withtool calls (file reads, edits, shell commands)and text output.*



SWE-chat的数据构建与分析方法遵循"采集→结构化→归因→分析"四阶段流程：

**输入**: GitHub公开仓库的Claude Code使用日志（用户主动选择分享的使用数据）。

**模块A - 会话提取与清洗（Session Extraction）**: 从原始日志中识别独立的用户-代理会话边界，过滤自动化脚本和非交互式使用。输入为原始Claude Code日志流，输出为结构化会话序列。

**模块B - 交互结构化（Interaction Structuring）**: 将会话解析为交替的user prompt / agent response / tool call三层结构。每个agent response可包含文本回复、代码建议、文件编辑、终端命令等工具调用。输入为清洗后的会话，输出为带时间戳的turn-level交互树。

**模块C - 代码归因引擎（Code Attribution）**: 核心创新模块。通过对比会话前后的代码状态，结合agent response中的编辑建议，逐行标注最终代码的人类/代理来源；并追踪代理生成代码在后续commit中的修改、删除或保留情况。输入为结构化会话+仓库代码历史，输出为带归因标签的代码演化图谱。

**模块D - 多维分析框架（Analytical Layer）**: 基于归因结果，从三个层面展开分析：(1) 会话层面——成功率分布、turn数量、工具调用模式；(2) 代码层面——agent代码占比、生存率、三种编码模式分类；(3) 行为层面——用户中断频率、agent主动请求澄清的比例。

```
原始Claude Code日志 → [A提取清洗] → 会话序列 → [B结构化] → Turn-level交互树
                                                          ↓
仓库Git历史 + 代码状态 ← [C代码归因] ← 代码演化图谱 ← 编辑操作序列
                                                          ↓
                                               [D多维分析] → 使用模式/失败模式/效率指标
```

## 核心模块与公式推导

### 模块1: 会话结构化与Turn定义（对应框架图 B位置）

**直觉**: 真实交互不是简单的请求-响应，而是嵌套了工具调用的复杂对话，需要明确定义"turn"作为分析原子单位。

**Baseline 公式** (传统对话分析): 通常将每次用户发送视为一个turn，即 $T = \{(u_1, a_1), (u_2, a_2), ...\}$，其中 $u_i$ 为用户utterance，$a_i$ 为agent回复。

符号: $u_i$ = 第i轮用户提示, $a_i$ = 第i轮agent响应, $\tau_{i,j}$ = 第i轮中第j个工具调用

**变化点**: 传统定义无法捕捉编码代理的核心行为——工具调用。Claude Code等代理在单轮响应中可能执行多次文件读写、终端命令、搜索等操作，这些工具调用与用户提示共同构成完整的"交互单元"。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{a}_i = (text_i, \{\tau_{i,j}\}_{j=1}^{m_i}) \quad \text{将agent响应扩展为文本+工具调用集合}$$
$$\text{Step 2}: \mathcal{T}_i = (u_i, \tilde{a}_i, \Delta_i) \quad \text{加入代码状态变化}\Delta_i\text{作为turn的输出痕迹}$$
$$\text{最终}: \mathcal{S} = \{\mathcal{T}_i\}_{i=1}^{N} \text{ with metadata } (repo, user\_type, timestamp, exit\_status)$$

**对应消融**: 

### 模块2: 代码归因与生存率计算（对应框架图 C位置）

**直觉**: 必须区分"代理生成了代码"和"代理生成的代码被保留"，后者才是真实效能指标。

**Baseline 公式** (SWE-smith等): 仅计算agent产生的代码差异量，即 $L_{base} = |\text{diff}_{agent}|$，不追踪后续命运。

符号: $c$ = 代码行, $author(c) \in \{H, A\}$ = 人类或代理, $t_0$ = 生成时间, $t_1, t_2, ...$ = 后续commit时间

**变化点**: 基线方法隐含假设所有agent生成代码等效采纳，但真实开发中大量代码被立即修改或很快删除。需要引入时间维度追踪"代码生存"。

**本文公式（推导）**:
$$\text{Step 1}: \hat{c} = (content, author, t_0, context) \quad \text{为每行代码建立带作者标签的实体}$$
$$\text{Step 2}: survival(c, t_k) = \mathbb{1}[\exists c' \in C_{t_k} : match(c, c') \wedge lineage(c') = c] \quad \text{追踪代码在后续版本中的存在性}$$
$$\text{Step 3}: \text{survival\_rate}_A = \frac{\sum_{c: author(c)=A} \int_{t_0}^{T} survival(c,t)dt}{|C_A| \cdot (T-t_0)} \quad \text{代理代码的时间加权平均生存率}$$
$$\text{最终}: \text{Code\_Survival} = \frac{|\{c : author(c)=A \wedge \exists t>t_0, survival(c,t)=1\}|}{|\{c : author(c)=A\}|} = 64.3\% \text{ (Table 3)}$$

**对应消融**: Table 3显示，按编码模式细分后，agent-dominated模式的代码生存率显著低于collaborative模式。

### 模块3: 会话成功率自动标注（对应框架图 D位置）

**直觉**: 缺乏人工标注时，需要可靠的自动方法评估会话质量，但简单规则会误判复杂的部分成功场景。

**Baseline 公式** (SWE-bench式): $success = \mathbb{1}[\text{test\_pass}]$，即测试通过为成功，否则失败。

符号: $s$ = 会话, $r_i$ = 第i轮LLM评定的成功程度, $y$ = 最终标注标签

**变化点**: 真实会话无预设测试，且成功是渐进的（部分任务完成、代码需后续调整）。引入LLM-as-judge进行turn-level和session-level双层评估。

**本文公式（推导）**:
$$\text{Step 1}: r_i = LLM_{judge}(prompt_i, response_i, \Delta_i, context) \in \{1,2,3,4,5\} \quad \text{每轮交互质量评分}$$
$$\text{Step 2}: R_{session} = \text{aggregate}(\{r_i\}) \text{ with turn-weighting by code impact} \quad \text{按代码影响加权聚合}$$
$$\text{Step 3}: y = \text{mode}(R_{session}) \text{ with calibration on human-labeled subset} \quad \text{在人类标注子集上校准}$$
$$\text{最终}: \text{Success\_Distribution} \sim \text{Left-skewed} \Rightarrow \text{most sessions rated 4-5 (Figure 6)}$$

**对应消融**: Figure 6显示评分分布左偏，但人工校验发现LLM对"agent错误被用户纠正"场景存在系统性高估。

## 实验与分析

SWE-chat的核心发现可归纳为以下定量结果：

| 指标 | SWE-chat实测 | 传统基准隐含假设 | 差距 |
|:---|:---|:---|:---|
| 会话数量 | 10,000+ | — | — |
| 覆盖仓库 | 8,000+ | 通常<500 | 16× |
| Agent代码平均占比 | 35.7% (全模式) / 更高 (agent-dominated) | ~100% (自主代理) | — |
| Agent代码生存率 | 64.3% (Table 3) | ~100% (默认采纳) | -35.7% |
| 用户中断频率 | 待补充 (Figure 8) | 0% (无用户) | — |
| 会话成功率(LLM评) | 左偏分布，多数4-5分 (Figure 6) | 二进制 | 揭示"部分成功"常态 |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/975f7a52-9ca6-46d2-bdd1-5587b2c54360/figures/Figure_5.png)
*Figure 5: Figure 5: Vibe coding in the wild. % of agent-authored code, structured into three codingmodes: human-only (0% agent-authored code), collaborative (0–99%), and vibe coding (≥99%).*



**关键发现分析**：

1. **代码生存率颠覆基准假设**：Table 3（Figure 7）显示代理生成代码的64.3%在后续被保留，意味着35.7%被修改或删除。这一数字远低于SWE-bench等基准中"解决方案即采纳"的隐含假设，说明真实场景中代理输出的"噪音成本"被系统性低估。

2. **三种编码模式的分化**：Figure 5揭示human-only、collaborative、agent-dominated三种模式并存。值得注意的是，agent-dominated模式并非主流，且其代码生存率可能更低——暗示过度依赖代理可能降低代码质量。

3. **Turn-level oversight揭示控制策略**：Figure 8显示用户在相当比例的turns中主动中断agent或agent主动请求澄清，这种"人在回路"的动态被基准测试完全抹除。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/975f7a52-9ca6-46d2-bdd1-5587b2c54360/figures/Figure_4.png)
*Figure 4: Figure 4: SWE-chat user-agent interaction statistics. (a) Distribution of turns per session. (b)Distribution of agent tool calls per turn. (c) Top 15 file types touched by agent tool calls.*



**消融与细分分析**：
- 按会话长度（Figure 4a）：多数会话为短交互（<10 turns），但长尾分布显示复杂任务需要50+ turns
- 按工具调用密度（Figure 4b）：每turn工具调用数呈幂律分布，少数高度自主的turn执行大量操作
- 按成功率分布（Figure 6）：左偏形态表明用户倾向于在"足够好"时停止，而非追求完美解决

**公平性检查**：
- **Baseline强度**：本文不 claim 超越任何agent模型，而是揭示评估体系的盲区，故baseline对比不适用传统sense
- **数据成本**：依赖GitHub公开数据+用户主动分享，存在选择偏倚（愿意分享的开发者可能更成熟）
- **失败案例**：Figure 2系统分类失败模式，包括agent误解意图、生成不可运行代码、陷入循环等；但LLM自动标注对"用户巧妙绕过限制"类成功案例可能误判为失败
- **伦理边界**：所有数据来自公开仓库且用户明确选择分享，但"公开"与"预期被研究"之间存在张力

## 方法谱系与知识库定位

**方法家族**: 软件工程实证研究 × AI代理评估 × 人机交互(HCI)数据集构建

**Parent Method**: SWE-bench (Jimenez et al., 2023) —— 本文直接继承其"从真实GitHub仓库提取评估信号"的核心思想，但彻底翻转了评估范式：从"代理能否独立解决"转向"代理如何与人类协作"。

**改变的插槽**:
- **data_curation**: 从人工筛选issue-PR对 → 被动采集真实交互痕迹
- **evaluation_metric**: 从二进制test pass → 连续代码生存率 + 多维成功率分布
- **human_role**: 从absent → central（用户提示、中断、决策全部保留）
- **attribution**: 从none → fine-grained line-level code authorship

**直接Baseline差异**：
| 方法 | 与SWE-chat的差异（1行） |
|:---|:---|
| SWE-bench | 移除人类参与者，用孤立任务替代迭代协作 |
| SWE-smith-trajectories | 保留agent轨迹但移除真实用户提示和代码归因 |
| CoderForge-Preview | 类似SWE-smith，缺乏人机贡献区分和长期代码追踪 |
| OpenHands/ODABA | 关注agent能力上限而非真实使用模式 |

**Follow-up方向**：
1. **纵向因果推断**: 当前为描述性分析，未来可设计干预实验验证"更多用户监督是否提升代码生存率"
2. **跨代理泛化**: SWE-chat目前仅覆盖Claude Code，需验证模式是否迁移至Copilot、Cursor等工具
3. **实时干预系统**: 基于识别的失败模式（Figure 2），构建预测性"agent困惑检测"并主动请求澄清

**知识库标签**: 
- modality: 代码/自然语言多模态交互
- paradigm: 数据驱动实证分析（非模型训练）
- scenario: 真实软件开发工作流
- mechanism: 代码归因追踪 + LLM-as-judge自动标注
- constraint: 依赖特定工具(Claude Code)的日志格式，存在选择偏倚

