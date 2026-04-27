---
title: Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.20987
aliases:
- COS-PLAY：共进化双代理框架实现长程任务中的技能发现与复用
- CLDSBA
- 核心洞察是：技能库和决策策略是相互依赖的——好的技能库需要好的轨迹
---

# Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks

[Paper](https://arxiv.org/abs/2604.20987)

**Topics**: [[T__Agent]], [[T__Reasoning]], [[T__Reinforcement_Learning]] | **Datasets**: Single-player games, Avalon, Diplomacy

> [!tip] 核心洞察
> 核心洞察是：技能库和决策策略是相互依赖的——好的技能库需要好的轨迹，好的轨迹需要好的技能库。COS-PLAY通过将两者显式建模为两个专职智能体并在闭环中联合优化，打破了这一鸡生蛋的困境。关键在于「协同演化」而非「先学技能再学策略」：技能库智能体从决策智能体的无标注轨迹中无监督提取技能，决策智能体则通过复合奖励被训练去有效使用这些技能，两者在GRPO框架下交替迭代，使得小模型（8B）能够通过结构化技能复用逼近甚至超越参数量大得多的前沿LLM。

## 基本信息

**论文标题**: Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks

**作者**: （未在提供的上下文中明确列出）

**发表场所**: （未明确标注）

**年份**: （未明确标注）

**代码/数据链接**: （未在提供的上下文中提供）

**基础模型**: Qwen3-8B（8B参数）

**核心方法**: COS-PLAY（Co-Evolving Skill-Play）

## 核心主张

COS-PLAY 提出了一种**双代理共进化架构**：决策代理（Decision Agent）负责技能检索与动作执行，技能库代理（Skill Bank Agent）负责从无标注轨迹中自动发现、验证并维护可复用的结构化技能。通过迭代共进化训练，一个**8B参数的Qwen3-8B模型**能够在长程游戏任务上**平均超越GPT-5.4等前沿大模型25.1%**。

**关键证据**: 在2048、Candy Crush、Tetris、Super Mario Bros.等单人游戏中，COS-PLAY相对GPT-5.4取得+25.1%平均奖励提升（Table 1，16次运行平均）。在Avalon和Diplomacy等多人社交推理游戏中声称匹配或接近前沿LLM表现。

**置信度评估**: 高置信度（0.95）——双代理架构和共进化训练流程有明确的方法论描述；但实验对比的公平性存在争议（见局限部分）。

## 研究动机

**问题**: 现有LLM智能体在长程任务中面临**一致性决策困难**——直接生成动作缺乏结构化规划，导致行为碎片化、难以维持跨时间步的目标一致性。

**重要性**: 长程游戏（如Diplomacy、Avalon）需要持续数十至数百步的策略执行与社交推理，是当前LLM代理的核心挑战场景。

**前人工作局限**:
- **GPT-5.4/Gemini-3.1-Pro/Claude-4.6-Sonnet**等前沿LLM：零样本或少样本推理，无任务特定适应，缺乏显式技能结构
- **CASCADE**：提出累积技能创建，但架构相对单一，未明确分离决策与技能发现
- **Optimus-1**：依赖混合多模态记忆，但未解决技能从无标注数据中自动发现的问题

**本文填补的空白**: 首次实现**完全无监督的技能发现与决策共进化**，无需人工标注技能边界，通过双代理互反馈机制持续改进。

## 方法流程

COS-PLAY 包含**8个核心模块**，形成闭环共进化流程：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Environment    │◄────│   Decision Agent│     │   Skill Bank    │
│  (Games)        │     │   AD            │     │   Agent AS      │
└────────┬────────┘     └────────┬────────┘     └─────────────────┘
         │                       │                       ▲
         │ o_t, r_t              │ τ (trajectory)        │ B^(u+1)
         ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐     ┌───────┴─────────┐
│ Skill Retrieval │     │ Action Generation│     │ 4-Stage Pipeline:│
│ π_skill(o_t,B)  │────►│ π_act(o_t,z_t,s̃_t)│     │ 1. Boundary      │
│  → s̃_t          │     │  → a_t            │     │    Proposal      │
└─────────────────┘     └─────────────────┘     │ 2. Segmentation  │
┌─────────────────┐                             │ 3. Contract      │
│ Intention Update│                             │    Learning      │
│ π_int(o_t,s̃_t)  │                             │ 4. Bank Update   │
│  → z_t          │                             │    Φ_S           │
└─────────────────┘                             └─────────────────┘
         ▲                                              │
         └──────────────────────────────────────────────┘
                    GRPO + Separate LoRA Adapters
                    (5 distinct functions)
```

**训练循环**: (1) 决策代理收集轨迹 → (2) 技能库代理分段并更新技能库 → (3) GRPO同时更新两个代理的5个独立LoRA适配器。

## 关键公式

**1. 扩展经验元组（新增意图标签）**
```latex
e_t = (o_t, a_t, r_t, o_{t+1}, d_t, z_t)
```
> 在标准元组基础上增加**意图标签 $z_t$**，为技能切换和边界检测提供信号。**本方法新增**。

**2. 技能检索策略**
```latex
\tilde{s}_t = \pi_\theta^{\text{skill}}(o_t, \mathcal{B})
```
> 根据观测和技能库检索活跃技能。**全新模块**，替代直接动作生成。

**3. 意图更新策略**
```latex
z_t = \pi_\theta^{\text{int}}(o_t, \tilde{s}_t)
```
> 生成自然语言意图标签，意图剧变时触发技能切换。**全新模块**。

**4. 技能条件动作生成**
```latex
a_t \sim \pi_\theta^{\text{act}}(\cdot \mid o_t, z_t, \tilde{s}_t)
```
> 以观测、意图、技能三重条件采样动作。**全新模块**，基线为 $a_t \sim \pi_\theta(\cdot \text{mid} o_t)$。

**5. 复合奖励目标**
```latex
\max_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=1}^{T} r_t\right]
```
> 其中 $r_t$ 包含环境反馈 + 技能跟随塑形 + 切换成本，**非单纯环境奖励**。

**6. 技能库演化算子（核心创新）**
```latex
\mathcal{B}^{(u+1)} = \Phi_S\!\left(\mathcal{B}^{(u)}, \mathcal{D}^{(u+1)}\right)
```
> 从当前库和新轨迹自动演化技能库。**全新公式**，实现共进化耦合。

## 实验结果

**单人游戏基准**（Table 1，16次运行平均）：
| 基准 | 指标 | COS-PLAY vs GPT-5.4 |
|------|------|---------------------|
| 2048, Candy Crush, Tetris, Super Mario Bros. | 平均奖励提升 | **+25.1%** |

**多人社交游戏**：
- **Avalon**：声称匹配或超过前沿LLM（Table 4，各角色胜率），但**具体数值未在上下文中明确给出**
- **Diplomacy**：声称接近前沿LLM表现（Table 5，各势力平均补给中心数），但**具体数值未明确给出**；论文指出COS-PLAY在Diplomacy中表现为"停滞而非崩溃"

**消融实验**（Table 6）：
- 移除**分离LoRA适配器**（改用共享适配器）：性能下降，支持模块化设计的必要性

**证据强度评估**：⚠️ **中等（0.6）**。主要问题：
1. 前沿LLM基线为零样本/少样本，无任务适应，对比不公平
2. 社交游戏训练时使用GPT-5-mini作为对手，提供基线没有的强监督信号
3. 冷启动使用GPT-5.4教师轨迹，赋予特权初始知识
4. 多数基准缺乏具体数值，仅百分比提升和定性声明

## 相关工作

**按角色分类**：

| 角色 | 方法 | 关系 |
|------|------|------|
| **主要基线** | GPT-5.4, GEMINI-3.1-PRO, CLAUDE-4.6-SONNET, GPT-OSS-120B | 零样本/少样本前沿LLM对比 |
| **次要基线/初始化** | Qwen3-8B with SFT only | 共享初始化点，SFT后未共进化 |
| **组件来源** | GRPO (Group-in-group policy optimization) | 双代理RL训练算法 |
| **组件来源** | LoRA (Low-rank adaptation) | 参数高效微调，5个独立适配器 |
| **相关/可能基线** | CASCADE | 累积技能创建，**谱系父节点** |
| **相关/可能基线** | Optimus-1 | 混合多模态记忆长程代理 |

**最重要的5篇参考文献**：
1. **CASCADE** — 直接前身，COS-PLAY继承其累积技能创建思想，但重构为双代理共进化
2. **GRPO** — 训练核心，用于同时优化决策代理和技能库代理
3. **LoRA** — 工程基础，实现5个功能模块的高效独立训练
4. **Optimus-1** — 长程记忆代理的对比基准
5. **GPT-5.4等前沿LLM** — 能力上限参照，证明小模型+结构可超越大模型

## 方法谱系

**谱系位置**: CASCADE → **COS-PLAY**

| 槽位 | CASCADE（推测基线） | COS-PLAY（本文） | 变更类型 |
|------|---------------------|------------------|----------|
| **架构** | 单一或较简单的技能创建架构 | **双代理架构**：决策代理AD + 技能库代理AS | 🔴 替换 |
| **推理策略** | 直接动作生成 | **技能增强推理**：π_skill → π_int → π_act 三阶段 | 🔴 替换 |
| **数据流程** | 可能依赖标注或简化处理 | **四阶段无监督管道**：边界提议→分割推理→契约学习→库维护 | 🔴 替换 |
| **训练方案** | 标准SFT/RL | **冷启动SFT + 迭代共进化**：GRPO + 5个独立LoRA | 🔴 替换 |
| **奖励设计** | 环境奖励为主 | **复合奖励**：环境反馈 + 技能跟随塑形 + 切换成本 | 🟡 修改 |
| **探索策略** | 隐式采样/温度 | **结构化探索**：谓词翻转、意图变化、奖励峰值、惊奇峰值 | 🟢 新增 |

**继承**: 累积技能创建的核心理念、长程任务场景
**变革**: 显式分离决策与技能发现、引入效果契约、完全无监督的轨迹分解、双代理互反馈机制

## 局限与展望

**论文明确/分析推断的局限**：

1. **冷启动依赖**：必须使用GPT-5.4教师轨迹进行SFT初始化，无法完全从零开始（privileged starting knowledge）

2. **对比公平性问题**：前沿LLM基线为零样本/少样本，而COS-PLAY经过环境交互和GRPO训练；社交游戏中使用GPT-5-mini作为训练对手，获得基线没有的强监督信号

3. **Diplomacy表现瓶颈**：出现"停滞而非崩溃"现象，说明在高度开放的多人谈判场景中技能发现仍不充分

4. **计算成本未披露**：训练计算量、推理延迟、GPU类型等关键工程细节缺失

5. **迭代次数限制**：每游戏最多25次迭代用于少样本适应，可能限制深度优化

**未来方向**：
- 消除对教师模型的依赖，实现完全自举的冷启动
- 将效果契约扩展到更复杂的时序逻辑规范
- 探索三代理及以上架构（如引入对手建模代理）
- 在更多开放世界环境（Minecraft等）验证可扩展性

## 知识图谱定位

**任务节点**：
- 🎮 **长程游戏智能体**（核心）
- 🔍 **技能发现与复用**（核心创新）
- 🤝 **多人社交推理**（Avalon, Diplomacy）
- 🧩 **单人谜题求解**（2048, Candy Crush, Tetris, Super Mario Bros.）

**方法节点**：
- ⭐ **COS-PLAY**（新增中心节点）
- 🔗 **GRPO**, **LoRA**（支撑技术）
- 📌 **CASCADE**（谱系父节点）

**机制节点**（本文新增4个）：
- **技能增强推理** — 显式技能检索+意图维护+条件生成
- **协同进化训练** — 双代理互反馈训练范式
- **效果契约学习** — 技能语义的形式化验证
- **边界提议启发式** — 无监督轨迹分割信号

**数据集节点**：
- 无标注轨迹（输入）
- GPT-5.4教师轨迹（冷启动依赖）

**结构贡献**：COS-PLAY在知识图谱中建立了**"小模型+显式技能结构 > 大模型零样本"**的新路径，将技能发现从监督/半监督推向完全无监督，为长程代理领域提供了可复用的双代理架构模板。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/39bf8fc5-801e-4494-ac67-1d9bbd15e97b/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of COS-PLAY. COS-PLAY is a multi-agent co-evolution framework that couples gameplay with skill learning.*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/39bf8fc5-801e-4494-ac67-1d9bbd15e97b/figures/Figure_2.png)
*Figure 2 (pipeline): Skill bank agent pipeline on one Diplomacy episode (Austria).*


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/39bf8fc5-801e-4494-ac67-1d9bbd15e97b/figures/Figure_3.png)
*Figure 3 (result): Skill bank evolution over Diplomacy training.*


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/39bf8fc5-801e-4494-ac67-1d9bbd15e97b/figures/Table_2.png)
*Table 2 (quantitative): Cross-run skill reusability across games.*


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/39bf8fc5-801e-4494-ac67-1d9bbd15e97b/figures/Table_1.png)
*Table 1 (quantitative): We report the number of discovered skills (#Skills), number of categories (#Cats)...*


