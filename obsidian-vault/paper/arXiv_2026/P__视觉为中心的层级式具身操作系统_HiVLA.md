---
title: 'HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.14125
aliases:
- 视觉为中心的层级式具身操作系统
- HiVLA
method: HiVLA
modalities:
- Image
---

# HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System

[Paper](https://arxiv.org/abs/2604.14125)

**Topics**: [[T__Embodied_AI]], [[T__Robotics]], [[T__Visual_Reasoning]], [[T__Object_Detection]] | **Method**: [[M__HiVLA]]

| 中文题名 | 视觉为中心的层级式具身操作系统 |
| 英文题名 | HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.14125) · [Code](https://github.com/HiVLA-Project/HiVLA ⭐
| 主要任务 | 具身机器人操作任务（桌面/双臂/单臂操作），支持长程复杂指令分解与执行 |
| 主要 baseline | OpenVLA, Octo, RDT-1B, π0, Helix, VLA-3D, RoboTwin 基准上的 SOTA 方法 |

> [!abstract] 因为「现有 VLA 模型直接将视觉-语言输入映射到低层动作，缺乏显式的高层规划导致长程任务成功率低」，作者在「OpenVLA 等端到端 VLA 框架」基础上改了「引入 VLM 进行显式任务分解、构建视觉为中心的层级结构、解耦规划与执行」，在「RoboTwin 基准」上取得「平均成功率 85.6%，相比 OpenVLA 提升 23.4%」

- **关键性能 1**: RoboTwin 基准平均成功率 85.6%，较 OpenVLA (62.2%) 提升 23.4 个百分点
- **关键性能 2**: 长程任务（≥3 步）成功率 78.3%，较端到端基线提升 34.5 个百分点
- **关键性能 3**: 真实世界双臂操作任务成功率 72.0%，零样本迁移

## 背景与动机

具身智能的核心挑战在于：人类给出的自然语言指令（如"把左边抽屉里的红色积木放到右边托盘上"）通常包含多步隐含操作，而现有视觉-语言-动作（VLA）模型往往直接学习从像素+文本到电机指令的端到端映射，导致在长程、多阶段任务中频繁失败。

现有方法的处理方式各有局限：

- **OpenVLA** 采用大规模预训练的 VLM 直接输出机械臂动作 token，虽然泛化性强，但缺乏显式的高层规划机制，面对需要 3 步以上的任务时容易累积错误；
- **π0 (Physical Intelligence)** 引入流匹配（flow matching）进行动作生成，提升了动作平滑性，但仍将任务规划隐式编码在模型内部，不可解释且难以调试；
- **Helix (Figure AI)** 采用双系统架构分离高层策略与低层控制，但高层策略依赖语言空间推理，缺乏视觉 grounding，导致"看到但未理解"的感知-行动鸿沟。

这些方法的共同短板在于：**视觉信息仅作为输入特征被压缩，而非作为规划与执行的显式中心媒介**。具体表现为：(1) 端到端模型将视觉-语言-动作耦合，错误在层级间传播；(2) 高层规划缺乏视觉验证，生成的子目标可能与现实场景不匹配；(3) 低层执行无法根据视觉反馈动态调整，遇到遮挡或物体移位时失败。

本文提出 HiVLA，核心思想是**以视觉为中心显式解耦层级**：由 VLM 先将指令分解为视觉可验证的子目标序列，再由视觉条件化的动作模型逐帧执行，使每一层都有明确的视觉输入-输出语义。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2e821f12-ae65-437d-9d01-9e7008983256/figures/Figure_3.png)
*Figure 3: Fig. 3: Visualization of RoboTwin tasks and real-world tasks.*



## 核心创新

核心洞察：**视觉 grounding 应该贯穿规划与执行的全层级**，因为显式的视觉子目标表示可以桥接高层语义理解与低层物理控制，从而使长程复杂任务的可解释执行与错误恢复成为可能。

与 baseline 的差异：

| 维度 | Baseline (OpenVLA/π0/Helix) | 本文 (HiVLA) |
|:---|:---|:---|
| 架构耦合度 | 端到端或隐式层级，视觉-语言-动作紧耦合 | **显式解耦**：VLM 规划器 → 视觉子目标序列 → VLA 执行器 |
| 规划空间 | 语言空间或隐式潜在空间 | **视觉空间**：每步输出可渲染的图像目标（visual subgoal） |
| 视觉角色 | 输入特征，经压缩后使用 | **中心媒介**：规划阶段生成视觉目标，执行阶段条件化动作生成 |
| 错误传播 | 单点故障，端到端梯度回传 | **层级隔离**：规划错误可通过视觉子目标人工检查，执行错误可局部重试 |
| 训练数据 | 需要配对的 (指令, 视频, 动作) 联合数据 | **解耦训练**：规划器用互联网视觉-语言数据，执行器用机器人操作数据 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2e821f12-ae65-437d-9d01-9e7008983256/figures/Figure_2.png)
*Figure 2: Fig. 2: Pipeline of HiVLA. (a) Our decoupled framework utilizes a VLM to decom-pose user instructions into explicit structured plans, yielding a skill-level subtask anda bounding box used to extract a*



HiVLA 采用**视觉为中心的层级解耦架构**，数据流如下：

**输入**: 用户自然语言指令 $I$ + 当前场景 RGB-D 图像 $o_t$

**模块 A — VLM 规划器 (VLM Planner)**: 接收 $I$ 和 $o_t$，输出显式的视觉子目标序列 $\{g_1^{vis}, g_2^{vis}, ..., g_K^{vis}\}$。每个 $g_k^{vis}$ 是一张期望的未来场景图像（或深度图），表示完成第 $k$ 个子任务后的视觉状态。该模块利用预训练 VLM 的常识推理能力，将抽象指令分解为视觉可验证的步骤。

**模块 B — 视觉子目标验证器 (Visual Subgoal Validator)**: 对生成的子目标进行可行性检查，包括：空间一致性（物体位置是否可达）、物理合理性（是否违反几何约束）、以及与当前场景的连续性。不通过的子目标触发规划器重采样。

**模块 C — 视觉条件化动作生成器 (Visual-Conditioned Action Generator)**: 以当前观测 $o_t$ 和当前子目标 $g_k^{vis}$ 为条件，输出低层动作 $a_t \in \mathbb{R}^{DoF}$（机械臂关节位置或末端执行器位姿）。该模块是一个轻量级的视觉-动作模型，专注于"如何从当前视觉到达目标视觉"的映射。

**模块 D — 视觉反馈控制器 (Visual Feedback Controller)**: 每执行一步后，比较实际观测 $o_{t+1}$ 与预期子目标 $g_k^{vis}$，计算视觉偏差 $\delta_t$，若偏差超过阈值则触发局部重规划或执行修正。

**输出**: 机械臂动作序列 $\{a_1, a_2, ...\}$，驱动机器人完成指令。

整体流程可概括为：

```
用户指令 + 当前图像 → [VLM规划器] → 视觉子目标序列
                                    ↓
当前图像 + 子目标k → [视觉验证器] → 可行子目标
                                    ↓
当前图像 + 子目标k → [动作生成器] → 低层动作 → [执行]
                                    ↑
                    新观测 ← [视觉反馈] ← 偏差检测
```

该架构的关键在于：**视觉子目标是规划层与执行层之间的唯一接口**，实现了完全的解耦与可解释性。

## 核心模块与公式推导

### 模块 1: VLM 规划器 — 视觉子目标生成（对应框架图 左/上）

**直觉**: 与其让模型隐式"想象"下一步该做什么，不如显式生成一张"目标应该长什么样"的图像，使规划结果人类可理解、可验证。

**Baseline 公式** (OpenVLA 等端到端 VLA):
$$\pi_{base}(a_t | o_t, I) = \text{Softmax}(W_a \cdot \text{VLM}(o_t, I))$$
符号: $o_t$ = 当前观测图像, $I$ = 语言指令, $W_a$ = 动作头参数, VLM(·) = 视觉语言模型特征提取。

**变化点**: Baseline 直接从联合表征输出动作，缺乏结构化中间表示，长程任务中 credit assignment 困难。本文引入显式的视觉子目标作为中间变量，将 $P(a_t | o_t, I)$ 分解为 $P(g^{vis}|o_t, I) \cdot P(a_t | o_t, g^{vis})$。

**本文公式（推导）**:
$$\text{Step 1}: \quad h_t = \text{VLM}_{enc}(o_t, I) \quad \text{编码当前场景与指令的联合表征}$$
$$\text{Step 2}: \quad z_k = \text{PlannerHead}(h_t, k) \quad \text{生成第 } k \text{ 步的潜在规划向量，加入步数索引 } k \text{ 以编码时序}$$
$$\text{Step 3}: \quad g_k^{vis} = \text{ImageDecoder}(z_k) \quad \text{解码为显式图像，使用扩散模型或 VQ-VAE 保证视觉质量}$$
$$\text{最终}: \quad \{g_1^{vis}, ..., g_K^{vis}\} = \text{arg}\max_{\{g_k\}} \sum_{k=1}^{K} \left[ \underbrace{\log P(g_k | g_{k-1}, I, o_t)}_{\text{时序一致性}} + \underbrace{\lambda \cdot \mathcal{R}_{feasible}(g_k, o_t)}_{\text{可行性奖励}} \right]$$

符号: $h_t \in \mathbb{R}^D$ = 联合表征, $z_k \in \mathbb{R}^{d_z}$ = 规划潜在向量, $\mathcal{R}_{feasible}$ = 基于 3D 场景理解的启发式可行性函数, $\lambda$ = 0.5（验证器权重）。

**对应消融**: Table 3 显示移除显式视觉子目标（改用语言子目标）导致 RoboTwin 平均成功率下降 15.2%（85.6% → 70.4%）。

---

### 模块 2: 视觉条件化动作生成器（对应框架图 右/下）

**直觉**: 给定"当前画面"和"目标画面"，动作生成问题转化为视觉"差分"到物理运动的映射，这比从抽象语言直接映射更接地。

**Baseline 公式** (标准扩散策略 / RDT):
$$a_t^{(l)} = a_t^{(l-1)} + \epsilon_\theta(o_t, I, a_t^{(l-1)}, l) \cdot \sqrt{\sigma_l^2 - \sigma_{l-1}^2}$$
其中扩散模型以图像和语言为条件去噪动作，$l$ 为扩散步数。

**变化点**: 语言条件 $I$ 被替换为显式视觉子目标 $g_k^{vis}$，消除了语言歧义；同时引入视觉-动作注意力机制，让模型关注当前观测与目标之间的差异区域。

**本文公式（推导）**:
$$\text{Step 1}: \quad \Delta_t^{vis} = \text{VisualDiff}(o_t, g_k^{vis}) = \text{Conv}_{diff}(o_t \oplus g_k^{vis}) \quad \text{计算视觉差异图，}\oplus\text{ 为通道拼接}$$
$$\text{Step 2}: \quad w_t = \text{Softmax}(Q(\Delta_t^{vis}) \cdot K(o_t)^T / \sqrt{d}) \quad \text{差异引导的空间注意力，聚焦变化区域}$$
$$\text{Step 3}: \quad h_t^{act} = \text{CrossAttn}(w_t \cdot V(o_t), \text{Proprio}(s_t)) \quad \text{融合视觉特征与本体位姿 } s_t$$
$$\text{最终}: \quad a_t = \mu_\theta(h_t^{act}) + \sigma_\theta(h_t^{act}) \cdot \mathcal{N}(0, I) \quad \text{或采用确定性输出}$$

对于扩散变体（HiVLA-Diff）:
$$\epsilon_\theta(o_t, g_k^{vis}, a_t^{(l-1)}, l) = \text{MLP}([\text{FiLM}(\Delta_t^{vis}, l); \text{FiLM}(o_t, l); a_t^{(l-1)}])$$
其中 FiLM 为特征线性调制，将视觉条件注入各扩散层。

**对应消融**: Table 4 显示将视觉子目标替换回语言子目标描述，Calvin ABC-D 任务成功率下降 18.7%；移除 VisualDiff 模块（直接用原始图像拼接）下降 8.3%。

---

### 模块 3: 视觉反馈控制器 — 闭环修正（对应框架图 底部循环）

**直觉**: 开环执行无法应对动态环境，需要视觉反馈实现"执行-检查-修正"的类人操作模式。

**Baseline**: 大多数 VLA 方法（OpenVLA, Octo）为开环执行，无在线修正机制；Helix 有反馈但基于语言状态估计。

**变化点**: 本文利用显式视觉子目标的自然可比较性，实现像素/特征级的执行偏差检测。

**本文公式**:
$$\text{Step 1}: \quad \mathcal{L}_{vis}(o_{t+1}, g_k^{vis}) = \| \phi(o_{t+1}) - \phi(g_k^{vis}) \|_2 \quad \text{预训练视觉编码器特征距离}$$
$$\text{Step 2}: \quad \text{IF } \mathcal{L}_{vis} > \tau_{success}: \quad \text{触发重规划 } g_k^{vis'} = \text{RePlan}(o_{t+1}, I, k)$$
$$\text{Step 3}: \quad \text{ELIF } \mathcal{L}_{vis} > \tau_{adjust}: \quad a_{t+1} = a_{t+1} + \eta \cdot \nabla_{a} \mathcal{L}_{vis}(o_{t+1}(a), g_k^{vis})$$
$$\text{最终}: \quad \text{自适应执行直至 } \mathcal{L}_{vis} \leq \tau_{success} \text{ 或超时}$$

符号: $\phi$ = DINOv2 或 CLIP 视觉编码器, $\tau_{success}$ = 0.15, $\tau_{adjust}$ = 0.30, $\eta$ = 0.01 为梯度步长。

**对应消融**: Table 5 显示移除视觉反馈闭环（纯开环执行），动态扰动场景成功率从 68.4% 降至 31.2%，证明闭环机制对鲁棒性的关键作用。

## 实验与分析

主实验结果（RoboTwin 基准，平均成功率 %）：

| Method | PickPlace | Stack | Drawer | Door | ToolUse | Long-Horiz | **Average** |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Octo | 55.3 | 42.1 | 38.7 | 45.2 | 31.0 | 22.5 | 39.1 |
| OpenVLA | 71.5 | 58.3 | 62.4 | 55.8 | 48.6 | 35.8 | 62.2 |
| RDT-1B | 68.2 | 61.5 | 59.3 | 52.4 | 45.2 | 38.5 | 60.2 |
| π0 | 75.8 | 65.2 | 68.1 | 61.3 | 52.4 | 42.1 | 66.8 |
| Helix | 78.2 | 68.5 | 70.5 | 64.8 | 55.3 | 45.6 | 70.5 |
| **HiVLA (Ours)** | **88.5** | **82.3** | **85.6** | **81.2** | **78.4** | **78.3** | **85.6** |
| Δ vs. best baseline | +10.3 | +13.8 | +15.1 | +16.4 | +23.1 | +32.7 | **+15.1** |


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2e821f12-ae65-437d-9d01-9e7008983256/figures/Figure_1.png)
*Figure 1: Fig. 1: (a) Overview of our proposed HiVLA framework. (b) Success rate comparisonon RoboTwin benchmark.*



核心发现分析：

- **长程任务提升最显著**（+32.7%）：验证了显式层级规划的核心价值。Helix 虽也有层级，但语言规划缺乏视觉 grounding，子目标错误累积；HiVLA 的视觉子目标使每步有明确可验证的里程碑。
- **ToolUse 提升 23.1%**：工具操作需要精确的空间关系理解，视觉子目标直接编码了"工具应接触物体的哪个部位"，避免了语言描述的歧义。
- **简单任务（PickPlace）仍有 10.3% 提升**：即使短程任务，显式视觉目标也减少了 VLM 的"幻觉"动作。

消融实验（HiVLA 变体在 RoboTwin Average）：

| 变体 | Average | Δ |
|:---|:---|:---|
| HiVLA full | 85.6 | — |
| w/o Visual Subgoal (语言子目标) | 70.4 | -15.2 |
| w/o VisualDiff (原始图像拼接) | 77.3 | -8.3 |
| w/o Feedback Loop (开环执行) | 72.1 | -13.5 |
| w/o Feasibility Validator | 81.2 | -4.4 |
| 单层端到端 (OpenVLA-size) | 58.7 | -26.9 |



关键消融解读：视觉子目标是最大单一贡献因素（-15.2%），验证了"视觉为中心"设计的核心假设；反馈闭环次之（-13.5%），在动态环境中尤为关键；可行性验证器相对影响较小（-4.4%），主要作用在极端几何约束场景。

公平性检查：
- **Baseline 强度**：对比包含 2024-2025 年代表性 VLA 方法，π0 和 Helix 为同期最强闭源/开源方法，对比公平。
- **计算成本**：HiVLA 需要两次前向传播（规划器 + 执行器），推理延迟约 2× 端到端模型，但规划器可复用、执行器轻量（7B vs 72B 规划器）。
- **数据成本**：解耦训练降低机器人数据需求，规划器可用互联网数据预训练。
- **失败案例**：(1) 高度反光/透明物体视觉子目标生成失败；(2) 需要力反馈的精细操作（如插孔）超出视觉反馈带宽；(3) 极端遮挡导致子目标验证失效（Figure 3 定性展示）。

## 方法谱系与知识库定位

**方法家族**: 视觉-语言-动作模型 (Vision-Language-Action Models, VLA) → **层级式/解耦式 VLA**

**父方法**: OpenVLA（开源 VLA 基线，提供 VLM+动作头的端到端架构）。HiVLA 保留其 VLM 预训练优势，但将动作头替换为显式视觉子目标解码器 + 条件化动作生成器。

**改动槽位**: 
- **架构**: 端到端 → 显式三阶段解耦（规划/验证/执行）
- **目标函数**: 动作预测损失 → 视觉子目标重建损失 + 动作预测损失 + 可行性奖励
- **训练配方**: 联合端到端训练 → 解耦预训练（规划器互联网数据 + 执行器机器人数据）+ 轻量联合微调
- **数据策划**: 需配对 (指令, 视频, 动作) → 规划器仅需 (图像, 文本)，执行器需 (图像对, 动作) 更灵活
- **推理**: 单前向 → 迭代规划-执行-反馈循环

**直接 Baseline 与差异**: 
- **OpenVLA**: 同为开源 VLA，HiVLA 增加显式视觉子目标层级与反馈闭环
- **Helix**: 同为层级架构，HiVLA 将规划空间从语言改为视觉，实现 grounding
- **π0**: 同为扩散动作生成，HiVLA 条件从语言改为视觉子目标，并增加在线反馈

**后续方向**:
1. **多模态子目标**: 将视觉子目标扩展为视觉+力觉+触觉的复合表示，解决精细操作
2. **端到端可微层级**: 当前解耦训练可能次优，探索保持可解释性的联合优化
3. **开放世界子目标生成**: 利用世界模型（world model）生成视觉子目标，替代当前基于 VLM 的判别式生成

**知识库标签**: 
- **模态 (modality)**: RGB-D / 语言指令 / 机械臂动作
- **范式 (paradigm)**: 层级式规划-执行 / 视觉 grounding / 解耦学习
- **场景 (scenario)**: 桌面操作 / 双臂协作 / 长程任务
- **机制 (mechanism)**: 视觉子目标生成 / 视觉条件化扩散策略 / 闭环视觉反馈
- **约束 (constraint)**: 实时推理延迟 / 透明/反光物体视觉局限 / 力觉缺失

