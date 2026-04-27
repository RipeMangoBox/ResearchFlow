---
title: 'ProSpero: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhoods'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- ProSpero：超越野生型邻域的主动学习蛋白设计
- PROSPERO
- PROSPERO is an active learning fram
acceptance: Poster
cited_by: 1
code_url: https://github.com/szczurek-lab/ProSpero
method: PROSPERO
modalities:
- protein sequence
paradigm: active learning
---

# ProSpero: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhoods

[Code](https://github.com/szczurek-lab/ProSpero)

**Topics**: [[T__Few-Shot_Learning]] | **Method**: [[M__PROSPERO]] | **Datasets**: Protein design

> [!tip] 核心洞察
> PROSPERO is an active learning framework that guides a frozen pre-trained generative model with a surrogate updated from oracle feedback, using fitness-relevant residue selection and biologically-constrained Sequential Monte Carlo sampling to enable exploration beyond wild-type neighborhoods while preserving biological plausibility.

| 中文题名 | ProSpero：超越野生型邻域的主动学习蛋白设计 |
| 英文题名 | ProSpero: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhoods |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.22494) · [Code](https://github.com/szczurek-lab/ProSpero) · [Project](未提供) |
| 主要任务 | 小样本蛋白质序列设计（Few-Shot Protein Engineering） |
| 主要 baseline | PEX, AdaLead, GFN-AL-δCS, LatProtRL, MLDE, CloneBO |

> [!abstract] 因为「蛋白质工程在探索野生型邻域之外时，常产生生物物理不合理的序列且受限于替代模型误设」，作者在「CloneBO 的 SMC 推理框架」基础上改了「冻结预训练生成模型 + 替代模型引导的电荷兼容约束 SMC + 适应度相关残基目标掩码」，在「8 个蛋白质适应度景观」上取得「Mean pTM 0.763，超越 PEX 的 0.752 与 LatProtRL 的 0.743」

- **Mean pTM 0.763**：超越最强 baseline PEX (+0.011)、LatProtRL (+0.020)、AdaLead (+0.028)
- **Novelty 15.87**：较 PEX 提升 +11.48，较 AdaLead 提升 +9.92，实现深层探索
- **Maximum pTM 0.808**：与 PEX (0.806) 持平，略优于 LatProtRL (0.792)

## 背景与动机

蛋白质工程的核心挑战在于：如何在极少量实验预算内，设计出既具有高适应度（fitness）又远离野生型序列（high novelty）的新蛋白。传统方法往往困在野生型邻域——例如，局部进化搜索 PEX 通过逐位突变逐步改进，但探索范围受限；全局方法如 GFlowNets（GFN-AL-δCS）或潜空间强化学习（LatProtRL）虽能扩大搜索空间，却常采样到生物物理不合理的序列，导致实验验证失败。

现有三条技术路线各有局限：**PEX** [6] 采用近端探索，依赖贪婪局部搜索，novelty 仅 4.39；**CloneBO** 虽引入 SMC 推理，但需针对每个蛋白家族重新训练生成模型，且仅用软似然扭曲（likelihood twisting）降低不利残基概率，无法彻底排除电荷冲突突变；**GFN-AL-δCS** [9] 等基于 GFlowNet 的方法虽多样性高（15.82），但 mean fitness 仅 0.731，适应度-新颖性权衡不佳。更根本的问题是，所有方法都假设替代模型（surrogate）准确——当替代模型在低数据区域误设时，指导信号反而误导采样。

本文动机源于一个具体观察：在信噪比（SNR）极低的环境下，无约束的替代模型引导会损害性能，而野生型序列本身往往已具备基本功能稳定性。因此，作者提出将生物物理约束作为**硬约束**嵌入采样过程，而非事后筛选，同时利用冻结的通用预训练模型避免逐任务训练成本。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d87a0837-444f-4b5d-be1a-61f2c1f25550/figures/Figure_1.png)
*Figure 1: Figure 1: Overview of PROSPERO. Each active learning iteration begins with training a surrogatemodel on the current dataset (A). The surrogate is then used to identify fitness-relevant residueswithin*



本文将展示：如何通过适应度相关残基选择与电荷兼容的硬约束 SMC，实现数据高效且鲁棒的蛋白质设计。

## 核心创新

核心洞察：**将生物物理合理性作为采样过程的硬约束而非软偏好**，因为电荷兼容约束能在替代模型误设时保底野生型适应度，从而使冻结预训练模型 + 主动学习替代更新的组合成为可能。

| 维度 | Baseline (CloneBO/PEX/GFN) | 本文 (PROSPERO) |
|:---|:---|:---|
| 生成模型 | 任务特定训练（CloneBO）或冻结但无约束（GFN） | **冻结通用预训练 EvoDiff-OADM**，零微调适配任意蛋白家族 |
| 约束形式 | 软似然扭曲（降低概率但不排除）或无视约束 | **硬指示函数约束**：电荷不兼容残基直接禁止采样 |
| 残基选择 | 随机掩码或全序列突变 | **目标掩码**：仅对替代模型判定的适应度相关位点进行引导探索 |
| 推理机制 | 直接采样或标准 SMC | **替代模型近似重要性权重的约束 SMC**，桥接离散扩散与主动学习 |

## 整体框架



PROSPERO 是一个迭代式主动学习循环，包含五个核心模块：

1. **冻结预训练生成模型 (EvoDiff-OADM)**：输入掩码后的蛋白序列，输出掩码位置的氨基酸提议分布。该模型保持冻结，不随任务更新，提供跨蛋白家族的通用先验。

2. **替代模型 (Surrogate Model)**：输入蛋白序列，输出预测适应度分数。每轮主动学习后，用 Oracle 反馈的新数据重新训练。

3. **目标掩码选择器 (Targeted Masking Selector)**：输入当前序列种群与替代模型预测，输出下一轮需掩码的残基位置集合。仅选择替代模型判定为"适应度相关"的位点，集中计算预算。

4. **电荷兼容约束 SMC 采样器 (Charge-Compatible SMC Sampler)**：输入起始序列、替代模型、掩码位置，输出多样化候选序列。核心操作：用替代模型近似 SMC 重要性权重，同时以硬约束将提议分布限制为与野生型电荷兼容的残基。

5. **Oracle 评估**：输入候选序列，输出真实适应度值，加入训练数据集用于下一轮替代模型更新。

数据流概览：
```
当前数据集 D_t → 替代模型训练 → 起始序列选择 (top-k) → 目标掩码 → 
电荷兼容 SMC (EvoDiff-OADM 提议 + 替代权重) → 候选序列 → Oracle 评估 → 
D_{t+1} = D_t ∪ {(x, y)} → 循环
```

每轮迭代中，SMC 粒子沿序列位置逐步传播，重要性权重由替代模型近似，重采样步骤淘汰低适应度粒子，最终输出高适应度且生物物理合理的序列集合。

## 核心模块与公式推导

### 模块 1: 替代模型引导的 SMC 重要性权重近似（对应框架图 核心推理引擎）

**直觉**：真实适应度似然 p(y|x) 不可计算，需用可训练的替代模型作为 tractable proxy，使 SMC 在离散序列空间可行。

**Baseline 公式** (CloneBO [14] 的似然扭曲):
$$w_t^{(i)} \propto \frac{p_{\text{CloneLM}}(x_t^{(i)}|\text{high value sequences})}{p_{\text{base CloneLM}}(x_t^{(i)})}$$
符号：$p_{\text{CloneLM}}$ 为任务特定训练的克隆语言模型，分子为条件于高价值序列的扭曲分布，分母为基础分布。

**变化点**：CloneBO 需为每个蛋白家族训练专用生成模型，且扭曲操作仍允许低概率的不利残基。本文改用**冻结通用预训练模型 + 替代模型近似**。

**本文公式（推导）**:
$$\text{Step 1}: \quad w_t^{(i)} \propto \frac{p(y|x_t^{(i)})}{q(x_t^{(i)})} \approx \frac{\hat{f}_\theta(x_t^{(i)})}{q(x_t^{(i)})} \quad \text{用替代模型 } \hat{f}_\theta \text{ 替代真实似然}$$
$$\text{Step 2}: \quad \tilde{w}_t^{(i)} = \frac{\hat{f}_\theta(x_t^{(i)})}{q_{\text{EvoDiff}}(x_t^{(i)} | x_{t-1}^{(i)})} \cdot \mathbb{1}[\text{charge-compatible}] \quad \text{加入硬约束归一化}$$
$$\text{最终}: \quad w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_j \tilde{w}_t^{(j)}} \quad \text{保证粒子权重归一化}$$
符号：$\theta$ = 替代模型参数，$q_{\text{EvoDiff}}$ = EvoDiff-OADM 的提议分布，$\mathbb{1}[\cdot]$ = 电荷兼容指示函数。

**对应消融**：Figure 4D 显示，在极低 SNR 下，完整 PROSPERO 因约束保持稳健，而无约束版本性能显著下降。

---

### 模块 2: 电荷兼容约束提议分布（对应框架图 SMC 采样步骤）

**直觉**：氨基酸电荷剧变（如正电→负电）常破坏蛋白结构稳定性，硬约束比软偏好更能保证生物物理合理性。

**Baseline 公式** (CloneBO 软扭曲):
$$q(x_t | x_{t-1}) \cdot \exp(\lambda \cdot \text{soft preference for similar charge})$$
软偏好仅降低不利残基概率，极端情况下仍可能采样。

**变化点**：将指数软偏好替换为**硬指示函数**，彻底排除电荷不兼容残基，使约束独立于替代模型质量。

**本文公式（推导）**:
$$\text{Step 1}: \quad q_{\text{raw}}(x_t | x_{t-1}) = q_{\text{EvoDiff}}(x_t | x_{t-1}) \quad \text{原始预训练模型提议}$$
$$\text{Step 2}: \quad q_{\text{constrained}}(x_t | x_{t-1}) = q_{\text{raw}}(x_t | x_{t-1}) \cdot \mathbb{1}[\text{charge}(x_t[i]) = \text{charge}(x_{\text{wt}}[i]), \forall i \in \text{mutated}]$$
$$\text{Step 3}: \quad q_{\text{final}} = \frac{q_{\text{constrained}}}{Z} \quad \text{在支持集上重归一化，Z 为配分函数}$$
$$\text{最终}: \quad q_{\text{final}}(x_t | x_{t-1}) = \frac{q_{\text{EvoDiff}}(x_t | x_{t-1}) \cdot \prod_{i \in \mathcal{M}} \mathbb{1}[\text{charge}(x_t[i]) = \text{charge}(x_{\text{wt}}[i])}{\sum_{x'} q_{\text{EvoDiff}}(x' | x_{t-1}) \cdot \prod_{i \in \mathcal{M}} \mathbb{1}[\text{charge}(x'[i]) = \text{charge}(x_{\text{wt}}[i])]}$$

**对应消融**：Table 17（未在 figures_available 中完整展示）及文本指出，去掉电荷约束后低 SNR 性能"substantially"下降，约束是低信噪比鲁棒性的关键来源。

---

### 模块 3: 目标掩码集合选择（对应框架图 掩码决策模块）

**直觉**：并非所有残基对适应度同等重要，随机掩码浪费计算预算在无关位点上。

**Baseline 公式** (标准掩码语言模型):
$$\mathcal{M}(x) \sim \text{Uniform}(\{1, ..., L\})$$
均匀随机选择掩码位置，L 为序列长度。

**变化点**：利用替代模型的梯度或注意力响应，识别对预测适应度影响最大的残基，实现**信息聚焦的探索**。

**本文公式（推导）**:
$$\text{Step 1}: \quad s_i = \left|\frac{\partial \hat{f}_\theta(x)}{\partial x_i}\right| \text{ 或 } \text{Attention}_\theta(x)_i \quad \text{计算每位点重要性分数}$$
$$\text{Step 2}: \quad \mathcal{M}(x) = \{i : s_i > \tau_{\text{adaptive}}\} \quad \text{自适应阈值选择 top 相关位点}$$
$$\text{最终}: \quad \mathcal{M}(x) = \{i : \text{residue } i \text{ is fitness-relevant under surrogate } \hat{f}_\theta\}$$

**对应消融**：文本指出"fewer starting points drive deeper, more directed exploration, resulting in higher novelty and fitness but lower diversity"——目标掩码通过聚焦关键位点，在减少起始序列数时仍能深入探索， fitness-novelty 权衡优于均匀掩码。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d87a0837-444f-4b5d-be1a-61f2c1f25550/figures/Table_1.png)
*Table 1: Table 1: Maximum fitness values achieved by each method. Reported values are the mean andstandard deviation over 5 runs. Green denotes fitness improvement over wild-type xstart. Bold: thebest overall*



本文在 8 个蛋白质适应度景观上评估，核心指标为 AlphaFold2 预测的 pTM 分数（结构置信度， proxy for fitness）、多样性（Diversity）与新颖性（Novelty）。
![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d87a0837-444f-4b5d-be1a-61f2c1f25550/figures/Table_2.png)
*Table 2: Table 2: Mean fitness of top 100 sequences generated by leading methods. Reported values are themean and standard deviation over 5 runs. Green: fitness improvement over wild-type xstart. Bold: thebest*

 中 Table 1 与 Table 2 报告了主要结果：PROSPERO 在 Mean pTM 上达到 **0.763**，显著超越所有 baseline——较 PEX (0.752) 提升 +0.011，较 LatProtRL (0.743) 提升 +0.020，较 AdaLead (0.742)、GFN-AL-δCS (0.731)、MLDE (0.735) 提升 +0.021~+0.032。Maximum pTM 为 **0.808**，与 PEX (0.806) 基本持平，略优于 LatProtRL (0.792)。这表明 PROSPERO 的核心优势在于**整体序列质量的均值提升**，而非单点最优。

在新颖性方面，PROSPERO 的 Novelty **15.87** 远超 PEX (+11.48)、AdaLead (+9.92)、LatProtRL (+10.30)，仅略低于 MLDE (16.97)。结合 Figure 4A 的散点图（
![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d87a0837-444f-4b5d-be1a-61f2c1f25550/figures/Figure_4.png)
*Figure 4: Figure 4: (A) Trade-offs between validity, fitness and novelty across leading methods; each dotrepresents the outcome of a single run. (B) Structural quality of top 100 sequences generated byPROSPERO*

），PROSPERO 在 validity-fitness-novelty 三维权衡中占据优势区域：多数方法要么 novelty 低（PEX, AdaLead），要么 mean fitness 低（GFN-AL-δCS），而 PROSPERO 同时保持高 fitness 与高 novelty。多样性指标 11.25 处于中等水平，低于 GFN-AL-δCS (15.82) 但高于 PEX (6.77) 与 LatProtRL (6.25)，反映目标掩码的聚焦特性。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d87a0837-444f-4b5d-be1a-61f2c1f25550/figures/Table_4.png)
*Table 4: Table 4: Maximum and mean pTM scores of top 100 sequences generated by leading methods underdistribution shifts. Reported values are the mean and standard deviation over 5 runs. Bold: thebest overall*



消融实验（
![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d87a0837-444f-4b5d-be1a-61f2c1f25550/figures/Table_3.png)
*Table 3: Table 3: Average novelty of top 100 sequences generated by leading methods. Reported values arethe mean and standard deviation over 5 runs. Bold: the best overall novelty. Underline: second-best.PROSP*

 对应 Figure 4D）验证了各组件贡献：在低 SNR 条件下，**电荷兼容约束**是性能保底的关键——去掉约束后替代模型误设导致显著退化；**目标掩码**在高 SNR 下贡献增大，使 SMC 引导更有效；有趣的是，在极低 SNR 下 SMC 引导本身可能轻微损害性能，但电荷约束的保底效应维持了整体鲁棒性。Table 17 进一步显示起始序列数 k 的权衡：更少起始点带来更高 novelty 与 fitness 但更低 diversity，更多起始点则相反。

公平性审视：对比的 baseline 覆盖局部搜索（PEX, AdaLead）、全局 RL/GFN（GFN-AL-δCS, LatProtRL, MLDE）及 SMC 方法（CloneBO），较为全面。但未包含近期结构扩散模型如 RFDiffusion、Chroma，也未报告各方法的 Oracle 调用预算是否一致——这对主动学习方法至关重要。此外，Maximum pTM 相对 PEX 的优势 (0.808 vs 0.806) 在统计显著性上存疑。作者披露：极低 SNR 下引导可能轻微有害，性能依赖预训练生成模型质量。

## 方法谱系与知识库定位

**方法家族**：主动学习 + 序列蒙特卡洛（Active Learning × SMC for Discrete Sequence Design）

**父方法**：CloneBO [14] —— 同样采用 SMC 推理，但使用任务特定训练的生成模型与软似然扭曲。PROSPERO 继承其 SMC 框架，替换为冻结预训练模型，并将软扭曲升级为硬电荷约束，新增目标掩码与替代模型引导。

**直接 baseline 差异**：
- **PEX** [6]：局部贪婪搜索，无全局规划能力 → PROSPERO 用 SMC 全局采样 + 替代模型引导
- **GFN-AL-δCS** [9]：GFlowNet 全局探索但 mean fitness 低 → PROSPERO 以约束保证合理性的同时提升 fitness
- **LatProtRL** [10]：潜空间 RL，novelty 有限 → PROSPERO 在序列空间直接操作，novelty 提升 +10.30
- **CloneBO**：任务特定训练 + 软约束 → PROSPERO 零微调通用模型 + 硬约束

**后续方向**：(1) 扩展至结构条件设计（结合 RFDiffusion 等结构生成模型）；(2) 多目标优化（同时优化稳定性、活性、免疫原性）；(3) 实验验证闭环（将 PROSPERO 与真正的高通量实验平台集成）。

**标签**：modality=protein sequence | paradigm=active learning + generative model guidance | scenario=few-shot fitness optimization | mechanism=constrained SMC with surrogate approximation | constraint=charge-compatible biological plausibility

