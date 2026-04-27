---
title: Calibrated Self-Rewarding Vision Language Models
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- 视觉语言模型的校准自奖励优化
- CSR (Calibrated
- CSR (Calibrated Self-Rewarding)
acceptance: Poster
cited_by: 71
code_url: https://github.com/NishilBalar/Awesome-LVLM-Hallucination
method: CSR (Calibrated Self-Rewarding)
followups:
- 多模态模型评判器LLaVA-Cr_LLaVA-Critic
---

# Calibrated Self-Rewarding Vision Language Models

[Code](https://github.com/NishilBalar/Awesome-LVLM-Hallucination)

**Topics**: [[T__Visual_Question_Answering]], [[T__Captioning]], [[T__Self-Supervised_Learning]] | **Method**: [[M__CSR]] | **Datasets**: [[D__MM-Vet]] (其他: Average across all benchmarks (comprehensive + hallucination), LLaVAW, CHAIRS, MMEP)

| 中文题名 | 视觉语言模型的校准自奖励优化 |
| 英文题名 | Calibrated Self-Rewarding Vision Language Models |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2405.14622) · [Code](https://github.com/NishilBalar/Awesome-LVLM-Hallucination) · [DOI](https://doi.org/10.52202/079017-1630) |
| 主要任务 | 视觉语言模型（VLM）的幻觉降低、偏好优化、视觉问答 |
| 主要 baseline | LLaVA-1.5 (7B/13B), Vila 7B, Self-Rewarding Language Models |

> [!abstract] 因为「视觉语言模型在自奖励训练中过度依赖文本信号、忽视图像信息导致幻觉」，作者在「Self-Rewarding Language Models」基础上改了「引入视觉-文本校准模块与参数λ的奖励函数」，在「LLaVA-1.5 7B」上取得「平均性能从66.61提升至72.24（+7.62%），CHAIR幻觉指标从59.2提升至79.0（+49.50%）」

- **平均性能**: LLaVA-1.5 7B Iter-5 达 72.24，较 Iter-1 的 66.61 提升 +5.63 绝对分数（+7.62% 相对提升）
- **幻觉降低**: CHAIR 转换分数 59.2 → 79.0，LLaVAW 准确率 66.7% → 71.1%（+8.9%）
- **校准参数敏感性**: MMEP 感知分数 λ=0.9 时 1524.2，较 λ=0.1 的 1508.6 提升 +15.6

## 背景与动机

视觉语言模型（VLM）如 LLaVA-1.5 在生成图像描述或回答视觉问题时，经常出现「幻觉」——即描述图像中不存在的物体或关系。例如，模型可能声称图片中有「红色的停车标志」，而实际图像中并无此物。这种幻觉的根源在于：VLM 的生成过程往往过度依赖语言先验，而未能充分锚定视觉输入。

现有方法从三个方向试图缓解此问题。**LLaVA-1.5** 通过改进视觉指令调优增强对齐，但其训练仍依赖固定的人类标注数据，无法自适应地纠正模型自身的偏见。**Self-Rewarding Language Models** 提出让语言模型自生成偏好数据并迭代优化，但该框架纯基于文本信号，未考虑视觉-文本模态间的校准问题，直接迁移至 VLM 会导致图像信号被进一步稀释。**VCD（未在本文中对比）** 等幻觉专精方法通过对比解码抑制语言先验，但需外部工具辅助，无法融入端到端的自提升循环。

这些方法的共同短板在于：**自奖励机制缺乏显式的视觉-文本校准**。当 VLM 作为自身的裁判时，其判断仍偏向文本一致性而非视觉忠实性，形成「文本偏见自我强化」的恶性循环。具体而言，标准自奖励目标 $y_w = \text{arg}\max_y R_{self}(y)$ 完全忽略图像特征 $x_v$ 的结构化贡献，导致最优响应趋向文本主导（$y_w \approx V_2 x_t$）。

本文的核心动机正是填补这一缺口：将自奖励框架扩展至视觉-语言领域，并通过可证明的校准机制显式上采样（up-weight）图像信号，使模型在自我迭代中逐步增强视觉忠实性而非幻觉。

## 核心创新

核心洞察：自奖励的信号偏差可通过**投影空间中的视觉-文本校准**来纠正，因为 VLM 的幻觉本质上是响应 $y$ 与视觉特征 $x_v$ 在共享表示空间中未对齐，从而使**显式注入视觉信号的闭式最优解**成为可能。

| 维度 | Baseline (Self-Rewarding LMs) | 本文 (CSR) |
|:---|:---|:---|
| 奖励设计 | 纯文本自奖励 $R_{self}(y)$，无视觉项 | 加权组合：(1-λ)⟨U₁ᵀxᵥ, U₁ᵀy⟩ − λ‖y − V₁xᵥ + V₂xₜ‖² |
| 最优响应 | 文本主导 $y_w \approx V_2 x_t$ | 视觉增强 $y_w = \frac{1-\lambda}{\lambda} U_1 U_1^\text{top} x_v + V_1 x_v + V_2 x_t$ |
| 理论保证 | 无显式模态平衡机制 | Theorem 5.1：存在 λ∈(0,1) 使 CSR 表示误差严格小于基线 |
| 迭代行为 | 文本偏见自我强化 | 视觉信号逐轮增强，5 轮迭代持续改进（66.61→72.24）|

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/eb0275b3-76bf-4c64-b199-bd125c543e39/figures/fig_001.png)
*Figure: Benchmark performance comparison.*



CSR 采用**在线迭代自奖励**范式，核心数据流如下：

1. **输入**: 图像 $x_v$ + 文本指令 $x_t$ → VLM Backbone（LLaVA-1.5 或 Vila）
2. **候选生成**: 当前模型 $\pi_\theta$ 采样多个候选响应 $\{y^{(i)}\}$
3. **校准自奖励评分**（Calibrated Self-Reward Scorer）: 对每个 $y^{(i)}$ 计算 $R(y) = (1-\lambda)\langle U_1^\text{top} x_v, U_1^\text{top} y\rangle - \lambda\|y - V_1 x_v + V_2 x_t\|^2$，输出标量分数
4. **偏好对构造**（Preference Pair Generator）: 取最高分响应为 $y_w$（preferred），最低分为 $y_l$（dispreferred）
5. **DPO 训练更新**（Iterative DPO Trainer）: 以 $(y_w, y_l)$ 为监督信号，通过 Direct Preference Optimization 更新 $\pi_\theta \rightarrow \pi_{\theta'}$
6. **循环**: 用更新后的模型重新生成候选，重复步骤 2-5

```
(x_v, x_t) ──→ [VLM Backbone] ──→ {y^(1), ..., y^(k)}
                                      │
                                      ▼
                              [Calibrated Self-Reward]
                              R(y) = (1-λ)⟨U₁ᵀxᵥ,U₁ᵀy⟩ - λ‖y-V₁xᵥ+V₂xₜ‖²
                                      │
                                      ▼
                              [Preference Pair Gen]
                              y_w = argmax R(y),  y_l = argmin R(y)
                                      │
                                      ▼
                              [DPO Trainer] ──→ π_θ' ──→ (循环)
```

关键新组件为 **Calibrated Self-Reward Scorer**，其通过可学习的投影矩阵 $U_1, V_1, V_2$ 将视觉、文本、响应特征映射至共享空间，以参数 λ 显式权衡「视觉对齐」与「模态一致性」。

## 核心模块与公式推导

### 模块 1: 校准自奖励函数（对应框架图步骤 3）

**直觉**: 标准自奖励完全忽略视觉输入的结构化信息，CSR 通过二次型惩罚强制响应同时贴近视觉特征和文本特征，但以不同权重。

**Baseline 公式** (Self-Rewarding Language Models):
$$y_w = \text{arg}\max_y R_{\text{self}}(y) \quad \text{（纯文本信号，无显式 } x_v \text{ 项）}$$

符号: $x_v$ = 视觉特征, $x_t$ = 文本特征, $y$ = 候选响应, $R_{\text{self}}$ = 模型自评分数

**变化点**: Baseline 的最优解退化为 $y_w \approx V_2 x_t$（文本主导），视觉信号被完全抑制；CSR 引入双线性对齐项 + 模态一致性惩罚，并通过 λ 显式控制二者权重。

**本文公式（推导）**:
$$\text{Step 1}: \quad R(y) = (1-\lambda)\langle U_1^\text{top} x_v, U_1^\text{top} y\rangle - \lambda\|y - V_1 x_v + V_2 x_t\|^2 \quad \text{（定义校准奖励）}$$
$$\text{Step 2}: \quad \nabla_y R(y) = (1-\lambda) U_1 U_1^\text{top} x_v - 2\lambda(y - V_1 x_v + V_2 x_t) = 0 \quad \text{（对 } y \text{ 求梯度）}$$
$$\text{最终}: \quad y_w = \frac{1-\lambda}{\lambda} U_1 U_1^\text{top} x_v + V_1 x_v + V_2 x_t$$

**对应消融**: Table 10 显示 λ=0.9 时 MMEP 1524.2，λ=0.1 时 1508.6，Δ = +15.6；λ=0.5 时 1515.4，居中验证权衡必要性。

---

### 模块 2: 闭式最优响应与视觉信号上采样（对应框架图核心洞察）

**直觉**: 将最优响应显式解出后，可直观看到 CSR 如何通过 $(1-\lambda)/\lambda \cdot U_1 U_1^\text{top} x_v$ 项「注入」视觉信号。

**Baseline 公式** (标准 VLM 解码):
$$y_w = V_2 x_t \quad \text{（文本投影主导，视觉项系数为 0）}$$

符号: $U_1$ = 视觉→共享空间投影, $V_1$ = 视觉→响应空间投影, $V_2$ = 文本→响应空间投影, λ = 校准参数

**变化点**: Baseline 中 $x_v$ 仅通过 $V_1 x_v$ 间接影响，且通常 $\|V_1\| \ll \|V_2\|$；CSR 的闭式解显式添加与 λ 成反比的视觉增强项，当 λ→0 时视觉权重趋于无穷（理论极限），实际取 λ=0.9 平衡稳定性。

**本文公式（推导）**:
$$\text{Step 1}: \quad y_w^{\text{CSR}} = \underbrace{\frac{1-\lambda}{\lambda} U_1 U_1^\text{top} x_v}_{\text{视觉增强项}} + \underbrace{V_1 x_v + V_2 x_t}_{\text{基线响应}} \quad \text{（由模块1解出）}$$
$$\text{Step 2}: \quad \text{当 } \lambda \in (0,1), \frac{1-\lambda}{\lambda} > 0 \Rightarrow U_1 U_1^\text{top} x_v \text{ 系数恒正，视觉信号被显式放大}$$
$$\text{最终}: \quad y_w^{\text{CSR}} = V_1 x_v + V_2 x_t + \frac{1-\lambda}{\lambda} U_1 U_1^\text{top} x_v$$

**对应消融**: Table 10 中 λ=0.1（接近 baseline）时平均性能 66.61，λ=0.9 时 72.24，视觉增强项贡献 Δ = +5.63。

---

### 模块 3: 理论保证——Theorem 5.1 的表示误差界（对应框架图理论支撑）

**直觉**: 需证明 CSR 的表示确实优于纯文本基线，而非仅工程调参。

**Baseline 公式** (纯文本表示的预测误差下界):
$$\min_{\beta} \mathbb{E}[(\beta^{*\text{top}}(V_1^* x_v + V_2^* x_t) - \beta^\text{top} (V_2 x_t))^2] \geq \beta^{*\text{top}} V_1^* \text{Cov}(x_t) V_1^{*\text{top}} \beta^*$$

符号: $\beta^*$ = 最优线性预测器, $V_1^*, V_2^*$ = 真实投影矩阵, $\text{Cov}(x_t)$ = 文本特征协方差

**变化点**: Baseline 误差含 $V_1^* \text{Cov}(x_t) V_1^{*\text{top}}$ 项（视觉-文本耦合噪声），且无法通过优化消除；CSR 通过构造特定 $\beta_0$ 使视觉对齐项匹配，将误差压缩至正交补空间。

**本文公式（推导）**:
$$\text{Step 1}: \quad L(y) = \min_{\beta} \mathbb{E}[(\beta^{*\text{top}}(V_1^* x_v + V_2^* x_t) - \beta^\text{top} y)^2] \quad \text{（定义下游预测损失）}$$
$$\text{Step 2}: \quad \text{代入 } y = y_w^{\text{CSR}} \text{，构造 } \beta_0 \text{ s.t. } \frac{1-\lambda}{\lambda} U_1 U_1^\text{top} \beta_0 = U_1 U_1^\text{top} V_1^{*\text{top}} \beta^* \quad \text{（匹配视觉项）}$$
$$\text{Step 3}: \quad L(y_w^{\text{CSR}}) \leq \beta^{*\text{top}} V_1^* (I - U_1 U_1^\text{top}) \text{Cov}(x_t) (I - U_1 U_1^\text{top}) V_1^{*\text{top}} \beta^* \quad \text{（CSR 误差上界）}$$
$$\text{最终}: \quad \beta^{*\text{top}} V_1^* (I - U_1 U_1^\text{top}) \text{Cov}(x_t) (I - U_1 U_1^\text{top}) V_1^{*\text{top}} \beta^* < \beta^{*\text{top}} V_1^* \text{Cov}(x_t) V_1^{*\text{top}} \beta^*$$

因 $(I - U_1 U_1^\text{top})$ 为到 $U_1$ 正交补的投影算子，当 $U_1 \neq 0$ 且 $\text{Cov}(x_t) \text{succ} 0$ 时严格压缩二次型，证毕。

**对应消融**: 无直接数值消融，但 λ=0.9 的全面性能优势（Table 10）间接验证理论预测。

## 实验与分析



本文在多个视觉语言基准上评估 CSR，涵盖综合感知（MMEP、MM-Vet）、视觉问答（LLaVAW）及幻觉专精（CHAIR）三类任务。核心结果如 Table 10（及关联图表）所示：LLaVA-1.5 7B 经 5 轮 CSR 迭代后，**平均综合分数从 Iter-1 的 66.61 提升至 Iter-5 的 72.24，绝对增益 +5.63（相对 +7.62%）**；同架构 13B 模型亦获 +5.25% 相对提升，验证方法的可扩展性。幻觉指标上，CHAIR 转换分数从 59.2 跃升至 79.0，**相对改善达 +49.50%**，表明视觉信号上采样有效抑制了物体幻觉；LLaVAW 准确率从 66.7% 提升至 71.1%（+8.9%），显示校准机制对判别性任务同样有益。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/eb0275b3-76bf-4c64-b199-bd125c543e39/figures/fig_002.png)
*Figure: The CSR framework operates an iterative process of preference data generation and*





消融实验聚焦于校准参数 λ 与迭代轮次两个关键变量。λ 的敏感性在 Table 10 中清晰呈现：MMEP 感知分数 λ=0.1 时为 1508.6，λ=0.5 时 1515.4，λ=0.9 时 **1524.2**——λ=0.9 较 λ=0.1 提升 +15.6，较 λ=0.5 提升 +8.8，证实**视觉对齐权重需显著高于模态惩罚权重**方能释放 CSR 潜力。迭代轮次的消融（Figure 3 / Table 10）显示性能从 Iter-1 的 66.61 单调递增至 Iter-5 的 72.24，但增益边际递减（Iter-4→5 增幅收窄），暗示收敛趋势；作者因算力限制仅测试至 5 轮，未达完全收敛。

公平性审视：本文 baselines 以 LLaVA-1.5 和 Vila 为主，**未与同期幻觉专精方法（如 LURE、VCD、OPERA）或 VL 专用偏好学习方法（如 RLHF-V、HA-DPO）直接对比**，可能低估相对优势。此外，CHAIR 采用「100 − 原始 CHAIR」的转换计分，虽便于同向比较，但可能模糊绝对幻觉率。模型规模限于 7B/13B，未探索更大参数或 scaling law。作者亦承认此局限，指出未来需在更强 backbone 与更多迭代轮次上验证。

## 方法谱系与知识库定位

**方法家族**: 自提升（Self-Improving）/ 自奖励（Self-Rewarding）→ 视觉-语言偏好优化

**父方法**: **Self-Rewarding Language Models**（Yuan et al., 2024）。CSR 将其纯文本框架扩展至视觉-语言领域，核心改动三处：
- **reward_design**: 新增 Vision-Text Calibration Module，以 $U_1, V_1, V_2$ 投影矩阵实现模态对齐
- **objective**: 闭式最优解显式上采样图像信号，Theorem 5.1 提供理论保证
- **data_pipeline**: 自生成偏好对保留，但评分机制由纯文本扩展为视觉-文本联合校准

**直接 baselines 及差异**:
- **LLaVA-1.5**: 静态指令调优，无迭代自提升；CSR 在其上叠加校准自奖励循环
- **Vila 7B**: 替代 backbone 验证通用性，CSR 方法本身不变
- **Self-Rewarding LMs（纯文本）**: 无视觉校准，最优响应文本主导；CSR 解决其 VLM 迁移时的模态失衡

**后续方向**:
1. **Scaling**: 在 70B+ 模型及更多迭代轮次上验证收敛行为与性能上限
2. **跨模态扩展**: 将校准机制推广至视频、音频等多模态场景
3. **外部裁判融合**: 当前纯自奖励，未来可探索「自奖励 + GPT-4V/人类」的混合裁判以进一步提升数据质量

**标签**: 模态(视觉-语言) / 范式(自提升/迭代DPO) / 场景(幻觉降低/偏好优化) / 机制(投影空间校准/闭式最优解) / 约束(无外部裁判/算力受限≤5迭代)

## 引用网络

### 后续工作（建立在本文之上）

- [[P__多模态模型评判器LLaVA-Cr_LLaVA-Critic]]: Self-rewarding VLM with calibration; closely related method likely compared agai

