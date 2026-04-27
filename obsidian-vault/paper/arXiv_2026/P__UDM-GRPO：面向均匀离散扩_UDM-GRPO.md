---
title: 'UDM-GRPO: Stable and Efficient Group Relative Policy Optimization for Uniform Discrete Diffusion Models'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.18518
aliases:
- UDM-GRPO：面向均匀离散扩散模型的稳定高效群组相对策略优化
- UDM-GRPO
- UDM-GRPO的核心直觉是：**让RL训练的状态-动作分布尽可能贴近
method: UDM-GRPO
modalities:
- Image
paradigm: Reinforcement Learning
---

# UDM-GRPO: Stable and Efficient Group Relative Policy Optimization for Uniform Discrete Diffusion Models

[Paper](https://arxiv.org/abs/2604.18518)

**Topics**: [[T__Image_Generation]], [[T__Reinforcement_Learning]], [[T__Text_Generation]] | **Method**: [[M__UDM-GRPO]] | **Datasets**: GenEval, PickScore, OCR

> [!tip] 核心洞察
> UDM-GRPO的核心直觉是：**让RL训练的状态-动作分布尽可能贴近预训练时的分布**。动作重定义（$\hat{x}_1$）消除了中间预测的噪声梯度，使优化目标与奖励定义对齐；前向轨迹重建将训练状态拉回预训练流形，消除OOD暴露。两者共同作用的本质是：在离散扩散的RL微调中，**用已知干净样本的前向加噪路径替代模型自身生成的反向去噪路径**，从而在保持分布一致性的同时提供准确的优化信号。这一思路有效，因为它将RL探索的不确定性与训练稳定性的需求解耦。

## 基本信息

**论文标题**: UDM-GRPO: Stable and Efficient Group Relative Policy Optimization for Uniform Discrete Diffusion Models

**作者**: （未在提供的分析中提取完整作者列表）

**发表 venue**: （未明确提取）

**年份**: （未明确提取）

**代码/数据链接**: （未在提供的分析中提取）

**基础模型**: URSA (1.7B 参数均匀离散扩散模型)

**训练资源**: 32 × NVIDIA A100 40GB GPUs

## 核心主张

UDM-GRPO 是首个将群组相对策略优化（GRPO）成功应用于均匀离散扩散模型（UDM）的框架，通过**将动作重新定义为最终干净样本 x̂₁**（而非中间预测 xᵗ₁）和**采用前向扩散过程重构训练轨迹**，解决了直接集成 GRPO 导致的训练不稳定问题（奖励剧烈波动、KL 散度急剧增长）。

**关键证据**：
- GenEval 总体得分 **0.96**，超越 URSA 基线（0.69）提升 **39.1%**，略超最佳连续流方法 Flow-GRPO（0.95）
- PickScore **23.81**，超越所有对比方法
- 消融实验显示：反向轨迹+中间动作配置（naive 集成）GenEval 仅 0.84，且出现严重不稳定；最终方法达 0.96

**置信度评估**: 高（0.90），核心主张有充分实验支撑，但部分对比存在基线不对等限制。

## 研究动机

均匀离散扩散模型（UDM）在文本到图像生成中展现出潜力，但**缺乏有效的强化学习微调方法**以提升人类偏好对齐。现有工作存在两大空白：

1. **连续扩散的 RL 方法无法直接迁移**：Flow-GRPO 专为连续流匹配设计，其动作空间和轨迹构造依赖连续状态假设，离散 token 空间需要根本不同的数学处理。

2. **Naive 集成导致训练崩溃**：Figure 1 显示，直接将标准 GRPO 应用于 UDM 时，奖励先升后剧烈震荡，KL 散度急剧攀升——根源在于**早期去噪步骤的高熵使中间预测 xᵗ₁ 不可靠**，导致错误信用分配；同时**反向生成轨迹 X_backward 偏离预训练分布**，产生 OOD 状态。

本工作填补 UDM 与 RLHF 结合的空白，为离散扩散模型提供首个稳定高效的在线优化框架。

## 方法流程

UDM-GRPO 的四阶段流程（Figure 3）：

```
Prompt c + 组大小 G
    ↓
[Clean Sample Generation] → G 个干净样本 {x̂₁ⁱ}（反向去噪，10步）
    ↓
[Forward Trajectory Construction] → 前向扰动得 {x̂_{tᵢʲ}ⁱ} 
    │   （新模块：用前向扩散 p_{tᵢʲ}(x|x̂₁ⁱ) 重构噪声状态，替代反向轨迹）
    ↓
[Reward Computation] → 奖励 R(x̂₁ⁱ,c) + 组相对优势 Âᵢ
    ↓
[Policy Optimization] → 修改版 GRPO 更新 θ
    │   （新模块：基于前向状态和最终样本动作的重要性比率）
    │   + Reduced-Step：仅优化早期时间步（蓝虚线框，Section 4.3）
    │   + CFG-Free：训练时移除分类器自由引导（Section 4.4）
    ↓
Updated Policy θ
```

**核心创新模块**：Forward Trajectory Constructor、Final-Sample Action Redefinition、Reduced-Step Selector、CFG-Free Sampler。

## 关键公式

**1. 动作重定义（核心创新）**
```latex
a_t \triangleq \hat{x}_1, \quad \pi(a_t \mid s_t) \triangleq p_\theta(\hat{x}_1 \mid \hat{x}_t, c)
```
将动作从不可靠的中间预测 $x^t_1$ 改为最终干净样本 $\hat{x}_1$，与扩散预训练目标一致。

**2. 前向轨迹采样（核心创新）**
```latex
\hat{x}_{t_i^j}^i \sim p_{t_i^j}(x \mid \hat{x}_1^i)
```
通过前向扩散从干净样本重构噪声状态，对齐预训练分布。

**3. 修改的重要性比率（核心创新）**
```latex
r_{t_i^j}^i(\theta) = \frac{p_\theta(\hat{x}_1^i \mid \hat{x}_{t_i^j}^i, c)}{p_{\theta_{\text{old}}}(\hat{x}_1^i \mid \hat{x}_{t_i^j}^i, c)}
```
基于前向状态和最终样本动作计算，替代标准 GRPO 的中间状态比率。

**4. 裁剪策略目标（继承 GRPO）**
```latex
\mathcal{J}_{\mathrm{policy}}^{(t_i^j, i)} = \min\left(r_{t_i^j}^i(\theta)\hat{A}_i,\; \mathrm{clip}\left(r_{t_i^j}^i(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_i\right)
```

**5. 完整损失（适配修改）**
```latex
\mathcal{L} \leftarrow \mathcal{L} - \mathcal{J}_{\mathrm{policy}}^{(t_i^j, i)} + \beta\, D_{\mathrm{KL}}\!\left(p_\theta(\cdot \mid \hat{x}_{t_i^j}^i, c) \,\|\, p_{\mathrm{ref}}(\cdot \mid \hat{x}_{t_i^j}^i, c)\right)
```

**6. 组相对优势（继承 GRPO，未修改）**
```latex
\hat{A}_i = \frac{R(x_1^i, c) - \mathrm{mean}\left(\{R(x_1^i, c)\}_{i=1}^G\right)}{\mathrm{std}\left(\{R(x_1^i, c)\}_{i=1}^G\right)}
```

**新颖性标记**：公式 1-3 为本文创新；公式 4-6 继承自 GRPO/Flow-GRPO，但应用语境改变。

## 实验结果

**主实验结果**（Table 1, Table 2）：

| 基准 | 指标 | UDM-GRPO | 对比基线 | 提升 |
|:---|:---|:---|:---|:---|
| **GenEval** | Overall | **0.96** | URSA 0.69 / Flow-GRPO 0.95 / FLUX.1-Dev 0.66 | +0.27 vs URSA, **SOTA** |
| GenEval | Position | 0.97 | URSA 0.28 / Flow-GRPO 0.99 | +0.69 vs URSA |
| **PickScore** | Score | **23.81** | URSA 21.79 / SD3.5-L 22.91 | +2.02 vs URSA, **SOTA** |
| OCR | Accuracy | 0.57 | URSA 0.08 / SD3.5-L 0.68 / FLUX.1-Dev 0.59 | +0.49 vs URSA |

**关键消融**（Table 3, Figure 6）：
- **最终样本动作 vs 中间动作**：x̂₁ 配置 GenEval 0.89→0.94（前向轨迹下），KL 更低
- **前向 vs 反向轨迹**：前向 0.94 vs 反向 0.89，反向最终崩溃
- **CFG-Free vs CFG**：CFG-Free 最终 0.96 vs CFG 0.94，收敛更快、KL 更低
- **Naive 集成**（反向+xᵗ₁）：严重不稳定，GenEval 仅 0.84

**证据强度评估**：0.75/1.0。主要限制：仅单一基模型（URSA）验证；与 Flow-GRPO 的对比基于不同架构（SD3.5-M 2.5B vs URSA 1.7B）；OCR 仍显著落后连续扩散基线。

## 相关工作

**按角色分类的关键引用**：

**基线方法（直接对比）**：
- **URSA**（*Uniform discrete diffusion with metric path for video generation*）：主要基线，1.7B UDM 基础模型，UDM-GRPO 对其微调
- **Flow-GRPO**（*Flow-grpo: Training flow matching models via online rl*）：直接前驱，连续流匹配上的 GRPO 应用，GenEval 0.95
- **Show-o w/ Mask-GRPO**（*Show-o: One single transformer to unify multimodal understanding and generation*）：离散模型基线，但架构不同（Show-o vs URSA）

**组件来源**：
- **GRPO**（*Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning*）：核心 RL 算法基础，UDM-GRPO 修改其动作定义和轨迹构造

**重要关系**：UDM-GRPO 与 Flow-GRPO 构成**方法谱系中的连续-离散扩展关系**，而非简单并列竞争；与 Show-o/Mask-GRPO 的对比受限于基模型差异，公平性不足。

## 方法谱系

UDM-GRPO 位于 **GRPO → Flow-GRPO → UDM-GRPO** 演化链的末端：

```
GRPO (通用RL算法, DeepSeek-R1)
    ↓ builds_on [policy_network_architecture, trajectory_construction, state_space]
Flow-GRPO (连续流匹配适配)
    ↓ builds_on [action_definition, trajectory_construction, credit_assignment, 
                 training_recipe, inference_strategy]
UDM-GRPO (离散扩散适配) ← 本文
```

**从 Flow-GRPO 继承的 slot**：
- 群组相对优势估计（Âᵢ 计算）
- 裁剪策略目标函数结构
- 无 critic 的在线 RL 框架

**修改的 slot**：
| Slot | Flow-GRPO (基线值) | UDM-GRPO (新值) |
|:---|:---|:---|
| `action_definition` | 中间预测 $x^t_1$ | **最终样本 x̂₁** |
| `trajectory_construction` | 反向轨迹 X_backward | **前向重构 X_forward** |
| `credit_assignment` | 中间状态重要性比率 | **前向状态+最终样本比率** |
| `training_recipe` | 全时间步优化 | **Reduced-Step 早期时间步** |
| `inference_strategy` | 标准 CFG | **CFG-Free 训练采样** |

UDM-GRPO 是首个将 GRPO 框架成功拓展至**离散 token 空间**和**均匀离散扩散**的工作，填补了谱系中的关键空白。

## 局限与展望

**论文明确陈述的局限**：
- OCR 性能（0.57）仍显著低于连续扩散基线 SD3.5-L（0.68），视觉文本渲染能力有限
- 仅基于单一基模型 URSA（1.7B）验证，未在其他离散扩散架构上测试

**分析推断的额外局限**：
- **对比公平性不足**：Flow-GRPO 最佳结果使用 SD3.5-M（2.5B），与 URSA（1.7B）架构和规模均不同；Show-o/Mask-GRPO 基线模型不同
- **CFG 消融混淆**：CFG-Free 的 0.96 结果与 "URSA (w/o CFG)" 基线（20.46 PickScore）对比，但 CFG-Free 是训练策略而非基线模型，消融解释需谨慎
- **缺乏标准 RL 基线**：未与 URSA+PPO、URSA+REINFORCE 等对比，无法隔离 GRPO 本身的贡献
- **Reduced-Step 的理论依据不足**：早期时间步优化的理论最优性未严格证明

**未来方向**：
1. 扩展到更大规模的离散扩散模型（如 Show-o、LlamaGen）验证通用性
2. 结合专门的可读性奖励提升 OCR 性能
3. 探索 Reduced-Step 时间步选择的自适应策略
4. 开发离散-连续统一的 RL 框架，统一 Flow-GRPO 与 UDM-GRPO

## 知识图谱定位

UDM-GRPO 在知识图谱中连接以下关键节点：

**任务节点**：
- `text-to-image generation`（主任务）
- `human preference alignment` / RLHF（优化目标）
- `compositional image generation`（GenEval 评测维度）
- `visual text rendering`（OCR 评测维度，性能瓶颈）

**方法节点**：
- `UDM-GRPO`（新增核心方法节点）
- `GRPO` → `Flow-GRPO` → `UDM-GRPO`（演化链）
- `URSA`（被修改的基础模型节点）
- `forward trajectory reconstruction` / `final-sample-as-action` / `Reduced-Step training` / `CFG-Free sampling`（新增机制节点）

**基准/数据集节点**：
- `GenEval`（SOTA 0.96）、`PickScore`（SOTA 23.81）、`OCR`（0.57，非 SOTA）
- `Pick-a-Pic`（训练数据）

**领域结构贡献**：
UDM-GRPO 填补了"**离散扩散模型 × 在线强化学习**"的交叉空白，建立了从连续流匹配（Flow-GRPO）到离散扩散的方法迁移路径。其提出的"前向轨迹+最终样本动作"范式可能成为后续离散生成模型 RL 调优的标准组件，推动扩散模型 RL 方法从连续域向离散域的系统扩展。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8e8d099-ad08-4edb-8e63-fae24c3ec294/figures/Figure_1.png)
*Figure 1: Figure 1. Reward–step training curve. The baseline suffers fromoptimization collapse after 500 steps, characterized by violentreward oscillation and exploding KL divergence. In contrast, ourUDM-GRPO a*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8e8d099-ad08-4edb-8e63-fae24c3ec294/figures/Figure_2.png)
*Figure 2: Figure 2. Illustration of the three trajectories. Xbackward denoisesx0 via the reverse process to obtain ˆx1. In contrast, Xforward andXpretrain share the same forward diffusion process but differ in*


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8e8d099-ad08-4edb-8e63-fae24c3ec294/figures/Figure_4.png)
*Figure 4: Figure 3. Overview of UDM-GRPO. Given a prompt, we first sample G clean images ˆx1 using the reverse process of UDM. To solve theinstability caused by directly using this Xbackward as trajectory and x*


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8e8d099-ad08-4edb-8e63-fae24c3ec294/figures/Figure_5.png)
*Figure 5: Figure 5. Qualitative Comparison. We evaluate our model againstSD3.5-L, Flux.1 Dev and URSA using prompts from GenEval andPickScore, respectively.*


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8e8d099-ad08-4edb-8e63-fae24c3ec294/figures/Figure_6.png)
*Figure 6: Figure 7. Qualitative Comparison. We compare different methodsfor integrating GRPO into our base model. From left to right, theresults correspond to (a): backward + xt1, (b): backward + ˆx1, (c):forwa*


![Figure 8](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8e8d099-ad08-4edb-8e63-fae24c3ec294/figures/Figure_8.png)
*Figure 8: Figure 8. Qualitative Comparison. The prompts are taken from GenEval, PickScore respectively, where we compare the SD3.5-L andFlux.1 Dev with our model.*


![Figure 9](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8e8d099-ad08-4edb-8e63-fae24c3ec294/figures/Figure_9.png)
*Figure 9: Figure 9. Visualization for different method.*


![Figure 10](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8e8d099-ad08-4edb-8e63-fae24c3ec294/figures/Figure_10.png)
*Figure 10: Figure 10. We visualize the generated samples across successive training iterations during the optimization.*


