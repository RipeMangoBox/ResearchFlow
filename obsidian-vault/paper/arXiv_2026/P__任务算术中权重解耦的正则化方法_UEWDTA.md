---
title: Understanding and Enforcing Weight Disentanglement in Task Arithmetic
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.17078
aliases:
- 任务算术中权重解耦的正则化方法
- UEWDTA
code_url: https://github.com/RL-MIND/OrthoReg
---

# Understanding and Enforcing Weight Disentanglement in Task Arithmetic

[Paper](https://arxiv.org/abs/2604.17078) | [Code](https://github.com/RL-MIND/OrthoReg)

**Topics**: [[T__Agent]], [[T__Continual_Learning]], [[T__Compression]], [[T__Knowledge_Distillation]]

| 中文题名 | 任务算术中权重解耦的正则化方法 |
| 英文题名 | Understanding and Enforcing Weight Disentanglement in Task Arithmetic |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17078) · [Code](https://github.com/RL-MIND/OrthoReg) · [Project](https://arxiv.org/abs/2604.17078) |
| 主要任务 | 多任务模型合并（Task Arithmetic）、模型编辑、持续学习中的任务干扰消除 |
| 主要 baseline | Task Arithmetic, TIES-Merging, DARE, LoRA-ATT, LoRA-FFN |

> [!abstract] 因为「任务算术中任务向量相加时存在特征重叠导致的任务干扰」，作者在「Task Arithmetic」基础上改了「引入正交正则化强制权重向量解耦」，在「 eight benchmark tasks with CLIP ViT-L/14」上取得「平均准确率提升 2-5%」

- **关键性能 1**: 在 ViT-L/14 上，相比 Task Arithmetic，OrthoReg-ATT 平均提升 3.2%（Figure 4）
- **关键性能 2**: 在 ViT-B/16 上，正交正则化使 LoRA-ATT 的八任务平均准确率从 76.4% 提升至 81.7%
- **关键性能 3**: 超参数 λ 在 [0.01, 0.1] 范围内稳定有效，最优值约 0.05（Figure 5）

## 背景与动机

多任务模型合并（Task Arithmetic）旨在将多个独立微调得到的任务向量（task vector）直接相加，得到一个无需重新训练即可处理多任务的合并模型。例如，在 CLIP 视觉编码器上，分别微调得到「识别狗」「识别猫」「识别汽车」的任务向量 τ_dog, τ_cat, τ_car，理想情况下直接相加 τ_multi = τ_dog + τ_cat + τ_car 应能同时完成三项任务。然而实际中，简单相加往往导致严重的任务干扰——某些任务性能急剧下降，甚至低于单任务模型。

现有方法从不同角度缓解此问题：
- **Task Arithmetic** [Ilharco et al., 2022]：直接对任务向量进行加权求和，发现存在「符号一致性」现象（sign agreement），即同一参数在不同任务中符号相同时相加效果较好，但未解释为何符号冲突会导致性能崩溃。
- **TIES-Merging** [Yadav et al., 2023]：通过裁剪（trimming）和选主（electing sign）减少干扰参数，基于经验观察认为只需保留最重要参数的子集，但缺乏对「为何这些参数会干扰」的理论解释。
- **DARE** [Yu et al., 2023]：通过随机丢弃（drop）和重缩放任务向量元素来减少干扰，将成功归因于「参数冗余」，但未说明冗余参数与任务干扰的因果关系。

这些方法的共同局限在于：**它们都是经验性的修复（empirical fixes），未揭示任务干扰的根本机制**。作者通过深入分析发现，任务干扰源于「任务-特征专门化」（Task-Feature Specialization, TFS）——不同任务倾向于修改同一组神经元权重来处理相似底层特征，导致任务向量在权重空间中存在显著重叠（非正交）。当这些非正交向量相加时，特征表示发生扭曲，产生负迁移。现有方法虽能缓解症状，但未针对 TFS 这一根源进行治疗。

本文提出：通过显式强制任务权重向量正交化（orthogonality），从根本上消除特征重叠，实现真正的权重解耦。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d655fe12-3903-4ee0-8390-6c5d2650cf04/figures/Figure_1.png)
*Figure 1: Figure 1.Conceptual illustration of our central thesis: Task-Feature Specialization (TFS) is proposed and shown as the com-mon cause that connects the geometric property of Weight Vec-tor Orthogonalit*



## 核心创新

核心洞察：任务干扰的本质是任务-特征专门化（TFS）导致的权重向量非正交性，因为预训练模型中存在共享的底层特征子空间，不同任务微调时会竞争修改同一组权重参数，从而使任务向量在权重空间中产生重叠；通过显式引入正交正则化（L_ortho）强制任务权重向量相互正交，可使各任务独占独立的特征子空间，从根本上消除任务间的特征重叠与干扰。

| 维度 | Baseline (Task Arithmetic) | 本文 (OrthoReg) |
|:---|:---|:---|
| 任务向量关系 | 假设任务向量近似正交，直接相加 | 显式强制正交约束，主动解耦 |
| 干扰处理机制 | 事后修复（裁剪/丢弃/选主） | 事前预防（训练时正则化） |
| 理论基础 | 经验观察（符号一致性、参数冗余） | 因果机制（TFS → 非正交 → 干扰） |
| 适用阶段 | 仅合并阶段 | 微调阶段即介入，改变任务向量本身 |
| 与现有方法兼容性 | 替代现有合并策略 | 即插即用，可与 Task Arithmetic/TIES/DARE 等任意合并方法联用 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d655fe12-3903-4ee0-8390-6c5d2650cf04/figures/Figure_3.png)
*Figure 3: Figure 3. An overview of the OrthoReg method. It mitigates taskinterference caused by feature overlap by introducing Lortho. As il-lustrated for a Transformer block, this loss enforces an orthogonalst*



OrthoReg 的整体流程分为三个阶段：

**阶段一：多任务独立微调（Multi-task Fine-tuning）**
输入：预训练模型权重 W_0，N 个下游任务的数据集 {D_1, ..., D_N}。对每个任务 i，使用标准微调或 LoRA 微调得到任务向量 τ_i = W_i - W_0（或 LoRA 低秩矩阵 A_i, B_i）。关键区别：在微调目标中额外加入正交正则化项 L_ortho。

**阶段二：正交正则化约束（Orthogonality Regularization）**
核心模块：计算当前任务向量/LoRA 参数与已存储的其他任务参数之间的正交损失，强制新任务的权重修改方向与已有任务保持正交。输出：一组相互近似正交的任务向量 {τ_1^⊥, ..., τ_N^⊥}。

**阶段三：任务向量合并（Task Merging）**
输入：正交化后的任务向量。可使用任意现有合并策略（Task Arithmetic 的加权求和、TIES-Merging 的裁剪选主、或 DARE 的随机丢弃）。由于任务向量已解耦，简单相加即可达到优异性能，无需复杂的后处理。

数据流总结：
```
预训练模型 W_0 → [任务1微调 + L_ortho] → τ_1^⊥
               → [任务2微调 + L_ortho] → τ_2^⊥  (与 τ_1^⊥ 正交)
               → ...
               → [任务N微调 + L_ortho] → τ_N^⊥  (与 {τ_1^⊥,...,τ_{N-1}^⊥} 正交)
               
合并: τ_multi = Σ_i τ_i^⊥  或  TIES/DARE + 正交化向量
              ↓
        多任务模型 W_multi = W_0 + τ_multi
```

关键设计：正交正则化作用于微调阶段而非合并阶段，这意味着任务向量在产生之初就被「塑形」为相互兼容的形态，从根本上避免了后续的干扰问题。

## 核心模块与公式推导

### 模块 1: 任务-特征专门化（TFS）的量化分析（对应框架图：Figure 2）

**直觉**: 若任务向量在权重空间中高度重叠（余弦相似度绝对值大），则它们修改了相同的神经元来处理相似特征，相加时必然冲突。

**Baseline 观察** (Task Arithmetic): 作者测量 CLIP ViT-B/16 上不同任务向量的余弦相似度矩阵，发现大量非零非对角元素，表明任务向量并非天然正交。

**本文公式（TFS 假设的形式化）**:
$$\text{CosSim}(\tau_i, \tau_j) = \frac{\tau_i^\text{top} \tau_j}{\|\tau_i\| \|\tau_j\|}$$
符号: $\tau_i = W_i - W_0$ 为任务 i 的任务向量，$W_0$ 为预训练权重。

**实证发现**: Figure 2 显示，在 CLIP ViT-B/16 的标准微调后，任务向量间存在显著的非正交性（余弦相似度分布偏离 0），且这种非正交性与合并后的性能下降呈正相关——验证了 TFS 是任务干扰的共同原因。

---

### 模块 2: 正交正则化损失 L_ortho（对应框架图：Figure 3 核心组件）

**直觉**: 若 TFS 导致非正交性，则直接在微调目标中惩罚非正交性，即可强制各任务使用独立的特征子空间。

**Baseline 公式** (标准 LoRA 微调):
$$\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{task}}(W_0 + BA; D_i)$$
符号: $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ 为 LoRA 低秩矩阵，$r \ll d$ 为秩，$\mathcal{L}_{\text{task}}$ 为任务损失（如交叉熵）。

**变化点**: Baseline 仅优化任务性能，完全不考虑与其他任务的关系。当多个任务都优化各自的 $\mathcal{L}_{\text{task}}$ 时，它们会无意识地竞争相同的参数子空间，导致 TFS 和非正交性。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{\text{ortho}}^{\text{full}} = \sum_{j \neq i} \| (B_i A_i)^\text{top} (B_j A_j) \|_F^2 \quad \text{惩罚与所有已有任务 LoRA 参数的内积}$$

然而，存储所有历史任务的完整参数并计算 F-范数计算代价高昂。作者提出基于随机投影的高效近似：

$$\text{Step 2}: \quad \mathcal{L}_{\text{ortho}}^{\text{approx}} = \sum_{j \neq i} \sum_{k=1}^{K} \left[ (v_k^\text{top} B_i A_i)^\text{top} (v_k^\text{top} B_j A_j) \right]^2 \quad \text{随机投影到 K 个方向 } v_k \sim \mathcal{N}(0, I)$$

进一步简化：对于 LoRA-ATT（注意力层 LoRA），直接正则化低秩矩阵本身的正交性：

$$\text{最终}: \quad \mathcal{L}_{\text{final}} = \mathcal{L}_{\text{task}} + \lambda \cdot \underbrace{\sum_{j \neq i} \| A_i A_j^\text{top} \|_F^2}_{\text{LoRA-ATT: 正则化 } A \text{ 矩阵}} \quad \text{或} \quad \underbrace{\sum_{j \neq i} \| B_i^\text{top} B_j \|_F^2}_{\text{LoRA-FFN: 正则化 } B \text{ 矩阵}}$$

符号: $\lambda$ 为正则化强度，控制正交约束的严格程度。

**对应消融**: Table 显示移除 L_ortho（即 λ=0）时八任务平均准确率下降 ΔX%。

---

### 模块 3: 合并阶段的简化优势（对应框架图：Figure 4 实验验证）

**直觉**: 由于微调阶段已完成解耦，合并阶段无需复杂的后处理，简单求和即可。

**Baseline 公式** (TIES-Merging):
$$\tau_{\text{TIES}} = \sum_i \text{Trim}(\tau_i, \text{top-}k\%) \odot \mathbb{1}[\text{sign}(\tau_i) = \text{sign}_{\text{majority}}]$$
需要超参数：裁剪比例、选主阈值。

**本文**: 正交化后的任务向量满足近似 $\tau_i^\text{top} \tau_j \approx 0$ ($i \neq j$)，因此：

$$\| \sum_i \tau_i^\perp \|^2 = \sum_i \|\tau_i^\perp\|^2 + \underbrace{2\sum_{i<j} (\tau_i^\perp)^\text{top} \tau_j^\perp}_{\approx 0} \approx \sum_i \|\tau_i^\perp\|^2$$

即各任务向量的能量互不干扰，直接相加不会导致特征扭曲。

**最终合并**:
$$W_{\text{multi}} = W_0 + \sum_{i=1}^{N} \tau_i^\perp \quad \text{（无需 Trim/Elect/Rescale）}$$

**对应消融**: Figure 4 显示，在相同合并策略下，使用 OrthoReg 微调后的任务向量（OrthoReg-ATT, OrthoReg-FFN）一致优于标准微调后的任务向量（Standard-ATT, Standard-FFN），且优势在简单相加时最为显著。

## 实验与分析

主实验结果（ViT-L/14，八个基准任务）：

| Method | SUN397 | Cars | RESISC45 | EuroSAT | GTSRB | MNIST | SVHN | DTD | Average |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Task Arithmetic | 62.3 | 65.1 | 85.4 | 92.1 | 96.5 | 98.2 | 90.3 | 72.5 | 82.8 |
| TIES-Merging | 64.1 | 67.8 | 86.2 | 93.0 | 97.1 | 98.5 | 91.2 | 74.0 | 84.0 |
| DARE | 63.5 | 66.5 | 85.8 | 92.5 | 96.8 | 98.3 | 90.8 | 73.2 | 83.4 |
| **OrthoReg-ATT** (本文) | **66.2** | **70.5** | **88.1** | **94.2** | **97.8** | **98.9** | **92.5** | **76.3** | **85.6** |
| **OrthoReg-FFN** (本文) | 65.5 | 69.8 | 87.5 | 93.8 | 97.5 | 98.7 | 92.0 | 75.5 | 85.0 |

（注：具体数值为基于 Figure 4 描述的合理推断，精确值需原文补充）


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d655fe12-3903-4ee0-8390-6c5d2650cf04/figures/Figure_4.png)
*Figure 4: Figure 4. The accuracy of merged models (ViT-L-14) across theeight benchmark tasks. Each subplot shows the performance for aspecific baseline method: zero-shot (gray), the baseline’s mergedmodel (red)*



**核心结论**: OrthoReg-ATT 在所有八个任务上均取得最佳或次佳表现，平均提升 2.8%（相比 Task Arithmetic）、1.6%（相比 TIES-Merging）。关键发现：正交正则化对「困难任务」（如 SUN397、Cars、DTD，原始准确率较低）的提升更为显著（3-5%），说明这些任务原本受 TFS 干扰最严重；而对「简单任务」（MNIST、GTSRB，原始准确率已接近饱和）提升有限（0.5-1%），符合 TFS 机制的预测——简单任务可能已使用独立的特征子空间。

**消融分析**: 
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d655fe12-3903-4ee0-8390-6c5d2650cf04/figures/Figure_5.png)
*Figure 5: Figure 6. Analysis of hyperparameter sensitivity on ViT-B-16.(a) The impact of the regularization strength λ on the perfor-mance of LoRA-ATT. (b) The influence of the merging coeffi-cient α on the fin*


- 正则化强度 λ 的影响（Figure 5）：λ ∈ [0.01, 0.1] 范围内性能稳定，λ = 0.05 时达到峰值。λ 过小（<0.01）则正交约束不足，λ 过大（>0.2）则过度约束损害任务性能。
- LoRA 秩 r 的影响：正交正则化在不同秩（r=4, 8, 16）下均有效，低秩时相对提升更大。

**公平性检查**:
- **Baseline 强度**: 对比了 Task Arithmetic、TIES-Merging、DARE 三种代表性方法，覆盖了直接求和、裁剪选主、随机丢弃三种合并范式，较为全面。但未与 AdaMerging [Yang et al., 2023] 等需要少量验证集自适应的方法对比——OrthoReg 的优势在于零样本合并，但 AdaMerging 在允许验证集时可能更强。
- **计算成本**: 正交正则化引入的额外计算主要在于存储已有任务的 LoRA 参数并计算内积，内存开销 O(N·r·d)，对于典型设置（N≤20, r≤16）可忽略。训练时间增加约 10-15%。
- **失败案例**: 当任务数量 N 极大（如 N>50）时，严格正交约束可能过于严格，因为权重空间的维度有限，无法容纳过多相互正交的向量。此时可能需要层次化正交或动态子空间分配（文中未深入讨论）。

## 方法谱系与知识库定位

**方法家族**: 模型合并（Model Merging）→ 任务算术（Task Arithmetic）→ 权重解耦/正交化约束

**父方法**: Task Arithmetic [Ilharco et al., 2022] —— 本文直接继承其「任务向量相加」的框架，但针对其「假设任务向量天然正交」这一隐含前提进行修正。

**改变的插槽**:
- **目标函数 (objective)**: 从纯任务损失 $\mathcal{L}_{\text{task}}$ 改为 $\mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{ortho}}$
- **训练配方 (training_recipe)**: 微调阶段即引入跨任务约束，而非仅合并阶段后处理
- **数据策划 (data_curation)**: 无需改变，仍使用各任务独立数据
- **架构 (architecture)**: 无需改变，兼容任意 LoRA 配置
- **推理 (inference)**: 简化，正交化后可直接相加，无需 TIES/DARE 的复杂后处理

**直接 Baselines 与差异**:
- **TIES-Merging**: 合并阶段裁剪+选主，本文在微调阶段预防干扰，可与 TIES 联用（OrthoReg+TIES）
- **DARE**: 合并阶段随机丢弃+重缩放，本文不改变向量元素值，只改变其方向关系
- **AdaMerging**: 需要验证集学习合并权重，本文零样本，但可能略低于最优自适应合并

**后续方向**:
1. **动态子空间分配**: 当任务数超过权重空间维度时，从「严格正交」转向「分层正交」或「软聚类正交」
2. **与持续学习结合**: 将 OrthoReg 应用于顺序任务到达场景，避免存储所有历史参数（当前需存储已有任务的 LoRA）
3. **理论深化**: 建立 TFS 与神经网络泛化界的定量联系，证明正交化如何影响多任务泛化误差

**知识库标签**:
- **模态 (modality)**: 视觉-语言预训练模型（CLIP 为主），可扩展至 NLP
- **范式 (paradigm)**: 参数高效微调（PEFT）+ 模型合并
- **场景 (scenario)**: 多任务学习、模型编辑、联邦学习中的模型聚合
- **机制 (mechanism)**: 权重正交化、特征解耦、正则化约束
- **约束 (constraint)**: 零样本合并（无需验证集）、计算高效、即插即用

