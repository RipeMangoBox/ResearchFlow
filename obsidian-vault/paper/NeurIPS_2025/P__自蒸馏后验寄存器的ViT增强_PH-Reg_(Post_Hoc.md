---
title: Vision Transformers with Self-Distilled Registers
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 自蒸馏后验寄存器的ViT增强
- PH-Reg (Post Hoc
- PH-Reg (Post Hoc Registers)
- Post Hoc Registers (PH-Reg) can eff
acceptance: Spotlight
cited_by: 8
method: PH-Reg (Post Hoc Registers)
modalities:
- Image
paradigm: self-supervised
---

# Vision Transformers with Self-Distilled Registers

**Topics**: [[T__Semantic_Segmentation]], [[T__Depth_Estimation]] | **Method**: [[M__PH-Reg]] | **Datasets**: Open-vocabulary Semantic, Training Efficiency vs DVT, Inference Cost vs DVT

> [!tip] 核心洞察
> Post Hoc Registers (PH-Reg) can efficiently add register tokens to existing pre-trained ViTs without labeled data or full retraining, using self-distillation with test-time augmentation to generate artifact-free dense embeddings.

| 中文题名 | 自蒸馏后验寄存器的ViT增强 |
| 英文题名 | Vision Transformers with Self-Distilled Registers |
| 会议/期刊 | NeurIPS 2025 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.21501) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Semantic Segmentation, Depth Estimation, Open-Vocabulary Semantic Segmentation |
| 主要 baseline | DVT (Dense Vision Transformer), NACLIP, SCLIP, MaskCLIP |

> [!abstract] 因为「预训练ViT的artifact token破坏密集预测质量，而现有register方法需从头训练」，作者在「DVT」基础上改了「单阶段自蒸馏+测试时增强生成去噪教师目标+仅解锁部分学生权重」，在「8个开放词汇分割benchmark」上取得「mIoU提升且节省58.9%训练时间」

- 相比DVT节省 **58.9%** 训练时间，无需存储1.4TB中间neural field
- 在 **8个开放词汇语义分割benchmark** 上提升zero-shot分割性能
- 使用 **DINOv2 ViT-B** 作为统一初始化，公平对比DVT

## 背景与动机

Vision Transformers (ViTs) 在全局图像理解上表现优异，但其自注意力机制会产生**artifact token**——即与局部图像结构语义不一致的异常特征，严重损害细粒度定位任务。例如，在开放词汇语义分割中，背景区域可能被错误激活为前景物体，导致分割边界模糊。

现有解决方案沿三条路径展开：
- **Register tokens**（ViT with registers）：在输入序列中加入特殊寄存器token以吸收artifact项，但**必须从头训练**，无法直接应用于已预训练的大型模型如DINOv2；
- **DVT (Dense Vision Transformer)**：通过两阶段训练——先拟合Instant-NGP neural field生成密集监督，再蒸馏到ViT——实现artifact-free特征，但需存储**1.4TB中间neural field**，训练成本极高；
- **CLIP-based分割方法**（NACLIP/SCLIP/MaskCLIP）：利用文本-图像对齐特征进行zero-shot分割，但未根本解决ViT内部的artifact问题，密集特征质量受限。

核心瓶颈在于：**如何将register token的好处迁移到已预训练的ViT，同时避免DVT的高昂存储与两阶段训练？** 本文提出PH-Reg，以单阶段自蒸馏框架，无需标注数据、无需存储中间表示，直接为现有ViT注入register能力。

## 核心创新

**核心洞察**：测试时增强（TTA）的均值聚合天然是去噪的最优估计，因为MSE下的最优预测即为多扰动样本的经验均值，从而使"无artifact教师监督"成为可能；配合部分权重解锁，仅优化register token和少量adapter即可适配预训练 backbone，无需全量重训练。

| 维度 | Baseline (DVT) | 本文 (PH-Reg) |
|:---|:---|:---|
| 训练阶段 | 两阶段：neural field拟合 + ViT蒸馏 | 单阶段自蒸馏，端到端 |
| 存储需求 | 1.4TB neural field中间结果 | 零额外存储，目标实时计算 |
| 监督来源 | 梯度优化的neural field渲染 | TTA均值去噪的冻结教师特征 |
| 可训练参数 | 完整ViT backbone + neural field | 仅register token + 选定adapter权重 |
| 训练时间 | 基准 | **节省58.9%** |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8ac0d7e2-6d7f-47f7-9005-dff0f4a69de3/figures/fig_001.png)
*Figure: Effect of PH-Reg on Open-vocabulary Segmentation. For each image, we compare four*



PH-Reg的学习框架包含三个核心阶段：

1. **网络初始化**：教师（Teacher）与学生（Student）从**同一预训练ViT**（如DINOv2 ViT-B）初始化；教师全程**冻结**，学生则在其输入层**追加随机初始化的register token**。

2. **教师去噪**：对输入图像施加多种数据增强（随机裁剪、颜色抖动等），教师网络为每种增强生成密集特征；按**最优聚合公式**取均值，得到去噪后的artifact-free监督目标。

3. **学生蒸馏与部分更新**：学生网络处理含register token的完整序列，其输出密集特征与教师去噪目标计算MSE损失；**仅register token嵌入和选定adapter权重参与梯度更新**，主干backbone保持冻结。

数据流示意：
```
输入图像 x
    ├──→ [教师分支: 冻结ViT] + {增强1, 增强2, ..., 增强n} → 特征{f_1,...,f_n} → 均值聚合 → f* (去噪目标)
    └──→ [学生分支: ViT + 可学习Register Token] ──────────────────────────────→ f_student
                                                      ↓
                                              MSE(f_student, f*)
                                                      ↓
                                              仅更新: [Register Token, Adapter]
```

关键设计：register token作为"信息垃圾桶"吸收artifact项，TTA均值提供稳定监督，部分解锁保证高效适配。

## 核心模块与公式推导

### 模块 1: 测试时增强去噪（教师目标生成）

**直觉**：单一前向传播的教师特征仍含噪声，多增强样本的均值可在MSE意义下最优抑制随机扰动。

**Baseline（无去噪）**：直接使用教师单样本输出 $f_{\text{single}}$ 作为监督目标，隐含假设 $f_{\text{single}} = f_{\text{clean}}$，实际受artifact污染。

**变化点**：引入$n$种增强变换$\mathcal{T}_i$，教师生成$n$个特征；证明均值最小化总平方误差。

**本文公式（推导）**：
$$\text{Step 1:} \quad f_i = \text{Teacher}(\mathcal{T}_i(x)), \; i=1,...,n \quad \text{（多增强前向）}$$
$$\text{Step 2:} \quad f^* = \text{arg}\min_{\mu} \sum_{i=1}^n \|f_i - \mu\|^2 = \frac{1}{n}\sum_{i=1}^n f_i \quad \text{（最优聚合：经验均值去噪）}$$
$$\text{最终:} \quad f^* = \frac{1}{n}\sum_{i=1}^n f_i$$

符号：$f_i$ = 第$i$种增强下的教师密集特征，$f^*$ = 去噪后的最优监督目标，$n$ = 增强数量。

**对应消融**：Figure 5显示不同增强数量对性能的影响，验证均值聚合的有效性。

---

### 模块 2: 带Register Token的自蒸馏损失

**直觉**：学生需同时学习保持原始语义能力（通过register token吸收artifact）和匹配去噪后的密集结构。

**Baseline（DVT）**：两阶段损失——先优化neural field重建损失$\mathcal{L}_{\text{NGP}}$，再蒸馏$\mathcal{L}_{\text{distill}}$，需存储中间表示。

**变化点**：单阶段端到端，用MSE直接对齐学生register-augmented特征与教师去噪目标；register token作为可学习参数插入序列前端。

**本文公式（推导）**：
$$\text{Step 1:} \quad z_{\text{student}} = [r_1, r_2, ..., r_k; z_{\text{patch}}] \quad \text{（插入}k\text{个register token到patch token前）}$$
$$\text{Step 2:} \quad f_{\text{student}} = \text{Student}(z_{\text{student}}; \theta_{\text{reg}}, \theta_{\text{adapter}}) \quad \text{（仅register和adapter可梯度更新）}$$
$$\text{Step 3:} \quad \mathcal{L}_{\text{distill}} = \|f_{\text{student}} - f^*\|_2^2 \quad \text{（MSE蒸馏损失）}$$
$$\text{最终:} \quad \min_{\theta_{\text{reg}}, \theta_{\text{adapter}}} \mathbb{E}_{x \sim \mathcal{D}} \|f_{\text{student}}(x) - f^*(x)\|_2^2$$

符号：$r_j \in \mathbb{R}^d$ = 第$j$个可学习register token，$k$ = register数量，$\theta_{\text{reg}}$ = register嵌入参数，$\theta_{\text{adapter}}$ = 选定adapter权重，主干$\theta_{\text{backbone}}$冻结。

**对应消融**：Figure 5显示register数量$k$的敏感性分析，过多或过少均影响性能。

---

### 模块 3: 部分权重解锁策略

**直觉**：预训练ViT的通用表示已足够好，仅需微调少量参数来适应register机制，避免灾难性遗忘。

**Baseline（全量微调）**：更新所有参数$\theta$，导致过拟合和原始能力退化。

**变化点**：仅解锁register token和轻量adapter模块，backbone冻结，参数量降至~1%。

**本文公式**：
$$\theta_{\text{trainable}} = \{\theta_{\text{reg}}\} \cup \{\theta_{\text{adapter}}^{(l)}\}_{l \in \mathcal{S}}$$
$$\theta_{\text{frozen}} = \theta_{\text{backbone}} \text{setminus} \theta_{\text{trainable}}$$

其中$\mathcal{S}$为选定的adapter层索引集合。

**对应消融**：Table S.6（或相关表格）显示部分解锁相比全量微调在保持性能的同时大幅降低计算开销。

## 实验与分析



本文在**8个开放词汇语义分割benchmark**（含ADE20K, Pascal VOC 2012, Pascal Context, COCO Object等）以及深度估计任务上评估PH-Reg。核心结果：以DINOv2 ViT-B为统一backbone，PH-Reg在zero-shot开放词汇分割上相比DVT取得可比或更优的mIoU，同时**训练时间减少58.9%**；相比NACLIP、SCLIP、MaskCLIP等CLIP-based方法，PH-Reg通过根本消除artifact token获得更清晰的密集特征。具体而言，Table 1显示PH-Reg在多个benchmark上的mIoU提升，且无需任何标注数据。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8ac0d7e2-6d7f-47f7-9005-dff0f4a69de3/figures/fig_002.png)
*Figure: Learning Framework of PH-Reg. (a) Our framework begins by creating two networks*



定性结果（Figure 4及附录Figure S.1-S.3）直观展示了PH-Reg的分割边界清晰度：相比原始ViT特征和ClearCLIP，PH-Reg的激活图更紧密贴合真实物体轮廓，背景误激活显著减少。Figure 6进一步对比了原始特征与PH-Reg特征的范数分布，显示artifact token的异常高范数现象被有效抑制。



消融实验（Figure 5）聚焦两个关键超参：**register数量$k$**与**增强数量$n$**。实验表明，register数量存在最优区间（过少不足以吸收artifact，过多干扰正常patch表示）；增强数量$n$的增加单调提升教师目标质量，但边际收益递减。去掉TTA去噪（即$n=1$）导致性能显著下降，验证了均值聚合的必要性。

公平性核查：作者明确使用**相同DINOv2 ViT-B初始化**对比DVT，消除预训练差异；但基线选择集中于CLIP-based分割方法，未与SAM、SEEM等最新基础模型对比。此外，分类任务改进未被强调，PH-Reg主要面向密集预测。作者披露的限制包括：仅解锁少量参数可能限制极端域偏移场景的适配能力，且训练时需多次教师前向传播增加即时计算量。

## 方法谱系与知识库定位

**方法家族**：ViT Register Evolution（ViT寄存器演进）

**父方法**：DVT (Dense Vision Transformer) —— PH-Reg继承其"artifact-free密集特征"目标，但彻底重构实现路径：将两阶段neural field训练替换为单阶段自蒸馏，将1.4TB存储需求降为零。

**直接基线与差异**：
- **DVT**：需梯度优化Instant-NGP + 存储neural field；PH-Reg用TTA均值实时生成目标，无需存储
- **DeiT [13]**：提出ViT知识蒸馏范式；PH-Reg将其扩展为"自蒸馏"（师生同初始化）并引入register机制
- **标准ViT with registers [6]**：需从头训练；PH-Reg首次实现**后验添加**到预训练模型

**改动槽位**：architecture（追加register token）/ training_recipe（单阶段自蒸馏+部分解锁）/ data_pipeline（TTA实时生成目标，零存储）/ inference（保持标准ViT前向，无额外开销）

**后续方向**：
1. 将PH-Reg扩展到hierarchical ViT（如Swin）或CNN-Transformer混合架构
2. 探索register token的可解释性——是否学到语义明确的"垃圾回收"模式
3. 结合SAM等segmentation foundation model，验证artifact消除对交互式分割的增益

**标签**：modality=image | paradigm=self-supervised distillation | scenario=dense prediction, zero-shot transfer | mechanism=register token, test-time augmentation | constraint=no labeled data, limited trainable parameters

