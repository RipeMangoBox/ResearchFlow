---
title: 'FedLPA: Local Prior Alignment for Heterogeneous Federated Generalized Category Discovery'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 联邦GCD的局部先验对齐方法
- FedLPA
- FedLPA eliminates unrealistic assum
acceptance: Poster
cited_by: 28
method: FedLPA
modalities:
- Image
paradigm: federated self-supervised
---

# FedLPA: Local Prior Alignment for Heterogeneous Federated Generalized Category Discovery

**Topics**: [[T__Federated_Learning]], [[T__Few-Shot_Learning]] | **Method**: [[M__FedLPA]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]], [[D__ImageNet-1K]]

> [!tip] 核心洞察
> FedLPA eliminates unrealistic assumptions in federated generalized category discovery by grounding learning in client-specific structures and aligning predictions with locally derived priors through iterative local structure discovery and adaptive prior refinement.

| 中文题名 | 联邦GCD的局部先验对齐方法 |
| 英文题名 | FedLPA: Local Prior Alignment for Heterogeneous Federated Generalized Category Discovery |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [DOI](https://doi.org/10.1109/cvpr52733.2024.02715) |
| 主要任务 | Federated Generalized Category Discovery (Fed-GCD) |
| 主要 baseline | FedAvg + GCD, FedAvg + GCL, FedAvg + AGCL |

> [!abstract] 因为「联邦场景下现有方法假设已知新类别数量且使用全局固定先验，导致在非独立同分布数据上性能崩溃」，作者在「FedAvg + GCL」基础上改了「引入 Client-Level Concept Discovery (CLCD) 进行本地概念发现，并以 Local Prior Alignment (LPA) 正则化替代固定损失」，在「CIFAR-10 non-i.i.d. (α=0.2)」上取得「ACC All 92.3%，相比最佳 baseline FedAvg + GCL 的 68.2% 提升 +24.1%」

- **CIFAR-10 (α=0.2)**: ACC All 92.3% vs. FedAvg+GCL 68.2%, ACC New 91.2% vs. 70.1%, ACC Old 94.5% vs. 64.2%
- **ImageNet-100 (α=0.2)**: ACC All 71.7% vs. FedAvg+AGCL 67.5%, 提升 +4.2%
- **CIFAR-100 (α=0.2)**: ACC All 53.6% vs. FedAvg+GCL 52.5%, 提升仅 +1.1%

## 背景与动机

在联邦学习场景中，多个客户端持有异构的本地数据，需要协作训练一个能同时识别已知类别和发现新类别的模型——这就是联邦广义类别发现（Federated Generalized Category Discovery, Fed-GCD）。例如，多个医院各自拥有不同病种的患者影像，部分病种是已标注的已知类别，但各医院还可能存在未标注的新病种，且不同医院的病种分布差异极大（非独立同分布）。现有方法如何处理这一问题？

**FedAvg + GCD** [22] 直接将中心化 GCD 方法套用到联邦框架，使用标准交叉熵和对比损失，但假设全局已知类别数量固定；**FedAvg + GCL** [19] 作为首个专门的联邦 GCD 工作，引入图对比学习，但仍需预先知道新类别总数，且使用全局固定的类别先验；**FedAvg + AGCL** [19] 在此基础上增加增强图对比学习，同样未能摆脱对全局类别信息的依赖。

这些方法的致命缺陷在于：**它们假设服务器或客户端预先知道新类别的总数，并使用全局统一或固定的类别先验分布**。在非独立同分布场景下，各客户端的本地数据分布差异巨大，全局先验与本地实际分布严重错配，导致新类别发现性能急剧下降——尤其在客户端数据极度偏斜时（如 Dirichlet α=0.2），模型几乎无法正确聚类新类别。

本文提出 FedLPA，首次完全消除对已知新类别数量的假设，通过客户端本地结构发现和自适应先验对齐，解决联邦 GCD 中的异构性难题。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af14621a-2368-484b-9404-588aeaa3235b/figures/Figure_1.png)
*Figure 1 (pipeline): The overall framework of the proposed FedLPA, including (a) Confidence-guided Local Category Discovery (CLCD) and (b) Local Prior Alignment (LPA) with self-distillation.*



## 核心创新

核心洞察：**客户端本地的数据分布结构本身就蕴含了正确的类别先验信息**，因为 Infomap 聚类能从高置信度锚点构建的相似图中发现自然的社区结构，从而使无需全局类别数量的动态先验自适应成为可能。

| 维度 | Baseline (FedAvg + GCL) | 本文 (FedLPA) |
|:---|:---|:---|
| 新类别数量假设 | 必须预先已知全局新类别数 | **完全无需假设**，由 CLCD 本地自动发现 |
| 类别先验来源 | 全局固定或均匀分布 | **本地经验先验**，从 Infomap 聚类结果动态计算 |
| 训练流程 | 标准联邦轮询聚合 | **两阶段迭代**：warmup → 周期性 CLCD + LPA 本地训练 |
| 损失设计 | 交叉熵 + 对比损失 | 监督损失 + 自蒸馏 + **KL 散度先验对齐 (LPA)** |

## 整体框架



FedLPA 的整体流程采用**两阶段迭代架构**，核心数据流如下：

**输入**：各客户端持有的本地图像数据（包含标注的已知类别样本和未标注的新类别样本）

**阶段一：联邦 Warmup（20 轮）**
- 模块：标准 FedAvg 训练
- 输入/输出：全局模型 → 经多轮本地训练与聚合后的初始化全局模型
- 作用：建立基础表征能力，为后续结构发现提供稳定的特征空间

**阶段二：迭代训练循环（每 R 轮执行 CLCD）**
- **CLCD 模块（Client-Level Concept Discovery）**：输入本地特征和经置信度过滤的高置信度已知类别锚点，构建相似图并运行 Infomap 聚类，输出本地概念分配、经验类别先验分布 q 和原型向量
- **LPA 正则化本地训练**：输入本地数据、CLCD 生成的经验先验 q、全局模型，输出经先验对齐的本地模型；核心是在自蒸馏框架中加入 KL 散度项，使批次预测 p 匹配 q
- **全局聚合（FedAvg）**：输入各客户端的本地适配模型，输出更新后的全局模型

**输出**：能同时处理已知类别分类和新类别发现的自适应全局模型

```
全局模型 
  → [联邦 Warmup: 20轮标准训练] 
  → 循环 {
      [CLCD: 每R轮] → 相似图 → Infomap聚类 → 经验先验 q
      → [LPA本地训练] → 监督损失 + 自蒸馏 + ε·KL(p‖q)
      → [FedAvg聚合]
    } → 输出
```

## 核心模块与公式推导

### 模块 1: Client-Level Concept Discovery (CLCD)（对应框架图左半部分）

**直觉**：客户端本地的特征空间中，高置信度的已知类别样本可以作为可靠锚点，通过相似图传播发现自然的类别社区结构，从而无需知道全局有多少新类别。

**Baseline 公式** (FedAvg + GCL): 无本地图构建，直接使用全局固定的类别分配
$$\text{无显式结构发现，固定类别数 } C_{\text{global}} \text{ 预设}$$

**变化点**：Baseline 假设全局类别数已知且固定；本文改为从本地数据动态发现结构，用 Infomap 替代预设类别数。

**本文公式（推导）**:
$$\text{Step 1: } S_{ij} = \text{sim}(f(\mathbf{x}_i), f(\mathbf{x}_j)) \cdot \mathbb{1}[\text{confidence}(\mathbf{x}_i) > \tau_P] \cdot \mathbb{1}[\text{confidence}(\mathbf{x}_j) > \tau_P]$$
加入高置信度过滤，仅保留可靠的已知类别锚点构建相似图，抑制噪声边

$$\text{Step 2: } \{\hat{y}_i\}_{i=1}^{|B|} = \text{Infomap}(\{S_{ij}\})$$
运行 Infomap 随机游走社区发现，将样本划分为本地概念社区

$$\text{Step 3: } q_c = \frac{1}{|B|} \sum_{\mathbf{x} \in B} \mathbb{1}[\hat{y}(\mathbf{x}) = c]$$
从聚类结果计算经验类别先验分布 q，为 LPA 提供自适应目标

**对应消融**：Table 3 显示 CLCD 组件移除后性能显著下降；Figure 2 显示已知样本过滤百分位 P=0.5 时达到最优 57.7%，极端值导致退化。

---

### 模块 2: Local Prior Alignment (LPA) 正则化（对应框架图右半部分）

**直觉**：模型在本地训练时，其批次预测的类别分布应当与本地数据的真实结构（由 CLCD 发现）一致，而非盲目匹配全局假设。

**Baseline 公式** (FedAvg + GCL 的标准损失):
$$\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{contrastive}}$$
符号: $\mathcal{L}_{\text{ce}}$ = 已知类别交叉熵, $\mathcal{L}_{\text{contrastive}}$ = 全局对比损失（固定温度系数，无先验自适应）

**变化点**：Baseline 使用全局固定损失，无本地分布适应；本文引入自蒸馏框架，并以 KL 散度强制批次预测对齐本地经验先验。

**本文公式（推导）**:
$$\text{Step 1: } \mathcal{L}_{\text{self-distill}} = \|f_{\text{teacher}} - f_{\text{student}}\|^2 \quad \text{（继承 Flipped Classroom [14] 的教师-学生一致性）}$$
建立自蒸馏基础结构，提供稳定的表征学习目标

$$\text{Step 2: } \mathcal{L}_{\text{LPA}} = \varepsilon \cdot D_{\text{KL}}(p(\mathbf{x}) \| q(\mathbf{x})) = \varepsilon \sum_{c} p_c(\mathbf{x}) \log \frac{p_c(\mathbf{x})}{q_c} \quad \text{（加入 KL 散度对齐本地先验）}$$
其中 $p(\mathbf{x})$ 为当前批次模型预测的 softmax 分布，$q$ 为 CLCD 输出的经验先验，$\varepsilon$ 为权重超参数

$$\text{最终}: \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}} + \mathcal{L}_{\text{self-distill}} + \mathcal{L}_{\text{LPA}}$$
监督损失（仅已知类别）+ 自蒸馏一致性 + 先验对齐正则化，LPA 同时作用于已知和新类别发现

**对应消融**：Table 3 显示去掉 LPA（$\varepsilon=0$）后 ACC All 下降 -2.6%；Figure 2 显示 LPA 权重 $\varepsilon=0.5$ 时达到最优 57.7%。

---

### 模块 3: 两阶段训练调度（全局流程控制）

**直觉**：直接启动 CLCD 会因初始表征不稳定而产生错误聚类，需先 warmup 稳定特征空间。

**Baseline 公式**: 标准联邦训练从第 1 轮开始即执行完整流程

**变化点**：引入显式的 warmup 阶段，并控制 CLCD 执行频率 R 以平衡通信开销与结构更新时效性。

**本文公式**:
$$\text{Phase 1: } t \in [1, 20]: \quad \theta_{t+1} = \text{FedAvg}(\{\theta_t^k - \eta \nabla \mathcal{L}_{\text{base}}\}_k)$$
仅标准联邦训练，无 CLCD/LPA

$$\text{Phase 2: } t > 20: \quad \text{if } t \mod R = 0: \text{ 执行 CLCD 更新 } q^k; \quad \text{then 本地训练含 } \mathcal{L}_{\text{LPA}}$$
周期性概念发现与自适应训练交替进行

**对应消融**：Figure 2 显示 warmup 轮数 30 时达到 ACC All 60.5%（优于默认 20 轮配置），CLCD 频率 R=30 时最优；去除 warmup 导致 -3.3% 性能损失。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af14621a-2368-484b-9404-588aeaa3235b/figures/Table_2.png)
*Table 2 (result): Results on standard object recognition datasets with two different degrees of data heterogeneity.*



本文在标准图像识别数据集（CIFAR-10、CIFAR-100、ImageNet-100）和细粒度数据集（CUB-200、Stanford-Cars、Oxford-IIIT Pets）上评估 FedLPA，采用 Dirichlet 分布模拟非独立同分布场景（α=0.2 和 α=0.5）。
![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af14621a-2368-484b-9404-588aeaa3235b/figures/Table_1.png)
*Table 1 (result): Results on fine-grained datasets with different degrees of data heterogeneity.*



核心结果来自 Table 2：在 CIFAR-10 的强异构设置（α=0.2）下，FedLPA 达到 ACC All 92.3%，相比直接 baseline FedAvg + GCL 的 68.2% 提升 +24.1%，相比 FedAvg + GCD 的 63.4% 提升 +28.9%。这一巨大提升主要源于旧类别（ACC Old 94.5% vs. 64.2%）和新类别（ACC New 91.2% vs. 70.1%）的双重改善，说明 LPA 的先验对齐有效缓解了本地分布偏移导致的类别混淆。在 ImageNet-100 上，FedLPA 取得 71.7%，超越 FedAvg + AGCL 的 67.5%（+4.2%），验证了方法在更大规模数据上的有效性。然而，CIFAR-100 的结果显示 FedLPA 的 53.6% 仅比 FedAvg + GCL 的 52.5% 提升 +1.1%，表明当类别数量大幅增加（100 vs. 10）时，本地 Infomap 聚类的结构发现能力受限，细粒度区分难度上升。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af14621a-2368-484b-9404-588aeaa3235b/figures/Table_3.png)
*Table 3 (ablation): Component analysis of the proposed method on the non-i.i.d. settings on standard benchmarks.*



消融实验（Table 3 和 Figure 2）进一步验证各组件贡献：去掉 LPA 正则化（$\varepsilon=0$）导致 ACC All 下降 -2.6%；调整 CLCD 执行频率 R，发现 R=30 时性能最优，过于频繁（R=10）或稀疏（R=50）均导致退化，说明概念发现需要适度的新鲜度与稳定性平衡；已知样本过滤百分位 P=0.5 为最佳，过低引入噪声锚点，过高丢失有效结构信息。Figure 2 还显示联邦 warmup 轮数至关重要，30 轮 warmup 可达 60.5% ACC All，但默认 20 轮配置已能取得大部分增益。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af14621a-2368-484b-9404-588aeaa3235b/figures/Table_4.png)
*Table 4 (result): Results with increased number of clients (N = 10) on standard benchmarks under non-i.i.d. setting.*



公平性检查：本文对比的 baseline 均为 FedAvg 框架下的 GCD 变体（FedAvg+GCD/GCL/AGCL），但未包含其他联邦优化器如 Scaffold [9] 或 FedProx，也未与中心化 SOTA（如 SimGCD [23]、SPTNet [25]）对比以量化联邦化代价。CIFAR-100 的微小提升提示方法在复杂场景下的扩展性存在瓶颈。此外，CLCD 的 Infomap 聚类在极大规模本地数据上的计算开销未被充分讨论。作者披露的局限性包括：CLCD 周期性执行增加协调开销，以及 CIFAR-100 上性能差距相对有限。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af14621a-2368-484b-9404-588aeaa3235b/figures/Figure_2.png)
*Figure 2 (ablation): Ablation results of CLCD algorithm (left) and FedLPA under varying number of classes (right).*




![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af14621a-2368-484b-9404-588aeaa3235b/figures/Figure_3.png)
*Figure 3 (qualitative): Qualitative results of FedLPA improvement w.r.t. clients under non-i.i.d. setting.*

 定性结果（Figure 3）显示 FedLPA 在各客户端上的预测一致性显著改善，尤其在数据分布差异明显的客户端间。

## 方法谱系与知识库定位

**方法家族**：Federated Generalized Category Discovery（联邦广义类别发现）

**父方法**：FedAvg + GCL [19]（首个联邦 GCD 工作，建立联邦化 GCD 基础范式）

**改动槽位**：
- **data_pipeline**：新增 CLCD 模块，以相似图 + Infomap 聚类替代预设类别数
- **reward_design**：以 LPA（KL 散度正则化）替换标准对比损失，实现本地先验自适应
- **training_recipe**：改为两阶段调度（warmup + 周期性 CLCD/LPA 交替）
- **inference_strategy**：动态利用本地发现结构和自适应先验

**直接 Baseline 差异**：
- **FedAvg + GCD [22]**：直接联邦化中心化 GCD，无专门处理异构性的设计；本文通过 CLCD/LPA 根本消除其全局假设
- **FedAvg + GCL [19]**：需已知新类别数，用全局图对比学习；本文以本地结构发现完全替代该假设
- **FedAvg + AGCL [19]**：增强图对比但仍依赖全局信息；本文的自蒸馏+LPA 机制与之正交且更强
- **Flipped Classroom [14]**：中心化 GCD 的教师-学生对齐；本文将其扩展至联邦场景，并改为 KL 散度形式的本地先验对齐

**后续方向**：
1. 将 CLCD 的图聚类扩展至可微分形式（如 Gumbel-Softmax），实现端到端训练而非周期性离执行
2. 引入其他联邦优化器（Scaffold、FedProx）替代 FedAvg，验证 LPA 与更优聚合规则的兼容性
3. 探索跨模态联邦 GCD（如联邦医疗场景的多模态数据），扩展 CLCD 的相似图构建方式

**标签**：modality=图像 / paradigm=联邦自监督学习 / scenario=异构联邦学习 / mechanism=图聚类+KL散度正则+自蒸馏 / constraint=无需全局类别数假设

## 引用网络

### 直接 baseline（本文基于）

- Flipped Classroom: Aligning Teacher Attention with Student in Generalized Category Discovery _(NeurIPS 2024, 直接 baseline, 未深度分析)_: Recent NeurIPS 2024 GCD method with teacher-student alignment; cited across intr
- Federated Generalized Category Discovery _(CVPR 2024, 直接 baseline, 未深度分析)_: THE direct baseline - this is the first/only prior work on Federated GCD, the ex
- SPTNet: An Efficient Alternative Framework for Generalized Category Discovery with Spatial Prompt Tuning _(ICLR 2024, 实验对比, 未深度分析)_: Recent ICLR 2024 GCD method; cited across all sections. Likely appears in compar

