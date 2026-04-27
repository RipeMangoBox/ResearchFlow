---
title: Graph-Embedded Structure-Aware Perceptual Hashing for Neural Network Protection and Piracy Detection
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 图嵌入结构感知感知哈希保护神经网络
- Graph-Embedded S
- Graph-Embedded Structure-Aware Perceptual Hashing (GESAPH)
acceptance: poster
cited_by: 1
method: Graph-Embedded Structure-Aware Perceptual Hashing (GESAPH)
---

# Graph-Embedded Structure-Aware Perceptual Hashing for Neural Network Protection and Piracy Detection

**Topics**: [[T__Object_Detection]], [[T__Graph_Learning]] | **Method**: [[M__Graph-Embedded_Structure-Aware_Perceptual_Hashing]] | **Datasets**: Non-pirated model distinction, Pirated model

| 中文题名 | 图嵌入结构感知感知哈希保护神经网络 |
| 英文题名 | Graph-Embedded Structure-Aware Perceptual Hashing for Neural Network Protection and Piracy Detection |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [DOI](https://doi.org/10.1109/cvpr52734.2025.01878) · Code (⭐待补充) · Project (待补充) |
| 主要任务 | 深度模型知识产权保护、盗版模型检测、非盗版模型区分 |
| 主要 baseline | Xiong [28] (DNN-based perceptual hashing) |

> [!abstract]
> 因为「现有DNN感知哈希方法仅依赖参数信息而忽略模型结构信息，导致对结构变化敏感且误报率高」，作者在「Xiong [28]」基础上改了「引入三层DAG结构表示与GIN图嵌入的结构哈希模块，并通过融合层联合参数哈希与结构哈希」，在「TIMM模型集合上的盗版检测基准」上取得「微调/L1剪枝/随机剪枝检测准确率100%（T=0.1），非盗版区分误报率从9.50%降至2.69%」

- **非盗版区分误报率**: T=0.1 时从 9.50% 降至 2.69%，降低 7.81 个百分点
- **盗版检测准确率**: 微调、L1剪枝、随机剪枝三种攻击下 T=0.1 均达 100.0%，相比 Xiong [28] 分别提升 3.12/3.65/4.5 个百分点
- **结构表示复杂度**: 从 O(mn) 或 O(mnp) 降至 O(p×(m+n))

## 背景与动机

随着深度学习模型成为重要的数字资产，如何保护其知识产权、检测未经授权的盗版模型成为紧迫问题。例如，攻击者可能通过微调（fine-tuning）、剪枝（pruning）等方式窃取并修改他人训练好的模型，而模型所有者需要一种高效手段来验证所有权。现有方案主要依赖感知哈希（perceptual hashing）——为模型生成紧凑的指纹，使得相似模型具有相似哈希，不同模型哈希差异显著。

Xiong [28] 提出的 DNN-based perceptual hashing 是这一方向的代表性工作，其核心思想是直接对模型参数进行哈希编码。该方法通过训练一个参数哈希生成器，将模型权重映射为固定长度的二进制哈希码，在参数层面捕捉模型相似性。然而，这种方法存在根本性局限：**完全忽略了模型的结构信息**。当攻击者仅修改模型架构（如增减层、调整通道数）而保持参数统计特性相似时，纯参数哈希可能失效；反之，当两个合法独立的模型恰好参数分布相似时，又会产生高误报。

具体而言，Xiong [28] 的短板体现在三方面：其一，**结构盲性**——无法区分"同一模型的结构变种"与"不同模型的参数巧合相似"；其二，**表示效率低**——对卷积核等结构元素采用简单展平，丢失拓扑关联；其三，**融合机制缺失**——参数哈希与结构信息之间缺乏有效交互，难以联合优化。这些局限导致其在非盗版模型区分任务上误报率高达 9.5%（T=0.1），且在强结构修改攻击下的检测鲁棒性不足。

本文提出 GESAPH，首次将图神经网络嵌入引入模型感知哈希，通过三层有向无环图（DAG）显式编码模型架构拓扑，并以可学习的融合层联合参数与结构信息，实现更精准的模型指纹识别。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0965a4df-400e-47f7-a0ba-860625e6bf5d/figures/fig_001.png)
*Figure: An illustration of model perceptual hash for copyright*



## 核心创新

核心洞察：**模型架构的拓扑结构本身就是一种强身份信号**，因为合法独立模型的结构差异（层连接方式、卷积核维度配置）远大于参数随机扰动，而盗版模型往往保留原始结构骨架；通过将结构信息显式编码为三层DAG并用GIN学习其嵌入表示，可使结构感知哈希与参数哈希形成互补，从而在保持对盗版敏感的同时大幅降低对合法不同模型的误报。

| 维度 | Baseline (Xiong [28]) | 本文 GESAPH |
|:---|:---|:---|
| 结构表示 | 无，仅参数哈希 | 三层DAG（输入/激活/输出节点组），显式编码拓扑 |
| 图复杂度 | 不适用（无图结构） | O(p×(m+n))，相比传统两层表示 O(mnp) 显著降低 |
| 结构编码器 | 无 | GIN [29] 预训练于 MUTAG，对比学习微调于模型DAG |
| 多源融合 | 无（单一参数哈希） | 全连接融合层联合参数哈希与结构哈希为100-bit输出 |
| 损失设计 | 单一参数哈希损失 | 三任务联合损失 L = Lw + Ls + Lm，分别优化参数/结构/融合 |
| 训练策略 | 直接端到端训练 | MUTAG预训练 + 噪声注入对比学习微调（5%节点阈值） |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0965a4df-400e-47f7-a0ba-860625e6bf5d/figures/fig_002.png)
*Figure: Network framework of our proposed method. In part (a), model structure and model parameter matrices are firstly extracted.*



GESAPH 采用双分支架构，整体数据流如下：输入神经网络模型 → **模型分离**拆分为参数与结构两条路径 → 参数经预处理后由**参数哈希生成器**输出参数哈希；结构经**DAG提取模块**转换为三层有向无环图，再由**结构哈希生成器**（GIN-based，含MUTAG预训练与对比学习微调）输出128-bit结构哈希；最终**融合层**将两类哈希合并为100-bit感知哈希。

各模块角色详解：
- **模型分离**（输入：完整模型；输出：参数张量集合 + 结构配置信息）：将模型的可学习参数与架构描述（层类型、连接关系、维度配置）解耦，为双分支处理做准备。
- **参数预处理与参数哈希生成器**（输入：参数张量；输出：参数哈希）：对参数进行标准化等预处理，沿用类似Xiong [28]的编码网络生成参数侧哈希表示。
- **DAG提取模块**（输入：模型结构配置；输出：三层DAG）：核心创新模块，将全连接层和卷积层分别转换为具有输入节点组、激活节点组（卷积核各维度作为独立节点）、输出节点组的三层DAG，消除多边并保证有向无环性。
- **结构哈希生成器**（输入：DAG；输出：128-bit结构哈希）：基于GIN [29]的图神经网络，先在MUTAG分子图数据集上预训练图分类能力，再在自己生成的模型DAG数据集上以对比学习微调（相似对：同层添加≤5%节点；不相似对：添加>5%节点或新增层）。
- **融合层**（输入：参数哈希 + 128-bit结构哈希；输出：100-bit最终哈希）：全连接网络，学习参数与结构哈希的非线性交互，替代简单的拼接或硬编码。

```
模型 M
├─→ 参数 ──→ 预处理 ──→ 参数哈希生成器 ──→ 参数哈希 ─┐
│                                                    ├──→ 融合层 ──→ 100-bit Hash
└─→ 结构 ──→ DAG提取 ──→ GIN(预训练+对比微调) ──→ 结构哈希(128-bit)─┘
              ↑ MUTAG预训练
              ↓ 噪声注入对比学习 (≤5%节点阈值)
```

## 核心模块与公式推导

### 模块 1: 三层DAG提取（对应框架图 Figure 2 左下结构哈希分支）

**直觉**: 传统两层图表示将卷积核展平为边，导致多重边和空间复杂度 O(mnp)；将卷积核各维度提升为独立节点并分离输入/激活/输出三组，可将图简化为DAG并降低复杂度。

**Baseline 公式** (Xiong [28] 及传统图表示): 传统方法对 m×n 全连接层生成两层二分图，空间 O(mn)；对 m×n×p 卷积核生成带多重边的两层图，空间 O(mnp)。

符号: m = 输入维度, n = 输出维度, p = 卷积核尺寸（如 3×3=9）

**变化点**: Baseline 的两层设计无法区分卷积核内部维度关联，且多重边破坏DAG性质；本文将每个卷积核维度作为独立节点，形成输入组→激活组（含p个节点）→输出组的三层拓扑。

**本文公式（推导）**:
$$\text{Step 1}: \text{FC层: } G_{FC} = (V_{in} \cup V_{out}, E), \quad |V_{in}|=m, |V_{out}|=n$$
$$\text{但改为三层: } G_{FC}^{new} = (V_{in} \cup V_{act} \cup V_{out}, E'), \quad V_{act}=\emptyset \text{ (FC直接连接)}$$
$$\text{Step 2}: \text{Conv层: } G_{conv}^{new} = (V_{in} \cup V_{act} \cup V_{out}, E''), \quad |V_{act}|=p$$
$$\text{最终空间复杂度}: O(p \times (m + n)) \quad \text{（激活节点数p，每节点连接m输入+n输出）}$$

**对应消融**: Table 5 显示替换融合机制有影响，DAG设计本身为架构基础创新。

---

### 模块 2: 联合损失函数（对应框架图整体优化目标）

**直觉**: 单一损失无法同时约束参数相似性、结构相似性和融合一致性；分解为三个子任务可分别优化，再通过融合层统一。

**Baseline 公式** (Xiong [28]): 仅参数哈希损失，形式未明确给出，记为：
$$L_{base} = L_w$$
符号: $L_w$ = 参数哈希损失（如基于相似度矩阵的对比损失或二进制交叉熵）

**变化点**: Baseline 缺乏结构感知，无法区分"参数相似但结构不同"的合法独立模型；本文增加结构哈希损失 Ls 强制结构相似性，并增加融合损失 Lm 保证联合表示的判别性。

**本文公式（推导）**:
$$\text{Step 1}: L = L_w + L_s \quad \text{（加入结构哈希损失以编码拓扑信息）}$$
$$\text{Step 2}: L = L_w + L_s + L_m \quad \text{（再加入融合损失以保证100-bit输出的判别性）}$$
$$\text{最终}: L = L_w + L_s + L_m$$

符号详解: $L_w$ = 参数哈希损失（约束参数空间相似性）, $L_s$ = 结构哈希损失（GIN输出128-bit的图嵌入判别损失）, $L_m$ = 融合哈希损失（100-bit最终输出的相似度约束）

**对应消融**: Table 5 显示将融合层替换为拼接（concatenation）或硬编码（hard coding）均对性能产生负面影响，验证了 Lm 及可学习融合的必要性。

---

### 模块 3: 对比学习微调策略（对应结构哈希生成器训练）

**直觉**: 模型DAG数据集规模有限（2,569模型），直接训练易过拟合；利用MUTAG的通用图分类先验，再通过可控噪声生成相似/不相似对进行对比微调，可增强泛化性。

**Baseline 公式**: 无预训练，直接在目标数据上端到端训练：$\theta_{GIN} = \text{arg}\min_{\theta} L_s(G_{train})$

**变化点**: 缺乏预训练导致小样本下结构表示不稳定；本文引入两阶段训练，并设计基于节点增删的图扰动规则。

**本文公式（推导）**:
$$\text{Step 1 (预训练)}: \theta^* = \text{arg}\min_{\theta} L_{MUTAG}(G_{molecular}) \quad \text{（获得通用图嵌入能力）}$$
$$\text{Step 2 (相似对生成)}: G_s = \text{Noise}(G_o), \quad \text{s.t. } \frac{|\Delta V_{same\_layer}|}{|V_{same\_layer}|} \leq 5\%$$
$$\text{Step 3 (不相似对生成)}: G_d = \text{Noise}(G_o), \quad \text{s.t. } \frac{|\Delta V_{same\_layer}|}{|V_{same\_layer}|} > 5\% \text{ 或新增层}$$
$$\text{最终微调}: \theta^{**} = \text{arg}\min_{\theta} L_{contrastive}(G_s, G_d; \theta^*)$$

**对应消融**: 

## 实验与分析



本文在自建的 TIMM 模型集合（2,569个模型及其动态生成的盗版变体）上评估 GESAPH，涵盖两大核心任务：非盗版模型区分（降低误报）与盗版模型检测（提升检出）。关键结果来自 Table 1 和 Table 2。

**非盗版模型区分**（Table 1）是检验方法是否会将合法独立模型误判为盗版的关键指标。在阈值 T=0.1 下，GESAPH 的误报率为 2.69%，相比 Xiong [28] 的 9.50% 降低 7.81 个百分点；对应准确率从 91.51% 提升至 97.31%。在更宽松的 T=0.5 下，误报率仍保持 10.45%，低于 Xiong [28] 的 17.46%。这一提升直接验证了**结构信息对区分"参数巧合相似但结构不同"的独立模型至关重要**——仅靠参数哈希容易因权重分布相似而产生误判，而三层DAG拓扑表示能有效过滤此类假阳性。

**盗版模型检测**（Table 2）覆盖三种常见攻击场景：微调（fine-tuning）、L1剪枝、随机剪枝。在 T=0.1 的严格阈值下，GESAPH 对三种攻击的检测准确率均达到 100.0%，而 Xiong [28] 分别为 96.88%、96.35%、95.5%。值得注意的是，即使 baseline 在 T=0.5 时也能接近或达到 100%，但 GESAPH 在低阈值下的优势意味着**可用更严格的相似度标准实现可靠检测**，这对实际部署中减少调查成本具有重要意义。



消融实验（Table 5）聚焦于融合层设计的必要性。将融合层替换为简单拼接（concatenation）或硬编码（hard coding）均导致性能下降，尽管具体数值未在分析片段中完整给出，但作者明确标注"Negative impact on performance"。这支持了核心主张：**参数哈希与结构哈希的非线性交互不可被浅层替代**，100-bit 最终哈希的判别性依赖于可学习的融合机制。

**公平性检验**: 实验设计存在若干局限。首先，**仅与单一 baseline Xiong [28] 对比**，未纳入其他近期DNN感知哈希方法、密码学哈希基线或模型水印方法，结论的全面性受限。其次，结构哈希的测试集仅 227 个模型（TIMM的30%），规模偏小。第三，阈值 T 的选取（0.1, 0.5, 1.0, 2.0）缺乏理论论证或自适应机制说明。第四，数据集为自建生成，2,569模型的构建细节与动态盗版变体的生成协议未完全公开，可复现性存疑。训练成本方面，参数哈希生成器使用 4× RTX 4090 GPU 训练 5,000 迭代，属于中等计算开销。

## 方法谱系与知识库定位

GESAPH 属于**深度模型知识产权保护**方法族，直接继承自 **Xiong [28] 的 DNN-based perceptual hashing**，在五个关键 slot 上进行了扩展：

- **架构 (architecture)**: 新增三层DAG提取模块替换传统两层图表示，空间复杂度从 O(mnp) 降至 O(p×(m+n))；新增全连接融合层替代拼接/硬编码
- **数据流程 (data_pipeline)**: 引入 MUTAG 预训练 + 噪声注入对比学习微调（≤5%节点阈值），替代直接端到端训练
- **目标函数 (objective)**: 从单一参数损失扩展为 L = Lw + Ls + Lm 三任务联合损失

**直接基线对比**:
- **Xiong [28]**: 纯参数哈希，无结构感知，无预训练策略 → GESAPH 增加结构分支与融合机制，误报率降低 7.81%
- **GIN [29]**: 通用图同构网络，用于分子图分类 → GESAPH 将其适配为结构哈希生成器，增加 MUTAG 预训练与模型DAG对比微调

**后续方向**: （1）将DAG表示扩展至 Transformer 架构（注意力图、位置编码的图嵌入）；（2）引入自适应阈值 T 的学习机制，替代人工预设；（3）探索与其他模型保护技术（如动态水印、联邦学习指纹）的联合部署。

**标签**: 模态=model/weight | 范式=perceptual_hashing + graph_neural_network | 场景=intellectual_property_protection | 机制=contrastive_learning + pre-training_finetuning | 约束=binary_hash_output + black-box_detection

