---
title: 'R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- R-TPT：测试时提示调优增强VLM对抗鲁棒性
- R-TPT
acceptance: poster
cited_by: 23
method: R-TPT
---

# R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning

**Topics**: [[T__Adversarial_Robustness]], [[T__Cross-Modal_Matching]], [[T__Few-Shot_Learning]] | **Method**: [[M__R-TPT]] | **Datasets**: [[D__DTD]] (其他: UCF101)

| 中文题名 | R-TPT：测试时提示调优增强VLM对抗鲁棒性 |
| 英文题名 | R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2504.11195) · [Code] · [Project] |
| 主要任务 | 视觉语言模型(VLM)的对抗鲁棒性、测试时自适应(test-time adaptation)、零样本分类 |
| 主要 baseline | TPT, C-TPT, MEMO, MTA, CLIP, Ensemble |

> [!abstract] 因为「Vision-Language Models (VLMs) 在对抗攻击下极其脆弱，而现有测试时自适应方法未针对对抗鲁棒性设计」，作者在「TPT (Test-Time Prompt Tuning)」基础上改了「将边际熵替换为逐点熵最小化，并引入基于K近邻余弦相似度的可靠性加权集成」，在「DTD fine-grained」上取得「对抗准确率 30.3% vs TPT 的 2.3%，清洁准确率 52.0% vs TPT 的 30.5%」

- **DTD fine-grained 对抗准确率**: R-TPT 30.3% vs CLIP 0.8% / TPT 2.3% / C-TPT 1.8% / MTA 2.4%
- **DTD fine-grained 清洁准确率**: R-TPT 52.0% vs CLIP 29.5% / TPT 30.5% / C-TPT 30.0%
- **UCF101 对抗准确率**: R-TPT 41.0% vs APT+TeCoA (4shots) 39.4%，+1.6%

## 背景与动机

Vision-Language Models (VLMs) 如 CLIP 通过大规模对比学习获得了强大的零样本泛化能力，但在面对对抗攻击时表现出严重脆弱性——微小的、人眼不可见的图像扰动即可导致模型完全错误分类。例如，一张被精心扰动的"斑马"图像可能让 CLIP 以高置信度预测为"洗衣机"。这一问题在开放世界的零样本场景中尤为危险，因为攻击者无需知道下游任务的具体类别集合即可构造有效攻击。

现有应对思路主要分为三类：**(1) 对抗训练**（如 Robust CLIP）需要在训练阶段引入对抗样本，计算成本极高且难以扩展到 VLMs 的大规模预训练；**(2) 测试时增强防御**（如 MEMO）在推理时对单张图像做多视图增强并平均预测，但未利用文本提示的可调性；**(3) 测试时提示调优**（TPT）针对零样本泛化优化文本提示，通过边际熵最小化（marginal entropy minimization）使模型对增强视图更自信，但其设计目标并非对抗鲁棒性。

TPT 的核心局限在于两点：其一，**边际熵目标函数隐式包含 KL 散度约束**，强制每个样本的预测分布向批次平均分布靠拢，这在对抗场景下会"拖平"干净样本与对抗样本之间的信号差异；其二，**均匀集成（uniform ensemble）** 假设所有增强视图同等可靠，但对抗攻击下不同视图的受损程度差异巨大，盲目平均会引入噪声。因此，亟需一种能够区分视图可靠性、并直接优化单个样本置信度的测试时自适应机制。

本文提出 R-TPT，通过逐点熵最小化替代边际熵，并基于特征空间邻域相似度动态加权集成，在无需任何训练的前提下显著提升 VLMs 的对抗鲁棒性。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/398badcf-f035-4cc8-ab35-9ad83dbd2718/figures/fig_001.jpeg)
*Figure: Comparison between training-time and test-time defense*



## 核心创新

核心洞察：**逐点熵最小化配合可靠性加权集成**，因为边际熵的数学分解揭示了其隐含的 KL 散度对齐约束会压制个体样本的信号，而对抗攻击下不同增强视图的可靠性差异可通过特征空间局部结构有效度量，从而使测试时提示调优能够同时提升清洁精度与对抗鲁棒性。

| 维度 | Baseline (TPT) | 本文 (R-TPT) |
|:---|:---|:---|
| **优化目标** | 边际熵最小化 $\mathcal{L}_{marginal} = \mathcal{H}(\bar{p})$，隐含 KL 散度约束 | 逐点熵最小化 $\mathcal{L}_{point} = \frac{1}{|\mathcal{B}|}\sum_b \mathcal{H}(p^b)$，直接优化个体置信度 |
| **集成策略** | 均匀平均：所有增强视图权重相等 $w_i = \frac{1}{N+1}$ | 可靠性加权：权重基于 K 近邻余弦相似度 $r_i = \frac{1}{K}\sum_{k \in \mathcal{N}_i} \cos(f_i, f_k)$ |
| **设计目标** | 零样本泛化（domain generalization） | 对抗鲁棒性（adversarial robustness） |
| **对抗场景表现** | 对抗攻击下熵最小化失效，均匀集成放大噪声 | 可靠视图获更高权重，逐点优化保留个体判别信号 |

## 整体框架



R-TPT 的完整数据流如下：

1. **多视图增强（Multi-view Augmentation）**：对输入测试图像 $x$ 应用随机数据增强（如随机裁剪、颜色抖动），生成 $N$ 个增强视图 $\{x_0, x_1, ..., x_N\}$，其中 $x_0$ 为原始图像。

2. **特征提取（Feature Extraction）**：使用冻结的 CLIP 视觉编码器提取各视图特征 $\{f_0, f_1, ..., f_N\}$，$f_i \in \mathbb{R}^d$。

3. **可靠性计算（Reliability Computation）**【新增模块】：在特征空间中为每个样本 $i$ 寻找 K 近邻 $\mathcal{N}_i$，计算平均余弦相似度 $r_i = \frac{1}{K}\sum_{k \in \mathcal{N}_i} \cos(f_i, f_k)$ 作为可靠性分数。对抗受损严重的视图与其邻居相似度低，获得低权重。

4. **提示优化（Prompt Optimization）**：初始化可学习的文本提示向量，通过最小化逐点熵损失 $\mathcal{L}_{point}$ 更新提示参数。该过程仅优化提示嵌入，视觉编码器与文本编码器均冻结。

5. **加权集成预测（Weighted Prediction Aggregation）**【新增模块】：用优化后的提示计算各类别文本特征 $\{g_c\}$，对每个视图计算预测概率 $p_i$，最终以可靠性权重聚合：$\hat{p} = \frac{\sum_i r_i \cdot p_i}{\sum_i r_i}$。

```
测试图像 x
    ↓
[多视图增强] → {x_0, x_1, ..., x_N}
    ↓
[CLIP 视觉编码器] → {f_0, f_1, ..., f_N}
    ↓
    ├─→ [可靠性计算] ──→ {r_0, r_1, ..., r_N} ──┐
    │                                              ↓
    └─→ [文本编码器 + 可学习提示] ←──[逐点熵优化]──┘
                           ↓
                    {p_0, p_1, ..., p_N}
                           ↓
                    [加权集成] → 最终预测 ŷ
```

## 核心模块与公式推导

### 模块 1: 逐点熵最小化（对应框架图"提示优化"模块）

**直觉**：TPT 的边际熵在批次平均后再算熵，会强制个体预测向"共识"靠拢；对抗场景下这种"共识"可能被污染，应直接让每个样本各自最大化置信度。

**Baseline 公式** (TPT):
$$\mathcal{L}_{marginal} = \mathcal{H}(\bar{p}) = \mathcal{H}\left(\frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} p(x)\right)$$
符号: $\bar{p}$ = 批次平均预测分布, $\mathcal{B}$ = 增强视图批次, $\mathcal{H}(\cdot)$ = 香农熵, $p(x) \in \mathbb{R}^C$ = 样本 $x$ 的类别概率向量。

**变化点**：边际熵通过 Jensen 不等式可分解，发现其隐式包含额外的 KL 散度惩罚项，这在对抗样本存在时会抑制干净视图的强信号。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{H}(\bar{p}) = -\sum_{c=1}^{C} \bar{p}_c \log \bar{p}_c \quad \text{[熵的定义展开]}$$
$$\text{Step 2}: \mathcal{H}(\bar{p}) = \frac{1}{|\mathcal{B}|} \sum_{b=1}^{|\mathcal{B}|} \left( \mathcal{H}(p^b) + \mathcal{KL}(p^b \| \bar{p}) \right) \quad \text{[Jensen不等式与KL散度分解，揭示隐式约束]}$$
$$\text{最终}: \min \mathcal{L}_{point} = \frac{1}{|\mathcal{B}|} \sum_{b=1}^{|\mathcal{B}|} \mathcal{H}(p^b) \quad \text{[去掉KL散度项，仅保留逐点熵]}$$

**对应消融**：Table 7 显示使用边际熵替代逐点熵时，清洁准确率略升但对抗鲁棒性大幅下降，证明 KL 散度项在对抗场景下有害。

---

### 模块 2: 可靠性加权集成（对应框架图"可靠性计算"与"加权集成"模块）

**直觉**：对抗攻击对不同增强视图的破坏程度不同，均匀平均会让受损严重的视图污染最终预测；特征空间中相互靠近的视图更可能共享相同的"真实"信号。

**Baseline 公式** (TPT):
$$\hat{p}_{uniform} = \frac{1}{N+1} \sum_{i=0}^{N} p_i$$
符号: $p_i$ = 第 $i$ 个增强视图的预测概率, $N+1$ = 视图总数（含原始图像）。

**变化点**：均匀假设在对抗场景失效。本文引入基于局部特征结构的数据驱动可靠性度量：与邻居越相似，该视图越可靠。

**本文公式（推导）**:
$$\text{Step 1}: S_{i,j} = \cos(f_i, f_j), \quad i,j = 0,1,...,N \quad \text{[构建特征相似度矩阵]}$$
$$\text{Step 2}: r_{i} = \frac{1}{K} \sum_{k \in \mathcal{N}_i} S_{i,k} = \frac{1}{K} \sum_{k \in \mathcal{N}_i} \cos(f_i, f_k) \quad \text{[K近邻平均相似度作为可靠性分数]}$$
$$\text{Step 3}: w_i = \frac{r_i}{\sum_{j=0}^{N} r_j} \quad \text{[归一化权重]}$$
$$\text{最终}: \hat{p}_{weighted} = \sum_{i=0}^{N} w_i \cdot p_i = \frac{\sum_{i=0}^{N} r_i \cdot p_i}{\sum_{j=0}^{N} r_j}$$

符号: $f_i$ = 第 $i$ 个视图的特征向量, $\mathcal{N}_i$ = $f_i$ 的 K 近邻索引集（基于余弦相似度）, $r_i \in [0,1]$ = 可靠性分数, $K$ = 邻居数量超参数。

**对应消融**：Table 7 显示去掉加权机制（退化为均匀集成）时，防御能力降低且清洁性能反而提升，验证了可靠性加权对对抗鲁棒性的关键作用；完全去掉集成（单视图推理）则在 DTD fine-grained 上清洁准确率降至 53.2%→，对抗准确率仅 2.4%。

## 实验与分析



本文在多个基准上评估 R-TPT 的对抗鲁棒性与清洁精度。主要结果来自 DTD（Describable Textures Dataset，细粒度纹理分类）和 UCF101（动作识别）。在 DTD fine-grained 上，R-TPT 取得对抗准确率 30.3%、清洁准确率 52.0%，而基线 TPT 仅分别为 2.3% 和 30.5%——对抗鲁棒性提升超过 13 倍，清洁精度提升 70%。这一巨大差距表明 TPT 的边际熵与均匀集成在对抗场景下几乎完全失效，而 R-TPT 的两项改进协同恢复了有效信号。



消融实验（Table 7）量化了各组件贡献：去掉可靠性加权机制后，防御能力显著降低；将逐点熵替换回边际熵，清洁准确率略有上升但对抗鲁棒性大幅下降，印证了 KL 散度项的负面作用；最极端的是完全移除集成（单视图推理），在 DTD fine-grained 上对抗准确率仅 2.4%，在 ImageNet-X 上更降至 0.3%（清洁准确率 44.2%），证明多视图集成本身是防御的基础，而可靠性加权是集成有效性的关键。Figure 4(b) 进一步分析了邻居数量 K 的敏感性，显示 R-TPT 在不同 K 值下清洁与对抗准确率均保持稳定，对超参数选择不敏感。



效率方面，Table 6 显示 R-TPT 测试时间为 0.58s/图像（64 视图）、0.28s/图像（32 视图）、0.20s/图像（16 视图），对比需要训练的 APT+TeCoA（4shots）在 UCF101 上达到 39.4% 对抗准确率，R-TPT 以 41.0% 略超且无需任何训练。Figure 4(c) 分析了优化参数选择，证明优化文本提示在参数量与防御效果间取得最优平衡。

**公平性检查**：本文比较范围限于测试时自适应方法（TPT、C-TPT、MEMO、MTA），未与完整对抗训练方法（如 Robust CLIP、TeCoA 全量微调）直接对比；缺失基线包括 CoOp/Co-CoOp 等提示学习方法及更多对抗训练方案。此外，与 APT+TeCoA 的效率对比存在度量不一致（测试时间 vs 训练时间）。作者披露的局限包括：仅针对测试时自适应设定，不适用于可承担完整对抗训练的场景。

## 方法谱系与知识库定位

R-TPT 属于 **Test-Time Adaptation (TTA)** 方法家族，直接继承自 **TPT (Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models)**。谱系关系为：CLIP → TPT → R-TPT，其中 TPT 首次将提示调优引入测试时自适应，R-TPT 针对对抗鲁棒性做针对性改造。

**改变的插槽**（changed slots）：
- **objective（目标函数）**：TPT 的边际熵最小化 → R-TPT 的逐点熵最小化
- **inference_strategy（推理策略）**：TPT 的均匀平均集成 → R-TPT 的可靠性加权集成
- **training_recipe（训练/适配策略）**：TPT 面向零样本泛化 → R-TPT 明确针对对抗鲁棒性

**直接基线对比**：
- **TPT**：核心父方法，R-TPT 保留其测试时提示优化框架，替换目标与集成策略
- **C-TPT**：同期校准导向的 TPT 变体，R-TPT 不聚焦校准而聚焦对抗防御
- **MEMO**：测试时增强+自适应方法，R-TPT 额外利用文本提示可调性
- **MTA**：强训练-free CLIP 自适应基线，R-TPT 通过熵优化与加权集成超越

**后续方向**：(1) 将可靠性加权扩展至其他 TTA 目标函数（如 MEMO 的边际熵）；(2) 结合轻量对抗训练与测试时自适应的混合范式；(3) 探索 K 近邻可靠性在更复杂攻击（如 targeted attack、多模态攻击）中的有效性。

**标签**：modality=视觉-语言多模态 / paradigm=测试时自适应 / scenario=对抗防御 / mechanism=熵最小化+邻域相似度加权 / constraint=训练-free、零样本、仅调优文本提示

