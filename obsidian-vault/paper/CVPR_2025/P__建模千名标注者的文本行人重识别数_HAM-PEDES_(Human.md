---
title: Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 建模千名标注者的文本行人重识别数据合成
- HAM-PEDES (Human
- HAM-PEDES (Human Annotator Modeling for PEDestrian Description Synthesis)
acceptance: poster
cited_by: 13
method: HAM-PEDES (Human Annotator Modeling for PEDestrian Description Synthesis)
---

# Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification

**Topics**: [[T__Retrieval]], [[T__Image_Generation]], [[T__Few-Shot_Learning]] | **Method**: [[M__HAM-PEDES]] | **Datasets**: [[D__CUHK-PEDES]], [[D__ICFG-PEDES]], [[D__RSTPReid]] (其他: Direct Transfer CUHK-PEDES)

| 中文题名 | 建模千名标注者的文本行人重识别数据合成 |
| 英文题名 | Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.09962) · [Code] · [Project] |
| 主要任务 | Text-to-Image Person Re-identification (基于文本的行人重识别) |
| 主要 baseline | IRRA [20], RDE [37], LuPerson-MLLM [45], SYNTH-PEDES [68], AUL [ALBEF] |

> [!abstract] 因为「MLLM生成的行人描述缺乏真实人类标注的多样风格，导致预训练数据多样性不足」，作者在「LuPerson-MLLM的模板驱动生成」基础上改了「引入Human Annotator Modeling (HAM)学习1000个风格提示，配合UPS均匀采样策略」，在「CUHK-PEDES/ICFG-PEDES/RSTPReid」上取得「RDE+HAMPEDES Rank-1 77.99%/69.95%/72.5%，超越所有现有方法」

- **直接迁移性能**：HAM-PEDES(1.0M) 在 CUHK-PEDES 直接迁移设置下 Rank-1 达到 70.15%，相比 LuPerson-MLLM(1.0M) 的 57.61% 提升 **+12.54%**
- **微调后SOTA**：HAM-PEDES(1.0M) + RDE 在 CUHK-PEDES 上 Rank-1/mAP 为 **77.99%/69.72%**，超越此前最佳 AUL+ALBEF 的 77.23%/69.16%
- **推理效率**：生成每条文本描述仅需 **1.5ms**（3090 GPU），相比 AUL 的 123ms 提升约82倍

## 背景与动机

在基于文本的行人重识别（Text-based Person ReID）中，模型的泛化能力高度依赖于预训练阶段使用的图像-文本配对数据的质量与多样性。然而，大规模真实标注数据（如CUHK-PEDES）的获取成本极高，且不同人类标注者会自然产生风格迥异的描述——有人侧重颜色细节，有人偏好整体轮廓，有人使用复杂句式，有人简洁直接。这种"标注者风格多样性"恰恰是提升模型鲁棒性的关键。

现有方法如何应对这一挑战？**SYNTH-PEDES** [68] 采用规则模板生成合成描述，但模板数量有限导致风格单一；**LuPerson-MLLM** [45] 引入多模态大语言模型（MLLM），通过46个手工设计的提示模板增强输出多样性，然而这些模板仍是离散的、固定的，无法捕捉真实人类标注的连续风格分布；**IRRA** [20] 和 **RDE** [37] 等下游方法则专注于微调阶段的特征对齐，并未解决预训练数据本身的风格匮乏问题。

这些方法的共同瓶颈在于：**它们用"人工设计的模板多样性"替代了"真实人类标注的风格多样性"**。具体而言，LuPerson-MLLM的46个模板只能覆盖极小的风格子空间，而MLLM固有的输出模式坍塌（mode collapse）倾向使其倾向于生成安全、雷同的描述。更关键的是，真实数据中存在大量稀疏的、小众的标注风格（如极具个人特色的比喻或非常规语序），这些风格在特征空间中处于低密度区域，传统的KMeans等聚类方法几乎无法覆盖。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/916f2747-edb3-49be-97ea-718a6276f885/figures/Figure_1.png)
*Figure 1: Figure 1. Illustration of different schemes for enhancing the stylediversity for MLLM-generated textual descriptions. (Top) Exist-ing works typically rely on manually designed description tem-plates t*



因此，本文提出一个核心问题：**能否让MLLM直接"学习"数千名真实人类标注者的风格，而非依赖人工模板？** 基于此，作者提出了Human Annotator Modeling (HAM)框架，通过可学习的风格提示和两阶段聚类策略，将真实标注者的风格分布编码进MLLM的生成过程。

## 核心创新

核心洞察：**真实人类标注者的风格可以被参数化为可学习的连续提示向量**，因为MLLM的注意力机制能够将前置提示嵌入转化为生成行为的系统性偏移，从而使从稀疏特征区域中均匀采样风格、突破KMeans的密度偏见成为可能。

| 维度 | Baseline (LuPerson-MLLM) | 本文 (HAM-PEDES) |
|:---|:---|:---|
| 风格来源 | 46个手工设计模板 | 从真实标注提取的1000个可学习风格提示 |
| 风格空间覆盖 | 离散模板点，高密度区重复采样 | KMeans+UPS两阶段，显式覆盖稀疏区域 |
| 生成条件 | 随机选模板拼接指令 | 风格提示作为MLLM输入的独立模态嵌入 |
| 训练目标 | 标准图像描述损失 | 风格条件化的最大似然估计 |
| 聚类策略 | 无（模板无聚类） | KMeans捕获密集区 + UPS均匀探索稀疏区 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/916f2747-edb3-49be-97ea-718a6276f885/figures/Figure_2.png)
*Figure 2: Figure 2. Overview of our framework. In HAM, we first extract style features from human annotations and then perform clustering onthese features, as illustrated in (a). This allows us to group human a*



HAM-PEDES的整体数据生成流程包含五个核心阶段，形成从"真实标注分析"到"合成数据生成"再到"ReID预训练"的完整闭环：

**阶段一：风格特征提取（Style Feature Extraction）**
输入CUHK-PEDES训练集的真实人类标注，通过预训练语言模型编码器提取每条标注的风格特征表示 $\{F_s^n\}$。这些特征捕获了标注者在词汇选择、句式结构、描述粒度等方面的风格指纹。

**阶段二：两阶段聚类（KMeans + UPS）**
对风格特征空间执行KMeans聚类以识别高密度区域，随后对每个聚类应用UPS（Uniform Prompt Sampling）：在聚类均值附近按标准差范围均匀采样，主动探索KMeans忽略的稀疏边界区域。输出为1000个聚类/采样中心及其对应训练数据分配。

**阶段三：风格提示学习（Style Prompt Learning）**
将1000个聚类结果映射为1000个可学习的风格提示嵌入 $\{P_i\}_{i=1}^{1000}$，每个 $P_i \in \mathbb{R}^{M \times D}$ 包含M个token、维度D。这些提示与MLLM的视觉-语言适配器联合微调。

**阶段四：风格条件化生成（MLLM Caption Generation）**
对于每张行人图像，随机采样一个风格提示 $P_i$，构造MLLM输入 $[V, P_i, T]$（视觉特征V、风格提示$P_i$、文本指令T），生成具有特定人类标注者风格的描述。

**阶段五：ReID预训练（ReID Model Pre-training）**
将生成的图像-文本对组成HAM-PEDES数据集，用于预训练CLIP或专用ReID模型（如IRRA、RDE），随后在传统设置下微调。

```
真实人类标注 → 风格特征提取 → [KMeans聚类 + UPS均匀采样] → 风格提示学习
                                                    ↓
行人图像 ←—— 风格条件化MLLM生成 ←—— 采样风格提示 $P_i$
                                                    ↓
                                        HAM-PEDES 数据集 → ReID 预训练 → 下游微调
```

## 核心模块与公式推导

### 模块 1: MLLM输入构造与风格条件化（对应框架图 阶段四）

**直觉**: 标准MLLM将图像和指令直接拼接，导致生成风格由模型固有偏见主导；插入可学习的风格提示作为独立模态，可将生成行为"路由"到特定标注者风格。

**Baseline 公式** (标准MLLM输入):
$$Ins_{base} = [\mathbf{V}, \mathbf{T}]$$
符号: $\mathbf{V}$ = 视觉特征序列, $\mathbf{T}$ = 文本指令token嵌入

**变化点**: Baseline缺少显式风格控制，输出趋于平均化；本文将风格提示作为与视觉、文本并列的第三模态，使注意力机制能够跨模态关联风格与内容。

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{P}_i \in \mathbb{R}^{M \times D} \text{ 为第}i\text{个可学习风格提示，}M\text{为提示长度，}D\text{为隐藏维度} \quad \text{将风格参数化为连续嵌入}$$
$$\text{Step 2}: Ins = \left[ \mathbf{V}, \mathbf{P}_i, \mathbf{T} \right] \quad \text{三模态拼接，风格提示位于视觉与指令之间以形成条件化路径}$$
$$\text{最终}: \text{MLLM}_{\theta}(Ins) \rightarrow \{y_1, y_2, ..., y_L\} \text{ 风格条件化的描述序列}$$

**对应消融**: Table 1显示，仅微调视觉-语言适配器而不引入风格提示（即去掉$\{P_i\}$），性能显著下降9.2% Rank-1，证明风格提示的必要性。

---

### 模块 2: HAM训练损失（对应框架图 阶段三）

**直觉**: 风格提示必须通过最大似然训练才能学会"如何影响生成"；标准描述损失不区分风格来源，而条件化损失强迫模型将风格提示与特定生成模式绑定。

**Baseline 公式** (标准图像描述损失):
$$\mathcal{L}_{base} = -\mathbb{E}_{(x_i, y_i) \in \mathcal{D}} \left[ \sum_{m} \log p(y_{i,m} | x_i, y_{i,<m}) \right]$$
符号: $x_i$ = 图像, $y_i$ = 目标描述, $y_{i,m}$ = 第m个token, $y_{i,<m}$ = 历史token

**变化点**: Baseline仅条件于图像和历史token，生成风格由模型预训练偏见隐式决定；本文显式引入风格提示$\mathbf{P}_{s_i}$作为条件变量，使同一图像可生成多种风格变体。

**本文公式（推导）**:
$$\text{Step 1}: s_i = \text{ClusterAssign}(F_s^{(i)}) \quad \text{将样本}i\text{的风格特征分配到聚类}s_i\text{，确定使用哪个风格提示}$$
$$\text{Step 2}: \mathcal{L}_{HAM} = - \mathbb{E}_{(x_i, y_i, s_i) \in \mathcal{D}} \left[ \sum_{m} \log p\left(y_{i,m} \big| x_i, \mathbf{P}_{s_i}, y_{i,<m}\right) \right] \quad \text{风格条件化的自回归损失}$$
$$\text{最终}: \theta^*, \{P_i^*\} = \text{arg}\min_{\theta, \{P_i\}} \mathcal{L}_{HAM} \text{ 联合优化MLLM参数和风格提示}$$

**对应消融**: Table 1中，静态caption（无HAM）相比完整方法性能大幅下降；动态caption用46模板仅边际提升，用68,126模板仍远逊于HAM，证明连续风格提示优于离散模板扩展。

---

### 模块 3: UPS均匀提示采样（对应框架图 阶段二）

**直觉**: KMeans聚类倾向于在样本密集区域分配过多聚类中心，而忽略稀疏但独特的风格区域；通过在聚类统计量确定的范围内均匀采样，可主动探索这些"风格边疆"。

**Baseline 公式** (标准KMeans聚类):
$$\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i \quad \text{仅使用质心代表聚类，采样局限于已有成员或质心点}$$

**变化点**: KMeans的质心表示无法覆盖聚类边界外的潜在风格；UPS利用聚类内方差信息，在均值±标准差范围内均匀采样，将探索范围扩展到稀疏区域。

**本文公式（推导）**:
$$\text{Step 1}: \bm{\mu}_s = \frac{1}{N}\sum_{n=1}^{N} \mathbf{F}_s^n \quad \text{计算第}s\text{个聚类的风格特征均值}$$
$$\text{Step 2}: \bm{\sigma}_s = \sqrt{\frac{1}{N}\sum_{n=1}^{N}\left(\mathbf{F}_s^n - \bm{\mu}_s\right)^2} \quad \text{计算聚类标准差，衡量风格分散度}$$
$$\text{Step 3}: \mathbf{c}_i \sim \mathcal{U}\left(\bm{\mu}_s - \beta \cdot \bm{\sigma}_s, \; \bm{\mu}_s + \beta \cdot \bm{\sigma}_s\right) \quad \text{在}[\mu-\beta\sigma, \mu+\beta\sigma]\text{区间内均匀采样新风格中心}$$
$$\text{最终}: \text{采样得到的}\mathbf{c}_i\text{用于训练对应风格提示}P_i\text{，}\beta\text{为控制探索范围的超参数}$$

**对应消融**: Table 1显示，仅用KMeans相比KMeans+UPS在CUHK-PEDES上Rank-1下降3.77%（LLaVA1.6）；用DBSCAN替代则暴跌11.03%，证明UPS对稀疏风格覆盖的关键作用。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/916f2747-edb3-49be-97ea-718a6276f885/figures/Table_2.png)
*Table 2: Table 2. Comparisons with existing pre-training datasets under thedirect transfer setting defined in [45]. There are 2.53, 4.0, and 2.0captions per image on average for [68], [45], and ours, respec-ti*



本文在三个主流benchmark上评估HAM-PEDES：CUHK-PEDES（主流基准）、ICFG-PEDES（大规模）、RSTPReid（真实监控场景），同时测试直接迁移（direct transfer，无微调）和传统预训练-微调两种设置。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/916f2747-edb3-49be-97ea-718a6276f885/figures/Table_3.png)
*Table 3: Table 4. Comparisons with state-of-the-art ReID methods under the traditional evaluation setting.*



在传统微调设置下，HAM-PEDES(1.0M)配合RDE在CUHK-PEDES达到Rank-1 **77.99%**、mAP **69.72%**，超越此前最佳方法AUL+ALBEF（77.23%/69.16%）**+0.76%/+0.56%**。值得注意的是，AUL、APTM、RaSa等方法基于ALBEF（Swin-B/BERT-base），而HAM-PEDES+RDE基于CLIP-ViT，架构差异使得这一超越更具实质意义。在ICFG-PEDES和RSTPReid上，HAM-PEDES+RDE分别以**69.95%**和**72.5%**的Rank-1达到新SOTA，其中RSTPReid相比RDE基线提升**+7.20%**，相比LuPerson-MLLM+IRRA提升**+3.19%** Rank-1和**+1.99%** mAP。


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/916f2747-edb3-49be-97ea-718a6276f885/figures/Table_1.png)
*Table 1: Table 1. Ablation study on the key components of our methods, i.e., HAM and UPS, in the direct transfer setting defined in [45].*



直接迁移设置更能体现预训练数据本身的泛化能力。如Table 2所示，HAM-PEDES(1.0M)在CUHK-PEDES直接迁移达到**70.15%** Rank-1，相比同规模的LuPerson-MLLM(1.0M) **57.61%**提升**+12.54%**，相比SYNTH-PEDES(1.0M) **57.58%**提升**+21.85%**。即便仅用0.1M数据，HAM-PEDES已达60.74%，接近其他方法1.0M规模的表现，证明风格建模的数据效率优势。Figure 3进一步显示，随着预训练数据量增加，HAM-PEDES的性能曲线持续优于所有对比方法。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/916f2747-edb3-49be-97ea-718a6276f885/figures/Figure_3.png)
*Figure 3: Figure 3. Pre-training data size’s impact on direct transfer ReIDperformance. Best viewed with zoom-in.*



消融实验（Table 1）量化了各组件贡献：去掉风格提示（仅微调适配器）导致Rank-1下降**9.2%**；将UPS替换为纯KMeans下降**3.77%**（LLaVA1.6）；替换为DBSCAN则灾难性下降**11.03%**。模板方法的对比同样关键：46模板仅边际优于静态caption，68,126模板（穷举扩展）仍远不及HAM，证明连续风格提示的本质优势非离散扩展所能弥补。

公平性检查：作者披露的潜在局限包括——（1）ALBEF-based方法与CLIP-based方法的backbone差异使直接对比略失公平；（2）1.5ms vs 123ms的延迟比较未完全控制模型复杂度差异；（3）HAM-PEDES每张图像仅2.0条caption，少于LuPerson-MLLM的4.0条，虽有利于效率声明，但数据多样性或受影响。此外，缺少与更多近期MLLM-based方法（除[45]外）的直接对比。

## 方法谱系与知识库定位

HAM-PEDES属于**合成数据生成 → 视觉-语言预训练**的方法家族，直接父方法为 **LuPerson-MLLM** [45]（模板驱动的MLLM数据合成）和 **SYNTH-PEDES** [68]（规则模板合成）。

**改变的slots**（相比LuPerson-MLLM）：
- **data_pipeline**: 46个固定模板 → HAM人类标注者建模 + 1000可学习风格提示
- **training_recipe**: 随机模板选择 → KMeans+UPS两阶段聚类与采样
- **architecture**: 标准MLLM输入 → 三模态输入 [V, P_i, T] 含风格提示嵌入
- **inference_strategy**: 单一样式生成 → 每图像随机采样风格提示的条件生成

**直接baseline差异**：
- **vs LuPerson-MLLM**: 用连续风格提示替代离散模板，UPS替代随机选择，直接迁移Rank-1 +12.54%
- **vs SYNTH-PEDES**: 从规则模板升级到学习真实人类风格，同规模下+21.85%
- **vs IRRA/RDE**: HAM-PEDES是数据生成方法而非微调方法，可与IRRA/RDE正交组合

**后续方向**：（1）将HAM扩展到视频-文本行人重识别，建模时序标注风格；（2）探索风格提示的插值与编辑，实现可控的风格混合生成；（3）结合更强大的MLLM（如GPT-4V级模型）进一步提升描述质量与多样性。

**知识库标签**: 模态=cross-modal(图像+文本) | 范式=生成式数据增强+对比学习预训练 | 场景=行人重识别/智能监控 | 机制=prompt learning + clustering-based sampling | 约束=低标注成本、高效率推理

