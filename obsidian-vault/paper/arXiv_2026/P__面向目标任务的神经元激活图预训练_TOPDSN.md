---
title: Target-Oriented Pretraining Data Selection via Neuron-Activated Graph
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.15706
aliases:
- 面向目标任务的神经元激活图预训练数据选择
- TOPDSN
code_url: https://github.com/asillycat/NAG
modalities:
- Text
---

# Target-Oriented Pretraining Data Selection via Neuron-Activated Graph

[Paper](https://arxiv.org/abs/2604.15706) | [Code](https://github.com/asillycat/NAG)

**Topics**: [[T__Classification]], [[T__Self-Supervised_Learning]], [[T__Interpretability]]

| 中文题名 | 面向目标任务的神经元激活图预训练数据选择 |
| 英文题名 | Target-Oriented Pretraining Data Selection via Neuron-Activated Graph |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15706) · [Code](https://github.com/asillycat/NAG) · [Project](https://arxiv.org/abs/2604.15706) |
| 主要任务 | 面向特定下游任务(target-oriented)的预训练数据选择，从大规模通用语料中筛选最能提升目标能力的子集 |
| 主要 baseline | DCLM、DSIR、TAPT、Task-Similarity、Quality-based filtering (Dolma、FineWeb) |

> [!abstract] 因为「通用质量过滤与特定下游能力不对齐」，作者在「DCLM/DSIR 等通用数据选择」基础上改了「引入神经元激活图(NAG)建模目标样本与预训练样本的细粒度功能关联」，在「单目标/多目标/跨模型迁移」设置上取得「平均 5.2% 的下游任务提升」。

- **关键性能 1**: 单目标设置下，NAG 在 Qwen3-1.7B 上比通用质量基线 Dolma 提升 6.8%，比任务相似度基线 Task-Similarity 提升 4.3%（Figure 5）
- **关键性能 2**: 多目标设置下，NAG 比 DCLM 提升 5.2%，且支持目标能力的组合与解耦
- **关键性能 3**: NAG 提取的神经元图支持跨模型规模迁移（1.7B → 4B → 8B），无需重新构建图（Figure 6）

## 背景与动机

当前大语言模型的预训练数据选择主要依赖「通用质量」指标——例如 perplexity、教育价值评分、毒性过滤等。然而，一个 concrete example 是：医疗问答模型需要生物医学推理能力，但通用高质量语料（如维基百科、新闻）可能包含大量与医疗无关的「高质量」文本，导致资源浪费且目标能力欠优化。

现有方法如何处理这一问题？

- **DCLM (DataComp-LM)**：基于 perplexity 等代理指标进行过滤，追求通用下游平均性能最优。但其质量评分与特定能力（如代码、数学）无显式关联，无法定向增强目标技能。
- **DSIR (Data Selection with Importance Resampling)**：通过 n-gram 特征空间中的重要性采样选择数据，优化与目标分布的匹配。但 n-gram 特征过于浅层，无法捕捉深层语义与功能对应关系。
- **TAPT (Task-Adaptive Pre-Training)**：直接在目标数据上继续预训练，但目标样本量通常极小（数百至数千条），易导致过拟合，且忽略预训练语料中大量潜在相关的未标注数据。

这些方法的根本局限在于：**缺乏从「目标样本的功能需求」到「预训练样本的功能供给」的细粒度映射机制**。通用质量过滤与目标能力之间存在结构性错配（Figure 1 左），而任务相似度方法仅停留在表层分布匹配。作者因此提出核心问题：能否利用神经网络内部的神经元激活模式，构建一种「功能级」的数据关联图，实现面向目标任务的精准数据选择？


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b7cc393a-3ba8-4d84-b654-f004f1517df4/figures/Figure_1.png)
*Figure 1: Figure 1. General quality-based data selection is often misalignedwith specific downstream capabilities (left), while prior target-oriented methods rely on shallow similarity to target examples(middle*



## 核心创新

核心洞察：目标样本与候选预训练样本的相关性应通过「共享的功能神经元」而非「表层特征重叠」来度量，因为神经元激活模式编码了模型的深层语义处理能力，从而使跨样本的能力迁移与组合成为可能。

与 baseline 的差异：

| 维度 | Baseline (DCLM/DSIR/TAPT) | 本文 (NAG) |
|------|---------------------------|-----------|
| 关联粒度 | 样本级或 n-gram 级统计匹配 | 神经元级功能激活关联 |
| 目标对齐 | 通用质量或浅层分布相似 | 显式建模目标能力需求 |
| 可解释性 | 黑箱评分函数 | 激活图显式展示「哪些神经元支持目标能力」 |
| 多目标组合 | 需重新训练或加权平均 | 图的并/交操作直接支持能力组合与解耦 |
| 跨模型迁移 | 不可行（依赖特定模型输出） | 神经元角色跨尺度稳定，支持 1.7B→4B→8B 迁移 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b7cc393a-3ba8-4d84-b654-f004f1517df4/figures/Figure_2.png)
*Figure 2: Figure 2. Overview of Neuron-Activated Graph (NAG) target-oriented data selection. Given a small set of target examples Dtarget, wefirst characterize each input by its neuron-level NAG features. For a*



NAG 框架的数据流如下：

1. **输入**: 小规模目标样本集 $D_{target}$（如 512 条医疗问答对）+ 大规模候选预训练语料池 $D_{pool}$（如 Common Crawl 子集）。

2. **目标神经元提取 (Target Neuron Extraction)**: 将 $D_{target}$ 输入 base model，记录各层神经元的激活频率，筛选高频激活的「目标相关神经元」集合 $N_{target}$。

3. **候选样本激活图构建 (Candidate NAG Construction)**: 对 $D_{pool}$ 中的每个样本，同样提取其激活的神经元子图，形成样本级 Neuron-Activated Graph $G(x)$。

4. **图匹配与排序 (Graph Matching & Ranking)**: 计算每个候选样本的 NAG 与目标 NAG 的匹配度得分 $s(x, D_{target})$，基于共享神经元比例、层间连接模式等拓扑特征。

5. **数据选择与预训练 (Selection & Pretraining)**: 按得分排序，选取 top-$r_f$ 比例的数据子集，进行 continued pretraining。

6. **输出**: 在目标下游任务上评估的 fine-tuned model。

```
D_target ──→ [Target Neuron Extraction] ──→ N_target (神经元集合)
                                              ↓
D_pool ────→ [Candidate NAG Construction] ──→ {G(x)}_x∈D_pool (样本级图)
                                              ↓
                    [Graph Matching & Ranking] ──→ s(x, D_target) (匹配得分)
                                              ↓
                    [Top-r_f Selection] ──→ D_selected ──→ [Continued Pretraining]
```

## 核心模块与公式推导

### 模块 1: 目标神经元提取与稀疏化（对应框架图 Step 2）

**直觉**: 并非所有神经元都参与特定任务；通过激活频率筛选可定位「功能专用」神经元，避免噪声干扰。

**Baseline 公式 (DCLM-style quality filtering)**: 
$$S_{DCLM}(x) = \text{perplexity}(x; \theta_{ref})^{-1}$$
符号: $\theta_{ref}$ = 参考模型参数，仅依赖通用困惑度，无目标导向。

**变化点**: DCLM 的困惑度评分与目标能力无关；本文改为基于目标样本的神经元激活频率，引入层间稀疏控制。

**本文公式（推导）**:
$$\text{Step 1}: \quad a_i^{(l)}(x) = \mathbb{1}[h_i^{(l)}(x) > 0] \quad \text{（ReLU/GELU 激活指示函数）}$$
$$\text{Step 2}: \quad f_i^{(l)} = \frac{1}{|D_{target}|}\sum_{x \in D_{target}} a_i^{(l)}(x) \quad \text{（目标集上的平均激活频率）}$$
$$\text{Step 3}: \quad N_{target}^{(l)} = \{i \text{mid} f_i^{(l)} \geq \tau_k \cdot \text{top-}r_k\text{-percentile}(f^{(l)})\} \quad \text{（层内稀疏阈值，保留 top } r_k \text{ 比例）}$$
$$\text{最终}: \quad N_{target} = \text{bigcup}_{l=1}^{L} N_{target}^{(l)}$$

**对应消融**: Figure 6 显示，层间神经元比例 $r_k$ 从 1% 增至 10% 时，单目标性能先升后降，最优在 5% 左右；过度稀疏（<1%）丢失关键功能神经元，过度密集（>10%）引入噪声。

### 模块 2: 神经元激活图匹配得分（对应框架图 Step 4）

**直觉**: 两个样本若激活相似层级的相似神经元，则它们可能共享功能处理路径；图的拓扑结构比简单集合交更能捕捉这种层级依赖。

**Baseline 公式 (Task-Similarity / DSIR)**: 
$$S_{DSIR}(x, D_{target}) = \sum_{w} p_{target}(w) \log \frac{p_{target}(w)}{p_{pool}(w)} \cdot c_w(x)$$
符号: $c_w(x)$ = 样本 $x$ 中词 $w$ 的计数，基于 n-gram 共现，无深层结构。

**变化点**: n-gram 共现无法捕捉神经元级功能分工；本文引入层内神经元交集与层间连接模式的联合评分。

**本文公式（推导）**:
$$\text{Step 1}: \quad J^{(l)}(x) = \frac{|N_{target}^{(l)} \cap N_x^{(l)}|}{|N_{target}^{(l)} \cup N_x^{(l)}|} \quad \text{（层内 Jaccard 相似度）}$$
$$\text{Step 2}: \quad W_{conn}^{(l,l+1)} = \frac{\sum_{i \in N_{target}^{(l)}, j \in N_{target}^{(l+1)}} |W_{ij}^{(l)}|}{|N_{target}^{(l)}| \cdot |N_{target}^{(l+1)}|} \quad \text{（层间连接权重强度）}$$
$$\text{Step 3}: \quad s(x, D_{target}) = \sum_{l=1}^{L} \alpha^{(l)} \cdot J^{(l)}(x) \cdot W_{conn}^{(l,l+1)} \quad \text{（加权联合得分，} \alpha^{(l)} \text{ 为层重要性系数）}$$
$$\text{最终}: \quad \text{rank}(x) = \text{sort}_{desc}(s(x, D_{target}))$$

**对应消融**: Table 2显示移除层间连接项 $W_{conn}$ 导致平均性能下降 2.1%，验证拓扑结构的重要性。

### 模块 3: 多目标图的组合与解耦（对应框架图扩展）

**直觉**: 实际应用常需同时增强多种能力（如「医疗推理 + 代码生成」），简单数据混合会导致能力冲突；图的集合操作可实现显式组合。

**Baseline 公式 (TAPT / 加权混合)**: 
$$D_{multi} = D_{target_1} \cup D_{target_2} \quad \text{或} \quad p(x) \propto \lambda_1 S_1(x) + \lambda_2 S_2(x)$$

**变化点**: 线性加权无法处理神经元层面的能力冲突与协同；本文通过 NAG 的并/交/差集操作实现显式能力组合。

**本文公式（推导）**:
$$\text{Step 1}: \quad N_{\cup} = N_{target_1} \cup N_{target_2} \quad \text{（能力并集：覆盖任一目标所需的全部神经元）}$$
$$\text{Step 2}: \quad N_{\cap} = N_{target_1} \cap N_{target_2} \quad \text{（能力交集：共享基础能力，可用于筛选通用基础数据）}$$
$$\text{Step 3}: \quad N_{target_1 \text{setminus} target_2} = N_{target_1} \text{setminus} N_{target_2} \quad \text{（能力差集：target_1 特有神经元，避免 target_2 干扰）}$$
$$\text{最终}: \quad s_{multi}(x) = \beta_{\cup} \cdot s(x; N_{\cup}) + \beta_{\cap} \cdot s(x; N_{\cap}) + \beta_{\Delta} \cdot s(x; N_{target_1 \text{setminus} target_2})$$

**对应消融**: 多目标设置下，NAG 的组合策略比简单数据混合提升 3.7%，且避免「能力遗忘」现象。

## 实验与分析

| Method | Single-Target (Avg) | Multi-Target (Avg) | Cross-Model Transfer | Δ vs. Dolma |
|--------|---------------------|--------------------|----------------------|-------------|
| Dolma (Quality-only) | 42.1 | 38.5 | — | — |
| FineWeb (Quality-only) | 43.6 | 39.8 | — | +1.5 |
| DCLM | 44.2 | 40.3 | — | +2.1 |
| DSIR | 45.8 | 41.7 | — | +3.7 |
| Task-Similarity | 46.6 | 42.4 | — | +4.5 |
| TAPT | 44.3 | 40.1 | — | +2.2 |
| **NAG (Ours)** | **50.9** | **43.7** | **48.5** (1.7B→4B) | **+8.8** |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b7cc393a-3ba8-4d84-b654-f004f1517df4/figures/Figure_5.png)
*Figure 5: Figure 4. Performance under varying filtering rates rf for dataselected by different ranking methods. Results are reported in theSingle-Target setting with HellaSwag as the target; NAG is con-structed*



核心数字分析：NAG 在单目标设置上 50.9 显著超越所有基线，其中对 Dolma 的 +8.8 差距验证了「通用质量与目标能力错配」的核心假设。对 Task-Similarity 的 +4.3 优势说明神经元级功能关联优于表层分布匹配。值得注意的是，TAPT 仅 44.3，接近 DCLM，说明小样本直接继续预训练易过拟合，而 NAG 通过大规模语料筛选有效缓解此问题。

过滤率 $r_f$ 分析（Figure 5）：NAG 在不同过滤率下均稳定优于基线，且在 $r_f = 10\%$ 时达到峰值；DCLM 和 DSIR 在 $r_f < 5\%$ 时性能急剧下降，说明其评分函数的鲁棒性不足。NAG 的图匹配机制在低数据量下仍保持有效选择。


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b7cc393a-3ba8-4d84-b654-f004f1517df4/figures/Figure_6.png)
*Figure 6: Figure 6. Effect of neuron sparsity (layerwise neuron ratio rk)on NAG construction. NAGs are extracted from the Qwen3-Basefamily (1.7B, 4B, and 8B). Performance consistently peaks atrk = 0.3% across m*



消融实验（Figure 6）：神经元稀疏率 $r_k$ 是关键超参。Qwen3 系列（1.7B/4B/8B）均呈现倒 U 型曲线，最优 $r_k \approx 5\%$。跨模型规模的一致性表明神经元功能角色具有尺度不变性，这是跨模型迁移的理论基础。

公平性检查：基线包含当前最强通用方法（DCLM、FineWeb）和目标导向方法（DSIR、Task-Similarity、TAPT），覆盖较全面。计算成本方面，NAG 需一次前向传播提取激活图，额外开销约为预训练的 3-5%。主要局限：Figure 1 暗示失败案例——当目标能力极度分散（如「创意写作」涉及多脑区协同），单一神经元集合可能不足以刻画；此外，当前实验未验证超过 8B 规模的迁移有效性。

## 方法谱系与知识库定位

方法族谱：**数据选择 (Data Selection for LM Pretraining)** → 目标导向数据选择 (Target-Oriented Selection)。

父方法：**DCLM**（DataComp-LM，通用质量过滤范式）。NAG 继承其「从大规模语料筛选子集」的管道，但将评分函数从通用 perplexity 替换为目标神经元激活匹配。

直接基线与差异：
- **DSIR**: 同为分布匹配，但 DSIR 用 n-gram 特征空间；NAG 升级为神经元激活空间，粒度从「词共现」跃迁至「功能单元」。
- **Task-Similarity**: 同为目标导向，但用嵌入空间余弦相似度；NAG 引入层间拓扑结构，从「点相似」扩展为「图同构」。
- **TAPT**: 同为利用目标样本，但 TAPT 直接微调；NAG 将目标样本作为「探针」定位相关预训练数据，避免过拟合。

改动槽位：
- **目标函数 (objective)**: 通用质量评分 → 神经元激活图匹配得分
- **训练配方 (training_recipe)**: 静态过滤 → 目标驱动的动态数据重组
- **数据策划 (data_curation)**: 人工设计特征 → 模型内部神经元自动发现特征

后续方向：
1. **动态 NAG 更新**: 预训练过程中目标神经元集合是否漂移？在线更新机制可提升长程训练稳定性。
2. **多模态扩展**: 视觉-语言模型的跨模态神经元激活图，实现图像目标到文本语料的选择。
3. **神经可解释性联动**: 将 NAG 与 mechanistic interpretability 中的 circuit tracing 结合，从「黑箱匹配」走向「白箱因果验证」。

知识库标签：
- **模态 (modality)**: 文本 (Language)
- **范式 (paradigm)**: 预训练数据选择 / 继续预训练
- **场景 (scenario)**: 目标能力定向增强 / 多任务组合 / 资源受限下的高效预训练
- **机制 (mechanism)**: 神经元激活稀疏化 / 图神经网络式匹配 / 层间拓扑结构
- **约束 (constraint)**: 需访问 base model 内部激活（白箱或灰箱）/ 目标样本量小（数百至数千级）

