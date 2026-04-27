---
title: 'Less Attention is More: Prompt Transformer for Generalized Category Discovery'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- AptGCD：面向GCD的提示Transformer
- AptGCD
acceptance: poster
cited_by: 7
method: AptGCD
---

# Less Attention is More: Prompt Transformer for Generalized Category Discovery

**Topics**: [[T__Few-Shot_Learning]], [[T__Self-Supervised_Learning]] | **Method**: [[M__AptGCD]] | **Datasets**: SSB

| 中文题名 | AptGCD：面向GCD的提示Transformer |
| 英文题名 | Less Attention is More: Prompt Transformer for Generalized Category Discovery |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://doi.org/10.1109/cvpr52734.2025.02823) · [Code] · [Project] |
| 主要任务 | Generalized Category Discovery (GCD)，细粒度类别发现 |
| 主要 baseline | LegoGCD, CMS, CiPR, VPT, SPM |

> [!abstract]
> 因为「现有GCD方法在novel类别上表现不足，且标准视觉提示（如VPT）会引入噪声干扰」，作者在「LegoGCD」基础上改了「引入Meta Visual Prompt (MVP)聚焦显著特征，并在深层添加Prompt Transformer (PT)进行局部-全局交互」，在「SSB细粒度基准（CUB-200/Stanford Cars/FGVC-Aircraft平均）」上取得「New accuracy 57.3%，相比无提示baseline提升+4.0%，相比VPT提升+5.8%」

- **New accuracy**: SSB-Avg 57.3%（vs. LegoGCD 53.3%, +4.0%）
- **Transferability**: 集成至CMS提升+4.8%，集成至CiPR提升+5.2%，集成至LegoGCD提升+7.3%
- **参数增量**: 仅增加不到2%的prompt参数

## 背景与动机

Generalized Category Discovery (GCD) 旨在利用已标注的「旧类别」数据，从未标注数据中同时识别已知类别并发现「新类别」。一个典型场景是：给定100种鸟类的标注图像，模型需要在野外新收集的照片中识别这100种已知鸟类，同时发现另外50种从未见过的鸟类。这对细粒度识别尤为困难——例如区分CUB-200中外观极为相似的鸟种，需要模型精准定位羽毛纹理、喙形等细微差异。

现有方法如何处理这一问题？**LegoGCD** 作为代表性baseline，采用标准ViT-B/16架构，将[CLS] token直接送入分类器进行半监督聚类；**CMS** 和 **CiPR** 等后续方法则通过改进对比学习或原型网络来提升新类别发现能力。视觉提示领域，**VPT (Visual Prompt Tuning)** 在输入层插入可学习token以适配下游任务，**SPM** 则专为few-shot新类别设计提示策略。

然而这些方法存在关键缺陷：VPT等通用视觉提示在GCD任务中反而引入噪声——实验显示VPT将New accuracy从53.3%降至51.5%（-1.8%），因其未区分新旧类别的特征重要性差异；同时，深层特征缺乏对空间分布的适应性加权，导致细粒度特征的局部形态变化无法被全局上下文有效捕捉。LegoGCD的[CLS] token虽包含全局信息，却忽略了patch token在不同空间位置扮演的差异化角色。

本文提出AptGCD，通过「更少但更有针对性的注意力」解决上述问题：Meta Visual Prompt引导模型聚焦显著区域而非分散注意，Prompt Transformer在深层以局部-全局交互实现特征的自适应重加权。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee89fcd6-8b5d-480c-822c-3b093e49e5af/figures/fig_001.jpeg)
*Figure: Visulization of the attention across various methods on*



## 核心创新

核心洞察：GCD中新类别发现的关键不在于增加更多注意力，而在于让注意力「更精准」——通过输入层的Meta Visual Prompt预先锚定显著特征区域，再通过深层的Prompt Transformer让特征根据其空间分布角色自适应地参与全局交互，从而使细粒度特征的判别性表达成为可能。

| 维度 | Baseline (LegoGCD/VPT) | 本文 (AptGCD) |
|:---|:---|:---|
| 视觉提示设计 | 无提示 或 通用可学习token (VPT) | **Meta Visual Prompt**: 针对GCD优化的显著性引导提示 |
| 特征交互位置 | 仅[CLS] token分类 | **深层Prompt Transformer**: 局部-全局交互块 |
| 特征适应机制 | 固定权重，位置无关 | 空间自适应：同一特征在不同位置获得不同权重 |
| 参数效率 | 全模型微调 或 VPT全层插入 | 仅最后一块微调 + <2% prompt参数 |

与VPT的关键差异：VPT在输入层插入随机初始化的通用prompt，对GCD的新旧类别不加区分；MVP则通过元学习机制生成任务相关的显著性提示，主动抑制背景噪声。与SPM的差异：SPM针对few-shot新类设计，而MVP直接优化GCD的novel类别发现目标。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee89fcd6-8b5d-480c-822c-3b093e49e5af/figures/fig_002.jpeg)
*Figure: The architecture of AptGCD. Meta-prompt generation model generates prompts which are fed back into the input module,*



AptGCD的整体数据流遵循「提示增强输入 → 深层特征适应 → 统一分类」的三阶段范式：

1. **Image Input → Patch Embedding**: 原始图像经ViT标准patch embedding切分为序列化patch token，与[CLS] token拼接。

2. **Meta Visual Prompt (MVP)**: 在输入空间，可学习的MVP token被prepend/append至patch序列。这些prompt并非随机初始化，而是通过元生成机制优化，引导模型关注显著特征（如鸟类的羽毛纹理、汽车的格栅造型），而非背景噪声。

3. **ViT Backbone (冻结前层，仅微调最后一块)**: 处理prompt增强后的序列，输出各层特征。前层参数冻结以保持预训练知识，降低过拟合风险。

4. **Prompt Transformer (PT)**: 接入深层特征（实验显示越深层效果越好），通过local-global interaction block实现：局部路径捕捉特征的空间分布与形态变化，全局路径建立跨位置上下文关联，最终输出自适应加权后的判别性特征。

5. **Classifier (LegoGCD同款)**: 接收[CLS] token或适配后的特征，执行新旧类别的联合半监督聚类与分类。

```
Raw Image → [Patch Embed] → [+ MVP tokens] → [ViT Blocks 1~L-1 (frozen)] 
                                                          ↓
[CLS] + Patch Tokens ──────────────────────→ [ViT Block L (fine-tuned)] 
                                                          ↓
                                          [Prompt Transformer (deep layers)]
                                                          ↓
                                          [Classifier] → {Old Classes, New Classes}
```

关键设计：MVP作用于输入层解决「看什么」，PT作用于深层解决「如何加权」，两者解耦但协同。

## 核心模块与公式推导

### 模块 1: Meta Visual Prompt (MVP)（对应框架图 输入端）

**直觉**: 标准视觉提示在GCD中失效，因其对所有类别一视同仁地添加噪声；MVP通过元学习生成任务相关的显著性提示，使模型「预先知道」该关注哪些区域。

**Baseline 公式** (VPT-style prompting): 
$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \mathbf{x}_1; \cdots; \mathbf{x}_N; \mathbf{p}_1; \cdots; \mathbf{p}_M] + \mathbf{E}_{\text{pos}}$$
其中 $\mathbf{p}_i \in \mathbb{R}^D$ 为随机初始化的可学习prompt，与图像patch简单拼接。

**变化点**: VPT的$\mathbf{p}_i$与任务目标脱节，在GCD中导致旧类别过拟合、新类别欠拟合。MVP将prompt参数化为类别判别性的元知识：

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{P}_{\text{meta}} = g_\phi(\mathbf{S}) \quad \text{元生成器} \, g_\phi \, \text{从支持集} \, \mathbf{S} \, \text{提取显著性先验}$$
$$\text{Step 2}: \mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \mathbf{P}_{\text{meta}}; \mathbf{x}_1; \cdots; \mathbf{x}_N] + \mathbf{E}_{\text{pos}} \quad \text{将元提示置于关键位置}$$
$$\text{Step 3}: \mathcal{L}_{\text{MVP}} = \mathbb{E}_{(\mathbf{x},y)\sim\mathcal{D}_l}\left[ -\log p(y|\mathbf{z}_L^{(\text{cls})}) \right] + \lambda \cdot \text{Var}(\mathbf{A}_{\text{MVP}}) \quad \text{方差正则化防止注意力坍塌}$$
**最终**: MVP通过分类目标与注意力分散惩罚的联合优化，使$\mathbf{P}_{\text{meta}}$引导网络关注类判别性区域。

**对应消融**: Table 4显示，移除MVP（替换为None）New accuracy从57.3%降至53.3%（Δ-4.0%）；替换为VPT则降至51.5%（Δ-5.8%），证明通用提示在此任务中的负作用。

### 模块 2: Prompt Transformer (PT)（对应框架图 深层特征端）

**直觉**: 同一视觉特征（如「圆形」）在鸟头位置表示眼睛，在车身位置表示轮胎——PT根据特征的空间角色动态调整其全局贡献。

**Baseline 公式** (LegoGCD直接分类): 
$$\mathbf{y} = \text{Classifier}(\text{LN}(\mathbf{z}_L^{(\text{cls})}))$$
仅使用[CLS] token，所有patch token被压缩为单一全局表示，丢失空间细节。

**变化点**: 深层特征需要位置感知的动态重加权。PT引入local-global interaction，让prompt token与patch token在深层进行交互式注意力计算。

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{H}^{(l)} = \text{MSA}([\mathbf{P}_{\text{PT}}; \mathbf{Z}^{(l)}]) \quad \text{MSA为多头自注意力，} \mathbf{P}_{\text{PT}} \text{为PT专用提示}$$
$$\text{Step 2}: \mathbf{H}^{(l)}_{\text{local}} = \text{Conv}_{1\times1}(\mathbf{H}^{(l)}_{\text{patch}}) \quad \text{局部路径：1×1卷积捕捉空间邻域关系}$$
$$\text{Step 3}: \mathbf{H}^{(l)}_{\text{global}} = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{prompt}}\mathbf{K}_{\text{patch}}^\text{top}}{\sqrt{D/h}}\right)\mathbf{V}_{\text{patch}} \quad \text{全局路径：prompt-to-patch交叉注意力}$$
$$\text{Step 4}: \mathbf{Z}^{(l+1)} = \text{LN}(\mathbf{Z}^{(l)} + \text{FFN}([\mathbf{H}^{(l)}_{\text{local}}; \mathbf{H}^{(l)}_{\text{global}}])) \quad \text{局部-全局融合并残差连接}$$
**最终**: 输出适配后的特征$\mathbf{Z}^{(l+1)}$，其中每个patch的表示已融合其空间角色与全局上下文。

**对应消融**: Figure 5显示，PT放置于最深层（第12块）时New accuracy最优；移至浅层（第6块）性能显著下降，验证「深层特征更需要位置自适应」的假设。

### 模块 3: 联合训练目标（系统级优化）

**直觉**: MVP与PT需协同优化，但两者作用于不同层级，需平衡提示生成与特征适应的梯度流。

**本文公式**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}}^{\text{old}} + \mathcal{L}_{\text{cls}}^{\text{new}} + \lambda_1 \mathcal{L}_{\text{contrast}} + \lambda_2 \mathcal{L}_{\text{MVP-reg}}$$
其中$\mathcal{L}_{\text{cls}}^{\text{old}}$为有标注旧类的交叉熵，$\mathcal{L}_{\text{cls}}^{\text{new}}$为无标注新类的伪标签损失（匈牙利匹配后），$\mathcal{L}_{\text{contrast}}$为实例级对比损失，$\mathcal{L}_{\text{MVP-reg}}$为MVP注意力分散正则化。

**对应消融**: Figure 3的渐进式实验显示，Baseline → +MVP → +PT的逐步添加带来单调提升，两者存在互补效应而非冗余。

## 实验与分析



本文在细粒度语义基准SSB（Semantic Shift Benchmark）上开展系统评估，该基准平均CUB-200、Stanford Cars、FGVC-Aircraft三个细粒度数据集的指标。核心结果如Table 4所示：AptGCD在New accuracy上达到57.3%，相比直接移除所有提示机制的baseline（53.3%）提升+4.0个百分点，相比通用视觉提示VPT（51.5%）提升+5.8个百分点，甚至超过专为新类别设计的SPM（55.7%）+1.6个百分点。这一增益在细粒度场景尤为关键——CUB-200等数据集需要模型区分羽毛纹理、喙形等微观差异，MVP的显著性引导与PT的位置自适应恰好针对性解决了这一痛点。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee89fcd6-8b5d-480c-822c-3b093e49e5af/figures/fig_003.jpeg)
*Figure: Quantitative analysis on the contribution of MVP and*





消融实验进一步验证各组件的贡献。Figure 3的定量分析显示，MVP单独添加即可带来约2-3%的New accuracy提升，PT在此基础上再贡献约2-3%，两者叠加后旧类别(Old)准确率保持稳定、新类别(New)显著提升，有效缩小了新旧类别的性能差距。Figure 5关于PT放置位置的消融表明：将PT置于第12层（最深层）时New accuracy最高，移至第9层略有下降，第6层明显下降，验证了「深层特征更需要局部-全局交互」的设计假设。

Transferability分析（Table 7）展示AptGCD作为即插即用模块的价值：集成至CMS后New accuracy从52.7%提升至57.5%（+4.8%），集成至CiPR从47.9%提升至53.1%（+5.2%），集成至LegoGCD从53.3%提升至60.6%（+7.3%）。这一跨架构一致性增益说明MVP+PT的设计具有通用性。

公平性检查：对比基线中，LegoGCD为直接baseline，CMS/CiPR为同期先进方法，VPT/SPM为视觉提示领域的代表性方法。但论文未与XCon、SimGCD、PromptCAL、SPTNet等方法直接比较，这些可能是更强的竞争者。训练成本方面，仅使用单张NVIDIA GeForce RTX 3090，200 epoch，参数增量<2%，属于轻量适配方案。潜在局限：Herbarium19等不平衡数据集结果未完整报告；transferability实验未明确是完整重训练还是轻量适配。

## 方法谱系与知识库定位

**方法家族**: Visual Prompting → GCD Adaptation

**父方法**: **LegoGCD** [6] — AptGCD保留其分类器与基本训练流程，但在architecture、data_pipeline、inference_strategy三个slot进行扩展。

**直接基线对比**:
- **LegoGCD**: 标准ViT-B/16 + [CLS]分类 → AptGCD添加MVP输入提示与PT深层交互
- **VPT** [23]: 通用视觉提示，全层插入 → AptGCD的MVP针对GCD优化，且仅输入层使用
- **SPM** [33]: few-shot新类提示 → AptGCD直接优化GCD的半监督聚类目标，非few-shot场景
- **CMS/CiPR**: 先进GCD方法 → AptGCD作为即插即用模块可集成至其上，进一步提升性能

**改动slot总结**:
| Slot | 变更内容 |
|:---|:---|
| architecture | 深层添加Prompt Transformer with local-global interaction blocks |
| data_pipeline | 输入层插入Meta Visual Prompt替代标准patch序列 |
| inference_strategy | 从直接[CLS]分类改为prompt-guided adaptive feature weighting |
| training_recipe | 仅微调最后一块 + <2% prompt参数，SGD+cosine annealing |

**后续方向**:
1. 将MVP的元生成机制扩展至动态图提示，适应更复杂的类别层次结构
2. PT的局部-全局交互可借鉴状态空间模型（如Mamba）降低二次注意力复杂度
3. 探索AptGCD在开放词汇检测、持续学习等更广义新类别发现场景中的迁移

**标签**: 视觉/图像分类 | 提示学习范式 | 半监督/无监督新类发现 | 局部-全局注意力机制 | 轻量参数高效微调

