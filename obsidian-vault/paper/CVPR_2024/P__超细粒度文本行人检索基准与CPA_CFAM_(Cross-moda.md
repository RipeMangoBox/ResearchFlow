---
title: 'UFineBench: Towards Text-based Person Retrieval with Ultra-fine Granularity'
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 超细粒度文本行人检索基准与CPAM
- CFAM (Cross-moda
- CFAM (Cross-modal Fine-grained Alignment Model)
acceptance: Poster
cited_by: 58
code_url: https://github.com/Zplusdragon/UFineBench
method: CFAM (Cross-modal Fine-grained Alignment Model)
---

# UFineBench: Towards Text-based Person Retrieval with Ultra-fine Granularity

[Code](https://github.com/Zplusdragon/UFineBench)

**Topics**: [[T__Retrieval]], [[T__Benchmark_-_Evaluation]], [[T__Cross-Modal_Matching]] | **Method**: [[M__CFAM]] | **Datasets**: [[D__RSTPReid]], [[D__CUHK-PEDES]] (其他: UFine3C)

| 中文题名 | 超细粒度文本行人检索基准与CPAM |
| 英文题名 | UFineBench: Towards Text-based Person Retrieval with Ultra-fine Granularity |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2312.03441) · [Code](https://github.com/Zplusdragon/UFineBench) · [Project](待补充) |
| 主要任务 | 基于文本的行人检索（Text-based Person Retrieval）、细粒度跨模态对齐 |
| 主要 baseline | IRRA、PLIP、CLIP、DSSL、SSAN、LBUL、IVT、CFine |

> [!abstract] 因为「现有文本行人检索数据集和模型仅关注粗粒度全局匹配，无法处理'穿着红色耐克运动鞋、背着黑色双肩包'等超细粒度描述」，作者在「CLIP双编码器+InfoNCE」基础上改了「增加共享粒度解码器、多任务损失函数和UFine6926细粒度数据集」，在「RSTPReid」上取得「CFAM-L/14达到R@1 62.45，相比IRRA提升+2.25」

- **RSTPReid**: CFAM-L/14 R@1/R@5/R@10/mAP/mSD = 62.45/83.55/91.10/49.50/36.92，全面超越IRRA
- **CUHK-PEDES**: CFAM达到72.87/88.61/92.87/64.92/50.20，相比仅使用Lgs的基线提升R@1 +4.42
- **跨数据集泛化**: 在UFine6926上训练后，UFine3C上R@1达到62.84，显著优于粗粒度数据训练

## 背景与动机

现有文本行人检索（Text-based Person Retrieval）面临一个根本性瓶颈：当用户查询从"穿黑色上衣的男子"升级为"穿黑色连帽卫衣、戴银色项链、手持蓝色保温杯、脚穿白色空军一号的年轻男性"时，主流模型往往失效。这种超细粒度（ultra-fine granularity）描述包含多个局部属性及其精确组合关系，要求模型不仅能识别"有什么"，还要精确定位"在哪里、如何组合"。

现有方法如何处理这一问题？**IRRA**通过跨模态关系推理和全局特征对齐提升性能，但依赖单一嵌入向量，丢失空间细节；**PLIP**利用大规模行人数据预训练获得强泛化能力，仍停留在全局表征层面；**CLIP**作为视觉-语言预训练的标杆，采用双编码器架构和InfoNCE损失实现图像-文本对齐，但其对比学习仅优化全局相似度，无法捕捉细粒度对应关系。这些方法的根本局限在于：**将整张图像和整段文本压缩为单个向量进行匹配**，导致局部属性（如特定配饰、鞋款）的信息在池化过程中被淹没。

更深层的问题在于数据：现有基准如CUHK-PEDES、ICFG-PEDES、RSTPReid的文本描述粒度不足，缺乏对行人外观的系统性细粒度标注，既无法训练也无法公平评估超细粒度检索能力。为此，本文提出UFineBench基准（含UFine6926训练集和UFine3C/UFineIC测试集），并设计CPAM（Cross-modal Fine-grained Alignment Model，后文亦称CFAM）框架，通过显式的多粒度特征解码与层次化损失设计，实现从全局到局部的渐进式跨模态对齐。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ec26ed93-6ddb-4949-933c-e252d609325b/figures/Figure_1.png)
*Figure 1 (motivation): Comparisons between proposed UFineBench and existing fine-grained datasets.*



## 核心创新

核心洞察：**全局嵌入对齐存在粒度天花板**，因为单一向量无法编码"红色耐克鞋在左脚"这类空间-属性耦合信息，而引入共享的粒度解码器（Granularity Decoder）配合多任务损失，可以在不增加推理阶段计算复杂度的前提下，实现视觉-文本的细粒度对应学习，从而使超细粒度检索成为可能。

| 维度 | Baseline (CLIP+InfoNCE) | 本文 (CFAM) |
|:---|:---|:---|
| 特征表示 | 全局单向量 (768/1024-d) | 全局向量 + 16个粒度查询的多粒度特征 |
| 对齐方式 | 图像嵌入 ↔ 文本嵌入 点积 | 多粒度特征 ↔ 多粒度特征 跨模态匹配 |
| 损失函数 | InfoNCE (L_info) | Lgs + λ₁Lls + λ₂Lm + λ₃Lcid 四任务加权 |
| 训练数据 | CUHK-PEDES等粗粒度数据 | 新增UFine6926超细粒度标注数据 |
| 推理策略 | 全局特征最近邻检索 | 全局+多粒度特征融合匹配 |

与CLIP相比，CFAM保留预训练编码器作为特征提取器，但**新增两个可学习模块**（粒度解码器+匹配器）和**三个辅助损失**，形成"粗-细"层次化的对齐监督；与IRRA等专门化方法相比，CFAM不依赖复杂的关系图构建，而是通过Transformer查询机制实现可扩展的细粒度扩展。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ec26ed93-6ddb-4949-933c-e252d609325b/figures/Figure_3.png)
*Figure 3 (architecture): Overview of the proposed CPAM framework.*



CFAM框架的数据流如下：输入图像（384×128）和文本描述（最大长度168）分别经过预训练编码器提取高层语义特征，随后进入新增的细粒度对齐模块。

**模块1: CLIP视觉编码器 (ViT-B/16或L/14)** — 输入行人图像，输出视觉特征图/全局token，作为后续解码的语义基础。保留预训练权重以继承开放域视觉-语言知识。

**模块2: CLIP文本编码器** — 输入tokenized文本，输出文本特征序列及全局[EOS]表示，与视觉侧对称处理。

**模块3: 共享粒度解码器 (Granularity Decoder)** — 核心创新模块。输入视觉/文本特征 + 16个可学习的粒度查询（维度512），通过2层Transformer解码器输出多粒度特征。关键设计：**跨模态共享参数**，使视觉和文本在同一语义空间内解耦粒度信息。

**模块4: 匹配器 (Matcher)** — 输入对齐后的多粒度视觉-文本特征对，通过2层Transformer块+MLP（含Sigmoid激活）计算细粒度匹配分数，输出跨模态相似度。

推理阶段，模型同时利用全局特征和16个粒度查询的局部特征进行多层级匹配，兼顾检索效率与细粒度区分能力。整体流程可概括为：

```
图像 ──→ CLIP Visual Encoder ──┐
                                ├──→ [Granularity Decoder] ──→ [Matcher] ──→ 相似度分数
文本 ──→ CLIP Textual Encoder ──┘         (16 queries)           (2-layer Tx + MLP)
```

## 核心模块与公式推导

### 模块1: 多任务损失函数（对应框架图损失监督部分）

**直觉**: 单一InfoNCE仅约束"正负样本是否匹配"，无法监督"哪些局部区域/词语应该对应"，需引入层次化损失实现从粗到细的对齐。

**Baseline 公式** (CLIP / No.0基线):
$$L_{InfoNCE} = -\log\frac{\exp(s(v,t)/\tau)}{\sum_{t'}\exp(s(v,t')/\tau)}$$
符号: $v,t$ = 图像/文本全局特征, $s(\cdot)$ = 余弦相似度, $\tau$ = 温度系数。

**变化点**: InfoNCE的$s(v,t)$仅为全局点积，对"红色耐克鞋"等局部属性无显式约束。本文将其扩展为四任务损失，逐步引入细粒度监督。

**本文公式（推导）**:
$$\text{Step 1}: L_{gs} = -\log\frac{\exp(s(v_g,t_g)/\tau)}{\sum_{t'}\exp(s(v_g,t')/\tau)} \quad \text{保留全局对齐作为基础}$$
$$\text{Step 2}: L_{ls} = \sum_{i}\|f_v^{(i)} - f_t^{(i)}\|_2^2 \quad \text{粒度解码器输出局部特征} f_v^{(i)}, f_t^{(i)} \text{的L2约束}$$
$$\text{Step 3}: L_m = \text{BCE}(\text{Matcher}(\{f_v^{(i)}\}, \{f_t^{(i)}\}), y_{match}) \quad \text{显式二分类匹配监督}$$
$$\text{Step 4}: L_{cid} = \text{InfoNCE}(z_{id}^v, z_{id}^t) \quad \text{对比身份判别，增强身份级判别力}$$
$$\text{最终}: L = L_{gs} + \lambda_1 L_{ls} + \lambda_2 L_m + \lambda_3 L_{cid}$$

**对应消融**: Table 5显示，从No.0（仅$L_{gs}$）到No.1（加入$L_{ls}$），CUHK-PEDES上R@1提升+1.97；继续加入$L_m$再提升+1.27；最终完整模型R@1达72.87。

### 模块2: 粒度解码器与匹配器（对应框架图核心创新）

**直觉**: 固定数量的可学习查询（如DETR）能自适应发现数据中的频繁粒度模式，跨模态共享确保视觉-文本语义空间一致。

**Baseline 做法**: 无此模块，CLIP直接使用编码器输出。

**本文公式**:
$$\text{Granularity Decoder}: Q \in \mathbb{R}^{16 \times 512}, \quad F_v^{gran} = \text{Decoder}_2(Q, F_v^{enc}), \quad F_t^{gran} = \text{Decoder}_2(Q, F_t^{enc})$$
$$\text{Matcher}: s_{fine} = \text{MLP}_{sig}(\text{Transformer}_2([F_v^{gran}; F_t^{gran}]))$$
$$\text{最终相似度}: s_{total} = \alpha \cdot s(v_g, t_g) + (1-\alpha) \cdot s_{fine}$$

符号: $Q$ = 可学习查询, $F^{enc}$ = 编码器特征, $[;]$ = 拼接, $\alpha$ = 全局-局部融合权重。

**对应消融**: Table 5中移除共享机制（视觉/文本各用独立解码器）导致性能下降，验证跨模态参数共享的关键作用（具体数值待补充）。

## 实验与分析


![Table 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ec26ed93-6ddb-4949-933c-e252d609325b/figures/Table_7.png)
*Table 7 (comparison): Comparison with the state-of-the-art methods.*



本文在三个基准上验证CFAM：RSTPReid（标准细粒度测试）、CUHK-PEDES（大规模经典基准）、以及新提出的UFine3C/UFineIC（跨数据集泛化测试）。核心结果如Table 7和Table 8所示：在RSTPReid上，CFAM-L/14达到R@1 62.45 / R@5 83.55 / R@10 91.10 / mAP 49.50 / mSD 36.92，相比此前最优的IRRA（60.20/81.30/88.20/47.17/35.22）全面提升，R@1提升+2.25，mAP提升+2.33。值得注意的是，CFAM-B/16版本（59.40/81.35/88.50/46.04/34.27）在mAP和mSD上略低于IRRA，说明更大编码器（L/14）对细粒度任务有显著收益。在CUHK-PEDES上，完整模型达到72.87/88.61/92.87/64.92/50.20，相比仅使用全局损失$L_{gs}$的No.0基线（68.45/86.50/91.68/61.28/46.31），R@1提升+4.42，mSD提升+3.89，验证了细粒度组件的有效性。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ec26ed93-6ddb-4949-933c-e252d609325b/figures/Table_5.png)
*Table 5 (ablation): Ablation study on each component of CPAM.*



消融实验（Table 5 / Table 8）揭示了各损失组件的贡献：加入局部相似性损失$L_{ls}$带来最大单次增益（R@1 +1.97），说明显式局部特征约束是突破全局瓶颈的关键；匹配损失$L_m$进一步带来+1.27，表明可学习的匹配器比固定距离度量更适合细粒度对齐；方差感知损失$L_{vap}$（即$L_{cid}$相关组件）贡献+0.73，主要提升mSD指标，改善细粒度对齐的稳定性。训练数据方面，Table 2显示在UFine6926上训练相比传统粗粒度数据集，UFine3C上R@1从约55提升至62.84，验证了超细粒度数据对泛化的关键作用。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ec26ed93-6ddb-4949-933c-e252d609325b/figures/Figure_4.png)
*Figure 4 (qualitative): Categories of rank-10 retrieval results on UFineIC.*



公平性检查：本文对比的IRRA、CFine等为2022-2023年代表性方法，但未包含2023年后更新的专门化方法。CFAM-L/14与B/16基线使用不同编码器规模，部分指标的直接对比需谨慎。作者披露的主要局限包括：细粒度解码的计算开销（未报告具体推理延迟）、以及部分消融实验细节依赖补充材料。训练成本方面，使用单卡V100 32GB，60 epoch配合5 epoch warm-up，属于中等规模训练。

## 方法谱系与知识库定位

CFAM属于**CLIP扩展谱系**：以CLIP的双编码器预训练架构为父方法，通过新增模块实现从"全局对比"到"细粒度对齐"的演化。关键改动槽位包括：
- **架构**: 新增共享粒度解码器（2-layer Transformer + 16查询）+ 匹配器（2-layer Transformer + MLP/Sigmoid）
- **目标函数**: InfoNCE → 四任务加权损失（Lgs + Lls + Lm + Lcid）
- **数据管线**: 引入UFine6926超细粒度训练集，替换/补充传统粗粒度数据
- **推理策略**: 全局+多粒度特征融合匹配

**直接基线对比**:
- **IRRA**: 同样基于Transformer，但依赖关系推理和全局特征；CFAM通过查询机制显式解码粒度，无需构建关系图
- **PLIP**: 强调大规模预训练数据；CFAM强调数据粒度质量（UFine6926）和架构细粒度设计，两者互补
- **CLIP**: 父方法，提供预训练编码器；CFAM证明在保持编码器冻结/微调的情况下，新增轻量解码模块即可实现细粒度扩展

**后续方向**: (1) 将粒度查询机制扩展至视频行人检索，处理时序细粒度变化；(2) 结合大语言模型（LLM）生成更丰富的细粒度描述，反哺数据标注；(3) 探索粒度解码器的可解释性，可视化16个查询各自对应的语义属性（如鞋类、包类、配饰等）。

**标签**: 模态=跨模态(图像-文本) / 范式=对比学习+多任务学习 / 场景=行人检索、细粒度识别 / 机制=Transformer查询解码、层次化损失 / 约束=需配对的细粒度标注数据

