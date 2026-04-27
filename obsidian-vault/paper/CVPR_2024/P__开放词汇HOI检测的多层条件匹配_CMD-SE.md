---
title: Exploring the Potential of Large Foundation Models for Open-Vocabulary HOI Detection
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 开放词汇HOI检测的多层条件匹配与语义增强
- CMD-SE
acceptance: Poster
cited_by: 33
code_url: https://sites.google.com/view/cmd-se/
method: CMD-SE
---

# Exploring the Potential of Large Foundation Models for Open-Vocabulary HOI Detection

[Code](https://sites.google.com/view/cmd-se/)

**Topics**: [[T__Object_Detection]], [[T__Cross-Modal_Matching]], [[T__Visual_Reasoning]] | **Method**: [[M__CMD-SE]] | **Datasets**: [[D__SWIG-HOI]], [[D__HICO-DET]]

| 中文题名 | 开放词汇HOI检测的多层条件匹配与语义增强 |
| 英文题名 | Exploring the Potential of Large Foundation Models for Open-Vocabulary HOI Detection |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2404.06194) · [Code](https://sites.google.com/view/cmd-se/) · [DOI](https://doi.org/10.1109/CVPR52733.2024.01576) |
| 主要任务 | Open-vocabulary Human-Object Interaction (HOI) Detection |
| 主要 baseline | Open-vocabulary HOI detector [56], Zero-shot methods [34, 39] with CLIP, DETR-based detectors |

> [!abstract] 因为「开放词汇HOI检测中，单一特征层级难以捕捉不同空间尺度的人-物交互，且标准文本嵌入缺乏细粒度语义」，作者在「Open-vocabulary HOI detector [56]」基础上改了「引入多层解码(MD)、条件匹配(CM)软约束、以及基于GPT生成身体部位状态描述的语义增强(SE)」，在「SWIG-HOI」上取得「mAP 15.26，相对提升15.08%」

- **SWIG-HOI Full mAP**: 15.26，相对先前SOTA提升 15.08%
- **消融关键增量**: +MD 提升 1.42 mAP (11.45→12.87)，+CM 再提升 2.39 mAP (12.87→15.26)
- **SE对unseen类增益**: unseen mAP 从 7.32 提升至 10.70，相对提升 46.2%

## 背景与动机

人-物交互（HOI）检测旨在识别图像中「人在做什么」——例如「人骑在马背上」或「厨师用刀切菜」。传统方法只能识别训练时见过的固定交互类别，而开放词汇（open-vocabulary）HOI检测要求模型能泛化到从未见过的新交互类型，这对自动驾驶、视频监控等实际应用至关重要。

现有方法主要沿两条路径推进：一是基于DETR架构的零样本方法[34,39]，它们利用CLIP的文本嵌入进行交互分类，但依赖预训练的DETR权重且仅用单一层级特征解码；二是开放词汇方法[56]，它们摆脱了对检测数据集的预训练依赖，但仍采用标准的单层级特征解码策略。这两类方法共同面临一个核心盲区：HOI实例在空间尺度上差异巨大——「人拥抱人」需要精细的局部特征，「人远眺山脉」则需要全局上下文——而单层级特征无法同时满足这些矛盾需求。

更深层的问题在于匹配机制：标准二分匹配将预测与真值随机配对，完全忽视了「人-物距离」这一关键几何线索。近距离交互（如握手）与远距离交互（如投掷）被同等对待，导致特征层级与空间尺度错配。此外，现有文本嵌入仅使用粗粒度的交互名称（如"kick"），缺乏描述身体部位具体状态的细粒度语义，难以区分"踢足球"与"踢门"的微妙差异。

本文提出CMD-SE，通过三层递进式改进——多层解码显式利用多尺度特征、条件匹配建立距离-层级的软约束对应、语义增强引入GPT生成的身体部位状态描述——系统性地解决上述尺度错配与语义粗粒度问题。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9e6771c-aa69-478a-b7f6-c56949c2ab13/figures/Figure_1.png)
*Figure 1 (result): Performance comparison between our method and VIBE on HAKE. t-SNE visualization of HOI concepts in semantic space.*



## 核心创新

核心洞察：人-物交互的空间尺度（距离）与卷积特征层级存在天然对应关系——低层特征适合短距离精细交互，高层特征适合长距离全局交互——因为这一对应关系被显式编码为可学习的软约束，从而使无需预训练检测数据的开放词汇HOI检测成为可能。

| 维度 | Baseline [56] | 本文 CMD-SE |
|:---|:---|:---|
| 特征解码 | 单层级特征，所有HOI共享同一尺度 | 多层特征（MD），不同尺度HOI使用不同层级 |
| 匹配策略 | 标准二分匹配，距离无关 | 条件匹配（CM），按绝对距离约束特征层级选择 |
| 文本语义 | 交互名称或粗粒度描述 | GPT生成的身体部位状态描述（SE），细粒度且可重组 |
| 训练约束 | 仅边界框+分类+IoU损失 | 新增距离约束损失 $\mathcal{L}_d$，权重 $\lambda_d=5$ |

关键突破在于将「几何先验」（人-物距离）转化为「可优化目标」：不是硬编码规则，而是通过软约束损失让网络自主学习最优的层级-距离对应，同时利用大语言模型的知识生成可泛化的身体部位状态描述，实现语义空间的组合式扩展。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9e6771c-aa69-478a-b7f6-c56949c2ab13/figures/Figure_2.png)
*Figure 2 (pipeline): The framework of our CMUSE-SHO.*



CMD-SE的完整数据流如下：

**输入**: 图像 $I$ + 交互类别名称（如"kick"）

**模块1: CLIP视觉编码器 (ViT-B/16, 冻结)** — 输入图像，输出多层级视觉特征 $\{F_1, F_2, ..., F_L\}$，其中低层保留空间细节，高层聚合语义信息。

**模块2: 多层解码 (MD)** — 输入多层级特征，输出各层级的HOI预测集合 $\{\hat{Y}_1, \hat{Y}_2, ..., \hat{Y}_L\}$。替代了baseline的单层解码，使不同尺度交互有专属的特征表达通道。

**模块3: 条件匹配 (CM)** — 输入预测集合与真值标注（含人-物绝对距离 $d_{HO}$），输出距离感知的匹配结果。核心操作：根据 $d_{HO}$ 的大小，将短距离HOI的真值优先匹配到低层特征预测，长距离HOI匹配到高层特征预测。

**模块4: 语义增强 (SE)** — 输入交互类别，通过GPT生成该交互涉及的身体部位状态描述（如"kick: leg extended, foot in contact with ball"），经CLIP文本编码器输出增强嵌入 $t_{SE}$，用于最终的交互分类。

**输出**: HOI三元组 $\langle$人框, 物框, 交互类别$\rangle$

```
图像 I → [CLIP ViT-B/16] → {F_1,...,F_L}
                              ↓
                    [Multi-level Decoding]
                              ↓
                    {Ŷ_1,...,Ŷ_L} + GT(d_HO) → [Conditional Matching]
                              ↓
                         距离感知匹配结果
                              ↓
                    交互名称 → [GPT] → 身体部位状态描述
                              ↓
                    [CLIP_text] → t_SE → [分类头]
                              ↓
                         HOI三元组输出
```

## 核心模块与公式推导

### 模块1: 条件匹配损失 $\mathcal{L}_d$（对应框架图模块3）

**直觉**: 标准匹配随机分配预测与真值，导致低层特征被迫学习远距离HOI（需要大感受野）而高层特征被迫学习近距离HOI（需要精细定位），造成特征层级与任务需求错配。

**Baseline 公式** (标准二分匹配 [Carion et al., DETR]):
$$\min_{\sigma} \sum_{i} \mathcal{C}_{match}(y_i, \hat{y}_{\sigma(i)})$$
符号: $y_i$ = 第$i$个真值HOI，$\hat{y}_{\sigma(i)}$ = 排列$\sigma$下的第$i$个预测，$\mathcal{C}_{match}$ = 分类+边界框联合代价

**变化点**: 代价函数缺乏对「特征层级 $l$」与「人-物距离 $d_{HO}$」对应关系的约束。近距离HOI应由低层（高分辨率）特征检测，远距离HOI应由高层（大感受野）特征检测。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{C}_{dist}(d_{HO}, l_{feat}) = |d_{HO} - \mu(l_{feat})|$$
其中 $\mu(l)$ 为层级$l$的期望距离中心（低层对应小距离，高层对应大距离）。加入绝对距离软约束：
$$\text{Step 2}: \quad \min_{\sigma} \sum_{i} \mathcal{C}_{match}(y_i, \hat{y}_{\sigma(i)}) + \lambda_d \cdot \mathcal{C}_{dist}(d_{HO}^{(i)}, l_{\sigma(i)})$$
权重 $\lambda_d=5$ 经消融验证为最优，平衡匹配代价与距离约束。
$$\text{最终}: \quad \mathcal{L}_{total} = \lambda_b \mathcal{L}_b + \lambda_{iou} \mathcal{L}_{iou} + \lambda_{cls} \mathcal{L}_{cls} + \lambda_d \mathcal{L}_d$$
其中 $\lambda_b=5, \lambda_{iou}=2, \lambda_{cls}=5, \lambda_d=5$。

**对应消融**: Table 5显示$\lambda_d=0$时Full mAP为13.80，$\lambda_d=5$时提升至15.26，增益+1.46；$\lambda_d=10$时略降至15.08，验证$\lambda_d=5$为最优拐点。

---

### 模块2: 语义增强嵌入 $t_{SE}$（对应框架图模块4）

**直觉**: 标准CLIP文本嵌入仅用交互名称（如"kick"），无法区分不同场景下的同一动词；而身体部位状态描述（"leg extended, foot in contact with ball"）提供了可重组的细粒度语义，且GPT能自动为任意新交互生成此类描述。

**Baseline 公式** (标准CLIP文本编码):
$$t = \text{CLIP}_{text}(\text{prompt}), \quad \text{prompt} = \text{"a photo of [interaction]"}$$
符号: $t$ = 文本嵌入向量，CLIP$_{text}$ = 冻结的CLIP文本编码器

**变化点**: 提示词缺乏身体部位的细粒度状态信息，且对新交互的泛化依赖人工设计模板。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{Body-part Selection via GPT: } \{p_1, p_2, ..., p_k\} = \text{GPT}(\text{interaction}, \text{template})$$
GPT根据"Body-part Selection"流程生成涉及的身体部位及其状态描述（如{kicking: {leg: "extended", foot: "in contact with ball"}}）。
$$\text{Step 2}: \quad \text{prompt}_{SE} = \text{concatenate}(\text{interaction}, \{p_j\}_{j=1}^k)$$
$$\text{最终}: \quad t_{SE} = \text{CLIP}_{text}(\text{prompt}_{SE})$$

**对应消融**: Table 6显示使用身体部位状态描述时unseen mAP达10.70，而仅使用交互名称时为7.32，相对提升46.2%；使用动作定义或场景描述均不如身体部位状态精确。

---

### 模块3: 多层解码整合（对应框架图模块2）

**直觉**: 单层级特征解码强制所有HOI共享同一特征尺度，而视觉编码器的不同层级天然具有不同的感受野与分辨率。

**Baseline**: 仅使用最后一层特征 $F_L$ 进行HOI解码。

**本文**: 对各层特征独立应用解码头，预测结果聚合：
$$\hat{Y} = \text{bigcup}_{l=1}^{L} \text{Decoder}_l(F_l)$$
条件匹配自动将不同距离的HOI分配到最优层级，避免手工设计特征金字塔的复杂融合。

**对应消融**: Table 3（或4）显示Base模型（单层）Full mAP 11.45，+MD提升至12.87（+1.42），+CM进一步提升至15.26（+2.39），验证多层结构与条件匹配的协同效应。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9e6771c-aa69-478a-b7f6-c56949c2ab13/figures/Table_1.png)
*Table 1 (comparison): Comparison of our proposed CMUSE-SHO with state-of-the-art methods on HICO-DET.*



本文在SWIG-HOI和HICO-DET两个基准上评估CMD-SE。SWIG-HOI是包含丰富交互类别的大规模数据集，HICO-DET则用于模拟零样本评估（特定交互类别在训练时不可见）。

在SWIG-HOI上，CMD-SE取得Full mAP 15.26，相对先前开放词汇HOI检测器提升15.08%。这一增益并非来自单一模块：消融显示Base（单层解码+标准匹配+无语义增强）仅11.45；引入多层解码（MD）后提升至12.87（+1.42）；叠加条件匹配（CM）后跃升至15.26（+2.39，相对+MD）；完整模型含语义增强（SE）维持15.26但unseen类从7.32大幅改善至10.70。值得注意的是，CM的增益（+2.39）超过MD（+1.42），说明距离感知的匹配策略比单纯增加特征层级更为关键——它解决了「有多层但不会用」的问题。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9e6771c-aa69-478a-b7f6-c56949c2ab13/figures/Table_5.png)
*Table 5 (ablation): Ablation on the weight for the additional self-correlation term λ in the proposed CMUSE-SHO.*



条件匹配的内部机制经充分验证：使用绝对距离优于相对距离（seen mAP 16.79 vs 16.57，unseen 10.70 vs 10.30）；低层特征匹配短距离HOI（Low-Small）优于匹配长距离（Low-Large，seen 16.79 vs 16.61）。软约束权重$\lambda_d$的消融（Table 5）显示$\lambda_d=0$时Full mAP 13.80，$\lambda_d=5$时最优15.26，$\lambda_d=10$时略降至15.08，证明适度的距离约束有效，过强则干扰正常匹配。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9e6771c-aa69-478a-b7f6-c56949c2ab13/figures/Figure_4.png)
*Figure 4 (qualitative): Qualitative results of our method on SWIG-HOI test set.*



定性结果（Figure 4）展示了communicating、kissing、hurling、kicking等交互的检测效果，验证了方法在复杂场景下的可视化表现。

**公平性检视**: 作者明确承认与[34,39]在HICO-DET上的比较存在不公平性——这些方法预训练于COCO检测数据，而HICO-DET与COCO共享对象标签空间，CMD-SE则完全不使用检测预训练。Table 1和Table 2中部分基线数据缺失（如[16,18]的完整结果），且未与[56]在完全相同的设置下直接对比。训练成本方面，使用2×A100 GPU、batch size 128、80 epochs，属于中等规模。主要局限：CLIP ViT-B/16冻结限制了视觉特征的进一步适配；GPT生成描述的稳定性未量化分析；对极端长距离交互（如"人观看远山"）的性能未单独报告。

## 方法谱系与知识库定位

CMD-SE隶属于**开放词汇检测 → 开放词汇HOI检测**的方法谱系，直接继承自Open-vocabulary HOI detector [56]（不预训练检测数据的范式）。核心改动slot：inference_strategy（单层→多层解码+条件匹配）、architecture（标准CLIP文本嵌入→GPT增强的身体部位状态描述）、training_recipe（新增距离约束损失$\mathcal{L}_d$）。

**直接基线对比**:
- vs. [56] Open-vocabulary HOI detector: 同为无检测预训练，但[56]用单层解码+标准匹配；CMD-SE增加MD/CM/SE三层改进
- vs. [34,39] Zero-shot with CLIP: 依赖DETR预训练权重+COCO数据，CMD-SE完全摆脱检测预训练
- vs. DETR-based HOI detectors: 传统闭集方法，不支持开放词汇泛化

**后续方向**:
1. **动态层级选择**: 当前CM为软约束，可探索硬注意力或可学习的层级门控机制
2. **视觉端语义增强**: 当前SE仅增强文本侧，可对视觉特征引入部位级注意力对齐
3. **多模态大模型升级**: 将冻结CLIP替换为可微调的更大视觉语言模型（如LLaVA），联合优化检测与描述生成

**知识库标签**: 模态(vision-language) / 范式(open-vocabulary detection) / 场景(human-object interaction) / 机制(multi-scale feature decoding, soft constraint optimization, LLM-generated prompts) / 约束(no detection pretraining, frozen CLIP backbone)

