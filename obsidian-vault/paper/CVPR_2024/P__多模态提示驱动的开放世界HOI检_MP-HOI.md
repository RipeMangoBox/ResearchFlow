---
title: Open-World Human-Object Interaction Detection via Multi-modal Prompts
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 多模态提示驱动的开放世界HOI检测
- MP-HOI
acceptance: Poster
cited_by: 42
method: MP-HOI
---

# Open-World Human-Object Interaction Detection via Multi-modal Prompts

**Topics**: [[T__Object_Detection]], [[T__Few-Shot_Learning]], [[T__Cross-Modal_Matching]] | **Method**: [[M__MP-HOI]] | **Datasets**: [[D__HICO-DET]] (其他: V-COCO, V-COCO Zero-shot, V-COCO 10% fine-tuning)

| 中文题名 | 多模态提示驱动的开放世界HOI检测 |
| 英文题名 | Open-World Human-Object Interaction Detection via Multi-modal Prompts |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2406.07221) · [Code] · [Project] |
| 主要任务 | Human-Object Interaction (HOI) Detection, Open-World HOI Detection, Zero-Shot HOI Detection |
| 主要 baseline | GEN-VLKT (primary), RLIPv2, CDN, QPIC, HOTR, SCG, IDN, FCL |

> [!abstract] 因为「传统HOI检测器只能识别预定义固定类别，无法处理开放世界中任意文本/视觉提示描述的新颖交互」，作者在「GEN-VLKT」基础上改了「引入Stable Diffusion与CLIP双分支视觉特征、场景感知适配器、对象/交互对比损失，以及统一Magic-HOI训练数据」，在「HICO-DET/V-COCO」上取得「HICO-DET rare mAP 35.48，V-COCO Scenario 1 AP 66.2（SOTA）」

- **HICO-DET**: MP-HOI-L 达到 44.53 mAP（新SOTA），rare 设置提升 6.20 mAP
- **V-COCO**: 100% 训练数据下 Scenario 1/2 AP 为 66.2/67.6，超越 GEN-VLKT 的 62.4/64.5；仅用 10% 数据微调即达 57.7/60.2
- **Zero-shot V-COCO**: Scenario 1/2 AP 37.5/44.2，展示开放世界检测能力

## 背景与动机

Human-Object Interaction (HOI) 检测旨在从图像中识别出〈人，物，交互〉三元组，例如「某人正在骑摩托车」或「某人在切苹果」。然而，现有方法面临一个根本性瓶颈：它们被限制在预定义的固定类别集合内，无法应对真实世界中组合爆炸式的交互描述。例如，同一张野外照片中，一个人可能同时「蹲坐在摩托车上」且「手持头盔」——这种复合交互在标准训练集中极为罕见，传统检测器几乎必然漏检。

现有方法如何处理这一问题？GEN-VLKT [30] 通过视觉-语言知识蒸馏将CLIP的语义知识迁移到检测框架，但其知识蒸馏组件复杂且仍绑定固定类别；RLIPv2 [60] 通过超大规模预训练（2200K+数据）提升泛化，但需昂贵微调且不具备真正的开放词汇推理能力；CDN、QPIC、HOTR 等方法则专注于改进集合预测或关系建模的架构设计，均未突破封闭类别的桎梏。

这些方法的共同短板在于：**视觉表征与语义空间未实现真正的对齐**。它们要么依赖预训练模型的间接知识迁移（如GEN-VLKT的蒸馏），要么仅靠数据堆砌覆盖长尾分布，无法让模型直接理解「任意文本提示」或「视觉示例」所描述的交互概念。更关键的是，Stable Diffusion 等生成模型蕴含丰富的场景先验、CLIP 具备强大的跨模态对齐能力，但这些资源在HOI检测中尚未被有效利用。

本文提出 MP-HOI，首次将多模态提示（文本+视觉）显式注入HOI检测流程，通过双分支特征提取与对比学习实现开放世界推理。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cdd1fe11-e6f8-4e84-9d2d-5ef3b79770fd/figures/fig_001.png)
*Figure: We show (a) the coexisting composited interactions within the same person in an in-the-wild (e.g., A man is squatting on the*



## 核心创新

**核心洞察**：HOI检测的开放世界能力瓶颈在于视觉特征与语义提示的错位对齐，因为Stable Diffusion的内部特征蕴含细粒度场景生成先验、CLIP图像编码器具备跨模态语义对齐能力，从而使「任意文本/视觉提示直接驱动检测」成为可能。

| 维度 | Baseline (GEN-VLKT) | 本文 (MP-HOI) |
|:---|:---|:---|
| 视觉特征 | 标准CNN/Transformer视觉 backbone | Stable Diffusion $F_{sd}$ + CLIP $F_{img}^{clip}$ 双分支 |
| 特征适配 | 无显式适配 | Scene-Aware Adaptor $\alpha$ / $\beta$ 任务对齐 |
| 语义对齐 | 知识蒸馏（间接、固定类别） | 对象/交互对比损失 $L_o^c$ / $L_i^c$（直接、开放词汇） |
| 推理方式 | 固定类别分类 | 任意文本提示 + 视觉提示开放检测 |
| 训练数据 | 单一数据集（HICO-DET） | Magic-HOI 统一六数据集 + SynHOI 合成增强 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cdd1fe11-e6f8-4e84-9d2d-5ef3b79770fd/figures/fig_002.png)
*Figure: Illustration of a) HOIPrompts and b) how HOIPrompts*



MP-HOI 的整体流程可概括为「双分支视觉编码 → 场景感知适配 → 多模态提示融合 → 开放检测输出」：

1. **Stable Diffusion 特征提取器**：输入原始图像，在 $t=0$ 时间步提取内部特征 $F_{sd}$，利用生成模型的场景先验增强稀有类别表征；
2. **CLIP 图像编码器**：输入同一图像，提取 $F_{img}^{clip}$ 特征，提供与语言空间预对齐的语义表征；
3. **Scene-Aware Adaptor $\alpha$**：接收 $F_{sd}$，通过可学习的适配模块将其从生成空间映射到HOI检测空间；
4. **Scene-Aware Adaptor $\beta$**：接收 $F_{img}^{clip}$，类似地对齐CLIP特征与检测任务需求；
5. **多模态提示编码器**：编码任意文本提示（如"Princess Diana and Prince Charles"）和/或视觉提示（参考图像），生成提示嵌入；
6. **HOI Detection Head**：融合适配后的双分支视觉特征与提示嵌入，输出〈人，物，交互〉三元组检测结果。

```
Input Image ──┬──→ Stable Diffusion ──→ F_sd ──→ Adaptor α ──┐
              │                                               ├──→ Detection Head ──→ HOI Triplets
              └──→ CLIP Image Encoder ──→ F_img^clip ──→ Adaptor β ──┘
                                           ↑
Text Prompt / Visual Prompt ──→ Multi-modal Prompt Encoder ──┘
```

## 核心模块与公式推导

### 模块 1: Scene-Aware Adaptor（对应框架图左侧双分支）

**直觉**: 预训练的 Stable Diffusion 和 CLIP 并非为HOI检测任务设计，其特征空间与检测需求存在领域鸿沟，需要轻量适配器进行任务对齐。

**Baseline 公式** (GEN-VLKT): 直接使用 backbone 提取的视觉特征 $F_{backbone}$ 输入检测头，无显式适配：
$$F_{det} = F_{backbone}$$

**变化点**: GEN-VLKT 的标准视觉特征缺乏生成先验和跨模态对齐能力；本文引入双分支并各自添加可学习适配器。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{F}_{sd} = \text{Adaptor}_\alpha(F_{sd}) = F_{sd} + \Delta_\alpha(F_{sd}) \quad \text{残差适配保留生成先验}$$
$$\text{Step 2}: \tilde{F}_{img}^{clip} = \text{Adaptor}_\beta(F_{img}^{clip}) = F_{img}^{clip} + \Delta_\beta(F_{img}^{clip}) \quad \text{残差适配保留语义对齐}$$
$$\text{最终}: F_{fuse} = \text{Fusion}(\tilde{F}_{sd}, \tilde{F}_{img}^{clip})$$

**对应消融**: 去掉 Adaptor $\alpha$ 后 full mAP 从 34.41 降至 34.08（-0.33）；去掉 Adaptor $\beta$ 后降至 33.76（-0.65），说明 CLIP 分支的适配更为关键。

---

### 模块 2: 对象与交互对比损失（对应框架图训练阶段）

**直觉**: 为了让检测器理解「任意提示描述的交互」，需要在对象级别和交互级别分别建立视觉-语言对齐，而非仅依赖最终的分类损失。

**Baseline 公式** (GEN-VLKT): 标准检测损失组合，无显式多模态对齐：
$$\mathcal{L}_{base} = \mathcal{L}_{cls} + \mathcal{L}_{box} + \mathcal{L}_{distill}^{VLKT}$$
其中 $\mathcal{L}_{distill}^{VLKT}$ 为视觉-语言知识蒸馏损失，间接迁移CLIP知识。

**变化点**: 知识蒸馏是单向、间接的；本文改为直接对比学习，让视觉查询与多模态提示在嵌入空间显式对齐。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}_o^c = -\frac{1}{N_o}\sum_{i=1}^{N_o} \log\frac{\exp(\text{sim}(q_o^i, p_o^i)/\tau)}{\sum_{j}\exp(\text{sim}(q_o^i, p_o^j)/\tau)} \quad \text{对象查询-提示对齐}$$
$$\text{Step 2}: \mathcal{L}_i^c = -\frac{1}{N_i}\sum_{k=1}^{N_i} \log\frac{\exp(\text{sim}(q_i^k, p_i^k)/\tau)}{\sum_{l}\exp(\text{sim}(q_i^k, p_i^l)/\tau)} \quad \text{交互查询-提示对齐}$$
$$\text{最终}: \mathcal{L}_{final} = \mathcal{L}_{det} + \lambda_o \mathcal{L}_o^c + \lambda_i \mathcal{L}_i^c$$
符号: $q_o^i, q_i^k$ = 对象/交互视觉查询特征；$p_o^j, p_i^l$ = 多模态提示嵌入；$\tau$ = 温度系数；$\lambda_o, \lambda_i$ = 损失权重。

**对应消融**: 加入 $\mathcal{L}_o^c$ 和 $\mathcal{L}_i^c$ 后，rare mAP 从 31.07 提升至 31.87（+0.80），其中交互对比损失对稀有类别提升最为显著。

---

### 模块 3: 多模态提示推理（对应框架图右侧）

**直觉**: 开放世界检测需要同时支持文本描述的精确语义和视觉示例的直观参考，二者互补可解决文本歧义问题。

**Baseline 公式** (GEN-VLKT): 固定类别嵌入，推理时只能输出预训练类别：
$$\hat{c} = \text{arg}\max_{c \in \mathcal{C}_{fixed}} P(c|F_{det})$$

**变化点**: 固定类别集合限制开放应用；本文将类别嵌入替换为动态提示嵌入。

**本文公式（推导）**:
$$\text{Step 1}: p_{text} = \text{CLIP}_{text}(t) \quad \text{文本提示编码}$$
$$\text{Step 2}: p_{vis} = \text{CLIP}_{image}(v) \quad \text{视觉提示编码}$$
$$\text{Step 3}: p_{fuse} = \text{Combine}(p_{text}, p_{vis}) \quad \text{多模态融合}$$
$$\text{最终}: \hat{y} = \text{arg}\max_{y} \text{sim}(q_y, p_{fuse}) \quad \text{开放词汇匹配}$$

**对应消融**: 仅使用文本提示时 full mAP 为 34.82，加入视觉提示后提升至 35.18（+0.36），验证视觉提示对消除语义歧义的作用。

## 实验与分析



本文在 HICO-DET、V-COCO、SWiG-HOI、HCVRD 四个基准上评估 MP-HOI。核心结果来自 Table 2（HICO-DET）与 Table 5（V-COCO）。在 HICO-DET 上，MP-HOI 使用 Magic-HOI + SynHOI 训练后达到 36.50 full mAP / 35.48 rare mAP / 36.80 non-rare mAP；其中 MP-HOI-L 变体更达到 44.53 mAP，刷新该基准 SOTA。这一成绩超越了 RLIPv2——后者虽在 2200K 数据上预训练，MP-HOI 仍以更少依赖大规模预训练的优势取得更优 rare 性能。在 V-COCO 上，MP-HOI 以 100% 数据训练取得 Scenario 1 AP 66.2 / Scenario 2 AP 67.6，显著领先 GEN-VLKT 的 62.4/64.5，以及 CDN（61.7/63.8）、QPIC（58.8/61.0）等专家模型。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cdd1fe11-e6f8-4e84-9d2d-5ef3b79770fd/figures/fig_003.png)
*Figure: Overview of MP-HOI, comprising three components: Representative Feature Encoder, Sequential Instance and Interaction*



更具说服力的是低数据与零样本场景：V-COCO 上仅使用 10% 训练数据微调，MP-HOI 即达到 57.7/60.2，超过部分全量训练的强基线；零样本设置下仍取得 37.5/44.2，证明开放世界推理能力。Figure 5 与 Figure 6 展示了开放世界定性结果，模型可接受「Princess Diana and Prince Charles」等复杂文本描述，或结合视觉提示（交互定义）与文本提示（对象定义）进行灵活检测。



消融实验（对应 Table 3/4 类表格）揭示了各组件的贡献：去掉 Stable Diffusion 特征 $F_{sd}$ 后 rare mAP 从 31.29 骤降至 29.63（-1.66），验证生成先核对稀有类别的关键作用；去掉 CLIP 特征 $F_{img}^{clip}$ 后 full mAP 从 34.41 降至 32.92（-1.49），影响非稀有性能；去掉 Magic-HOI 统一数据后 full mAP 降 0.75（35.93→35.18）；去掉 SynHOI 合成数据后降 0.57（36.50→35.93），稀有类别受损更明显。值得注意的是，本文 baseline GEN-VLKT 被刻意移除了其核心知识蒸馏组件，这一处理可能削弱对比的公平性；同时 RLIPv2 的 HICO-DET 具体数值未在提供文本中明确列出，仅声明「超越」。此外，与更近期的开放词汇检测方法（如 GLIP、Grounding DINO 等）的直接对比缺失，限制了结论的完备性。

## 方法谱系与知识库定位

**方法族**: HOI Detection → 开放词汇/开放世界检测

**父方法**: GEN-VLKT [30]（CVPR 2022）。MP-HOI 继承其检测头设计，但彻底改造了特征提取、训练目标与推理范式。

**变更槽位**:
- **architecture**: 标准视觉 backbone → Stable Diffusion $F_{sd}$ + CLIP $F_{img}^{clip}$ 双分支 + Scene-Aware Adaptor $\alpha$/$\beta$
- **objective**: 知识蒸馏 → 对象/交互对比损失 $L_o^c$ / $L_i^c$
- **data_curation**: HICO-DET 单数据集 → Magic-HOI 六数据集统一 + SynHOI 合成增强
- **inference**: 固定类别 → 任意文本/视觉提示开放检测

**直接基线差异**:
- **GEN-VLKT**: 本文移除其知识蒸馏，改为直接多模态对比对齐；双分支特征替代单分支
- **RLIPv2**: 不依赖 2200K 预训练，通过架构设计实现 competitive 性能
- **CDN/QPIC/HOTR 等**: 本文唯一实现真正的开放词汇推理，支持任意提示

**后续方向**:
1. 将多模态提示机制扩展至视频 HOI 检测，利用时序一致性增强开放世界推理
2. 探索更高效的适配器设计（如 LoRA）替代全量微调，降低 Stable Diffusion/CLIP 分支的计算开销
3. 构建更大规模的开放世界 HOI 基准，验证模型在真实长尾分布中的泛化边界

**标签**: 模态=vision+language | 范式=open-vocabulary detection | 场景=human-object interaction | 机制=contrastive learning + multi-modal prompting | 约束=需预训练 SD 与 CLIP 权重

