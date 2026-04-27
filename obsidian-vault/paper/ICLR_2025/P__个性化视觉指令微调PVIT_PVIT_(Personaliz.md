---
title: Personalized Visual Instruction Tuning
type: paper
paper_level: C
venue: ICLR
year: 2025
paper_link: null
aliases:
- 个性化视觉指令微调PVIT
- PVIT (Personaliz
- PVIT (Personalized Visual Instruction Tuning)
acceptance: Poster
cited_by: 131
code_url: https://github.com/sterzhang/PVIT
method: PVIT (Personalized Visual Instruction Tuning)
---

# Personalized Visual Instruction Tuning

[Code](https://github.com/sterzhang/PVIT)

**Topics**: [[T__Visual_Question_Answering]], [[T__Captioning]] | **Method**: [[M__PVIT]] | **Datasets**: P-Bench MC questions

| 中文题名 | 个性化视觉指令微调PVIT |
| 英文题名 | Personalized Visual Instruction Tuning |
| 会议/期刊 | ICLR 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.07113) · [Code](https://github.com/sterzhang/PVIT) · [Project](https://github.com/sterzhang/PVIT) |
| 主要任务 | 个性化视觉问答（Personalized Visual Question Answering）、特定人物识别与不可回答查询拒绝 |
| 主要 baseline | LLaVA-7B, Qwen-VL-7B, LLaVA-OneVision-7B, VILA1.5-7B, InternVL-Chat-V1.5-26B, Deepseek-VL-7b-chat, mPLUG-Owl2 |

> [!abstract] 因为「现有MLLMs无法准确识别图像中特定人物且容易在目标人物缺席时幻觉回答」，作者在「LLaVA」基础上改了「三阶段个性化数据构建流程（含PhotoMaker人脸增强与对抗不可回答样本）与个性化视觉指令微调训练策略」，在「P-Bench」上取得「单图场景98.71%准确率，相比最强baseline LLaVA-OneVision-7B提升+10.0」

- **P-Bench MC单图准确率**: P-LLaVA 98.71% vs. LLaVA-OneVision-7B 88.72%，提升 **+10.0**
- **P-Bench MC多图（≥4人）准确率**: P-LLaVA 95.32% vs. LLaVA-OneVision-7B 76.31%，提升 **+19.0**
- **不可回答查询拒绝准确率**: 对抗样本训练后 99.78%，去掉对抗样本后暴跌至 1.12%，**Δ-98.66**

## 背景与动机

现有多模态大语言模型（MLLMs）如LLaVA、Qwen-VL等已能处理广泛的视觉理解任务，但在一个看似简单的场景面前却集体失效：用户上传一张包含多人的聚会照片，询问"穿红色衣服的是不是Alice？"——模型要么认错人，要么在Alice根本不在场时胡乱编造答案。这种**个性化视觉问答**（Personalized Visual Question Answering）要求模型不仅能理解场景，还能**精准识别特定个体**并**在目标人物缺席时拒绝回答**，而非幻觉生成。

现有方法如何处理这一任务？**LLaVA**（Liu et al., 2023a）作为视觉指令调优的代表，使用通用图像-文本对训练，缺乏针对特定人物识别的专门设计；**Qwen-VL-7B**和**LLaVA-OneVision-7B**虽支持多图输入，但在场景人物增多时性能显著衰减；**InternVL-Chat-V1.5-26B**参数量达26B，却因架构与数据不匹配，在个性化任务上反而落后于7B模型。这些方法的共同短板在于：**训练数据缺乏人物级别的精细标注、没有针对"不可回答"场景的拒答能力训练、以及人脸多样性不足导致的泛化性差**。

具体而言，当场景图像中人物数量从1人增至4人及以上时，现有SOTA MLLMs的准确率急剧下降（LLaVA-OneVision-7B从88.72%降至76.31%），且几乎完全无法在目标人物缺席时正确拒绝回答。本文正是针对这一空白，提出了一套从数据构建到训练策略的完整个性化视觉指令微调方案PVIT，并构建了专门的评测基准P-Bench和大规模数据集PVIT-3M。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e37d857-d95d-43af-9c04-f00eddcc30bf/figures/Figure_3.png)
*Figure 3: Statistics of PVIT-3M, a large-scale personalized visual instruction tuning dataset.*



## 核心创新

核心洞察：**个性化视觉理解的关键瓶颈不在模型架构，而在训练数据的构造方式**——因为现有通用视觉指令数据缺乏"人物-场景-关系"的三元结构化信息，且未覆盖"目标人物缺席"的对抗场景，从而使MLLMs既无法建立稳定的人物身份表征，也无法学会合理的认知拒答。

| 维度 | Baseline (LLaVA) | 本文 (PVIT) |
|:---|:---|:---|
| 数据构造 | 通用图像-文本对，无人物层级信息 | 三阶段流水线：整体场景信息→特定人物信息→融合生成个性化描述 |
| 人脸多样性 | 原始人脸图像，身份覆盖有限 | PhotoMaker人脸增强，显著扩充输入个体多样性 |
| 不可回答处理 | 无专门训练，易幻觉生成 | 对抗样本注入：场景图不含目标人物时训练模型拒绝回答 |
| 评测基准 | 通用VQA基准（如VQAv2） | 自建P-Bench：MC问答+个性化描述，专门评测人物级理解 |

与单纯扩大模型规模或改进架构不同，PVIT通过**数据驱动的精细化设计**，在保持7B参数量的同时实现了对26B模型的显著超越。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e37d857-d95d-43af-9c04-f00eddcc30bf/figures/Figure_1.png)
*Figure 1 (pipeline): The Personalized Visual Instruction Tuning (PVT) framework consists of three phases.*



PVIT框架包含**数据构造**与**模型训练**两大阶段，数据侧采用三阶段生成流水线，训练侧基于LLaVA-7B进行个性化视觉指令微调，最终产出P-LLaVA模型。

**数据构造流水线（三阶段）**：
1. **整体信息生成（Holistic Information Generation）**：输入场景图像，输出全局场景描述（人物关系、活动、环境等上下文），建立"场景骨架"；
2. **特定人物信息生成（Specific Person Information Generation）**：输入裁剪的人物面部图像，输出该个体的外貌特征、穿着、姿态等细粒度描述，建立"身份锚点"；
3. **信息融合（Information Fusion）**：将前两阶段输出融合为**个性化描述**——既包含场景中"谁在哪里做什么"的全局信息，又嵌入"目标人物的具体外貌特征"的局部信息，形成人物-场景关联的完整表征。

**数据增强与对抗组件**：
- **PhotoMaker人脸增强**：对人物面部图像进行风格/姿态多样化增强，扩充训练身份的视觉多样性；
- **对抗样本生成（Adversarial Sample Generation）**：构造"场景图中不含目标人物"的负样本，配套"无法回答"的标准回复，训练模型的拒答能力。

**模型训练**：
- **P-LLaVA训练**：以LLaVA-7B为基座，在PVIT-3M数据集（300万样本，多类别覆盖）上进行个性化视觉指令微调，支持多图输入与不可回答查询处理。

```
场景图像 ──→ [整体信息生成] ──┐
                               ├──→ [信息融合] ──→ 个性化描述 ──┐
人物图像 ──→ [PhotoMaker增强] ──→ [特定人物信息生成] ──┘      │
                                                              ├──→ PVIT-3M ──→ P-LLaVA训练
负样本场景 ──→ [对抗样本生成] ──→ "无法回答"标注 ──────────────┘
```

## 核心模块与公式推导

本文方法以数据工程创新为核心，未引入复杂的数学损失函数重构，因此本节重点阐述**三阶段数据生成**与**对抗训练机制**的设计原理及其与标准指令调优的对比。

### 模块 1: 三阶段信息融合生成（对应框架图 左侧数据流水线）

**直觉**: 通用视觉指令数据将"图像-描述"视为扁平映射，但个性化任务需要解耦"场景上下文"与"人物身份"再重新关联，否则模型难以在多人场景中定位特定个体。

**Baseline 形式 (标准视觉指令调优, LLaVA)**:
$$\mathcal{L}_{\text{base}} = -\sum_{t} \log P_\theta(x_t | x_{<t}, \mathbf{v}, \mathbf{q})$$
其中 $\mathbf{v}$ 为视觉特征，$\mathbf{q}$ 为问题文本，$x_t$ 为生成token。数据构造为直接配对：图像 → 通用描述/回答。

**变化点**: 标准LLaVA的数据构造缺乏**显式的人物身份建模**。当问题涉及"Alice是否在图中"时，模型无先验的Alice外貌表征可比对，只能依赖模糊的视觉-语言关联，导致多人场景下的身份混淆。

**本文设计（分步推导）**:

$$\text{Step 1: 整体信息提取} \quad \mathbf{H} = \text{GPT-4V}(\mathbf{I}_{\text{scene}}; \mathbf{p}_{\text{holistic}}) \quad \text{生成场景级描述，建立人物关系拓扑}$$

$$\text{Step 2: 特定人物信息提取} \quad \mathbf{S}_i = \text{GPT-4V}(\text{Crop}(\mathbf{I}_{\text{scene}}, \mathbf{b}_i); \mathbf{p}_{\text{person}}) \quad \text{对第}i\text{个人物裁剪图提取外貌锚定特征}$$

$$\text{Step 3: 信息融合} \quad \mathbf{D}_i = \text{GPT-4V}(\mathbf{H}, \mathbf{S}_i; \mathbf{p}_{\text{fuse}}) \quad \text{将人物}i\text{的外貌嵌入场景上下文，生成个性化描述}$$

其中 $\mathbf{p}_{\text{holistic}}, \mathbf{p}_{\text{person}}, \mathbf{p}_{\text{fuse}}$ 为各阶段专用提示词模板（对应文中Table 6-8），确保生成内容的结构一致性。

**对应消融**: Table 4（数据消融）显示，完整三阶段数据相比简化构造在各项任务上均有提升，其中对抗样本与人脸增强的贡献最为关键。

### 模块 2: PhotoMaker人脸增强与对抗不可回答训练（对应框架图 中下支路与右下支路）

**直觉**: 模型需要足够 diverse 的人脸样本以建立鲁棒的身份表征，同时必须学会"知之为知之，不知为不知"——在目标人物缺席时不臆测。

**Baseline 形式 (标准数据增强)**: 原始图像随机裁剪、颜色抖动等，无针对人脸的专门增强，亦无负样本训练。

**变化点**: 
- **人脸多样性不足**：标准增强不改变人物身份的外观本质特征，模型见过的人物身份数量有限；
- **无拒答机制**：所有训练样本均为"可回答"，导致测试时遇到不可回答情况必然幻觉。

**本文设计（推导）**:

$$\text{Step 1: PhotoMaker增强} \quad \mathbf{I}_{\text{face}}^{(k)} = \text{PhotoMaker}(\mathbf{I}_{\text{face}}; \mathbf{c}_k), \quad k=1,...,K \quad \text{生成}K\text{种风格/姿态变体，扩充身份视觉空间}$$

$$\text{Step 2: 对抗负样本构造} \quad (\mathbf{I}_{\text{scene}}^-, \mathbf{q}_{\text{target}}) \rightarrow \mathbf{y}_{\text{reject}} = \text{"I cannot find [Target] in the image."} \quad \text{目标人物} \notin \text{场景时强制拒答}$$

$$\text{最终训练目标}: \mathcal{L}_{\text{PVIT}} = -\underbrace{\sum_{(I,q,y^+) \in \mathcal{D}_{\text{pos}}} \log P_\theta(y^+|v(I), q)}_{\text{正样本：标准个性化回答}} - \underbrace{\sum_{(I^-,q,y^-) \in \mathcal{D}_{\text{neg}}} \log P_\theta(y^-|v(I^-), q)}_{\text{负样本：拒答模式学习}}$$

其中 $\mathcal{D}_{\text{pos}}$ 为含目标人物的PVIT-3M正样本，$\mathcal{D}_{\text{neg}}$ 为对抗构造的负样本，$y^-$ 统一为拒绝回答模板。

**对应消融**: Table 4 显示，去掉PhotoMaker人脸增强（Aug）后Augment准确率从95.32%降至88.43%，**Δ-6.89**；去掉对抗样本（Adv）后不可回答准确率从99.78%暴跌至1.12%，**Δ-98.66**，证明两项组件分别对泛化性与拒答能力具有决定性作用。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e37d857-d95d-43af-9c04-f00eddcc30bf/figures/Table_1.png)
*Table 1 (quantitative): MC questions on P-Bench. P-LLaVA trained with PVIT significantly outperforms other MLLMs.*



本文在自建的**P-Bench**基准上进行评测，该基准包含多选题（MC）与个性化描述两类任务，覆盖单图/多图、不同人物数量（Cnt=1,2,3,≥4）及不可回答场景。
![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e37d857-d95d-43af-9c04-f00eddcc30bf/figures/Table_3.png)
*Table 3 (ablation): The performance of MC questions with more images.*



主实验结果如Table 1与Table 3所示。在**单图场景（Cnt=1）**中，P-LLaVA达到**98.71%**的MC准确率，相比最强7B baseline LLaVA-OneVision-7B的88.72%提升**+10.0**，且显著优于所有对比模型包括26B的InternVL-Chat-V1.5-26B（56.21%）。随着场景复杂度增加——人物数量增至2人、3人、≥4人——现有方法性能急剧衰减：LLaVA-OneVision-7B分别降至83.23%、80.37%、76.31%，而P-LLaVA保持**95.03%、94.9%、95.32%**的高准确率，与最强baseline的差距分别拉大至**+11.8、+14.5、+19.0**。这一趋势揭示了核心发现：**人物数量增加对现有MLLMs是致命挑战，但对经PVIT训练的P-LLaVA影响甚微**，验证了个性化数据构造对复杂场景鲁棒性的关键价值。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e37d857-d95d-43af-9c04-f00eddcc30bf/figures/Table_4.png)
*Table 4 (ablation): Two segmentation losses on ME questions.*



消融实验（Table 3/Table 4）进一步量化各组件贡献。**PhotoMaker人脸增强（Aug）**的移除导致Augment准确率下降6.89个百分点（95.32%→88.43%），表明人脸多样性直接决定模型对陌生身份的泛化能力。**对抗样本（Adv）**的移除则造成灾难性后果：不可回答查询的准确率从99.78%崩溃至1.12%，**Δ-98.66**，说明没有显式负样本训练，模型完全不具备拒答意识，必然幻觉生成。这一对比凸显了对抗训练在可信AI中的不可替代性。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e37d857-d95d-43af-9c04-f00eddcc30bf/figures/Figure_4.png)
*Figure 4 (qualitative): Qualitative examples of P-LLaVA results including user queries, images, and model responses.*



定性结果（Figure 4）展示P-LLaVA在实际查询中的响应质量，包括正确识别特定人物、处理多图输入及合理拒绝不可回答问题。

**公平性检验**：本文对比的baselines均为开源7B级别模型（除InternVL-Chat-V1.5-26B），与P-LLaVA规模大体可比；但未包含GPT-4V/GPT-4o、Gemini Pro Vision等商业API模型，以及专门面向人物识别的视觉模型。P-Bench由作者自建，存在潜在benchmark bias。此外，InternVL-Chat-V1.5-26B参数量远超其他模型却表现最差，提示其架构或训练目标与个性化任务存在根本性错配，而非公平规模比较。训练计算成本（GPU类型、时间）未在文中披露。

## 方法谱系与知识库定位

**方法家族**: 视觉指令调优（Visual Instruction Tuning）→ 个性化视觉理解

**父方法**: **LLaVA**（Liu et al., 2023a）—— 本文直接在LLaVA-7B基础上进行slot-level改造，未修改视觉编码器（CLIP ViT-L/14）或语言模型（Vicuna-7B）的架构，全部创新集中于数据流水线与训练策略。

**改动插槽**:
| 插槽 | 改动内容 |
|:---|:---|
| data_pipeline | 通用图像-文本对 → 三阶段个性化数据构造（整体/人物/融合）+ PhotoMaker增强 + 对抗负样本 |
| training_recipe | 标准视觉指令调优 → 个性化视觉指令调优，支持多图输入与不可回答处理 |
| inference_strategy | 单图/多图通用推理 → 多图个性化人物识别 + 认知拒答 |

**直接Baselines与差异**:
- **LLaVA-OneVision-7B**: 同为LLaVA系列多图扩展，但无个性化数据构造，多人场景性能衰减显著
- **Qwen-VL-7B / VILA1.5-7B / Deepseek-VL-7b-chat / mPLUG-Owl2**: 通用MLLMs，缺乏人物级身份建模与拒答机制
- **InternVL-Chat-V1.5-26B**: 规模更大但架构不匹配，反证数据策略优于纯参数扩张

**后续方向**:
1. **动态身份库扩展**：将PVIT从封闭集人物识别扩展到开放世界持续学习新身份
2. **多模态身份融合**：结合语音、文本描述等多模态线索增强人物识别鲁棒性
3. **安全与隐私**：PhotoMaker增强与对抗训练的边界——防止模型被恶意用于深度伪造或绕过身份验证

**标签**: 
- **modality**: 视觉-语言（vision-language）
- **paradigm**: 指令调优（instruction tuning）
- **scenario**: 个性化理解（personalized understanding）、人物识别（person recognition）
- **mechanism**: 数据工程驱动（data-centric）、对抗训练（adversarial training）、认知拒答（knowledge rejection）
- **constraint**: 7B参数效率、多图输入、不可回答处理

