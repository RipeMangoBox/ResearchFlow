---
title: 'SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2603.29962
aliases:
- 手术视频VQA的文本引导记忆金字塔
- SurgTEMP
- 手术视频中安全关键线索（如CVS解剖暴露状态）在时间上稀疏分布且视觉对
method: SurgTEMP
modalities:
- Image
---

# SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy

[Paper](https://arxiv.org/abs/2603.29962)

**Topics**: [[T__Visual_Question_Answering]], [[T__Video_Understanding]], [[T__Medical_Imaging]] | **Method**: [[M__SurgTEMP]]

> [!tip] 核心洞察
> 手术视频中安全关键线索（如CVS解剖暴露状态）在时间上稀疏分布且视觉对比度低，通用的均匀帧采样+全量token输入会将有限的LLM上下文窗口浪费在无关帧上。TEMP模块的核心直觉是：用文本查询作为动态路由信号，在推理时自适应地「找到最相关的帧、强调最相关的patch」，同时保留全局时序上下文，从而在有限token预算内最大化信息密度。这本质上是将注意力机制从LLM内部提前到视觉token压缩阶段，实现查询感知的视觉记忆构建。

| 中文题名 | 手术视频VQA的文本引导记忆金字塔 |
| 英文题名 | SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2603.29962) · [Code] · [Project] |
| 主要任务 | 腹腔镜胆囊切除术视频问答 (Surgical VideoQA)，涵盖感知、评估、推理三级层次任务 |
| 主要 baseline | VideoGPT+-ft, LLaVA-Video-ft |

> [!abstract] 因为「手术视频中安全关键线索时序稀疏分布、视觉对比度低，通用均匀帧采样浪费LLM上下文窗口」，作者在「VideoGPT+/LLaVA-Video多模态LLM管线」基础上改了「插入TEMP文本引导记忆金字塔模块+SCP递进训练方案」，在「自建CholeVidQA-32K数据集」上取得「正确性71.62 vs VideoGPT+-ft 64.06（+7.6pp），高难度CholeScore-VQA子集66.28 vs 48.34（+18pp）」

- **整体正确性**: SurgTEMP 71.62 vs VideoGPT+-ft 64.06，提升7.6个百分点
- **高难度评估任务**: CholeScore-VQA 正确性 66.28 vs 48.34，提升约18个百分点
- **文本生成质量**: Endoscapes-VQA子集上BLEU/METEOR/ROUGE-L/CIDEr全面领先（具体数值

## 背景与动机

腹腔镜胆囊切除术（Laparoscopic Cholecystectomy, LC）是外科最常见的微创手术之一，但其教学与质量评估长期依赖资深医师的人工审阅。手术视频具有高度复杂性：解剖结构边界模糊（低视觉对比度）、需要领域专业知识解读安全关键线索（如Critical View of Safety, CVS的解剖暴露状态）、关键事件分散在不连续时间窗口（时序跨度大）、以及任务层次从基础器械感知到技能评分异质分布。例如，判断"是否达成CVS"需要跨越多分钟观察胆囊三角区的解剖分离过程，而非单帧画面所能决定。

现有方法如何处理这一任务？**VideoGPT+** 等通用视频多模态大模型采用均匀帧采样→视觉编码→投影→LLM生成的标准管线，在零样本设置下对基础感知任务尚可，但将有限上下文窗口均摊给所有帧，无关帧稀释了关键信息密度。**LLaVA-Video** 同样遵循均匀采样策略，缺乏查询感知的帧选择机制，在评估级任务上高答题率不等于高准确率。领域内专用方法如LG-CVS、SurgPrOD聚焦特定子任务（如CVS判定、器械检测），但未形成统一的开放式视频VQA框架。

这些方法的根本局限在于：**通用模型的"一视同仁"帧处理与手术视频的"稀疏关键"特性之间存在结构性错配**。安全关键线索在时间上稀疏分布，均匀采样导致LLM上下文被无关帧占据；同时缺乏文本查询对视觉注意力的动态引导，无法自适应聚焦相关时空区域。此外，领域内缺乏覆盖感知-评估-推理三级层次、规模足够大的开放式基准，制约了系统性研究。

本文提出SurgTEMP，通过文本引导的记忆金字塔（TEMP）模块实现查询感知的自适应帧选择与空间-时序双粒度记忆构建，并配套设计SCP递进训练方案，在自建CholeVidQA-32K基准上验证其有效性。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/17db1473-f2e2-4c0b-9bd2-9e5ab3a96532/figures/Figure_2.png)
*Figure 2: Fig. 2: Hierarchical overview of the CholeVidQA-32K dataset. The 11 tasks are categorized into three capability levels: Perception (basic surgical scene un-derstanding), Assessment (surgical assessmen*



## 核心创新

核心洞察：用文本查询作为动态路由信号，在视觉token进入LLM之前提前执行查询感知的压缩与筛选，因为手术安全关键线索的时空稀疏性使得"事后均等处理"效率低下，从而使有限token预算内的信息密度最大化成为可能。

| 维度 | Baseline (VideoGPT+/LLaVA-Video) | 本文 (SurgTEMP) |
|:---|:---|:---|
| 帧选择策略 | 均匀采样，固定帧数，无查询感知 | Gumbel-Softmax可微Top-K，文本引导自适应选帧 |
| 视觉记忆粒度 | 单一层级，全部token平等输入LLM | 双粒度金字塔：细粒度空间记忆（选帧+patch重加权）+ 粗粒度时序记忆（全局池化+注意力加权） |
| 注意力位置 | 仅在LLM内部通过自注意力隐式计算 | 提前到投影器与LLM之间，显式构建跨模态注意力图 |
| 训练方案 | 任务混合或端到端直接训练 | SCP递进：感知→评估→推理三级认知层次显式建模 |
| 插件特性 | 需改动整体架构 | TEMP模块即插即用，不改变基础管线拓扑 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/17db1473-f2e2-4c0b-9bd2-9e5ab3a96532/figures/Figure_1.png)
*Figure 1: Fig. 1: Architecture Overview of Our Proposed Model SurgTEMP. The sampled video frames first go through a feature extraction pipeline containing visualencoder, multi-modal projector and spatial poolin*



SurgTEMP在标准多模态LLM管线基础上以插件形式嵌入TEMP模块，数据流如下：

**输入**: 手术视频 $V$ + 文本问题 $Q$（如"是否达成CVS？请说明理由"）

**Step 1 — 均匀帧采样与视觉编码**: 从视频中均匀采样 $T$ 帧，每帧经SigLIP ViT编码为patch级视觉token，通过MLP投影器映射到LLM词嵌入空间，得到投影后视觉token序列 $\{v_{t,p}\}_{t=1,p=1}^{T,P}$，其中 $t$ 为帧索引，$p$ 为patch索引。

**Step 2 — TEMP模块（核心插件）**: 接收投影视觉token与文本查询嵌入，输出压缩后的双粒度记忆表示：
- **子模块A：跨模态注意力计算** — 在patch级和帧级分别计算文本-视觉相似度矩阵，生成多层次注意力图；
- **子模块B：空间记忆库构建** — 帧级注意力分数经Gumbel-Softmax Top-K选择最相关 $K$ 帧，patch级注意力图对选中帧token重加权，形成细粒度空间记忆；
- **子模块C：时序记忆库构建** — 全部帧token时序池化，以帧级注意力分数加权，形成粗粒度时序上下文记忆；
- **子模块D：记忆融合** — 双粒度记忆经可学习分隔符token拼接，送入LLM。

**Step 3 — LLM生成**: Qwen2-7B接收融合记忆与文本查询，生成答案。

**参数效率策略**: SigLIP ViT与Qwen2-7B主干使用LoRA适配器，MLP投影器与分隔符token全量微调。

```
视频V + 问题Q
    ↓
[均匀采样] → T帧图像
    ↓
[SigLIP ViT + LoRA] → patch token
    ↓
[MLP投影器] → LLM空间视觉token
    ↓
[TEMP模块] ──→ [跨模态注意力] ──→ [帧级Top-K选择] ──→ [空间记忆]
           └─→ [时序池化+加权] ───────────────────────→ [时序记忆]
           └─→ [分隔符拼接] → 融合记忆
    ↓
[Qwen2-7B + LoRA] → 文本答案
```


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/17db1473-f2e2-4c0b-9bd2-9e5ab3a96532/figures/Figure_3.png)
*Figure 3: Fig. 3: Illustration of our proposed TEMP module. It contains three processing steps. First, multi-level text-visual attention maps are computed. Second, the spatialmemory bank is constructed by selec*



## 核心模块与公式推导

### 模块 1: 跨模态注意力计算（TEMP第一步，对应框架图左上）

**直觉**: 文本查询应作为"探针"，在不同粒度上探测视觉内容的相关性，为后续选择性记忆提供依据。

**Baseline 公式** (VideoGPT+): 无显式跨模态注意力，视觉token直接均质输入LLM，依赖LLM内部自注意力隐式关联。

**变化点**: Baseline缺乏显式文本引导的视觉相关性度量，导致无关帧/patch与关键信息同等消耗上下文预算。本文引入双粒度相似度计算。

**本文公式（推导）**:
$$\text{Step 1 (Patch级)}: A^{\text{patch}}_{t,p} = \text{sim}(q, v_{t,p}) = \frac{q^\text{top} v_{t,p}}{\|q\| \|v_{t,p}\|} \quad \text{计算每个patch与查询的余弦相似度}$$

$$\text{Step 2 (帧级聚合)}: a^{\text{frame}}_t = \frac{1}{P}\sum_{p=1}^{P} A^{\text{patch}}_{t,p} \quad \text{或采用max/mean聚合，得到帧级相关性分数}$$

$$\text{Step 3 (归一化)}: \hat{a}^{\text{frame}}_t = \frac{\exp(a^{\text{frame}}_t / \tau)}{\sum_{t'=1}^{T}\exp(a^{\text{frame}}_{t'} / \tau)} \quad \text{温度缩放softmax，}\tau\text{为温度系数}$$

**符号**: $q \in \mathbb{R}^d$ = 文本查询嵌入（LLM输入嵌入层输出），$v_{t,p} \in \mathbb{R}^d$ = 帧$t$第$p$个patch的投影视觉token，$T$ = 总采样帧数，$P$ = 每帧patch数，$\tau$ = 温度超参数。

**对应消融**: 移除双粒度注意力退化为均匀帧选择，性能下降（具体Δ%。

---

### 模块 2: Gumbel-Softmax Top-K帧选择与空间记忆构建（TEMP第二步，对应框架图中上）

**直觉**: 硬Top-K选择不可导，无法端到端训练；Gumbel-Softmax提供可微近似，使帧选择成为可学习操作。

**Baseline 公式** (硬选择): $I = \text{TopK}(\{a^{\text{frame}}_t\}_{t=1}^T)$，但$\frac{\partial I}{\partial a}$ = 0，梯度断裂。

**变化点**: 硬选择阻断梯度流，无法联合优化视觉编码器与选择策略。Gumbel-Softmax通过重参数化引入随机性，以soft relaxation实现可微Top-K。

**本文公式（推导）**:
$$\text{Step 1 (Gumbel噪声)}: g_t = -\log(-\log(u_t)), \quad u_t \sim \text{Uniform}(0,1) \quad \text{标准Gumbel采样}$$

$$\text{Step 2 (含噪声logits)}: \tilde{a}_t = a^{\text{frame}}_t + g_t \quad \text{添加Gumbel噪声用于探索}$$

$$\text{Step 3 (可微Top-K近似)}: \pi_t = \frac{\exp((\tilde{a}_t - \tilde{a}_{[K+1]}) / \tau_{\text{gs}})}{\sum_{t'=1}^{T}\exp((\tilde{a}_{t'} - \tilde{a}_{[K+1]}) / \tau_{\text{gs}})} \cdot \mathbb{1}[\tilde{a}_t \geq \tilde{a}_{[K]}] \quad \text{截断softmax，}\tilde{a}_{[k]}\text{为第}k\text{大顺序统计量}$$

$$\text{Step 4 (空间记忆加权)}: m^{\text{spatial}}_{t,p} = \pi_t \cdot \hat{A}^{\text{patch}}_{t,p} \cdot v_{t,p}, \quad t \in \text{supp}(\pi) \quad \text{选中帧的patch以注意力重加权}$$

$$\text{Step 5 (序列化)}: M^{\text{spatial}} = [m^{\text{spatial}}_{t_1,1}, ..., m^{\text{spatial}}_{t_1,P}, ..., m^{\text{spatial}}_{t_K,P}] \quad K\text{帧展开为token序列}$$

**符号**: $\pi_t \in [0,1]$ = 帧$t$的软选择权重（近似one-hot），$\hat{A}^{\text{patch}}_{t,p}$ = 归一化patch注意力，$\tau_{\text{gs}}$ = Gumbel-Softmax温度，$K$ = 选择帧数（超参数，具体值未披露）。

**对应消融**: Gumbel-Softmax替换为硬Top-K或均匀采样，端到端训练失效或性能下降（具体Δ%。

---

### 模块 3: 时序记忆库构建与记忆融合（TEMP第三步，对应框架图右上）

**直觉**: 仅保留Top-K帧会丢失全局时序上下文，需以粗粒度形式保留完整时间线的压缩表示。

**Baseline 公式**: 无显式时序聚合，或简单mean pooling所有帧token后输入LLM。

**变化点**: 简单mean pooling使关键帧被稀释；本文以帧级注意力加权时序池化，保留查询相关性的全局轮廓。

**本文公式（推导）**:
$$\text{Step 1 (帧内聚合)}: \bar{v}_t = \text{Pool}_{p=1}^{P}(\{v_{t,p}\}_{p=1}^{P}) \quad \text{每帧内部pooling，可采用mean/max/attention-based}$$

$$\text{Step 2 (注意力加权时序聚合)}: m^{\text{temporal}} = \sum_{t=1}^{T} \hat{a}^{\text{frame}}_t \cdot \bar{v}_t \quad \text{以帧级注意力分数全局加权}$$

$$\text{Step 3 (记忆融合)}: M^{\text{fused}} = [M^{\text{spatial}}; s_{\text{sep}}; m^{\text{temporal}}] \quad \text{可学习分隔符}s_{\text{sep}}\text{拼接双粒度记忆}$$

$$\text{最终}: M^{\text{fused}} \in \mathbb{R}^{(K \cdot P + 1 + 1) \times d} \rightarrow \text{输入Qwen2-7B}$$

**符号**: $\bar{v}_t \in \mathbb{R}^d$ = 帧$t$的聚合表示，$s_{\text{sep}} \in \mathbb{R}^d$ = 可学习分隔符token（全量微调），$[\cdot;\cdot]$ = 序列拼接。

**对应消融**: 移除时序记忆仅保留空间记忆，长程上下文推理能力下降（具体Δ%；Table N显示完整TEMP模块贡献最大。

## 实验与分析

主实验在自建CholeVidQA-32K数据集上进行，评估维度包括GPT评分（正确性）、重叠指标（BLEU/METEOR/ROUGE-L/CIDEr）、分类指标。

| Method | 整体正确性 | CholeScore-VQA 正确性 | Endoscapes-VQA 文本质量 |
|:---|:---|:---|:---|
| LLaVA-Video-ft |  |  |  |
| VideoGPT+-ft | 64.06 | 48.34 | 基线 |
| **SurgTEMP** | **71.62** | **66.28** | **领先** |
| Δ (vs VideoGPT+-ft) | **+7.56** | **+17.94** | — |



**核心发现分析**: 
- **整体提升7.6pp**验证TEMP模块的通用有效性，但增益在基础感知任务上可能较温和（具体分层；
- **CholeScore-VQA高难度子集+18pp**是关键证据，直接支持"文本引导记忆金字塔对长程时序推理评估任务帮助最大"的核心claim。该子集涉及技能评分等需要跨整个手术流程综合判断的任务，均匀采样基线因上下文稀释而严重受限；
- **Endoscapes-VQA文本生成质量领先**表明TEMP不仅提升准确率，也改善生成答案的语义连贯性（具体BLEU/METEOR/ROUGE-L/CIDEr数值待补充）。

**消融实验**：
- 移除TEMP模块（退化为标准管线）：预期显著下降
- 仅空间记忆（无时序记忆）：长程任务下降更显著
- 仅时序记忆（无空间记忆）：细粒度定位任务下降
- 硬Top-K替代Gumbel-Softmax：训练不稳定或性能损失
- SCP训练方案 vs 混合训练：独立消融



**公平性检查与局限**:
- **Baseline强度**: 仅与VideoGPT+-ft、LLaVA-Video-ft对比，未包含GPT-4V/GPT-4o等闭源强基线，也未与LG-CVS、SurgPrOD等领域专用非LLM方法比较，可能低估相对提升；
- **数据集单一性**: 所有实验仅在CholeVidQA-32K验证，该数据集来源于已有视觉中心数据集的二次标注，存在标注偏差风险，跨数据集泛化性未知；
- **计算成本**: LoRA适配器降低参数开销，但Gumbel-Softmax Top-K引入额外前向计算，具体FLOPs/显存开销未披露；
- **超参数透明度**: K值、LoRA rank、Gumbel-Softmax温度等关键超参数未披露，复现难度增加；
- **SCP方案贡献模糊**: 缺乏独立消融，无法区分TEMP模块与训练方案的各自贡献；
- **失败案例**: 零样本提示词敏感性未分析，极端长视频（远超平均时长）的截断策略未讨论。

## 方法谱系与知识库定位

**方法家族**: 多模态大语言模型 (MLLM) → 视频理解 → 手术领域适配

**父方法**: VideoGPT+ / LLaVA-Video（标准管线：视觉编码器→多模态投影器→LLM）

**改动插槽**:
- **架构**: 在投影器与LLM之间插入TEMP插件模块（+双粒度记忆金字塔）
- **目标/损失**: 保持自回归语言建模损失，SCP训练方案改变任务采样分布与顺序
- **训练配方**: SCP递进训练（感知→评估→推理），LoRA+全量混合微调策略
- **数据策展**: 自建CholeVidQA-32K三级层次数据集（但为二次标注，非原始采集）
- **推理**: 相同，TEMP模块前向激活

**直接基线与差异**:
- **VideoGPT+**: 均匀帧采样+全token输入；SurgTEMP以TEMP实现查询感知的自适应压缩
- **LLaVA-Video**: 类同VideoGPT+策略；SurgTEMP引入显式跨模态注意力前置与可微帧选择
- **LG-CVS / SurgPrOD**: 领域专用非LLM方法，聚焦单任务；SurgTEMP提供统一开放式VQA框架但未直接对比

**后续方向**:
1. **跨手术泛化**: 将TEMP模块迁移至其他术式（如结直肠、妇科腹腔镜），验证领域自适应能力；
2. **实时视频流处理**: 当前针对离线整段视频，扩展至术中实时流式推理需重新设计记忆更新机制；
3. **SCP方案独立验证与自动化**: 探索任务难度自动评估以替代人工三级划分，或验证该递进策略在通用视频QA中的迁移性。

**知识库标签**: 
- modality: video + text
- paradigm: 插件式模块增强 / 记忆增强LLM
- scenario: 手术视频分析 / 医学教育 / 技能评估
- mechanism: 跨模态注意力 / Gumbel-Softmax可微选择 / 层次化记忆金字塔
- constraint: 计算效率（LoRA）/ 长视频上下文限制 / 领域数据稀缺

