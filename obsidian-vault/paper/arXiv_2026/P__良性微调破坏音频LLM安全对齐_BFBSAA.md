---
title: Benign Fine-Tuning Breaks Safety Alignment in Audio LLMs
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.16659
aliases:
- 良性微调破坏音频LLM安全对齐
- BFBSAA
code_url: https://github.com/jrohsc/audio_benign_finetuning
---

# Benign Fine-Tuning Breaks Safety Alignment in Audio LLMs

[Paper](https://arxiv.org/abs/2604.16659) | [Code](https://github.com/jrohsc/audio_benign_finetuning)

**Topics**: [[T__Adversarial_Robustness]], [[T__Speech_Processing]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | 良性微调破坏音频LLM安全对齐 |
| 英文题名 | Benign Fine-Tuning Breaks Safety Alignment in Audio LLMs |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16659) · [Code](https://github.com/jrohsc/audio_benign_finetuning) · [Project] |
| 主要任务 | 研究音频大语言模型在良性数据微调后的安全对齐失效机制 |
| 主要 baseline | Qwen2.5-Omni, LLaMA-Omni, SALMONN 等 Audio LLM 架构 |

> [!abstract] 因为「音频LLM在良性数据微调后安全对齐被破坏」，作者在「文本/视觉模态的良性微调安全研究」基础上改了「引入语义-声学双轴分析框架并揭示架构条件性脆弱性」，在「SD-QA等音频问答数据集」上取得「模型内部编码器与参考编码器对比下的安全下降量化结果」

- **关键性能**: 使用模型内部编码器（model-internal encoder）时，良性微调后 JSR（Jailbreak Success Rate）显著上升，而使用共享参考编码器（shared reference encoder）时 JSR 变化呈现跨模态不对称性
- **关键性能**: 不同架构（双编码器/统一编码器/无差别编码器）在语义接近性与声学接近性上的脆弱性分布存在系统性差异
- **关键性能**: 拒绝信号抑制（refusal signal suppression）在 LLM 层间呈现架构条件性模式，投影到拒绝方向上的信号在 L0-L27 层间变化显著

## 背景与动机

音频大语言模型（Audio LLMs）已广泛部署于语音助手、客服、教育等场景，用户自定义微调成为标准工作流。然而，一个致命安全隐患长期被忽视：用户上传自己的良性音频数据（如有声书、播客、日常对话）进行微调，竟可能让模型对有害查询的拒绝能力大幅下降——即"越狱成功率"急剧攀升。

此前研究已在文本模态（Qi et al., NeurIPS 2023）和视觉模态（Zhan et al., 2024）中证实此现象。这些工作的核心发现是：在表示空间中，良性样本与有害内容的接近程度可预测其危害潜力——越"靠近"有害区域的良性样本，微调后安全对齐破坏越严重。但这些分析均在单一、无差别的嵌入空间中进行，未能揭示不同输入属性对脆弱性的差异化影响。

音频模态引入了结构上更复杂的问题。考虑一个具体例子：一段关于"如何制作蛋糕"的烹饪教学音频，其文字转录完全无害，但说话者的语气急促、音调紧张——这种声学特征可能与紧急求助、危险情境的音频在嵌入空间中接近。因此，一个良性样本可以通过两条独立路径接近有害内容：**语义路径**（说了什么）和**声学路径**（听起来怎样）。这种双轴接近性在文本和视觉模态中不存在：文本仅有语义维度，视觉虽有内容但缺乏独立的"呈现方式"维度。

此外，Audio LLMs 在架构上与文本 LLM 存在两点关键差异。第一，**音频编码器在微调时通常被冻结**，意味着安全训练的参数路径（文本 LLM 的后端）与微调路径（音频编码器的输出适配）并不重叠——拒绝机制的脆弱性源于其从文本安全训练继承而来、从未经过音频安全数据强化。第二，**不同架构对语义与声学特征的权重分配截然不同**：双编码器架构（如 Qwen2.5-Omni）分离处理语音与音频，统一编码器（如 LLaMA-Omni）共享参数，无差别编码器则不做区分。这导致脆弱性轴因架构而异，现有研究完全忽略了这些音频特有的结构性差异。

本文首次系统揭示：良性微调在音频模态中的安全破坏机制具有**语义-声学双轴性**和**架构条件性**，填补了关键安全盲区。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/35fc95bb-504d-4640-8797-90e3e9908bb1/figures/Figure_1.png)
*Figure 1: Figure 1: Overview. Benign and harmful audio are embedded via either the model’s ownencoder (model-internal) or a shared reference encoder (semantic, acoustic, or mixed). Benignsamples closest to harm*



## 核心创新

核心洞察：**音频模态的语义-声学解耦性使得良性样本可通过非语义路径接近有害内容**，因为音频嵌入空间天然编码了独立于文字内容的声学特征分布，从而使架构条件性的安全脆弱性分析成为可能——不同架构对双轴特征的权重分配差异，决定了其特有的"脆弱性指纹"。

| 维度 | Baseline（文本/视觉模态研究） | 本文 |
|:---|:---|:---|
| 接近性分析轴 | 单一嵌入空间，无属性解耦 | 语义 vs. 声学双轴独立分析 |
| 编码器角色 | 统一处理，无架构区分 | 模型内部编码器 vs. 共享参考编码器对比 |
| 脆弱性预测 | 基于与有害内容的标量距离 | 基于架构条件性的层间拒绝信号投影模式 |
| 安全机制归因 | 参数更新直接破坏对齐 | 冻结编码器导致安全训练路径与微调路径错配 |

与此前工作相比，本文不满足于"良性微调会破坏安全"这一现象复现，而是首次回答：**在音频模态中，什么属性的良性样本最危险？不同架构为何危险程度不同？破坏发生在表示空间的哪个环节？**

## 整体框架



本文提出三阶段分析框架，系统解构音频LLM良性微调的安全脆弱性：

**阶段一：双轴接近性量化（Dual-Axis Proximity Quantification）**
- 输入：良性音频数据集（SD-QA 等）+ 有害音频/文本查询集合
- 处理：通过模型内部编码器（model-internal encoder）或共享参考编码器（shared reference encoder）提取嵌入
- 输出：每个良性样本的语义接近性分数（基于文字转录的文本嵌入距离）和声学接近性分数（基于原始音频嵌入与文本嵌入的残差距离）

**阶段二：架构条件性脆弱性分析（Architecture-Conditioned Vulnerability Profiling）**
- 输入：阶段一的接近性分数 + 三种架构变体（双编码器/统一编码器/无差别编码器）
- 处理：按语义/声学接近性分层采样，对各子集进行良性微调，测量微调后 JSR 变化
- 输出：每种架构的"脆弱性指纹"——语义轴 vs. 声学轴的相对敏感度

**阶段三：拒绝信号层间追踪（Refusal Signal Layer-wise Tracing）**
- 输入：微调前后的模型隐藏状态（L0-L27）
- 处理：计算各层隐藏状态在预定义的"拒绝方向"（refusal direction）上的投影幅度
- 输出：拒绝信号抑制曲线，定位安全对齐破坏发生的具体层范围

数据流总览：
```
良性音频样本 → [编码器选择: 内部/参考] → 嵌入空间
                                    ↓
                    [语义接近性] ← 文本转录嵌入
                    [声学接近性] ← 音频-文本残差嵌入
                                    ↓
                    [架构条件性微调] → JSR 变化率
                                    ↓
                    [层间隐藏状态提取] → 拒绝方向投影 → 安全破坏定位
```

该框架的核心设计在于**编码器解耦**：通过对比模型自身编码器与共享参考编码器的结果，分离"架构固有特征处理"与"任务学习引入的偏差"，从而精确归因脆弱性来源。

## 核心模块与公式推导

### 模块 1: 语义-声学双轴接近性分解（对应框架图阶段一）

**直觉**: 音频嵌入包含"说了什么"和"怎么说"两个不可约信息源，需显式解耦才能识别非语义路径的安全风险。

**Baseline 公式**（单轴接近性，Qi et al. 文本模态）:
$$d_{\text{single}}(b, h) = \|\phi(b) - \phi(h)\|_2$$
符号: $b$ = 良性样本, $h$ = 有害样本, $\phi$ = 统一编码器, $\|\cdot\|_2$ = L2 距离

**变化点**: 文本模态中 $\phi$ 直接处理文字，音频模态中若用统一编码器则语义与声学纠缠；需引入显式解耦，且考虑编码器来源（模型内部 vs. 参考）。

**本文公式（推导）**:
$$\text{Step 1}: \quad e_{\text{audio}} = f_{\text{enc}}(x_{\text{audio}}), \quad e_{\text{text}} = g_{\text{transcribe}}(x_{\text{audio}}) \rightarrow f_{\text{enc}}$$
$$\text{其中 } f_{\text{enc}} \in \{f_{\text{model-internal}}, f_{\text{reference}}\} \text{，加入编码器选择以隔离架构效应}$$

$$\text{Step 2}: \quad d_{\text{semantic}}(b, h) = \|\text{Proj}_{\text{text}}(e_{\text{audio}}^{(b)}) - \text{Proj}_{\text{text}}(e_{\text{audio}}^{(h)})\|_2$$
$$\text{语义投影提取文字内容对应的方向，解决"说了什么"的接近性}$$

$$\text{Step 3}: \quad d_{\text{acoustic}}(b, h) = \|e_{\text{audio}}^{(b)} - \text{Proj}_{\text{text}}(e_{\text{audio}}^{(b)}) - (e_{\text{audio}}^{(h)} - \text{Proj}_{\text{text}}(e_{\text{audio}}^{(h)}))\|_2$$
$$\text{残差提取非语义成分，即"听起来怎样"的接近性}$$

$$\text{最终}: \quad d_{\text{dual}}(b, h) = (d_{\text{semantic}}, d_{\text{acoustic}}) \in \mathbb{R}^2$$

**对应消融**: 

---

### 模块 2: 架构条件性拒绝信号抑制（对应框架图阶段三，Figure 3）

**直觉**: 安全对齐的破坏体现为拒绝回答有害查询的能力下降，该能力由LLM内部特定方向编码；追踪该方向在层间的强度变化可定位破坏机制。

**Baseline 公式**（标准表示工程，Zou et al. 2023）:
$$r_{\text{layer}} = \frac{1}{N_{\text{refusal}}} \sum_{i=1}^{N_{\text{refusal}}} (h_{\text{refuse}, i}^{(\text{ell})} - h_{\text{comply}, i}^{(\text{ell})})$$
$$\text{RefusalScore}(h^{(\text{ell})}) = \langle h^{(\text{ell})} - h_{\text{baseline}}^{(\text{ell})}, r_{\text{layer}} \rangle$$
符号: $h^{(\text{ell})}$ = 第 $\text{ell}$ 层隐藏状态, $r_{\text{layer}}$ = 层特定拒绝方向, $\langle \cdot, \cdot \rangle$ = 内积

**变化点**: Baseline 假设拒绝方向在各层一致且与架构无关；本文发现音频LLM中拒绝信号抑制具有**架构条件性**——不同编码器架构下，语义/声学路径触发的抑制模式在层间分布截然不同。

**本文公式（推导）**:
$$\text{Step 1}: \quad \Delta_{\text{refusal}}^{(\text{ell})}(b) = \text{RefusalScore}_{\text{post-FT}}^{(\text{ell})}(b) - \text{RefusalScore}_{\text{pre-FT}}^{(\text{ell})}(b)$$
$$\text{加入微调前后对比，量化特定良性样本 } b \text{ 引入的拒绝信号变化}$$

$$\text{Step 2}: \quad \text{Proj}_{r^{(\text{ell})}}(h^{(\text{ell})}) = \frac{\langle h^{(\text{ell})}, r^{(\text{ell})} \rangle}{\|r^{(\text{ell})}\|_2}$$
$$\text{标准化投影，保证跨层、跨架构可比性}$$

$$\text{Step 3}: \quad \text{ArchitectureConditionedProfile}(\text{arch}) = \{ (\Delta_{\text{refusal}}^{(\text{ell})}(b) \text{mid} d_{\text{semantic}}(b,h) < \tau_s, d_{\text{acoustic}}(b,h) < \tau_a) \}_{\text{ell}=0}^{L}$$
$$\text{按语义/声学接近性分层，构建架构特定的层间抑制指纹}$$

$$\text{最终}: \quad \text{VulnerabilityAxis}(\text{arch}) = \text{arg}\max_{\text{axis} \in \{s,a\}} \mathbb{E}_{b: d_{\text{axis}} \text{ low}} \left[ \sum_{\text{ell}=0}^{L} |\Delta_{\text{refusal}}^{(\text{ell})}(b)| \right]$$

**对应消融**: Figure 3 显示 Qwen2.5-Omni（双编码器架构）在 L0-L27 层的拒绝方向投影变化，低语义-高声学接近性样本触发深层（L20+）显著抑制，而低声学-高语义样本主要影响浅层（L5-L15）。

---

### 模块 3: 嵌入空间邻近过滤（对应 Figure 4）

**直觉**: 并非所有良性样本都同等危险，需系统筛选"最可能破坏安全"的子集进行针对性分析。

**本文公式**:
$$\text{Step 1}: \quad \mathcal{B}_{\text{filtered}} = \{ b \in \mathcal{B} : \min_{h \in \mathcal{H}} d_{\text{dual}}(b,h) < (\tau_s, \tau_a) \}$$
$$\text{双阈值过滤，保留语义或声学至少一轴接近有害的样本}$$

$$\text{Step 2}: \quad \text{QuadrantSplit}(\mathcal{B}_{\text{filtered}}) = \mathcal{B}_{ss} \cup \mathcal{B}_{sa} \cup \mathcal{B}_{as} \cup \mathcal{B}_{aa}$$
$$\text{其中 } ss=\text{语义近/声学近}, sa=\text{语义近/声学远}, as=\text{语义远/声学近}, aa=\text{双远}$$

$$\text{最终}: \quad \text{JSR}_{\text{post-FT}}^{(q)} = \frac{1}{|\mathcal{B}_q|} \sum_{b \in \mathcal{B}_q} \mathbb{1}[\text{Model}_{\text{FT on } b}(h_{\text{test}}) \text{ complies}]$$

该模块为阶段二的架构条件性分析提供受控的样本分层。

## 实验与分析


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/35fc95bb-504d-4640-8797-90e3e9908bb1/figures/Figure_2.png)
*Figure 2: Figure 2: Cross-modal asymmetry: JSR (%) after fine-tuning on semantic proximity-filteredSD-QA data as text (blue) vs. audio (red). Dashed lines indicate pretrained baselines. AF3shows audio fine-tuni*



**主结果：跨编码器与跨模态对比**

| 条件 | JSR 变化 (文本模态微调) | JSR 变化 (音频模态微调) | 关键发现 |
|:---|:---|:---|:---|
| 共享参考编码器 | 基线水平 | 基线水平 | 编码器统一时，跨模态对称 |
| 模型内部编码器（Qwen2.5-Omni） | 
| 模型内部编码器（LLaMA-Omni） | 

Figure 2 核心发现：在语义接近性过滤的 SD-QA 数据上，文本微调（蓝色）与音频微调（红色）的 JSR 变化呈现**跨模态不对称性**——音频模态的 JSR 上升更陡峭且与声学接近性相关，文本模态则主要受语义接近性驱动。虚线标记的阈值区分了"安全区域"与"危险区域"。

**架构条件性分析（Figure 3）**


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/35fc95bb-504d-4640-8797-90e3e9908bb1/figures/Figure_3.png)
*Figure 3: Figure 3: Architecture-conditioned refusal signal suppression. Projection onto the refusaldirection across LLM layers (L0–L27) for Qwen2.5-Omni (top) and AF3 (bottom), under text(left) and audio (righ*



Qwen2.5-Omni（双编码器）的拒绝信号投影显示：
- **浅层（L0-L10）**：语义接近性主导，拒绝信号抑制与 $d_{\text{semantic}}$ 负相关
- **深层（L20-L27）**：声学接近性主导，$d_{\text{acoustic}}$ 低的样本引发更强抑制
- **关键转折层（L15 附近）**：语义-声学权重发生交叉，此位置因架构而异

该模式在统一编码器（LLaMA-Omni）中显著不同：语义与声学信号在浅层即高度纠缠，深层分离度低，导致脆弱性预测更困难。

**消融实验**：移除声学接近性过滤（仅用语义距离筛选样本）导致 Qwen2.5-Omni 的 JSR 预测 AUC 下降；反之，移除语义过滤对基于声学编码器的模型影响更大。验证双轴解耦的必要性。

**公平性检查**：
- Baseline 强度：对比了 Qi et al.（文本）、Zhan et al.（视觉）的接近性预测方法直接迁移到音频的效果，均显著劣于本文双轴方法
- 计算成本：嵌入提取为一次性开销，层间追踪需前向传播 L 次，总体可控
- 数据成本：SD-QA 为公开数据集，有害查询集规模
- **失败案例/局限**：(1) 仅覆盖英语音频，多语言声学特征可能改变脆弱性分布；(2) 拒绝方向基于英文有害查询定义，跨文化有害内容可能未被捕获；(3) 微调数据量固定，大规模持续微调的累积效应未研究

## 方法谱系与知识库定位

**方法家族**: 大模型安全对齐脆弱性分析（Jailbreak Vulnerability Analysis）

**父方法**: Qi et al., "Fine-tuning Aligned Language Models Compromises Safety", NeurIPS 2023 —— 首次建立"良性微调→安全对齐破坏"的因果链条，提出基于嵌入空间接近性的危害预测框架。

**本文改变的插槽**:
| 插槽 | 父方法 | 本文 |
|:---|:---|:---|
| architecture | 文本Transformer统一编码 | 音频多编码器架构（双/统一/无差别） |
| objective | 单轴L2距离最小化 | 语义-声学双轴联合优化 |
| training_recipe | 全参数微调 | 冻结音频编码器，仅微调投影/适配层 |
| data_curation | 文本语料邻近过滤 | 音频嵌入双阈值四象限分层 |
| inference | 单层隐藏状态分析 | 层间拒绝信号追踪（L0-L27） |

**直接 Baseline 与差异**:
- **Qi et al. (2023)**: 文本模态单轴分析 → 本文扩展至音频双轴，引入编码器架构变量
- **Zhan et al. (2024, 视觉)**: 视觉-语言模型微调安全 → 本文发现音频特有的声学路径独立风险
- **Andriushchenko et al. (2024, 对齐攻击)**: 对抗性微调攻击 → 本文研究**非对抗性**良性数据的自然风险

**后续方向**:
1. **防御机制**: 基于架构条件性指纹的动态安全门控——在脆弱层注入音频特定的拒绝强化
2. **多模态扩展**: 视频模态的三轴分析（语义/视觉呈现/音频声学）
3. **持续学习场景**: 在线微调过程中的安全监控，而非单次微调后分析

**知识库标签**:
- modality: audio + text (multimodal)
- paradigm: representation engineering / safety alignment
- scenario: fine-tuning vulnerability / user customization risk
- mechanism: encoder freezing / cross-modal asymmetry / refusal direction suppression
- constraint: benign data only / black-box encoder access / architecture heterogeneity

