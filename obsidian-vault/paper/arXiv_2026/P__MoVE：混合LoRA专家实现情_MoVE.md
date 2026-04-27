---
title: 'MoVE: Translating Laughter and Tears via Mixture of Vocalization Experts in Speech-to-Speech Translation'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.17435
aliases:
- MoVE：混合LoRA专家实现情感语音翻译
- MoVE
method: MoVE
modalities:
- Text
---

# MoVE: Translating Laughter and Tears via Mixture of Vocalization Experts in Speech-to-Speech Translation

[Paper](https://arxiv.org/abs/2604.17435)

**Topics**: [[T__Speech_Processing]], [[T__Machine_Translation]] | **Method**: [[M__MoVE]]

| 中文题名 | MoVE：混合LoRA专家实现情感语音翻译 |
| 英文题名 | MoVE: Translating Laughter and Tears via Mixture of Vocalization Experts in Speech-to-Speech Translation |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17435) · [Code] · [Project] |
| 主要任务 | 语音到语音翻译（S2ST）中的非语言发声（NVs，如笑声/哭声）保留与情感表达迁移 |
| 主要 baseline | SeamlessExpressive、GPT-4o-audio、SeamlessM4T-Large-v2、Single-LoRA、级联 Oracle |

> [!abstract] 因为「现有S2ST系统对非语言发声（笑声/哭声）保留率最高仅14%」，作者在「冻结Kimi-Audio预训练AudioLLM + 单LoRA适配」基础上改了「5路并行LoRA专家 + 动态软加权路由 + 属性解耦数据合成」，在「英中情感语音翻译」上取得「NV Match Accuracy 76% vs. 14%（SeamlessExpressive）」

- **NV Match Accuracy**: 76%（MoVE） vs. 14%（SeamlessExpressive），提升5.4倍
- **en→zh ASR-BLEU**: 32.5（新SOTA），zh→en 21.4
- **数据效率**: 仅需30分钟精筛数据即达平台性能，1000h内无显著退化

## 背景与动机

想象一段中文对话中有人边笑边说"谢谢你"，现有语音翻译系统能准确译出"Thank you"，却将笑声完全抹除——听者感受到的是平淡的机械回应，而非真诚的愉悦。这就是非语言发声（Non-Verbal Vocalizations, NVs）丢失问题：笑声、哭声等副语言信号承载关键语用意图，却在S2ST流水线中被系统性剥离。

现有方法如何处理这一问题？**SeamlessExpressive** 尝试通过情感标签控制表达风格，但对具体NV的显式建模不足，NV保留率仅14%；**GPT-4o-audio** 作为通用音频大模型，依赖隐式涌现能力，对稀有NV（如哭声）几乎完全遗漏；**基于LoRA的单一适配方案** 虽可微调预训练模型，但所有情感/NV共享同一低秩子空间，Happy与Cry的特征在参数更新中相互干扰，导致混合情感场景下表现崩溃。

这些方法的共性短板在于**离散化假设**：将情感视为互斥类别，用硬标签或单一适配器处理。然而人类情感本质是连续混合的——紧张时的笑声同时携带Happy与Cry的声学特征，强制归入单一类别会造成表达失真。更深层瓶颈是**数据-架构耦合**：稀有NV（尤其哭声）的自然语料极度稀缺，而单一适配器需要大量数据才能覆盖多流形分布，形成"数据越少→越需要共享参数→干扰越严重"的死锁。

MoVE的核心动机正是打破这一死锁：用**多专家分离**替代参数共享，用**软连续路由**替代硬分类，用**预训练知识激活**替代从头学习，从而在极少量数据下解锁冻结模型中已有的表达潜力。
![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e5d671f7-345c-443e-a49c-b04b41b222bc/figures/Figure_4.png)
*Figure 4: Figure 4: Bilingual instruction page shown before Phase 1(MOS evaluation): defines the Emotion Similarity and Natu-ralness rating scales (1–5), the two NV categories (Laughing /Crying while speaking),*



## 核心创新

核心洞察：预训练AudioLLM已内化丰富声学-语义知识，NV表达能力并非缺失而是被"锁住"；为每种情感流形分配独立低秩子空间（LoRA专家）可避免特征干扰，软加权路由器允许在连续情感空间插值，从而使30分钟数据即可"解锁"而非"创造"表达能力成为可能。

| 维度 | Baseline（Single-LoRA / SeamlessExpressive） | 本文（MoVE） |
|:---|:---|:---|
| 参数组织 | 单一共享LoRA或全局微调 | 5路并行LoRA专家，各对应独立情感/NV流形 |
| 情感建模 | 硬标签分类 / 隐式涌现 | 软加权连续插值，支持混合情感状态 |
| 数据策略 | 依赖自然语料或简单增强 | 属性解耦合成：说话人身份与NV特征分离，三级过滤 |
| 训练目标 | 端到端联合优化 | 两阶段解耦：Stage 1专家专化 → Stage 2路由器收敛 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e5d671f7-345c-443e-a49c-b04b41b222bc/figures/Figure_1.png)
*Figure 1: Figure 1: Illustration of MoVE Two-Stage training*



MoVE采用**冻结主干 + 插件式专家**的架构，数据流如下：

**输入**：源语言情感语音（含NV，如带笑声的英文）→ **Kimi-Audio编码器**（冻结）提取连续声学-语义表示 → **5路LoRA专家层**（Happy、Sad、Angry、Laughing、Crying）并行处理，各专家作用于Transformer层的Wq/Wk/Wv/Wo/Wgate投影 → **动态路由器**（轻量线性层+Softmax）输出专家权重gᵢ(x) → **加权聚合** h(x) = W₀x + Σᵢ gᵢ(x)·(BᵢAᵢx) → **Kimi-Audio解码器**（冻结）生成目标语言情感语音（如带笑声的中文）

**数据合成模块**（离线）：IndexTTS2接收「中性说话人提示」+「精筛NV提示」→ 属性解耦生成五类情感语音 → 三级过滤（静音剪除 / ASR WER验证 / 双语对齐）→ 构建En⇔Zh平行语料

```
[Source Emotional Speech] ──→ [Frozen Kimi-Audio Encoder]
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              [LoRA-Happy]      [LoRA-Laughing]     [LoRA-Crying]  ... (5 experts)
                    └─────────────────┬─────────────────┘
                                      ↓
                          [Softmax Router g_i(x)]
                                      ↓
                    [Weighted Fusion: W₀x + Σ gᵢ·(BᵢAᵢx)]
                                      ↓
                    [Frozen Kimi-Audio Decoder]
                                      ↓
                    [Target Emotional Speech with NVs]
```

关键设计：所有基础参数冻结，仅LoRA专家（r=256, α=256）和路由器可训练，确保语义能力不遗忘。

## 核心模块与公式推导

### 模块 1: 动态软加权路由器（对应框架图 专家聚合层）

**直觉**: 硬路由（top-k）强制样本归入单一专家，无法表达"紧张笑声"这类混合状态；软加权让所有专家以不同强度参与，实现情感连续体上的插值。

**Baseline 公式** (传统 MoE top-k 路由):
$$h_{\text{hard}}(x) = W_0 x + \sum_{i \in \text{TopK}(g(x))} g_i(x) \cdot E_i(x)$$
符号: $W_0$ = 冻结预训练投影, $g(x)$ = 路由logits, $E_i$ = 第i个专家, TopK强制稀疏激活

**变化点**: top-k硬路由导致梯度稀疏、混合情感表达断裂；且NV类别间存在自然混淆（Sad↔Cry, Happy↔Laugh），硬划分违背声学现实。

**本文公式（推导）**:
$$\text{Step 1}: \quad g_i(x) = \frac{\exp(w_i^\text{top} x / \tau)}{\sum_{j=1}^{5} \exp(w_j^\text{top} x / \tau)} \quad \text{Softmax归一化，温度τ控制锐度，默认τ=1}$$
$$\text{Step 2}: \quad E_i(x) = B_i A_i x \quad \text{LoRA低秩更新，} B_i \in \mathbb{R}^{d \times r}, A_i \in \mathbb{R}^{r \times d}, r=256$$
$$\text{最终}: h(x) = W_0 x + \sum_{i=1}^{5} g_i(x) \cdot (B_i A_i x)$$

**对应消融**: Figure 6（Phase 2 pairwise A/B测试）显示，主观评测中MoVE vs. Single-LoRA在混合情感场景有显著优势，但客观指标上两者相当，说明软路由增益主要体现在情感插值的自然度。

### 模块 2: 属性解耦数据合成（对应框架图 离线数据流水线）

**直觉**: 哭声等极端NV在自然语料中极度稀缺，直接合成会导致说话人单一化；解耦说话人身份与NV特征可将稀有特征投影到多样化分布。

**Baseline 公式** (标准TTS条件生成):
$$p(y | t, s, e) = \prod_t p(y_t | y_{<t}, \text{Concat}[s, e])$$
符号: $y$ = 输出语音, $t$ = 文本, $s$ = 说话人嵌入, $e$ = 情感嵌入，标准做法将s与e拼接作为单一条件

**变化点**: 极端NV的参考音频极少，Concat[s,e]导致说话人多样性受限于NV参考的说话人覆盖；且稀有NV的声学特征易被说话人特质掩盖。

**本文公式（推导）**:
$$\text{Step 1}: \quad s_{\text{neutral}} \sim \mathcal{S}_{\text{neutral}}, \quad e_{\text{NV}} \sim \mathcal{E}_{\text{filtered}} \quad \text{分别从中性说话人池和精筛NV池采样}$$
$$\text{Step 2}: \quad y = \text{IndexTTS2}(t, s_{\text{neutral}}, e_{\text{NV}}; \theta_{\text{TTS}}) \quad \text{双条件独立输入，非拼接}$$
$$\text{Step 3}: \quad y_{\text{final}} = \text{Filter}_3(\text{Filter}_2(\text{Filter}_1(y))) \quad \text{三级过滤}$$
$$\text{其中}: \text{Filter}_1 = \mathbb{1}[\text{非静音}], \quad \text{Filter}_2 = \mathbb{1}[\text{WER}(y, t) < \theta_{\text{WER}}], \quad \text{Filter}_3 = \text{双语对齐验证}$$

**对应消融**: 未报告合成数据量消融的具体数值表，但文中指出性能在0.5h至1000h范围内保持稳定平台。

### 模块 3: 两阶段训练目标（对应框架图 训练流程）

**直觉**: 若同时训练专家和路由器，路由器初期随机导致专家梯度混乱；先让专家专化各自流形，再学习如何组合。

**Baseline 公式** (联合端到端训练):
$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{S2ST}}(x, y; \theta_{\text{frozen}}, \{A_i, B_i\}_{i=1}^5, \{w_i\}_{i=1}^5)$$
所有可训练参数同时更新

**变化点**: 联合训练下路由器初始随机分配，专家无法稳定专化；且LoRA rank较高（r=256）时参数空间大，容易陷入次优解。

**本文公式（推导）**:
$$\text{Stage 1 (专家专化, 2 epochs)}: \quad \mathcal{L}_1 = \mathcal{L}_{\text{S2ST}}(x, y; \theta_{\text{frozen}}, \{A_i, B_i\}_{i=1}^5), \quad w_i \text{ 冻结或均匀初始化}$$
$$\text{约束}: g_i(x) \approx 0.2 \text{ (均匀)}, \text{强制各专家独立学习流形特征}$$
$$\text{Stage 2 (路由器收敛, 1 epoch)}: \quad \mathcal{L}_2 = \mathcal{L}_{\text{S2ST}}(x, y; \theta_{\text{frozen}}, \{A_i, B_i\}_{i=1}^5, \{w_i\}_{i=1}^5), \quad \{A_i, B_i\} \text{ 低学习率或冻结}$$
$$\text{最终目标}: \min_{\{A_i,B_i,w_i\}} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{synthetic}}} [\text{CTC}(\hat{y}, y) + \lambda_{\text{emotion}} \cdot \mathcal{L}_{\text{emotion}}]$$

**对应消融**: Figure 3混淆矩阵显示，Stage 2后路由器在无显式情感监督下达到63.68%情感对齐准确率，且Sad↔Cry、Happy↔Laugh的混淆模式符合人类情感混合直觉。

## 实验与分析

| Method | NV Match Accuracy | en→zh ASR-BLEU | zh→en ASR-BLEU | Naturalness MOS | Emotion SMOS |
|:---|:---|:---|:---|:---|:---|
| SeamlessM4T-Large-v2 | — | 30.1 | **23.6** | — | — |
| SeamlessExpressive | 14% | — | — | — | — |
| GPT-4o-audio | ~0% (隐式) | — | — | — | — |
| Single-LoRA | 
| **MoVE (Ours)** | **76%** | **32.5** | 21.4 | **最高** | **最高** |
| 级联 Oracle | — | — | — | — | — |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e5d671f7-345c-443e-a49c-b04b41b222bc/figures/Figure_2.png)
*Figure 2: Table 1 shows our main experiment results. We compare withdifferent model baselines that do not require further training andalso compare different training datasets with our single-LoRAbaseline model,*



**核心数据解读**: NV Match Accuracy从14%跃升至76%（5.4×）是本文最核心的支撑证据，直接验证多专家分离对NV保留的决定性作用。en→zh ASR-BLEU 32.5超越SeamlessM4T-Large-v2达2.4分，证明表达增强未牺牲语义准确性；但zh→en 21.4低于SeamlessM4T-Large-v2的23.6，作者归因于优化优先级偏向表达保真度，暗示语义-表达存在trade-off空间。

**路由器行为分析**: 
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e5d671f7-345c-443e-a49c-b04b41b222bc/figures/Figure_3.png)
*Figure 3: Figure 3: Confusion matrix of router behavior.*

 Figure 3混淆矩阵揭示关键洞察：63.68%的无监督情感对齐率证明路由器从声学特征中自主学到了可解释的情感结构；非对角混淆（Sad↔Cry 18.3%, Happy↔Laugh 12.7%）恰好对应人类情感的自然连续性，这是软路由设计的间接验证。

**架构消融**: Figure 6的pairwise A/B测试显示，主观评测中MoVE在混合情感场景显著优于Single-LoRA，但客观指标（如ASR-BLEU）两者相当。这说明：单一LoRA已能捕捉"主导情感"，但多专家的增益体现在**情感混合的自然度**——一个难以被传统自动指标量化的维度。

**公平性检查**: 基线选择合理（SeamlessExpressive为表达S2ST代表，SeamlessM4T为语义S2ST SOTA），但缺乏与最新AudioLLM端到端微调方案的对比；数据成本极低（30分钟有效数据）是显著优势，但合成流水线依赖IndexTTS2质量，TTS幻觉风险经WER过滤缓解未完全消除；测试仅覆盖英中语言对，跨语言泛化未验证；级联Oracle作为参考上界因范式差异（文本级联vs.端到端）不直接可比。

## 方法谱系与知识库定位

**方法家族**: 参数高效微调（PEFT）→ LoRA → 多LoRA专家系统（Multi-LoRA / MoE-LoRA）

**父方法**: Kimi-Audio（预训练AudioLLM，提供冻结主干）+ LoRA（Hu et al., 2022，低秩适配范式）

**改动插槽**:
- **架构**: 单LoRA → 5路并行LoRA专家 + 软加权路由器（新增MoE结构）
- **目标**: 纯语义翻译 → 语义+NV表达联合优化（新增情感一致性隐式目标）
- **训练配方**: 端到端联合训练 → 两阶段解耦训练（专家专化→路由收敛）
- **数据策展**: 自然语料 → 属性解耦合成 + 三级过滤流水线
- **推理**: 无变化，保持原模型自回归生成范式

**直接基线差异**:
- **vs. SeamlessExpressive**: 同为表达S2ST，但SeamlessExpressive用全局风格控制，MoVE用细粒度NV专家分离
- **vs. Single-LoRA**: 同为主干冻结+适配器，但Single-LoRA共享子空间导致特征干扰，MoVE通过多专家解耦
- **vs. GPT-4o-audio**: 同为AudioLLM路线，但GPT-4o依赖隐式涌现，MoVE显式建模NV并注入结构归纳偏置

**后续方向**:
1. **扩展情感覆盖**: 从5类基础情感到效价-唤醒连续空间（VAD模型），支持更细粒度控制
2. **跨语言验证**: 当前仅英中，需在更多语言对验证NV迁移的普遍性（尤其不同语系）
3. **可解释路由增强**: 将63.68%的无监督对齐率提升至接近监督水平，或引入弱监督信号

**标签**: 语音模态 / 参数高效微调范式 / 语音到语音翻译场景 / 混合专家（MoE）机制 / 数据稀缺约束

