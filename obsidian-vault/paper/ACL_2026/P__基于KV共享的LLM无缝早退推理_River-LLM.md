---
title: 'River-LLM: Large Language Model Seamless Exit Based on KV Share'
type: paper
paper_level: B
venue: ACL
year: 2026
paper_link: https://arxiv.org/abs/2604.18396
aliases:
- 基于KV共享的LLM无缝早退推理
- River-LLM
acceptance: accepted
method: River-LLM
modalities:
- Text
---

# River-LLM: Large Language Model Seamless Exit Based on KV Share

[Paper](https://arxiv.org/abs/2604.18396)

**Topics**: [[T__Text_Generation]] (其他: Efficiency) | **Method**: [[M__River-LLM]]

| 中文题名 | 基于KV共享的LLM无缝早退推理 |
| 英文题名 | River-LLM: Large Language Model Seamless Exit Based on KV Share |
| 会议/期刊 | ACL 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18396) · [Code] · [Project] |
| 主要任务 | 大语言模型推理加速（Early Exit / Speculative Decoding方向） |
| 主要 baseline | Token-level Early Exit, LayerSkip, CALM, SkipDecode |

> [!abstract] 因为「decoder-only LLM中token-level early exit存在KV Cache缺失问题导致无法实际加速」，作者在「Token-level Early Exit」基础上改了「引入KV共享机制实现seamless exit」，在「Llama3.2 1B/3B on GSM8K/HumanEval」上取得「Score 0.26时ms/token从baseline的约2.5x降至接近1.0x（即无额外延迟）」

- **关键性能1**: KV共享策略下，relaxed threshold (Score≈0.15) 时ms/token接近vanilla推理（Figure 3a），strict threshold时仍优于naive early exit（Figure 3b）
- **关键性能2**: 首层state transition similarity与末层backbone-exit value vector similarity相关系数 r = 0.5536（Figure 7），验证KV共享的理论基础
- **关键性能3**: 最优token-level exit position分布显示大量token可在浅层退出（Figure 2a, Score=0.26）

## 背景与动机

大语言模型（LLM）推理成本高昂，early exit（早退）是一种有吸引力的加速思路：让"简单"token在浅层网络退出，避免计算全部层。然而，在decoder-only架构（如Llama、GPT系列）中，一个根本性的障碍使这一思路难以落地——KV Cache缺失问题。

具体而言，decoder-only LLM采用自回归生成：每个token的KV Cache需要缓存前面所有token的key和value。假设token t在第l层early exit，则后续token t+1, t+2, ... 在计算attention时，需要token t的完整KV Cache（即所有层的K/V）。但token t只计算到第l层就退出了，深层（l+1到L）的K/V并不存在。这导致后续token必须重新计算token t的深层K/V，或者放弃使用KV Cache，二者都严重损害推理效率。

现有方法如何处理这一问题？**Token-level Early Exit**（如CALM, SkipDecode）在分类/编码器模型上有效，但在decoder-only LLM中因上述KV Cache问题，实际加速比远低于理论预期（Figure 3显示naive策略导致ms/token显著上升）。**LayerSkip**采用layer skipping而非exit，避免了KV缺失但需要训练时特殊设计。**Speculative Decoding**系列（如Medusa, Eagle）用小模型草稿+大模型验证，不直接解决early exit的KV问题。

核心痛点：现有token-level early exit在decoder-only LLM中"理论上该快，实际上更慢"——Figure 3显示naive KV策略下ms/token不降反升。作者的关键观察是：如果能让early exit的token也能提供完整深度的KV表示，就能实现"seamless"（无缝）退出，既保留early exit的节省，又不破坏后续token的KV Cache依赖。

本文提出River-LLM，通过**KV共享机制**让exit layer与backbone深层共享KV表示，从根本上解决decoder-only LLM的early exit KV缺失难题。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/57d9051e-de40-4caf-9d15-9ec107841726/figures/Figure_1.png)
*Figure 1: Figure 1: KV Cache Absence problem for Early Exit indecoder-only LLM.*



## 核心创新

核心洞察：exit layer与backbone末层的value vector存在显著相关性（r=0.5536），因为浅层state transition已蕴含深层表示的足够信息，从而使exit token复用深层KV Cache而不重新计算成为可能。

| 维度 | Baseline (Token-level Early Exit) | 本文 (River-LLM) |
|:---|:---|:---|
| KV Cache来源 | 仅计算到exit layer，深层K/V缺失 | exit layer与backbone深层**共享KV**，后续token可直接使用 |
| 对后续token影响 | 需重新计算或放弃KV Cache，延迟剧增 | **无额外延迟**，seamless衔接 |
| 退出条件 | 固定阈值或学习分类器 | 基于首层state transition similarity预测exit时机 |
| 训练修改 | 通常需训练exit classifer | KV共享机制，最小化训练改动 |

与LayerSkip等layer skipping方法不同，River-LLM保持backbone完整，仅让部分token提前产生输出；与speculative decoding不同，无需draft model，避免了验证开销和内存占用。

## 整体框架


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/57d9051e-de40-4caf-9d15-9ec107841726/figures/Figure_4.png)
*Figure 4: Figure 4: Seamless exit architecture and inference paradigm: River-LLM. (a) KV-shared exit layer. (b) Inferencewith seamless exit.*



River-LLM的整体架构包含两个核心组件，数据流如下：

**输入** → **Backbone LLM（前向传播）** → **KV-shared Exit Layer（可选分支）** → **Seamless Exit决策** → **输出token / 继续深层计算**

各模块详解：

1. **Backbone LLM**: 标准decoder-only Transformer（如Llama3.2），负责常规的前向传播。所有token默认流经全部L层。

2. **KV-shared Exit Layer**（Figure 4a）: 插入在特定中间层的轻量分支。关键设计：该exit layer的**value vectors与backbone深层共享**——即exit token的V Cache直接作为后续token在深层attention中的V输入，无需重新计算。这是"seamless"的核心。

3. **Exit决策模块**: 基于首层（layer 1）的state transition similarity预测当前token是否可在exit layer安全退出。利用Figure 7观察到的相关性（r=0.5536），浅层动态即可可靠预测深层行为。

4. **Seamless Inference调度器**（Figure 4b）: 运行时协调——exit token从分支输出并复用共享KV；未exit token继续backbone深层计算。后续token的attention不受干扰，因为KV Cache完整。

```
[Token t] ──→ [Layer 1] ──→ [Exit?] ──Yes──→ [KV-shared Exit Layer] ──→ [Output]
                              │                    │ (shared V)
                              No                   ↓
                              └──→ [Layer 2..L-1] ──→ [Layer L] ──→ [Output]
                                                         ↑
                                    [Token t+1's attention uses shared V from exit]
```

## 核心模块与公式推导

### 模块 1: KV Cache缺失问题的形式化与Baseline困境（对应框架图 底层）

**直觉**: 先明确为什么naive early exit在decoder-only LLM中失效，才能理解KV共享的必要性。

**Baseline公式** (Token-level Early Exit):
对于token位置t，设其在layer l退出，输出为：
$$\hat{y}_t = \text{Softmax}(W_{\text{head}} \cdot h_t^{(l)})$$

符号: $h_t^{(l)}$ = token t在layer l的hidden state; $W_{\text{head}}$ = 输出头权重; KV Cache $K_t = [k_t^{(1)}, ..., k_t^{(L)}]$, $V_t = [v_t^{(1)}, ..., v_t^{(L)}]$。

**变化点**: 由于token t只计算到layer l，其深层KV $k_t^{(l+1:L)}, v_t^{(l+1:L)}$ 缺失。后续token $t' > t$ 的attention计算为：
$$\text{Attention}(Q_{t'}, K_{\leq t'}, V_{\leq t'}) = \text{softmax}\left(\frac{Q_{t'} K_{\leq t'}^T}{\sqrt{d_k}}\right) V_{\leq t'}$$

其中 $K_{\leq t'}, V_{\leq t'}$ 需要所有先前token的全部层KV。缺失 $v_t^{(l+1:L)}$ 导致必须：(a) 重新计算token t的深层，或 (b) 使用不完整的KV Cache。二者都使实际ms/token远超理论值（Figure 3）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \tilde{v}_t^{(L)} = f_{\text{share}}(h_t^{(l)}) \quad \text{通过共享映射生成深层等效V}$$
$$\text{Step 2}: \quad V_t^{\text{cache}} = [v_t^{(1)}, ..., v_t^{(l)}, \underbrace{\tilde{v}_t^{(l+1)}, ..., \tilde{v}_t^{(L)}}_{\text{shared from exit layer}}]$$
$$\text{最终}: \quad V_{\leq t'}^{\text{complete}} = \text{bigoplus}_{i \leq t'} V_i^{\text{cache}} \quad \text{保证后续token attention的KV Cache完整}$$

**对应消融**: 

---

### 模块 2: KV-shared Exit Layer设计（对应框架图 Figure 4a）

**直觉**: Exit layer产生的表示需同时满足两个角色——(a) 作为当前token的early output，(b) 作为后续token attention中的value来源。

**Baseline公式** (标准Transformer Layer):
$$\text{FFN}(\text{Attn}(X, X, X)) + X$$
其中Q/K/V均来自同一输入X的自注意力。

**变化点**: 标准exit layer独立计算，其V与backbone深层V分布不一致，无法直接复用。本文通过**value vector共享约束**强制对齐。

**本文公式（推导）**:
$$\text{Step 1}: \quad h_t^{\text{exit}} = \text{LayerNorm}(h_t^{(l)} + \text{Attn}(h_t^{(l)}, h_t^{(l)}, h_t^{(l)}))$$
$$\text{Step 2}: \quad v_t^{\text{shared}} = W_V^{\text{exit}} \cdot h_t^{\text{exit}} \quad \text{其中 } W_V^{\text{exit}} \text{ 与backbone深层 } W_V^{(L)} \text{ 共享/对齐}$$
$$\text{Step 3}: \quad \tilde{v}_t^{(l+1:L)} = \text{Repeat/Project}(v_t^{\text{shared}}) \quad \text{生成各层所需V的近似}$$
$$\text{最终输出}: \quad \hat{y}_t = \text{Softmax}(W_{\text{head}} \cdot \text{FFN}(h_t^{\text{exit}}))$$
$$\text{最终KV贡献}: \quad V_t^{\text{to cache}} = [v_t^{(1)}, ..., v_t^{(l)}, v_t^{\text{shared}}, ..., v_t^{\text{shared}}]$$

**对应消融**: Figure 3显示，采用KV共享策略后，relaxed threshold (Score≈0.15) 时ms/token接近vanilla（约1.0x），strict threshold时仍显著优于naive策略。

---

### 模块 3: 基于首层State Transition的Exit决策（对应框架图 决策模块）

**直觉**: 为避免运行完整exit layer再决定是否退出（这本身有开销），利用Figure 7发现的相关性，从首层动态预测exit可行性。

**Baseline公式** (传统Confidence-based Exit):
$$\text{Exit if } \max(\text{Softmax}(W_{\text{head}} \cdot h_t^{(l)})) > \tau$$
仅依赖当前层输出概率，未考虑深层行为预测。

**变化点**: 传统阈值对decoder-only LLM不可靠，因浅层概率分布与最终分布差异大。本文用**首层state transition similarity**作为深层backbone-exit value vector similarity的代理。

**本文公式（推导）**:
$$\text{Step 1}: \quad \Delta h_t^{(1)} = h_t^{(1)} - h_{t-1}^{(1)} \quad \text{首层state transition}$$
$$\text{Step 2}: \quad s_t = \text{sim}(\Delta h_t^{(1)}, \Delta h_{\text{ref}}^{(1)}) \quad \text{与参考transition pattern的相似度}$$
$$\text{Step 3}: \quad \hat{r}_t = g(s_t) \approx r_t = \text{sim}(v_t^{\text{backbone}(L)}, v_t^{\text{exit}}) \quad \text{预测深层V对齐度}$$
$$\text{最终决策}: \quad \text{Exit if } \hat{r}_t > \tau_{\text{score}} \text{（如Score=0.15或0.26）}$$

其中 $g(\cdot)$ 为轻量预测器（可为线性层或小MLP），利用Figure 7的统计相关性 $r=0.5536$ 训练。

**对应消融**: Figure 2a显示最优exit position分布，Score=0.26时约26% token可exit；Figure 2b显示token-level exit的显著加速潜力（理论值）。

## 实验与分析

| Method | GSM8K (ms/token, Score=0.15) | GSM8K (ms/token, Score=0.26) | HumanEval  |
|:---|:---|:---|:---|
| Vanilla (no exit) | ~1.0x baseline | ~1.0x baseline |  |
| Naive Token-level Exit | >>1.0x (显著更慢) | >>1.0x |  |
| **River-LLM (KV-shared)** | **~1.0x** (Figure 3a) | **~1.0x** (Figure 3b) |  |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/57d9051e-de40-4caf-9d15-9ec107841726/figures/Figure_3.png)
*Figure 3: Figure 3: Average ms/token of Token-level Exit usingdifference KV Cache Strategy on GSM8K. (a) Relaxedthreshold, Score ≈0.15. (b) Strict threshold, Score ≈0.25.*



**核心结果解读**: Figure 3是本文最关键的证据。横轴为不同KV Cache策略，纵轴为平均ms/token。关键发现：(a) relaxed threshold下，KV共享策略使ms/token降至接近vanilla水平（与理论加速匹配）；(b) strict threshold下，naive策略延迟爆炸，而KV共享仍可控。这直接验证了"KV缺失是early exit失效根源，KV共享是解药"的核心claim。

**消融分析**: 
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/57d9051e-de40-4caf-9d15-9ec107841726/figures/Figure_2.png)
*Figure 2: Figure 2: (a) Distribution of optimal Token-level Exitposition for Llama3.2 1B on GSM8K. Score = 0.26.(b) Token-level Exit significantly outperforms Sequence-level Exit on GSM8K.*


- KV共享机制 vs. 不共享：Figure 3中最显著的对比，不共享时ms/token随threshold严格化而急剧上升
- Exit决策质量：Figure 2a显示exit position分布集中于浅层（1-10层），说明大量token确实"简单"；Figure 7的r=0.5536相关性保证基于首层的决策可靠性

**公平性检查与局限**:
- **Baselines强度**: 对比了naive token-level exit（自有实现）和vanilla推理，但未与LayerSkip、Eagle等SOTA推理加速方法直接对比ms/token
- **模型规模**: 实验集中于Llama3.2 1B/3B，更大模型（7B/70B）的KV Cache压力更大，但验证待补充
- **任务范围**: GSM8K（数学推理）和HumanEval（代码），长文本生成任务未报告
- **Score阈值含义**: 文中Score定义待更明确（似为exit ratio或confidence阈值）
- **失败案例**: 未报告何时KV共享会失效（如深层语义突然转折的token）

**计算/数据成本**: 训练改动最小化（仅exit layer和预测器），但具体训练开销。

## 方法谱系与知识库定位

**方法家族**: Early Exit / Dynamic Neural Networks → 专用于Decoder-only LLM推理加速

**Parent Method**: Token-level Early Exit（CALM, SkipDecode等）。River-LLM继承其"逐token自适应计算深度"的核心思想，但通过KV共享机制解决了其在decoder-only架构上的根本不适配问题。

**直接Baselines及差异**:
| Baseline | 本文差异 |
|:---|:---|
| CALM / SkipDecode | 解决其KV Cache缺失问题，使token-level exit在decoder-only LLM上实际可行 |
| LayerSkip | LayerSkip跳过整个层，需训练时特殊设计；River-LLM保持backbone完整，token选择性exit |
| Medusa / Eagle (Speculative Decoding) | 无需draft model，避免验证开销和额外内存；直接嵌入backbone而非外挂系统 |
| Mixture of Depths (MoD) | MoD在训练时学习token路由，计算节省固定；River-LLM推理时动态exit，更灵活 |

**改动槽位**: 
- Architecture: 新增KV-shared exit layer（轻量分支）
- Objective: 增加value vector对齐约束
- Training recipe: 最小改动，主要利用预训练backbone
- Data curation: 无特殊要求
- Inference: 核心创新——seamless exit调度，基于首层预测动态决策

**Follow-up方向**:
1. **Multi-exit层级**: 当前似为单层exit，扩展至多层渐进exit，进一步加速简单token
2. **与Speculative Decoding融合**: KV共享的early exit作为draft机制，与验证阶段协同
3. **长上下文场景**: KV Cache压力随序列长度指数增长，KV共享在100K+上下文中的扩展性验证

**知识库标签**: 
- Modality: Text / Language Model
- Paradigm: Autoregressive Generation (Decoder-only)
- Scenario: Inference Acceleration, Edge Deployment
- Mechanism: Early Exit, KV Cache Optimization, Dynamic Computation
- Constraint: Minimal Training Overhead, Exact Output Preservation (非有损加速)

