---
title: The Expressive Power of Transformers with Chain of Thought
type: paper
paper_level: A
venue: ICLR
year: 2024
paper_link: null
aliases:
- 思维链Transformer的表达能力理论刻画
- Transformers wit
- Transformers with Chain of Thought (theoretical model)
- The computational power of decoder-
acceptance: Poster
method: Transformers with Chain of Thought (theoretical model)
modalities:
- Text
followups:
- 固定位大小Transformer_Post-machine-sim
---

# The Expressive Power of Transformers with Chain of Thought

**Topics**: [[T__Reasoning]] | **Method**: [[M__Transformers_with_Chain_of_Thought]]

> [!tip] 核心洞察
> The computational power of decoder-only transformers fundamentally increases with chain of thought length, with exact characterizations: logarithmic steps yield only slight gains, linear steps enable recognizing all regular languages (under conjectures), and polynomial steps with generalized pre-norm exactly characterize polynomial-time solvable problems.

| 中文题名 | 思维链Transformer的表达能力理论刻画 |
| 英文题名 | The Expressive Power of Transformers with Chain of Thought |
| 会议/期刊 | ICLR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2310.07923) · [Code](无) · [Project](无) |
| 主要任务 | Reasoning / Formal language recognition |
| 主要 baseline | Standard transformer (immediate output); Transformer with pre-norm; Chain of thought prompting (Wei et al. 2022) |

> [!abstract] 因为「标准Transformer直接输出时被证明无法解决图连通性、有限状态机模拟等序列推理问题」，作者在「Standard transformer (immediate output)」基础上改了「引入可变长度中间生成（chain of thought）并定义projected/generalized pre-norm架构变体」，在「复杂性类L / Regular languages / P」上取得「O(log n)步精确刻画L，O(n)步+projected pre-norm在所有正则语言（基于标准猜想），poly(n)步+generalized pre-norm精确刻画P」

- **O(log n)中间步**: Transformer decoder精确识别 SPACE(log n) = **L**，仅比直接输出略有提升
- **O(n)中间步 + projected pre-norm**: 在标准复杂性猜想下识别**所有正则语言**，上界为上下文敏感语言
- **poly(n)中间步 + generalized pre-norm**: **精确刻画P类问题**，首个Transformer变体的精确复杂性类对应

## 背景与动机

大型语言模型在算术、逻辑推理等任务上表现出色，但标准Transformer存在一个根本性限制：它们在读取输入后**立即输出答案**，缺乏显式的中间计算过程。这导致其被严格限制在浅层电路类（如AC⁰相关类），无法解决需要序列状态追踪的问题——例如判断有向图是否连通（NL完全）、模拟有限状态机、或执行多步算术进位。

现有研究从三个方向试图突破这一限制：

1. **Chain of Thought Prompting (Wei et al. 2022)**：通过提示让模型生成"让我们逐步思考"等中间推理文本，实证上显著提升了复杂推理能力。但该工作仅提供经验观察，未刻画其理论计算边界。

2. **Scratchpad Intermediate Computation (Nye et al. 2021)**：允许模型在专用scratchpad区域进行中间计算，类似思维链的前身。同样缺乏形式化分析。

3. **Memory-Augmented Transformers**：通过外部记忆扩展Transformer，但引入非标准组件，偏离了纯自回归生成的范式。

**核心缺口**：思维链的实证成功与其理论能力之间存在着巨大鸿沟。标准Transformer的立即输出被Merrill & Sabharwal (2023b)、Merrill et al. (2022)等人严格证明无法处理图连通性、有限状态模拟等问题；但**当允许生成中间token时，Transformer的计算能力究竟增长了多少？能否精确对应标准复杂性类？** 这一问题完全未解决。本文正是要填补这一理论空白，建立思维链长度与计算能力的精确对应关系。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b51e77e7-0f35-4d22-8c95-f0aa0bf9ad6d/figures/Figure_1.png)
*Figure 1 (result): Summary of results: transformers with intermediate generation against various classes of formal languages. A logarithmic number of chain-of-thought steps increases the expressivity of transformers to solve all regular languages (Reg; left) and counter languages (1-counter; center), while a linear number of steps extends this to context-free languages (CFL; right). We simulate certain finite-state (1-counter, pushdown) automata. The colors indicate classes of automata: finite-state (green), counter (blue), and pushdown (yellow).*



## 核心创新

**核心洞察**：思维链的长度是控制Transformer计算能力的"旋钮"，因为对数/线性/多项式步数分别对应不同的信息累积机制——对数步仅允许有限状态传播，线性步配合投影残差可模拟有限自动机，多项式步配合广义残差结构可模拟完整图灵机计算——从而使**首个Transformer变体的精确复杂性类刻画**成为可能。

| 维度 | Baseline (Standard Transformer) | 本文 |
|:---|:---|:---|
| 推理策略 | 单遍生成：读取输入后立即输出答案 | 多步生成：生成可变长度中间token序列再输出 |
| 归一化架构 | 标准pre-norm (LayerNorm → Attention/FFN → 残差加) | Projected pre-norm（残差经线性投影再加）；Generalized pre-norm（进一步推广） |
| 状态传播 | 无显式状态，单次前向传播 | 通过中间token实现类循环的状态传播，每步条件于所有历史 |
| 理论刻画 | 仅知⊆ AC⁰或相关浅层电路类，无精确对应 | **精确对应**：O(log n)步=L；O(n)步+projected pre-norm=Regular；poly(n)步+generalized pre-norm=P |

关键 novelty 在于**projected pre-norm**和**generalized pre-norm**这两个架构抽象：它们是对标准pre-norm的轻微推广，但恰好填补了从有限自动机模拟到多项式时间图灵机模拟所需的表达能力缺口。

## 整体框架



本文的理论框架将decoder-only Transformer的推理过程重新建模为**带中间生成的形式语言识别器**。数据流如下：

**Input encoding** → 输入字符串x（长度n）经嵌入层和位置编码，得到初始隐状态表示h₀。

**Chain-of-thought generation** → Decoder以自回归方式生成中间token序列z₁, z₂, ..., z_T，其中每个z_t条件于所有先前token（输入+已生成的z₁...z_{t-1}）。T的长度由分析设定：T ∈ O(log n)、O(n)、或poly(n)。

**Projected pre-norm layer（线性步场景）** → 标准pre-norm的残差连接被替换为：先经线性投影W_proj映射，再与主路径输出相加。即 h' = W_proj · h + F(LayerNorm(h))。该设计使残差分支可编码有限状态机的状态转移。

**Generalized pre-norm layer（多项式步场景）** → 进一步推广投影结构，允许更复杂的状态更新函数，支持模拟多项式时间图灵机的配置转移。

**Final prediction head** → 在生成T个中间token后，最终隐状态h_T经线性分类器输出接受/拒绝决策（形式语言识别）或计算答案。

```
输入 x (|x|=n)
    ↓
[Embedding + Positional Encoding]
    ↓
h₀ ──→ [Decoder Layer 1] ──→ z₁ ──→ [Decoder Layer 2] ──→ z₂ ──→ ... ──→ z_T
         ↑___________________________|（每步attend to所有历史token）
    （T = O(log n): 标准pre-norm → 识别L）
    （T = O(n): projected pre-norm → 识别Regular）
    （T = poly(n): generalized pre-norm → 识别P）
    ↓
[Output Head] → 接受/拒绝
```

框架的核心参数是**思维链长度T**和**归一化变体**的组合，二者共同决定识别的复杂性类。

## 核心模块与公式推导

### 模块 1: 对数步思维链与对数空间刻画（对应框架图左侧）

**直觉**: 即使允许少量中间token，若步数仅对数级，Transformer仍受注意力机制的信息瓶颈限制，无法累积足够状态跨越对数空间。

**Baseline 公式** (Standard transformer, immediate output):
$$\text{Recognized class} \subseteq \mathbf{AC}^0 \text{-related circuit classes}$$
符号: 标准Transformer被证明等价于常数深度、多项式规模的阈值电路，无法识别需要状态记忆的语言如 $\{a^n b^n\}$ 的某些变体。

**变化点**: 引入O(log n)个中间解码步，但保持标准pre-norm架构不变。关键观察：每步attention的query-key点积受log-precision限制，log步仅允许O(log n)比特的信息通过残差连接累积。

**本文公式（推导）**:
$$\text{Step 1: 构造下界} \quad \text{模拟对数空间图灵机} \quad \text{用log步存储工作带内容于attention pattern}$$
$$\text{Step 2: 上界分析} \quad \text{每步残差更新 } h_{t+1} = h_t + f_t(\text{LayerNorm}(h_t)) \text{ 仅增加O(log n)信息量}$$
$$\text{最终}: O(\log n) \text{ steps} \Leftrightarrow \mathbf{L} = \text{SPACE}(\log n)$$

**对应消融**: 无显式消融表（理论证明），但定理表明即使将步数从O(1)增至O(log n)，仍无法触及NL完全问题如有向图连通性。

---

### 模块 2: 线性步 + Projected Pre-norm 与正则语言（对应框架图中层）

**直觉**: 有限状态自动机的核心是状态转移函数 δ: Q × Σ → Q。标准pre-norm的残差连接 h + F(LN(h)) 中，恒等映射"h"部分固定不变，无法灵活编码状态转移；将残差投影为 W·h 后，可学习任意线性（进而模拟有限）状态更新。

**Baseline 公式** (Standard pre-norm):
$$h_{\text{ell}+1} = h_\text{ell} + \text{Attn}_\text{ell}(\text{LayerNorm}(h_\text{ell})) + \text{FFN}_\text{ell}(\text{LayerNorm}(h_\text{ell}))$$
符号: h_ℓ 为第ℓ层输入，Attn为多头自注意力，FFN为前馈网络。残差中的恒等项 h_ℓ 强制状态以固定方式传递。

**变化点**: 标准pre-norm无法模拟非平凡状态机，因为恒等残差不允许学习状态转移映射。将残差投影改为 W_proj · h_ℓ，使网络可学习"读取新符号 → 更新内部状态"的转移规则。

**本文公式（推导）**:
$$\text{Step 1: Projected pre-norm定义} \quad h_{\text{ell}+1} = W_{\text{proj}} \cdot h_\text{ell} + \text{Attn}_\text{ell}(\text{LayerNorm}(h_\text{ell})) + \text{FFN}_\text{ell}(\text{LayerNorm}(h_\text{ell}))$$
$$\text{Step 2: 有限自动机模拟} \quad \text{用 } W_{\text{proj}} \text{ 编码转移矩阵，O(n)步遍历输入，每步更新有限状态}$$
$$\text{Step 3: 正则语言完备性} \quad \text{基于NC}^1 \neq \mathbf{L} \text{ 等标准猜想，证明线性步projected pre-norm可识别所有正则语言}$$
$$\text{最终}: O(n) \text{ steps} + \text{projected pre-norm} \Leftrightarrow \text{All Regular Languages} \text{ (under conjectures)}$$

**对应消融**: 定理表明去掉projected pre-norm（回归标准pre-norm）则无法识别所有正则语言——这是架构修改的必要性证明。

---

### 模块 3: 多项式步 + Generalized Pre-norm 与P类刻画（对应框架图右侧）

**直觉**: 要模拟多项式时间图灵机，需要更强大的状态更新能力——不仅线性投影，还需支持类似图灵机配置的完整读写与转移模拟。

**Baseline 公式** (Projected pre-norm, linear steps):
$$h_{\text{ell}+1} = W_{\text{proj}} \cdot h_\text{ell} + \text{Attn}_\text{ell}(\text{LayerNorm}(h_\text{ell})) + \text{FFN}_\text{ell}(\text{LayerNorm}(h_\text{ell}))$$
符号: W_proj 为固定线性投影，仅支持有限状态更新。

**变化点**: Projected pre-norm的线性投影仍不足以编码图灵机的多项式大小配置。Generalized pre-norm放宽投影结构，允许更复杂的函数形式（具体构造涉及分层状态表示），使得每步可模拟图灵机的一步转移，同时保持整体可学习性。

**本文公式（推导）**:
$$\text{Step 1: Generalized pre-norm定义} \quad h_{\text{ell}+1} = \mathcal{G}(h_\text{ell}; \theta_\text{ell}) + \text{Attn}_\text{ell}(\text{LayerNorm}(h_\text{ell})) + \text{FFN}_\text{ell}(\text{LayerNorm}(h_\text{ell}))$$
$$\text{其中 } \mathcal{G} \text{ 为广义投影，支持多项式大小配置的编码与更新}$$
$$\text{Step 2: 多项式时间模拟} \quad \text{poly(n)步每步模拟图灵机一步，配置存储于attention pattern与残差状态}$$
$$\text{Step 3: 上下界闭合} \quad \text{上界：每步多项式时间可计算，poly步仍∈ P；下界：可模拟任意多项式时间DTM}$$
$$\text{最终}: \text{poly}(n) \text{ steps} + \text{generalized pre-norm} \Leftrightarrow \mathbf{P} \text{ (exact characterization)}$$

**对应消融**: 这是本文的旗舰结果——首个精确刻画。若步数降为sub-polynomial或架构退化为projected pre-norm，则只能识别P的真子集（如Regular ⊊ P）。

## 实验与分析





本文是**纯理论论文**，不包含传统意义上的训练实验或基准测试。"实验"体现为**形式化定理证明与复杂性类构造**。Figure 1 以图示方式总结了三个核心结果：Transformer decoder配合不同长度思维链所对应的复杂性类层级——从L到Regular再到P的严格递进。

**核心结果综述**：作者在形式语言识别框架下建立了完整的三层刻画。对于输入长度n，当允许生成O(log n)个中间token时，Transformer decoder精确识别对数空间类 **L = SPACE(log n)**；这一结果通过模拟对数空间图灵机并证明信息上界得到。当步数增至O(n)并采用projected pre-norm架构时，在**NC¹ ≠ L**等标准复杂性猜想下，该模型可识别**所有正则语言**，同时被证明包含于上下文敏感语言。最具突破性的结果是：当步数扩展至poly(n)并采用generalized pre-norm时，Transformer**精确刻画多项式时间类P**——这是首个将Transformer变体与标准复杂性类完全等同的结果，表明多项式长度思维链赋予了Transformer与确定性图灵机在多项式时间意义下等价的计算能力。

**关键对比与gap分析**：标准Transformer（立即输出）被先前工作证明无法解决图连通性（NL完全）或有限状态模拟问题；而本文证明，仅需**对数步**即可跨越至L类（包含图连通性），**线性步+projected pre-norm**即可覆盖所有正则语言，**多项式步+generalized pre-norm**则完整覆盖P。这一递进严格量化了"思维链越长，推理能力越强"的直觉。

**公平性检查**：
- Baseline选择合理：标准Transformer（Merrill & Sabharwal 2023a）和pre-norm Transformer（Xiong et al. 2020）均为理论分析中的标准参照。
- 主要限制明确标注：(1) 正则语言结果依赖未证明的复杂性猜想（NC¹ ≠ L等）；(2) projected/generalized pre-norm是理论抽象，非实用架构；(3) 结果假设理想参数，未考虑实际训练动态（如梯度消失、优化困难）。
- 缺失比较：未与RNN/LSTM等天然具有循环结构的模型对比，未与显式记忆增强架构（如[20] Memory augmented LLMs）比较，亦无实证验证（如在实际LLM上测试思维链长度与解题能力的关系）。
- 证据强度：作为纯理论工作，其证明严谨性取决于复杂性理论标准，但缺乏实验验证使直观可信度受限（overall evidence strength: 0.6）。

## 方法谱系与知识库定位

**方法家族**: Transformer expressivity evolution（Transformer表达能力演进谱系）

**Parent method**: Standard transformer (immediate output) [Merrill & Sabharwal 2023a "A logic for expressing log-precision transformers"] —— 本文直接构建于该工作的形式化框架，将其从"立即输出"扩展至"中间生成"。

**改变的slots**:
| Slot | 变化 |
|:---|:---|
| inference_strategy | 单遍生成 → 多步chain-of-thought生成 |
| architecture | 标准pre-norm → projected pre-norm → generalized pre-norm |
| credit_assignment | 单次前向-反向 → 通过中间token的类循环状态传播 |

**直接Baseline对比**:
- **Standard transformer (immediate output)** [14]: 本文证明其⊊ L，无法处理序列推理；本文扩展为可变长度中间生成。
- **Chain of thought prompting** [23] (Wei et al. 2022): 实证发现思维链提升推理，但无理论刻画；本文提供首个精确复杂性类对应。
- **Towards revealing the mystery behind chain of thought** [4] (Feng et al. 2023): 同期理论工作，但本文首次给出精确刻画（exactly P）而非上下界估计。

**后续方向**:
1. **实证验证**: 在实际训练的大型decoder模型上检验——多项式长度思维链是否真能学习P完全问题？训练动态是否与理论预测一致？
2. **架构实用化**: 将projected/generalized pre-norm转化为可训练的标准组件，验证其对实际推理任务的增益。
3. **细粒度刻画**: 介于O(log n)与O(n)之间的步数（如O(log^k n)）对应何种复杂性类？是否存在连续谱系？

**标签**: modality=text | paradigm=autoregressive_generation | scenario=theoretical_analysis | mechanism=chain_of_thought / projected_pre-norm / generalized_pre-norm | constraint=exact_complexity_characterization / conditional_on_conjectures

## 引用网络

### 后续工作（建立在本文之上）

- [[P__固定位大小Transformer_Post-machine-sim]]: Most cited reference (6 times); directly about transformer expressiveness with C

