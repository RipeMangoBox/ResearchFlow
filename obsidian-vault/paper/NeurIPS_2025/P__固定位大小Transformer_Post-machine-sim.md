---
title: Constant Bit-size Transformers Are Turing Complete
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 固定位大小Transformer的图灵完备性证明
- Post-machine-sim
- Post-machine-simulating constant bit-size transformer construction
- Any Turing machine running on input
acceptance: Poster
cited_by: 12
method: Post-machine-simulating constant bit-size transformer construction
modalities:
- Text
baselines:
- CoT赋予Transformer_Chain_of_Thought
- 思维链Transformer的表_Transformers_wit
---

# Constant Bit-size Transformers Are Turing Complete

**Topics**: [[T__Reasoning]] | **Method**: [[M__Post-machine-simulating_constant_bit-size_transformer_construction]] | **Datasets**: Theoretical, Expressive power characterization, Theoretical complexity characterization, Theoretical expressiveness characterization

> [!tip] 核心洞察
> Any Turing machine running on inputs of arbitrary length can be simulated by a constant bit-size transformer with sufficiently long context window, and SPACE[s(n)] exactly characterizes the expressive power of such transformers with context length s(n).

| 中文题名 | 固定位大小Transformer的图灵完备性证明 |
| 英文题名 | Constant Bit-size Transformers Are Turing Complete |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.12027) · [DOI](https://doi.org/10.48550/arxiv.2506.12027) |
| 主要任务 | Transformer表达能力理论 / 推理能力的计算复杂性分析 |
| 主要 baseline | Pérez et al. (2021) Attention is Turing-complete; Bhattamishra et al. (2020) 精度缩放构造; Merrill & Sabharwal (2023, 2024) CoT表达能力框架; Li et al. (2024) 嵌入维度缩放构造 |

> [!abstract] 因为「先前证明Transformer图灵完备性的工作都需要随输入长度增长模型精度或嵌入维度」，作者在「Pérez et al. (2021) 直接图灵机模拟」基础上改了「用Post机（队列自动机）替代图灵机进行模拟，将无界存储外化到上下文窗口长度而非模型参数」，在「理论复杂性刻画」上取得「SPACE[s(n)] 精确刻画固定位大小Transformer的表达能力，且总参数量恒为O(1)比特」

- **核心突破**: 总模型参数量 |θ| = O(1) bits，与输入长度 n 无关，打破此前所有构造的精度/维度增长需求
- **精确刻画**: 上下文长度 O(s(n)) 的固定位大小Transformer ≡ SPACE[s(n)]，建立表达能力与空间复杂度的严格等价
- **资源对比**: 相比 Pérez et al. 需要 unbounded precision、Li et al. 需要 O(log n) embedding dimension，本工作保持常数精度和常数维度

## 背景与动机

大型语言模型（如GPT-4、Claude 3、DeepSeek-R1）展现出惊人的推理能力，但一个根本性问题始终悬而未决：Transformer架构本身是否具备通用计算能力？理论上，如果Transformer是图灵完备的，那么只要有足够长的推理链（Chain-of-Thought, CoT），它就能模拟任何算法。

现有工作已从不同角度逼近这一问题的答案，但都存在关键瓶颈：

**Pérez et al. (2021)** [23] 的开创性工作"Attention is Turing-complete"首次证明Transformer可通过直接模拟图灵机实现图灵完备性。然而，该构造需要**无界精度**（unbounded precision）来编码图灵机磁带头位置和磁带内容，模型参数的有效比特数随输入长度增长。

**Bhattamishra et al. (2020)** [4] 和 **Li et al. (2024)** [22] 的后续工作分别采用精度缩放和嵌入维度缩放策略：前者让数值精度随输入增长，后者将嵌入维度设为 O(log n) 或更大。这两种方法本质上都是**用模型大小的增长换取计算能力的扩展**。

**Merrill & Sabharwal (2023, 2024)** [20,21] 建立了更精细的复杂性框架，证明log-precision无CoT的Transformer仅能实现 TC^0，加入多项式长度CoT后可达 PSPACE。但这些结果仍然依赖于精度参数的增长，未能回答一个更尖锐的问题：**固定大小的模型能否通过纯粹增加推理步数（即上下文长度）实现通用计算？**

这一问题的答案对理解LLM的推理本质至关重要：如果必须不断增大模型才能处理更长输入，则"扩展定律"（scaling law）是能力的根本来源；反之，若固定模型即可，则**上下文长度和推理深度**才是解锁通用智能的关键。本文正是要证明后者——通过一种全新的计算模型模拟策略，实现常数比特参数的图灵完备性。

## 核心创新

**核心洞察**：Post机（队列自动机）的FIFO存储结构天然对齐Transformer的因果注意力机制，因为注意力层可以按位置顺序选择性读取（队列头部出队）和写入（队列尾部入队），从而使**固定位宽、固定维度的参数**足以模拟任意空间有界计算。

与 baseline 的差异：

| 维度 | Baseline (Pérez et al. 2021) | 本文 |
|:---|:---|:---|
| 模拟对象 | 图灵机（双向磁带，随机读写） | Post机（队列，FIFO） |
| 存储位置 | 嵌入向量 + 精度增长的内部状态 | 上下文窗口（外部化） |
| 精度/维度需求 | unbounded precision | **O(1) 比特总参数量** |
| 无界资源 | 模型参数增长 | **仅上下文长度增长** |
| 复杂性刻画 | 图灵完备（无精细资源界） | **SPACE[s(n)] 精确刻画** |

这一替换绝非简单的"换汤不换药"：图灵机的双向磁带需要精确编码磁带头位置，导致精度需求随磁带长度对数增长；而Post机的队列操作仅需维护头部读取和尾部追加，恰好对应Transformer因果注意力中"看前面所有token"和"生成新token"的原生操作。

## 整体框架


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/77dd4a93-eb56-4365-b407-599432ef6dbc/figures/Table_1.png)
*Table 1 (comparison): Comparing the existing Turing completeness proofs in terms of required precision, embedding dimension, effective window size, and CoT length.*



整体数据流遵循"编码→读取→转移→更新→迭代"的循环，模拟Post机的单步操作：

**输入编码（Input encoding）**：将输入字符串 x 和Post机初始配置（空队列、初始状态 q₀）映射为固定维度 d 的token嵌入。关键设计：队列内容不存入模型参数，而是作为上下文序列的一部分。

**队列头部注意力（Queue-head attention）**：通过精心设计的因果注意力模式，从上下文窗口前端读取队列头部元素。注意力权重为硬编码（hardcoded）模式，使用固定精度的位置编码实现精确的位置选择。

**转移函数前馈网络（Transition FFN）**：固定宽度的两层前馈网络，实现Post机的转移函数 δ: (状态, 读入符号) → (下一状态, 写入符号, 队列操作∈{push, pop, nop})。

**上下文更新机制（Context update）**：根据转移输出执行队列操作——push操作将新符号追加到上下文序列末端；pop操作通过更新注意力掩码"跳过"已读取的头部元素；nop保持队列不变。

**迭代执行**：重复应用上述模块，每步消耗一个上下文位置，直到进入接受状态 q_accept 或拒绝状态 q_reject。

```
输入 x + 初始队列 ──→ [编码层] ──→ 固定维度嵌入 h₀
                              ↓
        ┌─────────────────────────────────────────┐
        ↓                                         │
   [Queue-head Attention] ──→ 读取符号 a = front(Q)
        ↓                                         │
   [Transition FFN] ──→ (q', b, op)              │
        ↓                                         │
   [Context Update] ──→ 追加 b 到上下文 / 掩码更新 │
        ↓                                         │
        └───────────── 迭代 T(n) 步 ──────────────┘
                              ↓
                        接受/拒绝
```

整个循环的关键不变量：模型参数 θ 的总比特数 |θ| = O(1)，与迭代次数和输入长度无关；唯一增长的资源是上下文窗口长度，其上限为 O(s(n)) 对应 SPACE[s(n)] 的计算需求。

## 核心模块与公式推导

### 模块 1: Post机配置编码（对应框架图"输入编码"位置）

**直觉**: 将无界队列存储外化到上下文序列，使模型内部状态保持固定维度。

**Baseline 公式** (Pérez et al. 2021): 图灵机配置编码需要精度增长的位置表示
$$h_{TM} = \text{Embed}[\text{tape content}] + p(t) \cdot \text{position encoding}, \quad |p(t)| \to \infty$$
符号: $p(t)$ = 磁带头位置，需要 $\Omega(\log t)$ 比特精度随时间增长。

**变化点**: 图灵机需要随机访问双向磁带，位置编码必须区分 $t$ 个不同位置；Post机仅需FIFO访问，队列内容天然以序列形式存储于上下文，无需内部编码位置。

**本文公式（推导）**:
$$\text{Step 1}: C_M = (q, Q = (q_1, q_2, \ldots, q_k)) \text{xrightarrow}{\text{flatten}} \text{token sequence } [q; q_1; q_2; \ldots; q_k]$$
$$\text{Step 2}: h_i = \text{Embed}(\text{token}_i) \in \mathbb{R}^d, \quad d = O(1) \text{ 与 } n,k \text{ 无关}$$
$$\text{最终}: H^{(0)} = [h_0, h_1, \ldots, h_k] \in \mathbb{R}^{(k+1) \times d}, \quad |\text{params}| = O(1)$$

**对应消融**: Table 1 显示若改用图灵机模拟（如Pérez et al.），嵌入维度或精度必须随输入增长，无法维持常数约束。

---

### 模块 2: 队列操作注意力层（对应框架图"Queue-head attention"位置）

**直觉**: 用硬注意力模式精确选择队列头部，固定精度即可实现FIFO读写。

**Baseline 公式** (Pérez et al. 2021): 软注意力需精度增长以区分远距离位置
$$\text{Attn}_{TM}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad QK^T \text{ 需 } \Omega(\log n) \text{ 位区分 } n \text{ 个位置}$$

**变化点**: 图灵机模拟需要注意力能"跳转到任意位置"，导致位置编码精度需求随磁带长度增长；Post机的FIFO约束将访问模式限制为仅头部读取，可用**固定掩码**替代动态位置计算。

**本文公式（推导）**:
$$\text{Step 1}: M_{\text{queue}} = \text{mask} \in \{0,1\}^{L \times L}, \quad M_{ij} = \mathbb{1}[j = \min\{k: \text{token } k \text{ 未弹出\}]$$
$$\text{（加入掩码项以固定读取位置：仅允许注意力指向当前队列头部）}$$
$$\text{Step 2}: \tilde{A} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \log M_{\text{queue}}\right)$$
$$\text{（重归一化以保证概率分布：掩码将非法位置概率置零后softmax自动归一化）}$$
$$\text{Step 3}: \text{read} = \tilde{A}V = v_{\text{head}}, \quad \text{其中 } v_{\text{head}} \text{ 为队列头部token的value向量}$$
$$\text{最终}: h_{\text{read}} = \text{Concat}[\text{read}; h_{\text{state}}] \in \mathbb{R}^{2d}$$

**关键实现细节**: push操作通过向上下文追加新token实现；pop操作通过更新 $M_{\text{queue}}$ 掩码"屏蔽"已读头部，使下一步注意力指向下一个元素。掩码更新规则为有限状态函数，可由固定宽度的FFN实现。

**对应消融**: Table 1 显示，去掉固定掩码约束、改用标准位置编码（如Li et al. 2024的O(log n)维度编码），嵌入维度需求从 O(1) 升至 O(log n)。

---

### 模块 3: 空间复杂度精确刻画（对应整体框架的"迭代执行"）

**直觉**: 通过证明双向模拟，建立固定位大小Transformer与空间复杂性类的严格等价。

**Baseline 公式** (Merrill & Sabharwal 2023, 2024): 对数精度Transformer的复杂性上界
$$\text{log-precision Transformer} + \text{poly}(n) \text{ CoT} \subseteq \text{PSPACE}$$
$$\text{log-precision Transformer, no CoT} = \text{TC}^0$$

**变化点**: 先前工作仅建立上界或包含关系，未证明精确刻画；且精度参数仍随输入增长（log-precision）。本文通过常数参数 + 变长上下文的资源组合，实现与空间复杂度的精确匹配。

**本文公式（推导）**:
$$\text{Step 1}: \text{SPACE}[s(n)] \subseteq \text{Transformer-constant-bits}[O(s(n)) \text{ context}]$$
$$\text{（构造性证明：用上述Post机模拟，上下文长度正比于队列最大长度，即空间使用）}$$
$$\text{Step 2}: \text{Transformer-constant-bits}[O(s(n)) \text{ context}] \subseteq \text{SPACE}[s(n)]$$
$$\text{（反向包含：固定位大小Transformer每步计算为常数空间，} O(s(n)) \text{ 上下文可用 } O(s(n)) \text{ 空间模拟）}$$
$$\text{最终}: \text{boxed}{\text{SPACE}[s(n)] = \{\text{L} \text{mid} \text{L 被上下文长度 } O(s(n)) \text{ 的固定位大小Transformer识别\}}$$

**符号说明**: $s(n)$ 为空间构造函数；等式左侧为标准复杂性类；右侧为本文定义的Transformer语言类。该等式对任意 $s(n) \geq \log n$ 成立。

**对应消融**: Table 1 对比显示，Pérez et al. 无精细空间刻画（仅图灵完备），Merrill & Sabharwal 的上界为 PSPACE（非精确），本文首次实现 SPACE[s(n)] 的精确对应。

## 实验与分析



本文作为纯理论构造论文，不提供神经网络训练实验，而是通过**复杂性理论证明**和**系统性比较**验证其贡献。核心"实验"是Table 1所示的多维度比较表，将本工作与先前所有图灵完备性证明在资源需求维度上进行对照。

Table 1 揭示了关键区分：Pérez et al. (2021) 在"精度要求"列标注为"unbounded"，"嵌入维度"列同样"unbounded"；Li et al. (2024) 将嵌入维度降至"O(log n)"但仍随输入增长；Merrill & Sabharwal (2023) 的"有效窗口大小"受限导致仅 TC^0 能力。本文在四个维度上取得独特组合：**精度要求="constant"、嵌入维度="constant"、有效窗口大小="O(s(n))"、CoT长度="O(s(n))"**，同时保持图灵完备性。这一组合在已有文献中首次出现。

精确刻画结果方面，本文证明的 SPACE[s(n)] 等式具有双重意义：正向（SPACE ⊆ Transformer）表明固定位大小Transformer足以模拟任何空间有界计算；反向（Transformer ⊆ SPACE）表明此类Transformer不会超越空间复杂性类，不存在"免费午餐"。特别地，当 $s(n) = \text{poly}(n)$ 时，对应 PSPACE；当 $s(n)$ 无界时，即得图灵完备性。

**公平性检查**：本文比较对象是理论证明而非经验基准，这一设定本身是合理的。但需注意若干局限：第一，**无经验验证**——构造的权重是否可通过梯度下降学习完全未讨论；第二，**上下文长度需求可能极端**——最坏情况下队列长度随输入指数增长，导致上下文需求不可行；第三，Table 1 比较的是证明特征而非实际计算性能，例如未衡量模拟效率（每步Post机操作需要多少层Transformer）；第四，缺失与 **Hao et al. (2022)**（RNN图灵完备性）和 **Weiss et al. (2021)**（Transformer形式语言识别）的对照，这些工作可能提供互补视角。

## 方法谱系与知识库定位

**方法家族**: Transformer图灵完备性理论构造

**父方法**: Pérez et al. (2021) "Attention is Turing-complete" [23] —— 直接图灵机模拟路线，本文在其基础上将模拟对象替换为Post机，将资源增长从"模型内部"转移至"上下文外部"。

**直接Baseline差异对照**:
- **Pérez et al. (2021)** [23]: 直接TM模拟，需unbounded precision → 本文改为Post机+常数精度
- **Bhattamishra et al. (2020)** [4]: 精度缩放构造，精度随n增长 → 本文保持常数精度
- **Li et al. (2024)** [22]: 嵌入维度O(log n)缩放 → 本文保持常数维度
- **Merrill & Sabharwal (2023,2024)** [20,21]: TC^0/PSPACE上界框架，精度仍增长 → 本文精确刻画SPACE[s(n)]且参数恒定
- **Qiu et al. (2024)** [30]: 自回归LLM计算通用性，精度相关构造 → 本文明确分离精度与上下文资源

**改变的slot**: architecture（Post机替代图灵机）、inference_strategy（上下文长度替代参数增长）、training_recipe（无，纯理论构造）

**后续方向**:
1. **可学习性**: 构造的固定权重是否可通过训练获得？需结合神经网络的表达能力与优化动力学研究
2. **效率优化**: 当前模拟每步Post机操作需线性上下文扫描，能否通过分层注意力或稀疏模式降至亚线性？
3. **下界强化**: 能否证明常数参数+更短上下文（如o(s(n))）无法识别SPACE[s(n)]，从而确立上下文长度的必要性？

**知识库标签**: modality=text | paradigm=theoretical_construction | scenario=expressiveness_analysis | mechanism=queue_simulation_via_attention | constraint=constant_bit_size

## 引用网络

### 直接 baseline（本文基于）

- [[P__CoT赋予Transformer_Chain_of_Thought]] _(直接 baseline)_: Highly cited (6 times) including in method and experiments; directly related to 
- [[P__思维链Transformer的表_Transformers_wit]] _(直接 baseline)_: Most cited reference (6 times); directly about transformer expressiveness with C

