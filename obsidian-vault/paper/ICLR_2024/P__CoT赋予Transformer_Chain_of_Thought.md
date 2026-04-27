---
title: Chain of Thought Empowers Transformers to Solve Inherently Serial Problems
type: paper
paper_level: C
venue: ICLR
year: 2024
paper_link: null
aliases:
- CoT赋予Transformer串行计算能力的理论证明
- Chain of Thought
- Chain of Thought (CoT) for Theoretical Expressiveness Analysis
- CoT empowers constant-depth transfo
acceptance: Poster
cited_by: 253
method: Chain of Thought (CoT) for Theoretical Expressiveness Analysis
modalities:
- Text
paradigm: supervised
followups:
- 固定位大小Transformer_Post-machine-sim
---

# Chain of Thought Empowers Transformers to Solve Inherently Serial Problems

**Topics**: [[T__Reasoning]], [[T__Math_Reasoning]] | **Method**: [[M__Chain_of_Thought_(CoT)_for_Theoretical_Expressiveness_Analysis]] | **Datasets**: Permutation Composition, Iterated Squaring, Circuit Value Problem, Modular Addition

> [!tip] 核心洞察
> CoT empowers constant-depth transformers with constant-bit precision to solve inherently serial problems by enabling serial computation through intermediate step generation, dramatically expanding their expressiveness from AC0 to problems solvable by boolean circuits of size T.

| 中文题名 | CoT赋予Transformer串行计算能力的理论证明 |
| 英文题名 | Chain of Thought Empowers Transformers to Solve Inherently Serial Problems |
| 会议/期刊 | ICLR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2402.12875) · [Code](未发布) · [Project](未发布) |
| 主要任务 | Reasoning, Math Reasoning（理论分析+合成任务验证） |
| 主要 baseline | Standard Transformer (without CoT), Constant-depth transformers with finite precision |

> [!abstract] 因为「Chain of Thought (CoT) 提示为何能提升Transformer推理能力缺乏理论解释」，作者在「标准常数深度Transformer」基础上改了「推理策略——引入多步中间token生成以模拟串行计算」，在「置换群合成(Permutation Composition)、迭代平方(Iterated Squaring)、电路求值(Circuit Value Problem)」上取得「无CoT时准确率接近零，有CoT时接近完美」

- **置换群合成(S5)**：无CoT准确率≈0%，有CoT准确率→近100%（Figure 3）
- **迭代平方(IS)**：无CoT无法完成串行模幂运算，有CoT实现高准确率（Figure 4）
- **电路求值问题**：无CoT无法求解，CoT逐步模拟门电路计算达成高准确率（Figure 5）

## 背景与动机

大型语言模型通过Chain of Thought (CoT) 提示——即要求模型"逐步思考"并生成中间推理步骤——在数学推理、逻辑推理等任务上取得了惊人的实证提升。然而，这一经验性成功背后缺乏严格的理论解释：为什么加入中间token就能让模型变"聪明"？一个常数深度、有限精度的Transformer，其计算能力的理论边界究竟在哪里？

现有研究从三个角度探索了这一问题：

- **Hao et al. [16] 与 Merrill et al. [6]**：通过电路复杂度理论证明，**常数深度、常数位精度（或对数精度）的Transformer在没有CoT的情况下，其表达能力上限为AC0**——即只能解决可被浅层并行电路计算的问题。AC0类问题无法处理需要串行依赖的计算，如迭代乘法或电路求值。

- **Feng et al. [13] "Towards revealing the mystery behind chain of thought"**：首次尝试从理论角度分析CoT，但未能给出CoT使Transformer突破AC0限制的完整刻画。

- **Giannou et al. [14] "Looped transformers as programmable computers"**：展示了通过循环（权重共享+迭代应用）可以让Transformer模拟图灵机，但需要特殊的循环架构，而非标准Transformer的简单提示策略。

这些工作的核心缺口在于：**没有人证明标准Transformer仅通过改变推理策略（生成中间CoT token）就能突破AC0的限制**。特别地，一个关键问题悬而未决：如果模型架构本身（常数深度、有限精度）被严格限制在AC0内，那么CoT这种"时间换深度"的策略究竟能让它达到多强的计算能力？本文正是要填补这一理论空白。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/189b9ee3-63ad-4af2-9080-64f61b72f5d1/figures/fig_001.png)
*Figure: Interestingly, we also show that logarithmically many CoT steps do*



本文的核心发现是：CoT通过将计算从空间维度（模型深度）展开到时间维度（生成步骤），使常数深度Transformer能够模拟任意大小为T的布尔电路——这是AC0完全无法触及的计算能力。

## 核心创新

核心洞察：**CoT的本质是"深度-时间权衡"（depth-time tradeoff）**，因为Transformer的每一层计算可以重用于处理前一步生成的token，从而使常数深度模型通过T步自回归生成累积出等效于T层电路的串行计算能力，最终突破AC0限制去模拟任意布尔电路。

| 维度 | Baseline（无CoT标准Transformer） | 本文（含CoT） |
|:---|:---|:---|
| 推理策略 | 单遍自回归生成，直接输出答案 | 多步生成，中间推理步骤显式化为token |
| 理论表达能力 | AC0（常数深度电路类） | 可模拟大小为T的任意布尔电路 |
| 嵌入维度需求 | 对数精度即受限 | O(log n)嵌入维度即足够支持T=poly(n)步 |
| 计算-深度权衡 | 深度固定，计算能力封顶 | 深度固定，时间步数扩展有效计算深度 |
| 架构改动 | 无 | 无，仅改变生成方式 |

这一洞察将CoT从经验性提示技巧提升为具有严格复杂度理论保证的计算机制：每一步CoT对应布尔电路的一层计算，T步CoT即T层电路的串行模拟。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/189b9ee3-63ad-4af2-9080-64f61b72f5d1/figures/fig_002.png)
*Figure: Illustration of Theorem 3.3 on a 2-gate and 2-input circuit.*



本文的理论-实证框架包含三个层次：

**1. 下界定理（无CoT的局限）**
输入：问题实例x（如置换序列、电路描述）；输出：AC0不可判定性结论。基于Hao et al. [16]的结果，证明常数深度、常数位精度Transformer的单遍生成能力严格受限。

**2. 上界定理（有CoT的能力）**
输入：问题实例x + CoT提示；输出：可模拟任意大小T的布尔电路。核心构造：将电路的每一层编码为CoT的一步输出，通过O(log n)维状态传递实现跨步信息流动。

**3. 实证验证层**
输入：合成任务数据（置换合成、迭代平方、电路求值、模加）；输出：准确率对比。使用监督微调训练小型Transformer，验证理论预测的"无CoT则失败，有CoT则成功"现象。

数据流示意：
```
问题实例 x → [Token化 + CoT提示模板] → Transformer Block（常数深度d）
                                              ↓
                                        生成 CoT_1（第1层电路状态）
                                              ↓
                                        [拼接(CoT_1, x) → Transformer Block] 
                                              ↓
                                        生成 CoT_2（第2层电路状态）
                                              ↓
                                        ... × T 次迭代 ...
                                              ↓
                                        生成最终答案 y
```

关键：Transformer Block本身深度不变，但**通过自回归展开，等效计算深度变为 d × T**。

## 核心模块与公式推导

### 模块1: 无CoT的AC0上界定理（Theorem 3.3）

**直觉**：先确立"基线有多弱"——证明没有CoT时，即使嵌入维度为poly(n)，常数深度Transformer也无法逃出AC0的牢笼。

**Baseline公式**（Hao et al. [16], Merrill & Sabharwal [6]）：
$$\text{depth-}d, O(\log n)\text{-precision transformer} \Rightarrow \text{expressiveness} \subseteq \text{AC}^0$$

符号：$d$ = 常数层数，$n$ = 输入长度，精度 = 每个数值的比特数（对数级或常数级），AC0 = 常数深度、无界扇入的AND/OR/NOT电路类。

**变化点**：本文将此结果强化到**常数位精度**（constant-bit precision），而非对数精度，使限制更严格；并明确此限制意味着无法解决迭代平方、电路求值等串行问题。

**本文公式**：
$$\text{depth-}d, \text{constant-bit precision transformer without CoT} \in \text{AC}^0 \text{subsetneq} \text{NC}^1 \subseteq \text{P}$$

**对应消融**：Figure 3-5中所有"无CoT"曲线验证此定理——准确率接近零。

---

### 模块2: CoT电路模拟定理（Theorem 4.1，核心结果）

**直觉**：证明CoT每一步可以模拟布尔电路的一层，T步模拟T层，从而将时间步数转化为有效的串行计算深度。

**Baseline公式**（单遍生成）：
$$y = f_\theta^{(d)}(x)$$
其中$f_\theta^{(d)}$表示深度为$d$的Transformer，输出直接为答案，无中间状态。

**变化点**：基线缺乏跨时间步的状态传递机制。本文将生成过程展开为迭代计算，将前一步的输出token作为下一步的输入，形成隐式状态机。

**本文公式（推导）**：

$$\text{Step 1: 状态初始化} \quad s_0 = \text{Embed}(x), \quad \text{将输入编码为初始状态}$$

$$\text{Step 2: CoT迭代} \quad s_t = f_\theta^{(d)}([s_{t-1}; \text{CoT}_{t-1}], x), \quad t = 1, \ldots, T$$
$$\quad \text{每一步将前一步状态与已生成CoT token拼接，通过常数深度计算更新}$$

$$\text{Step 3: 电路层模拟} \quad \text{CoT}_t = g_t(s_t) \approx \text{CircuitLayer}_t(s_{t-1})$$
$$\quad \text{通过精心设计的嵌入，使} f_\theta^{(d)} \text{能计算任意布尔函数的一层}$$

$$\text{最终: } \text{CoT生成} \Rightarrow \text{simulate boolean circuit of size } T$$

**关键构造**：使用$O(\log n)$维二进制向量编码电路门的$O(n)$个可能状态——这是可行的，因为每一步只需传递当前层的$O(1)$个门输出，而非整个电路状态。

**对应消融**：Figure 6-9中嵌入维度的消融显示，当嵌入维度低于$O(\log n)$时，CoT的模拟能力崩溃。

---

### 模块3: 对数嵌入维度的充分性（嵌入效率定理）

**直觉**：与直觉相反，模拟大小为T=poly(n)的电路不需要poly(n)的嵌入维度——$O(\log n)$足够，因为每步CoT只传递局部状态。

**Baseline公式**（先前电路模拟结果，如Looped Transformers [14]）：
$$\text{embedding size} = \text{poly}(n) \text{ for circuit simulation}$$

**变化点**：先前工作需要大嵌入来同时编码整个计算状态。本文利用CoT的**时间分解**特性，将全局状态分散到T个时间步，每步只需对数维的局部状态。

**本文公式**：
$$\text{embedding size} = O(\log n) \text{ suffices for } T = \text{poly}(n) \text{ CoT steps}$$

**推导要点**：
- 电路有$O(n)$个门，但第$t$层只有$O(1)$个门依赖前一层输出
- 用$\log_2(n)$位即可索引任意一个门的状态
- Transformer的注意力机制可以在$O(\log n)$精度下选择性地读取前一步的相关位

**对应消融**：Figure 6-9中对比了不同嵌入维度的效果，验证了$O(\log n)$的充分性和过小维度的失败。

## 实验与分析




![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/189b9ee3-63ad-4af2-9080-64f61b72f5d1/figures/fig_003.png)
*Figure: Permutation Composition (S5). The label is the composition of all the permutations, where given*



本文在四个合成任务上验证了理论预测，所有实验使用**浅层Transformer**（常数深度，如2-4层），通过监督微调训练。核心发现一致：**无CoT时模型完全失败，有CoT时模型近乎完美**——这与理论预言的AC0限制 vs. 电路模拟能力完全吻合。

**置换群合成（Permutation Composition, S5）**：任务要求计算多个置换的复合。这是一个** inherently serial**问题：置换必须按顺序应用，无法并行化。Figure 3显示，无CoT时准确率接近0%，而加入CoT后准确率跃升至近100%。这是因为AC0无法计算置换群的复合（需要串行累积），而CoT的T步生成恰好模拟了T次置换应用的串行过程。

**迭代平方（Iterated Squaring, IS）**：计算$x^{2^T} \mod p$，需要T次连续的模平方操作。Figure 4显示，无CoT时模型无法完成任何有效计算；有CoT时，模型通过生成中间平方结果逐步逼近最终答案，实现高准确率。该任务属于NC1，严格超出AC0，是CoT突破复杂度类的直接证据。

**电路求值问题（Circuit Value Problem）**：给定布尔电路描述和输入，计算输出。这是P完全问题，Figure 5验证了CoT可以逐步模拟每个门的计算——每步CoT对应一个门的求值，最终输出与电路等价的结果。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/189b9ee3-63ad-4af2-9080-64f61b72f5d1/figures/fig_004.png)
*Figure: Modular Addition(C7). The label is the sum of the inputs modulo a positive integer, which is 7*



**消融分析**（Figures 6-9）：最关键的消融是**嵌入维度的影响**。当嵌入维度从$O(\log n)$降至常数时，CoT的电路模拟能力显著下降；反之，增大到poly(n)并无必要，验证了理论的最优性。另一关键消融是**CoT步数**：当提供的CoT步数T小于电路深度时，准确率断崖式下跌；T达到理论下限后趋于饱和。

**公平性审视**：
- Baseline选择合理："无CoT的标准Transformer"是理论上的自然对照，完美对应AC0限制定理。
- **缺失的强baseline**：未与深层Transformer（非恒定深度，如100层）对比——深层模型可能无需CoT即可解决部分串行问题；未与Looped Transformers [14]直接 empirical 对比，后者通过权重循环而非提示策略实现串行计算。
- **实验尺度**：使用小型合成任务和小模型，未在GSM8K [10]等真实数学推理基准上验证——尽管论文引用了该数据集，实际实验未采用。
- **披露的限制**：作者明确承认理论结果针对常数深度Transformer，对标准深层Transformer的实践意义需进一步探索；未发布代码影响可复现性。

## 方法谱系与知识库定位

**方法家族**：Chain of Thought Prompting → **父方法**：Zero-shot-CoT [21]（Kojima et al., "Large language models are zero-shot reasoners"）

本文是对Zero-shot-CoT的**理论奠基性延伸**：从经验观察走向严格的电路复杂度证明。改变的slot主要是**theoretical_foundation**和**expressiveness_characterization**——保留了CoT的推理策略形式，但赋予其全新的复杂度理论解释。

**直接Baseline与差异**：

| 方法 | 与本文的差异 |
|:---|:---|
| Standard Transformer (无CoT) [16] | 本文证明其AC0限制，并通过CoT突破之 |
| Zero-shot-CoT [21] | 本文提供其首次严格的复杂度理论表征 |
| Looped Transformers [14] | 需修改架构（权重循环），本文用标准架构+提示策略达到类似能力 |
| Universal Transformers [11] | 需自适应计算时间和特殊终止机制，本文用固定深度+固定步数 |
| Feng et al. [13] | 同期理论工作，但本文给出了更紧的上界（AC0）和更完整的电路模拟构造 |

**后续方向**：
1. **真实任务验证**：将理论预测扩展到GSM8K、MATH等自然语言数学推理基准，检验CoT的"电路模拟"机制是否同样主导
2. **最优CoT长度**：理论给出T步模拟T层电路，但实际推理中如何自动确定最小充分T值
3. **深度-时间的连续权衡**：探索非恒定深度Transformer中，CoT步数与模型深度的最优组合

**标签**：
- **modality**: text
- **paradigm**: chain-of-thought prompting, theoretical analysis
- **scenario**: reasoning, synthetic tasks, complexity theory
- **mechanism**: depth-time tradeoff, circuit simulation, auto-regressive state passing
- **constraint**: constant-depth transformer, constant-bit precision, O(log n) embedding

## 引用网络

### 后续工作（建立在本文之上）

- [[P__固定位大小Transformer_Post-machine-sim]]: Highly cited (6 times) including in method and experiments; directly related to 

