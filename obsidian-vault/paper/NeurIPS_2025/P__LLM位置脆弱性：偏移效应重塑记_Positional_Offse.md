---
title: 'Positional Fragility in LLMs: How Offset Effects Reshape Our Understanding of Memorization Risks'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- LLM位置脆弱性：偏移效应重塑记忆化风险
- Positional Offse
- Positional Offset Evaluation Protocol for Memorization
- Memorization in LLMs exhibits posit
acceptance: Poster
cited_by: 1
method: Positional Offset Evaluation Protocol for Memorization
modalities:
- Text
paradigm: supervised
---

# Positional Fragility in LLMs: How Offset Effects Reshape Our Understanding of Memorization Risks

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Text_Generation]] | **Method**: [[M__Positional_Offset_Evaluation_Protocol_for_Memorization]] | **Datasets**: Sparse Gutenberg memorization, Sparse Gutenberg text, Downstream task performance, Prefix length effect, Offset position effect

> [!tip] 核心洞察
> Memorization in LLMs exhibits positional fragility: verbatim recall is strongest for prefixes at the start of the context window and degrades sharply with offset, making positional offset a critical overlooked axis for evaluating memorization risks.

| 中文题名 | LLM位置脆弱性：偏移效应重塑记忆化风险 |
| 英文题名 | Positional Fragility in LLMs: How Offset Effects Reshape Our Understanding of Memorization Risks |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.13171) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 语言模型记忆化评估、文本生成 |
| 主要 baseline | Carlini et al. 提取攻击、Quantifying Memorization Across Neural Language Models、Goldfish Loss |

> [!abstract] 因为「LLM记忆化风险被系统性低估——现有方法仅从序列开头探测，隐含假设记忆化在上下文窗口中均匀分布」，作者在「Carlini et al. 提取攻击框架」基础上改了「引入位置偏移变量（前缀长度、偏移量、曝光频率）的长上下文评估协议与BOD token因果干预」，在「Sparse Gutenberg记忆化基准（1B/3B/8B LLaMA，83B tokens预训练）」上取得「发现位置脆弱性现象：开头短前缀触发最强逐字记忆，偏移>0时记忆急剧衰减」

- **关键性能 1**：短前缀（1-10 tokens）在偏移0处触发最高Rouge-L逐字记忆分数，随前缀长度增加反而下降——反直觉现象
- **关键性能 2**：将敏感内容偏移放置至上下文更深处，可抑制可提取记忆化与文本退化，提供预训练时缓解策略
- **关键性能 3**：2% token dropping（Goldfish Loss变体）作为缓解策略，下游任务性能基本保持（Table 5/6）

## 背景与动机

大型语言模型（LLM）在预训练过程中会记忆大量训练数据，这带来了严重的隐私与版权风险——攻击者可能通过精心构造的提示提取出训练集中的敏感内容，如个人信息或受版权保护的书籍片段。然而，当前领域对「记忆化风险究竟有多大」的理解可能存在系统性偏差。

现有主流的记忆化探测方法，以 Carlini et al. [5] 的提取攻击为代表，其核心做法是：从训练序列的**开头**截取一段前缀，输入模型让其续写，然后比对生成结果与原始训练数据的重合程度。类似地，Nasr et al. [6] 的量化框架也主要关注「模型是否记住了某条数据」，而未系统考察**在序列的哪个位置**记住。这些方法隐含假设：记忆化在上下文窗口中是均匀分布的——无论从开头、中间还是末尾探测，结果都应相近。

但这一假设存在根本缺陷。实际预训练数据通常以长文档形式出现，敏感内容可能出现在上下文的任意位置；而模型的注意力机制、位置编码以及对 BOS/BOD（beginning-of-document）token 的依赖，都可能导致记忆化强度随位置剧烈变化。如果记忆化实际上高度集中于序列开头，那么仅从头探测会**高估**整体风险；反之，若某些位置的记忆化被严重低估，则现有缓解策略（如差分隐私、unlearning、解码约束）可能保护不足。

具体而言，现有方法的短板可归纳为三点：（1）**探测位置单一**——仅从头开始，未覆盖偏移场景；（2）**序列长度不足**——使用短序列打包，无法模拟真实长文档中敏感内容的嵌入方式；（3）**缺乏因果机制分析**——未隔离特定位置标记（如BOD token）的作用。本文正是针对这些局限，提出了一套系统性的位置偏移评估协议，首次将「位置」作为记忆化分析的核心维度。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/dee0dc5e-66b9-4d32-b309-4f25de5726d5/figures/Figure_2.png)
*Figure 2 (pipeline): Illustration of how offset effects can lead to underestimated memorization.*



## 核心创新

**核心洞察**：LLM的记忆化具有位置脆弱性（Positional Fragility）——模型将序列最早token作为检索锚点，导致开头短前缀触发最强逐字记忆，而偏移量增加或前缀变长时记忆急剧衰减；因为位置编码与BOD token形成了强位置依赖的检索机制，从而使「通过策略性偏移放置敏感内容来预训练时抑制记忆泄露」成为可能。

| 维度 | Baseline (Carlini et al. [5], Nasr et al. [6]) | 本文 |
|:---|:---|:---|
| 探测位置 | 仅从序列开头（offset=0） | 系统变化前缀长度1-500、偏移量0-3000+ |
| 序列长度 | 短序列（通常<1024 tokens） | 长序列4096-8192 tokens，10倍于 prior work |
| 数据构造 | 均匀混合，无位置控制 | 控制频率f的公开书籍在指定偏移处插入 |
| 评估指标 | 标量记忆分数 | 位置依赖的Rouge-L/Perplexity/多样性曲面 |
| 因果机制 | 无 | BOD token注意力掩码隔离锚定效应 |
| 缓解策略 | 后训练（unlearning/解码约束） | 预训练时偏移放置 + 2% token dropping |

## 整体框架



本文的整体框架围绕「位置偏移评估协议」展开，包含五个核心模块，数据流如下：

**模块 1：长上下文数据构造（Long-context Data Construction）**
输入为网络规模语料 + 公开领域书籍（Gutenberg Books）；输出为训练序列，其中书籍片段以控制频率f插入到指定偏移位置o。这是「Sparse Gutenberg」设置——模拟版权内容在真实数据中的出现模式；另有「Swapped Gutenberg」作为对照（书籍内容被打乱，区分真实记忆与统计模式补全）。

**模块 2：偏移控制预训练（Offset-Controlled Pretraining）**
输入为构造好的长上下文数据；输出为从头训练的LLaMA模型（1B/3B/8B参数，83B tokens总量）。关键参数：书籍曝光频率f、插入偏移o、是否启用2% token dropping缓解。

**模块 3：可变前缀探测（Variable-Prefix Probing）**
输入为训练好的模型 + 测试前缀（变化长度p和偏移o）；输出为模型续写结果。这是核心评估环节——系统扫描(p, o, f)三维空间。

**模块 4：BOD掩码生成（BOD-Masked Generation）**
输入为训练好的模型 + 注意力掩码（屏蔽BOD token）；输出为无BOD锚定影响的生成结果。用于因果验证：若移除BOD后记忆下降，则证实位置锚定机制。

**模块 5：位置脆弱性分析（Positional Fragility Analysis）**
输入为各条件下的生成输出；输出为记忆化指标随位置变化的曲线/曲面，包括Rouge-L（逐字回忆）、Perplexity（文本退化）、多样性指标。

```
Web语料 + Gutenberg书籍 → [数据构造: 控制f, 偏移o插入] → 长上下文训练数据
                                    ↓
                    [预训练: LLaMA 1B/3B/8B, 83B tokens]
                                    ↓
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
    [标准生成: 前缀(p,o)]    [BOD掩码生成]          [Swapped对照]
              ↓                     ↓                     ↓
    [指标计算: Rouge-L,        [对比BOD效应]         [区分记忆vs模式补全]
     Perplexity, 多样性]
              ↓
    [位置脆弱性曲面: M(p,o,f)]
```

## 核心模块与公式推导

### 模块 1: 位置记忆化度量 M(p, o, f)（对应框架图「指标计算」层）

**直觉**：记忆化不是标量，而是前缀长度、偏移位置、曝光频率的函数——必须显式纳入位置变量才能捕捉脆弱性。

**Baseline 公式** (Carlini et al. [5], Nasr et al. [6]):
$$M_{base}(p) = \text{Rouge-L}\left(\text{Gen}(x_{p,0}),\; y_{suffix}\right)$$
符号: $p$ = 前缀长度, $x_{p,0}$ = 从偏移0处截取的前缀, $y_{suffix}$ = 原始训练后缀, Gen = 自回归生成。

**变化点**: Baseline 固定偏移 $o=0$，无法检测位置效应；且仅用Rouge-L单一指标，未覆盖文本退化。

**本文公式（推导）**:
$$\text{Step 1}: \; M(p, o, f) = \text{Rouge-L}\left(\text{Gen}(x_{p,o}),\; y_{suffix}\right) \quad \text{加入偏移量} o \text{ 和频率} f \text{ 作为显式变量}$$
$$\text{Step 2}: \; \text{扩展为多维评估: } \mathbf{M}(p,o,f) = \left[\text{Rouge-L},\; \text{Perplexity}^{-1},\; \text{Diversity}\right] \quad \text{同时测量逐字记忆与生成质量退化}$$
$$\text{最终}: \; M_{final}(p, o, f) = \text{Rouge-L}\left(\text{Gen}(x_{p,o}),\; y_{suffix}\right) \;\text{with}\; x_{p,o} = \{x_{o+1}, \ldots, x_{o+p}\}$$

**对应消融**: Table 2 显示 Swapped（非记忆对照）vs Sparse（真实记忆）设置下，Sparse的Rouge-L显著更高，且偏移增大时两者差距缩小。

---

### 模块 2: BOD掩码生成概率 P_BOD(x)（对应框架图「BOD掩码生成」分支）

**直觉**：若模型依赖BOD token作为检索锚点，则屏蔽该token应削弱位置开头的记忆优势——这是验证「位置锚定」机制的因果干预。

**Baseline 公式** (标准自回归):
$$P(x) = \prod_{t=|o|+1}^{T} P(x_t \text{mid} x_{<t})$$
符号: $x_t$ = 第t个token, $x_{<t}$ = 历史上下文, $|o|$ = 偏移量。

**变化点**: 标准生成允许模型自由 attend 到BOD token；本文假设BOD通过注意力形成位置锚定，需隔离其影响。

**本文公式（推导）**:
$$\text{Step 1}: \; h_{BOD} = \text{Attention}(Q_{BOD}, K_{<t}, V_{<t}) \quad \text{识别BOD token的隐藏状态}$$
$$\text{Step 2}: \; \text{mask}(h_{BOD}) = \mathbf{0} \quad \text{注意力掩码置零，阻断BOD信息传播}$$
$$\text{最终}: \; P_{BOD}(x) = \prod_{t=|o|+1}^{T} P\left(x_t \text{mid} x_{<t},\; \text{mask}(h_{BOD})\right)$$

**对应消融**: Figure 4 显示包含/排除BOS token对4B模型记忆能力的影响，验证BOD锚定效应。

---

### 模块 3: 位置脆弱性指数 Fragility(o)（对应框架图「位置脆弱性分析」输出）

**直觉**：需要一个归一化指标，量化「偏移对记忆化的抑制强度」，便于跨模型规模比较。

**Baseline**: 无——prior work未定义位置脆弱性度量。

**本文公式（推导）**:
$$\text{Step 1}: \; \Delta M(o) = M(p_{short}, 0, f) - M(p_{short}, o, f) \quad \text{计算偏移} o \text{ 导致的记忆损失}$$
$$\text{Step 2}: \; \text{归一化以消除绝对记忆强度差异: } \frac{\Delta M(o)}{M(p_{short}, 0, f)} \quad \text{保证跨模型可比性，值域}[0,1]$$
$$\text{最终}: \; \text{Fragility}(o) = \frac{M(p_{short}, 0, f) - M(p_{short}, o, f)}{M(p_{short}, 0, f)}$$

符号: $p_{short}$ = 短前缀（通常1-10 tokens，记忆最敏感区域）, $o$ = 偏移量, $f$ = 固定曝光频率。

**对应消融**: Figure 3 展示LLaMA 3B/7B/4B模型在不同上下文长度下的位置脆弱性曲线；Table 9 给出1B和8B模型的偏移衰减具体数值。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/dee0dc5e-66b9-4d32-b309-4f25de5726d5/figures/Table_1.png)
*Table 1 (quantitative): Optimal length and sample of perfect matching by model size and repetition frequency.*



本文在自建的 Sparse Gutenberg 记忆化基准上进行了大规模实证研究，训练了 LLaMA 架构的 1B/3B/8B 模型（总计 83B tokens）。核心发现如下：记忆化强度并非均匀分布，而是呈现显著的位置脆弱性——**短前缀（1-10 tokens）在序列开头（offset=0）触发最强逐字记忆**，Rouge-L 分数最高；当前缀长度增加或偏移量增大时，记忆化急剧衰减。Table 1 展示了不同模型规模和重复频率下的「完美匹配」样本数：更大模型和更高频率确实提升记忆，但偏移的抑制作用贯穿所有条件。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/dee0dc5e-66b9-4d32-b309-4f25de5726d5/figures/Table_2.png)
*Table 2 (comparison): Comparison of key memorization metrics under Stripped and Pristine (Baseline) Settings.*



Table 2 对比了「Stripped」（去除敏感内容）与「Pristine/Baseline」设置下的关键记忆指标，同时包含 Swapped（打乱对照）与 Sparse（真实记忆）两种条件。数据显示：Sparse 设置的 Rouge-L 显著高于 Swapped，证实模型确实发生了逐字记忆而非统计模式补全；而当偏移量增加时，Sparse 与 Swapped 的差距缩小，说明**偏移放置可有效降低可提取记忆**。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/dee0dc5e-66b9-4d32-b309-4f25de5726d5/figures/Figure_1.png)
*Figure 1 (result): Positional fragility in LLMs measured with two complementary metrics. (a) Rouge-L scores. (b) MAUVE scores.*



Figure 1 以 Rouge-L 和 MAUVE 两个互补指标展示了位置脆弱性的全貌：(a) Rouge-L 捕获逐字回忆精度，(b) MAUVE 测量生成文本与真实分布的匹配度。Figure 5 和 Figure 6 进一步将 TF-IDF、Rouge-L、MAUVE 与曝光频率关联，揭示频率阈值效应——低频内容（f较小）即使位于开头也难以被记忆，而高频内容对位置更敏感。



消融实验聚焦于三个关键变量。Figure 3（对应 Table 8/9）展示上下文长度对位置脆弱性的影响：在 LLaMA 3B/7B/4B 上，更长上下文并未缓解脆弱性，反而因绝对偏移范围扩大而呈现更复杂的衰减模式。Figure 4 隔离 BOS token 效应：包含 BOS 时记忆能力显著强于排除 BOS 的情况，验证 BOD 锚定假设。Table 9 的数值显示，**偏移从0增至3000+ tokens 时，Rouge-L 出现断崖式下降**。

缓解策略验证方面，Table 5/6 对比了 2% token dropping（Goldfish Loss 变体）与标准训练：下游任务（HellaSwag 等）性能基本保持，说明预训练时偏移放置 + token dropping 的组合可在不牺牲模型效用的前提下降低记忆风险。

**公平性检查**：本文 baselines 限于 Carlini et al. [5] 和 Nasr et al. [6] 的标准探测框架，以及 Goldfish Loss 的下游对比。未纳入的更强 baseline 包括：差分隐私训练、模型 unlearning、解码时约束（如语义哈希）；架构上仅覆盖 LLaMA，未验证 GPT/Mistral 等系列；模型规模止于 8B，更大模型的缩放趋势未知。作者明确披露：公开书籍作为版权内容代理可能不完全反映真实法律敏感数据的记忆动态；且实验室中的精确偏移放置难以直接迁移到不可控的真实预训练场景。

## 方法谱系与知识库定位

**方法家族**: 训练数据提取攻击与记忆化量化 → 位置感知评估协议

**父方法**: Carlini et al. [5] "Extracting training data from large language models" —— 本文直接继承其「前缀-续写-比对」的提取攻击范式，但将标量评估扩展为位置依赖的多维分析。另一父方法为 Nasr et al. [6] "Quantifying memorization across neural language models" —— 扩展其指标框架，加入位置变量。

**改动插槽**:
- **data_pipeline**: 短序列开头探测 → 长序列（4096-8192 tokens）+ 控制频率f的偏移放置
- **evaluation_protocol**: 固定offset=0 → 系统扫描前缀长度p × 偏移量o × 频率f
- **inference_strategy**: 标准自回归 → BOD token注意力掩码因果干预
- **training_recipe**: 标准预训练 → 2% token dropping + 策略性敏感内容偏移放置

**直接 baselines 及差异**:
- **Carlini Extraction Attack [5]**: 本文继承其攻击形式，但揭示「仅从头探测」会系统性高估/低估风险；加入位置维度后重新校准评估
- **Quantifying Memorization [6]**: 本文扩展其Rouge-L指标为 M(p,o,f) 函数，新增Perplexity/多样性联合评估
- **Goldfish Loss**: 本文将其作为下游性能对照，但核心缓解策略是「偏移放置」而非「token dropping」

**后续方向**:
1. **跨架构验证**: 测试 GPT、Mistral、DeepSeek 等非 LLaMA 模型是否同样呈现位置脆弱性，以及 BOS/BOD 机制的普遍性
2. **更大规模缩放**: 8B 以上模型（如 70B+）的位置脆弱性曲线是否遵循相同规律，或出现涌现的「深度记忆」
3. **真实版权数据**: 用实际受版权保护内容替代公开书籍，验证偏移缓解策略的法律有效性；结合差分隐私或 unlearning 形成多层防御

**标签**: text / autoregressive generation / pretraining safety / memorization evaluation / positional bias / copyright risk mitigation / causal intervention / long-context modeling

