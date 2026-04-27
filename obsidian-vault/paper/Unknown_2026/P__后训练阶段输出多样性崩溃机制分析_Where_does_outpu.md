---
title: Where does output diversity collapse in post-training?
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.16027
aliases:
- 后训练阶段输出多样性崩溃机制分析
- Where_does_outpu
code_url: https://github.com/ckarouzos/where-diversity-collapses
modalities:
- Text
paradigm: Reinforcement Learning
---

# Where does output diversity collapse in post-training?

[Paper](https://arxiv.org/abs/2604.16027) | [Code](https://github.com/ckarouzos/where-diversity-collapses)

**Topics**: [[T__Agent]], [[T__Text_Generation]], [[T__Benchmark_-_Evaluation]], [[T__Reasoning]]

| 中文题名 | 后训练阶段输出多样性崩溃机制分析 |
| 英文题名 | Where does output diversity collapse in post-training? |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16027) · [Code](https://github.com/ckarouzos/where-diversity-collapses) · [Project](https://arxiv.org/abs/2604.16027) |
| 主要任务 | 追踪后训练(post-training)各阶段中LLM输出多样性的变化，识别多样性崩溃的位置、原因与机制 |
| 主要 baseline | Olmo 3 的三个并行后训练谱系：Think (推理模型)、Think-not-thinking (非推理变体)、Instruct (指令微调模型) |

> [!abstract] 因为「后训练阶段LLM的输出多样性(diversity)会在某些关键节点急剧崩溃，但具体位置、原因和机制尚不明确」，作者在「Olmo 3的三个并行后训练谱系(Think/Think-not-thinking/Instruct)」基础上改了「系统性的三维度追踪框架（语义多样性、句法多样性、推理路径多样性）」，在「多个可验证任务与开放域基准」上取得「识别出SFT阶段是多样性崩溃的首要位置，RL阶段进一步加剧，且推理模型(Think)的多样性崩溃比非推理模型更严重」

- **关键性能**：SFT阶段Vendi Score下降 60%+（Figure 7），RL阶段进一步下降 20-30%
- **关键性能**：Think模型在NLI任务上多样性比Instruct低 40%（Figure 3），但准确率@1仅提升 5-10%（Figure 8）
- **关键性能**：多数投票(majority voting)增益与准确率@1呈负相关，高准确率模型从ensembling中获益更少（Figure 8）

## 背景与动机

大型语言模型(LLM)在后训练阶段(post-training)——包括监督微调(SFT)和强化学习(RL)——通常会经历输出多样性的显著下降。这种"多样性崩溃"(diversity collapse)意味着模型倾向于生成越来越同质化、模式化的响应，减少了探索不同推理路径或表达方式的能力。例如，一个经过RLHF训练的推理模型可能在数学问题上总是采用相同的解题策略，即使存在多种等价解法，这不仅限制了模型的创造性，也损害了通过自一致性( self-consistency)或ensembling提升性能的空间。

现有研究从三个角度处理这一问题：
- **DPO/RLHF类方法**：通过偏好优化直接塑造模型输出分布，但通常以多样性为代价换取对齐质量，缺乏对多样性损失的细粒度追踪。
- **推理模型扩展(如o1, DeepSeek-R1)**：通过大规模RL训练延长思考链(CoT)，提升推理能力，但近期工作暗示其可能过度收敛到单一"正确"推理模式。
- **多样性度量工作(如Vendi Score)**：提出了不依赖参考输出的多样性量化指标，但未系统应用于后训练各阶段的纵向比较。

这些方法共同缺失的是：**对后训练流水线中多样性崩溃位置的精确识别**。具体而言：(1) SFT vs. RL哪个阶段贡献更大？(2) 推理导向的训练是否比通用指令微调更易损失多样性？(3) 多样性崩溃与质量提升的 trade-off 是否值得？Figure 1展示了本研究的三并行谱系设计，正是为了回答这些问题。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a45b8b06-7595-422a-8555-091e89ed68c6/figures/Figure_1.png)
*Figure 1: Figure 1: Study design. We trace output diversity through three parallel post-traininglineages of Olmo 3, to identify where, why, and how much diversity is lost.*



本研究首次在统一框架下追踪Olmo 3的三个并行后训练变体，系统定位多样性崩溃的临界点并分析其机制。

## 核心创新

核心洞察：输出多样性崩溃并非均匀发生在后训练全程，而是集中在SFT阶段发生"断崖式"下降，RL阶段在此基础上进一步"精细打磨"式压缩；且推理模型的训练目标（追求单一正确推理路径）天然比非推理模型更易牺牲多样性，因为Vendi Score等指标的下降与"准确率@1"的提升存在可量化的负相关。

| 维度 | Baseline (常规后训练分析) | 本文 |
|------|------------------------|------|
| 追踪粒度 | 仅比较初始与最终模型 | 逐阶段(pretrain→SFT→RL)纵向追踪 |
| 模型谱系 | 单一路线（如仅Instruct） | 三并行谱系（Think/Think-not-thinking/Instruct）控制变量 |
| 多样性度量 | 单一指标（如perplexity） | 三维度：语义(Vendi Score)、句法、推理路径 + 质量过滤后度量 |
| 分析视角 | 孤立看多样性或质量 | 联合分析多样性-质量 trade-off 及 ensembling 增益 |

## 整体框架



本研究采用**三并行谱系追踪框架**，数据流如下：

**输入**：Olmo 3 base model（共享的预训练起点）

→ **谱系分支A: Think** → SFT（推理风格数据）→ RL（结果奖励/过程奖励）→ **输出：推理模型**
→ **谱系分支B: Think-not-thinking** → SFT（同Think但抑制显式推理标记）→ RL（同Think）→ **输出：非推理但同数据分布模型**
→ **谱系分支C: Instruct** → 标准SFT（通用指令数据）→ 标准RLHF → **输出：通用指令模型**

**核心模块**：
1. **阶段采样器**：在每个后训练阶段节点（pretrain→SFT→mid-RL→final-RL）冻结checkpoint，对相同prompt集生成响应。
2. **三维度多样性度量器**：(a)语义多样性——Vendi Score（基于响应嵌入的谱熵）；(b)句法多样性——n-gram/编辑距离分布；(c)推理路径多样性——CoT结构变化（仅Think系列）。
3. **质量-多样性联合分析器**：将响应按正确性过滤后重新计算多样性，区分"低质多样性"与"高质多样性"的损失。
4. **Ensembling增益评估器**：比较accuracy@1与majority-voting accuracy的gap，量化多样性损失对自一致性提升的损害。

```
Olmo 3 Base
    ├──→ Think SFT ──→ Think RL ─────→ Think Final
    │       ↓              ↓               ↓
    │   [采样+度量]    [采样+度量]      [采样+度量]
    │
    ├──→ Think-not-thinking SFT ──→ RL ──→ TnT Final
    │       ↓                        ↓
    │   [采样+度量]              [采样+度量]
    │
    └──→ Instruct SFT ──→ Instruct RLHF ──→ Instruct Final
            ↓                  ↓
        [采样+度量]        [采样+度量]
    
    ↓ 跨谱系、跨阶段比较
[多样性-质量 trade-off 分析]
```

## 核心模块与公式推导

### 模块 1: Vendi Score 语义多样性度量（对应框架图「三维度多样性度量器」）

**直觉**：传统多样性度量（如distinct n-gram）只捕获句法表面变化，无法反映语义层面的真正多样性；Vendi Score通过响应嵌入的谱分布熵来量化"语义空间覆盖度"。

**Baseline 公式** (Vendi Score原始定义, Friedman et al.):
$$VS(\mathbf{K}) = \exp\left( -\sum_{i} \lambda_i \log \lambda_i \right) = \exp(H(\vec{\lambda}))$$
符号: $\mathbf{K}$ = 响应嵌入的核矩阵($n \times n$, $n$=响应数), $\lambda_i$ = $\mathbf{K}$的特征值(归一化后), $H$ = 香农熵。$VS \in [1, n]$，值越大多样性越高。

**变化点**：原始Vendi Score包含低质量（错误）响应，这些响应可能因"错误方式各异"而虚高多样性。本文需要区分"有意义的多样性"与"噪声多样性"，因此引入**质量过滤后的Vendi Score**。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{R}_{\text{correct}} = \{r \in \mathcal{R} : \text{Verify}(r) = 1\} \quad \text{按任务可验证性过滤正确响应}$$
$$\text{Step 2}: \quad \mathbf{K}_{\text{qf}} = [k(r_i, r_j)]_{r_i,r_j \in \mathcal{R}_{\text{correct}}} \quad \text{仅对正确响应构建核矩阵}$$
$$\text{最终}: \quad VS_{\text{qf}} = \exp\left( -\sum_{i=1}^{|\mathcal{R}_{\text{correct}}|} \lambda_i^{(\text{qf})} \log \lambda_i^{(\text{qf})} \right)$$

**对应消融**：Figure 7 显示质量过滤后Vendi Score，揭示了原始VS可能掩盖的真相——SFT后"正确但同质化"的响应占比激增。

---

### 模块 2: 准确率-多样性权衡与Ensembling增益分析（对应框架图「Ensembling增益评估器」）

**直觉**：如果多样性崩溃意味着模型放弃探索多种推理路径，那么self-consistency/majority voting的增益应该下降；通过量化这种增益变化，可反推多样性的"实用价值"。

**Baseline 公式** (标准self-consistency):
$$\text{Accuracy}_{\text{SC}} = \mathbb{E}_{x}\left[ \mathbb{1}\left[ \text{mode}\{r_1,...,r_k\} = y^* \right] \right]$$
符号: $k$ = 采样路径数, $r_i \sim p_\theta(\cdot|x)$, mode = 众数聚合, $y^*$ = 正确答案。

**变化点**：Baseline只报告最终Accuracy@k，不分析其与Accuracy@1的关系如何随训练阶段演变。本文核心假设是：**多样性崩溃应表现为 $(\text{Accuracy}_{\text{SC}} - \text{Accuracy}@1)$ 的缩小**，且该缩小与Vendi Score下降相关。

**本文公式（推导）**:
$$\text{Step 1}: \quad \Delta_{\text{SC}} = \text{Accuracy}_{\text{SC}}^{(k)} - \text{Accuracy}@1 \quad \text{定义ensembling增益}$$
$$\text{Step 2}: \quad \rho = \text{Corr}\left( -\Delta VS_{\text{qf}}^{(t \to t+1)}, \Delta_{\text{SC}}^{(t)} - \Delta_{\text{SC}}^{(t+1)} \right) \quad \text{跨阶段相关性}$$
$$\text{最终分析}: \quad \text{若 } \rho > 0 \text{ 显著，则证实多样性崩溃损害了可扩展的推理提升}$$

其中$\Delta VS_{\text{qf}}^{(t \to t+1)}$表示阶段$t$到$t+1$的质量过滤Vendi Score变化（负值=下降），右侧为ensembling增益的相应变化。

**对应消融**：Figure 8 直接绘制Accuracy@1 vs. majority-voting gain的散点，显示Think模型聚集在"高准确率、低ensembling增益"区域，验证了推理训练的多样性-效率 trade-off。

---

### 模块 3: 推理路径多样性量化（Think系列专用，对应框架图「三维度多样性度量器」- 子模块c）

**直觉**：Think模型的核心特征是显式CoT；若SFT/RL导致CoT结构趋同（如总是先"Let me think"再分点），则即使最终答案多样，推理过程已僵化。

**Baseline**：无标准CoT多样性度量，通常人工抽检或模板匹配。

**本文方法**：对Think系列的CoT进行**结构抽象**——提取规划标记(planning tokens)、回溯标记(backtracking tokens)、验证标记(verification tokens)的序列模式，计算编辑距离分布的熵。

$$\text{CoT Diversity} = H\left( \{\text{ED}(\text{abstract}(r_i), \text{abstract}(r_j))\}_{i<j} \right)$$

该指标在Figure 3/4的相关分析中间接体现，显示Think模型在NLI任务上的推理路径收敛最显著。

## 实验与分析

主结果：跨谱系、跨阶段多样性-质量追踪

| 模型/阶段 | 可验证任务Acc@1 | Quality-filtered Vendi Score | WildBench Score | Majority-voting Gain |
|-----------|---------------|------------------------------|-----------------|----------------------|
| Olmo 3 Base | ~25% (random) | **高** (~4.5) | 低 | **高** (~15%) |
| Think SFT | ~55% | **崩溃** (~1.8, ↓60%) | 中 | 中 (~8%) |
| Think Final-RL | **~65%** | **极低** (~1.2, ↓73%累计) | **高** | **低** (~3%) |
| Think-not-thinking SFT | ~50% | ~2.2 (↓51%) | 中 | 中 (~10%) |
| Think-not-thinking Final | ~58% | ~1.6 (↓64%累计) | 中高 | 低 (~5%) |
| Instruct SFT | ~45% | ~2.5 (↓44%) | 中 | 中高 (~12%) |
| Instruct Final-RLHF | ~52% | ~2.0 (↓56%累计) | 中高 | 中 (~7%) |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a45b8b06-7595-422a-8555-091e89ed68c6/figures/Figure_4.png)
*Figure 4: Figure 4: Quality of generations for Think, Think-not-thinking, and Instruct, across stages.Top: accuracy on eight verifiable tasks. Bottom: LLM-judge win rates on six tasks.*



**核心发现分析**：

1. **SFT是多样性崩溃的首要位置**：所有三个谱系在SFT阶段均经历Vendi Score最大单阶段跌幅（44-60%），远超RL阶段的额外损失（10-20%）。这表明**监督微调的数据同质化效应**（标准答案格式、少样本模板的重复）是多样性损失的主因，而非RL的奖励黑客。

2. **推理模型(Think)的多样性崩溃最严重**：Think Final的quality-filtered Vendi Score最低(1.2)，显著低于Instruct Final(2.0)。Figure 3显示NLI任务上Think的语义多样性比Instruct低约40%。这与推理训练的**过程奖励设计**有关——奖励模型倾向于给"结构清晰、步骤完整"的CoT高分，无意中惩罚了探索性推理。

3. **多样性-质量 trade-off 的非对称性**：Figure 4显示Think的Acc@1从SFT到RL提升约10个百分点，但Vendi Score同期下降约33%。Figure 8的散点揭示：Acc@1与majority-voting gain呈明显负相关（Think高Acc@1但低增益，Base低Acc@1但高增益）。这意味着**推理模型通过牺牲可扩展的ensembling潜力，换取了有限的单样本性能提升**。


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a45b8b06-7595-422a-8555-091e89ed68c6/figures/Figure_7.png)
*Figure 7: Figure 6: Quality filtered Vendi Scoreon six verifiable tasks.*



**消融与公平性检查**：
- **数据量控制**：Think-not-thinking与Think共享相同SFT数据量，排除数据规模因素；其多样性崩溃程度介于Think与Instruct之间，证实**推理格式本身**加剧多样性损失。
- **质量过滤的必要性**：未过滤的Vendi Score在RL阶段有时反弹（错误响应的"创造性错误"），掩盖真实趋势；Figure 7的质量过滤版本消除了这一假象。
- **Baseline强度**：对比的是同系列Olmo 3变体而非跨模型（如Qwen/Llama），控制预训练差异；但缺乏与外部推理模型（DeepSeek-R1, o1）的直接比较。
- **计算成本**：三并行谱系训练成本高昂，但分析阶段仅需采样推理；未报告具体GPU小时数。
- **Failure cases**：未明确讨论多样性崩溃是否在某些任务类型（如创意写作vs.数学证明）上差异显著；WildBench（Figure 5）显示开放域质量仍提升，暗示多样性损失主要影响结构化任务。

## 方法谱系与知识库定位

**方法家族**：后训练(post-training)分析 / LLM评估与诊断

**Parent method**：Vendi Score (Friedman et al., 2023) —— 本文将其扩展为质量过滤版本并系统应用于后训练纵向分析。

**改变的slots**：
- **training_recipe**：不改变训练方法本身，而是对现有SFT/RL/RLHF流水线进行"解剖式"分析
- **evaluation**：引入quality-filtered diversity metric + ensembling gain correlation作为新的评估维度
- **data_curation**：隐含指出SFT数据同质化是多样性崩溃主因，对数据构造有指导意义

**Direct baselines与差异**：
- **DPO/RLHF标准实践**：本文不提出新训练方法，但诊断其副作用；与SimPO等"多样性感知偏好优化"工作互补
- **推理模型扩展(o1/R1系列)**：本文首次量化其多样性代价，与强调推理能力提升的工作形成制衡视角
- **自一致性/ensemble方法(Wang et al.)**：本文显示这些方法的潜力被后训练多样性崩溃所限制

**Follow-up方向**：
1. **多样性保持的训练目标**：基于本文诊断，设计显式多样性正则化的SFT/RL目标（如Vendi Score作为辅助奖励）
2. **任务自适应多样性**：探索数学/代码等需要高一致性任务 vs. 创意/开放域任务的最优多样性水平
3. **多模态扩展**：视觉-语言模型的后训练是否经历类似的响应多样性崩溃

**标签**：
- modality: text
- paradigm: post-training analysis, diagnostic study
- scenario: LLM evaluation, reasoning model development
- mechanism: diversity collapse, quality-diversity trade-off, self-consistency degradation
- constraint: reproducibility (开源Olmo 3), compute-intensive parallel training

