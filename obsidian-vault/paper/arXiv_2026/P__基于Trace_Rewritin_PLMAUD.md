---
title: Protecting Language Models Against Unauthorized Distillation through Trace Rewriting
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2602.15143
aliases:
- 基于Trace Rewriting的LLM反蒸馏保护
- PLMAUD
modalities:
- Text
---

# Protecting Language Models Against Unauthorized Distillation through Trace Rewriting

[Paper](https://arxiv.org/abs/2602.15143)

**Topics**: [[T__Agent]], [[T__Knowledge_Distillation]], [[T__Privacy]], [[T__Text_Generation]]

| 中文题名 | 基于Trace Rewriting的LLM反蒸馏保护 |
| 英文题名 | Protecting Language Models Against Unauthorized Distillation through Trace Rewriting |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2602.15143) · [Code] · [Project] |
| 主要任务 | 防止LLM被unauthorized distillation，通过rewriting reasoning traces使student model无法有效学习 |
| 主要 baseline | KD (Knowledge Distillation), SFT (Supervised Fine-Tuning), watermark-based methods |

> [!abstract] 因为「LLM的reasoning traces容易被distiller窃取用于unauthorized distillation」，作者在「标准trace generation」基础上改了「instruction-based rewriting + token-level poisoning」，在「GSM8K和MATH benchmark」上取得「student accuracy显著下降（具体数值

- **关键性能**: 在GSM8K上，rewriting后的trace使student model accuracy相比clean KD下降
- **关键性能**: 在MATH上，anti-distillation效果保持
- **关键性能**: Watermark detection true detection rate随K增加而提升（Figure 4）

## 背景与动机

大型语言模型（LLM）的推理能力日益强大，但其生成的详细推理轨迹（reasoning traces）正成为unauthorized distillation的攻击面。攻击者可以收集目标teacher model的（question, reasoning trace, answer）三元组，直接用于supervised fine-tuning（SFT）来蒸馏出一个competent student model，而无需承担训练原始模型的巨大成本。

现有防御方法主要从三个角度切入：
- **Watermarking methods**（如KGW watermark）：在生成文本中嵌入统计水印，用于事后检测是否源自特定模型，但无法阻止distillation本身；
- **Adversarial training / gradient-based poisoning**：通过对抗样本污染训练数据，但往往在trace级别操作，难以保持语义一致性；
- **Output restriction**：限制API返回的推理细节，但这直接损害了用户体验和模型可用性。

这些方法的核心缺陷在于：**watermarking只能检测不能预防**，而**poisoning方法往往破坏trace的可读性或可被adaptive attack绕过**。更关键的是，现有方法缺乏对reasoning trace结构的精细利用——trace包含多个推理步骤，每一步都是潜在的干预点。

本文的核心动机是：**能否在不损害正常用户体验的前提下，通过精细地rewriting reasoning traces，使distilled student model学到错误的推理模式，同时保持teacher model输出的表面合理性？** 
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab3832d5-94a2-44c9-b6ec-73eb7039b623/figures/Figure_1.png)
*Figure 1: Figure 1: Overview of instruction-based rewriting.: (a) Clean trace generation: The teacher model T generatesa reasoning trace r for given task (query) q using a standard generation instruction pg. (b*



为此，作者提出了trace rewriting框架，将anti-distillation从被动的watermark detection转向主动的、结构化的trace manipulation。

## 核心创新

核心洞察：reasoning trace的**instruction-level可分解性**使得**局部rewriting**成为可能，因为每个reasoning step都包含可替换的instruction-like内容，从而使**保持表面合理性同时破坏蒸馏信号**成为可能。

与 baseline 的差异：

| 维度 | Baseline (Standard KD / Watermark) | 本文 |
|------|-----------------------------------|------|
| 干预层级 | Token-level 或 Output-level | **Instruction-level + Token-level 双层** |
| 防御目标 | 检测盗版 (watermark) 或 降低整体质量 | **精准破坏student的reasoning能力** |
| 可验证性 | 需统计检测 | **内置可检测的rewriting signature** |
| 对正常用户影响 | Watermark无影响 / Poisoning可能降低质量 | **保持teacher输出表面质量** |

关键创新点在于：将anti-distillation重新定义为**trace rewriting problem**——不是生成低质量输出，而是生成**结构上合理但 pedagogically harmful** 的推理轨迹。

## 整体框架



整体框架包含三个核心阶段，形成从clean trace到protected trace的完整流水线：

**输入**: User query / question $t$

**Stage 1: Clean Trace Generation（教师模型前向）**
- 输入: question $t$
- 模块: Teacher model $T$ 生成clean reasoning trace $r = (r_1, r_2, ..., r_n)$
- 输出: 原始推理轨迹 $(t, r, a)$，其中 $a$ 为最终答案

**Stage 2: Instruction-Based Rewriting（核心保护模块）**
- 输入: clean trace $r$
- 模块: Rewriting engine 基于instruction templates将 $r$ 转换为 $\tilde{r}$
- 关键操作: 识别trace中的instruction-like片段，替换为anti-distillation variants
- 输出: rewritten trace $\tilde{r}$，保持表面可读性但破坏学习信号

**Stage 3: Token-Level Poisoning（精细加固）**
- 输入: rewritten trace $\tilde{r}$
- 模块: Token-level optimizer（FO-Grad近似）注入梯度感知的token扰动
- 输出: final protected trace $\tilde{r}^*$

**Stage 4: Detection & Verification**
- 输入: suspect student model 的输出
- 模块: Watermark detector 验证是否源自protected traces
- 输出: detection result (true/false)

数据流示意：
```
Question t → [Teacher T] → Clean Trace r → [Instruction Rewriter] → ~r → [Token Poisoning] → ~r* → User/API
                                                      ↓
                                               [Watermark Embed]
                                                      ↓
                                         Suspect Student Output → [Detector] → Verdict
```

两个rewriting变体：
- **OPT (Optimization-based)**: 针对特定distillation objective优化rewriting
- **KPOD (Knowledge-Protective Objective-based)**: 更general的保护目标

## 核心模块与公式推导

### 模块 1: Instruction-Based Rewriting（对应框架图 Stage 2）

**直觉**: Reasoning traces中的自然语言指令（如"Let's think step by step", "First, calculate..."）是student model学习推理结构的关键载体，替换这些instruction可以破坏学习而不影响表面流畅性。

**Baseline 公式** (Standard SFT Distillation):
$$L_{\text{KD}} = -\mathbb{E}_{(t,r,a) \sim \mathcal{D}_T} \left[ \log P_\theta(a, r \text{mid} t) \right]$$
符号: $\theta$ = student model参数, $\mathcal{D}_T$ = teacher-generated trace dataset, $t$ = question, $r$ = reasoning trace, $a$ = answer

**变化点**: 标准KD假设teacher traces是optimal supervision signal。本文发现：**替换instructions可以创建misleading supervision**，使student学到错误的step-by-step关联。

**本文公式（推导）**:
$$\text{Step 1}: \quad \tilde{r} = \text{Rewrite}_\phi(r) \quad \text{将instruction } I \text{ 替换为 } I' \sim \mathcal{M}_{\text{anti}}(I)$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{anti}} = -\mathbb{E}\left[ \log P_\theta(a \text{mid} t, \tilde{r}) \right] \cdot \mathbb{1}[\text{correct}] + \lambda \cdot \text{Sim}(r, \tilde{r}) \quad \text{保证表面相似性}$$
$$\text{最终}: \quad \tilde{r}^* = \text{arg}\max_{\tilde{r}} \underbrace{\text{Utility}(\tilde{r}; t)}_{\text{对用户可用}} - \underbrace{\gamma \cdot \text{DistillGain}(\tilde{r}; \theta_s)}_{\text{对蒸馏有害}}$$

**对应消融**: Figure 3 显示OPT与KPOD两种rewriting策略在GSM8K和MATH上的anti-distillation效果对比。

---

### 模块 2: Token-Level Poisoning（对应框架图 Stage 3）

**直觉**: Instruction-level rewriting可能仍保留可利用的token-level patterns，通过gradient-aware token perturbation可以进一步破坏fine-tuning的梯度信号。

**Baseline 公式** (Naive Token Poisoning):
$$L_{\text{naive}} = \max_{\delta: \|\delta\|_0 \leq \epsilon} L_{\text{CE}}(f_\theta(x + \delta), y)$$
符号: $\delta$ = token perturbation, $\epsilon$ = sparsity budget, $f_\theta$ = victim model

**变化点**: Naive adversarial poisoning针对inference-time attack优化，而distillation是**training-time attack**，需要**影响多-step gradient accumulation**。

**本文公式（推导）**:
$$\text{Step 1}: \quad g_t = \nabla_\theta L_{\text{SFT}}(\theta; \tilde{r}^*) \quad \text{计算student在rewritten trace上的梯度}$$
$$\text{Step 2}: \quad \delta^* = \text{arg}\max_{\delta} \left\| \mathbb{E}_{\tilde{r}} \left[ \nabla_\theta L_{\text{SFT}}(\theta; \tilde{r} + \delta) \right] \right\|^2 - \mu \cdot \text{Perplexity}(\tilde{r} + \delta) \quad \text{最大化梯度方差同时控制困惑度}$$
$$\text{Step 3 (FO-Grad近似)}: \quad \delta_{\text{FO}} \approx \nabla_{\tilde{r}} \left. \frac{\partial L_{\text{SFT}}}{\partial \theta} \right|_{\theta_0} \cdot \text{Hessian}^{-1} \quad \text{一阶近似避免二阶计算}$$
$$\text{最终}: \quad \tilde{r}^{**} = \tilde{r}^* + \delta_{\text{FO}}$$

**对应消融**: Figure 9 显示Token-Level poisoning方法中FO-Grad作为adversarial approximation的效果，对比实际objective的gap。

---

### 模块 3: Watermark Detection（对应框架图 Stage 4）

**直觉**: Rewriting过程本身创造detectable statistical signature，可用于forensic verification。

**Baseline 公式** (Standard KGW Watermark):
$$s \sim \text{Bernoulli}(\sigma(\gamma \cdot (h \cdot g_k))) \quad \text{green/red token split}$$

**变化点**: 本文watermark不是事后添加，而是**rewriting process的固有副产品**——instruction替换模式形成natural signature。

**本文公式（推导）**:
$$\text{Step 1}: \quad K = \text{length of suspect output sequence}$$
$$\text{Step 2}: \quad T_K = \sum_{i=1}^{K} \mathbb{1}[\text{token}_i \in \text{RewritePattern}(\phi)]$$
$$\text{最终}: \quad \text{Detect} = \mathbb{1}\left[ T_K > \tau(K) \right], \quad \tau(K) = \mathbb{E}[T_K^{\text{clean}}] + z_\alpha \sqrt{\text{Var}(T_K^{\text{clean}})}$$

**对应消融**: Figure 4 显示true detection rate和false alarm rate随K的变化，llama3.1-8B作为suspect student model。

## 实验与分析

主实验结果对比（GSM8K / MATH）：

| Method | GSM8K (Student Acc ↓) | MATH (Student Acc ↓) | 备注 |
|--------|------------------------|----------------------|------|
| Clean KD (baseline) | 高 | 高 | 无保护 |
| Watermark only | 无影响 | 无影响 | 仅检测 |
| Instruction Rewriting (OPT) | ↓ | ↓ | Figure 3 left |
| Instruction Rewriting (KPOD) | ↓ | ↓ | Figure 3 right |
| + Token-Level Poisoning | 进一步↓ | 进一步↓ | Figure 9 |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab3832d5-94a2-44c9-b6ec-73eb7039b623/figures/Figure_3.png)
*Figure 3: Figure 2: Comparison of our rewriting approaches for anti-distillation on GSM8K (left) and MATH (right).*



**核心发现分析**:
- **主效应**: Figure 3 显示在GSM8K和MATH两个数学推理benchmark上，OPT和KPOD两种rewriting策略均有效降低student model accuracy，验证了instruction-level干预的anti-distillation有效性
- **方法对比**: KPOD在setting下表现更robust，OPT在setting下distillation degradation更强
- **叠加效应**: Token-level poisoning（Figure 9）在instruction rewriting基础上提供additional margin，FO-Grad近似有效降低计算成本

**消融实验**:
- **Rewriting component ablation**: 单独instruction rewriting vs. 单独token poisoning vs. 联合使用
- **Adaptive attack robustness**: Figure 7 显示对paraphrase attack的抵抗能力——distiller先paraphrase OPT traces再fine-tuning，KPOD variant保持有效性

**Fairness & Limitation检查**:
- **Baselines强度**: 对比了standard KD和SFT，但未与最新的adaptive distillation方法（如MetaDistil, GKD）对比
- **Compute cost**: Rewriting需要additional forward/backward passes，latency overhead
- **Failure cases**: 对very large student model（如70B+）的anti-distillation效果；对chain-of-thought length variation的sensitivity
- **Ethical consideration**: 正常用户可能接收到slightly altered reasoning traces，虽保持正确性但可能影响可解释性信任


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab3832d5-94a2-44c9-b6ec-73eb7039b623/figures/Figure_7.png)
*Figure 7: Figure 5: Robustness of anti-distillation to adaptive attacks. Paraphrased: distiller paraphrases our OPT tracesbefore fine-tuning. KPOD: distiller applies keypoint-based progressive distillation on o*



## 方法谱系与知识库定位

**方法家族**: LLM Security → Model Protection → Anti-Distillation / Watermarking

**Parent Method**: Knowledge Distillation (Hinton et al.) 的对抗视角——本文不是改进KD，而是**破坏KD的有效性**。

**Changed slots** (相比标准watermarking/poisoning):
- **Architecture**: 无模型结构改变，纯data-side intervention
- **Objective**: 从"检测盗版"(watermark) 或 "降低质量"(naive poisoning) 转向"精准破坏reasoning学习"
- **Training recipe**: Instruction-based rewriting + FO-Grad token poisoning 双层pipeline
- **Data curation**: 动态rewriting而非静态dataset poisoning
- **Inference**: Teacher model需运行rewriting engine，增加latency

**Direct baselines & differences**:
- **KGW Watermark** (Kirchenbauer et al.): 本文watermark是rewriting副产品，非显式嵌入；且具备preventive而非仅detective功能
- **Gradient-based Dataset Poisoning** (e.g., Bullseye Polytope): 本文针对sequence-level reasoning traces，非image classification；且保持utility constraint
- **Output Restriction (API-level)**: 本文不限制输出内容，而是surgically alter输出结构

**Follow-up directions**:
1. **Adaptive distiller arms race**: 如何防御知道trace rewriting机制的adaptive attacker？（Figure 7初步探索paraphrase attack）
2. **Multi-turn conversation protection**: 当前针对single reasoning trace，multi-turn dialog的temporal consistency如何保持？
3. **Theoretical characterization**: Rewriting对student model sample complexity的影响——需要多少poisoned traces才能有效防御？

**Tags**: 
- Modality: Text / Language Model
- Paradigm: Supervised Fine-Tuning Defense, Trace Rewriting
- Scenario: API Protection, Unauthorized Distillation Prevention
- Mechanism: Instruction Substitution, Gradient-Aware Token Poisoning, Implicit Watermarking
- Constraint: Utility-Preserving, Detectable, Robust to Paraphrase

