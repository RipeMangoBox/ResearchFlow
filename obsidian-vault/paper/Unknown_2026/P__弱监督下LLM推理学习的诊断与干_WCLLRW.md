---
title: When Can LLMs Learn to Reason with Weak Supervision?
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.18574
aliases:
- 弱监督下LLM推理学习的诊断与干预
- WCLLRW
- 弱监督下 RLVR 的成败不取决于 RL 算法本身
paradigm: Reinforcement Learning
---

# When Can LLMs Learn to Reason with Weak Supervision?

[Paper](https://arxiv.org/abs/2604.18574)

**Topics**: [[T__Agent]], [[T__Reasoning]], [[T__Few-Shot_Learning]], [[T__Self-Supervised_Learning]]

> [!tip] 核心洞察
> 弱监督下 RLVR 的成败不取决于 RL 算法本身，而取决于模型在 RL 开始前是否具备「推理忠实度」——即中间步骤真正支持最终答案的能力。缺乏忠实度的模型会走捷径：快速找到表面正确的答案模式，导致奖励饱和但不泛化。Thinking SFT 通过在显式推理轨迹上微调，将忠实推理的先验注入模型，使 RL 有「真正可以强化的东西」；CPT 则通过领域对齐进一步放大这一效果。本质上，这是一个「先装好引擎，再踩油门」的逻辑。

| 中文题名 | 弱监督下LLM推理学习的诊断与干预 |
| 英文题名 | When Can LLMs Learn to Reason with Weak Supervision? |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18574) · [Code] · [Project] |
| 主要任务 | 弱监督下RLVR（可验证奖励强化学习）的泛化诊断与pre-RL干预 |
| 主要 baseline | Base模型、CPT（持续预训练）、Instruct、Non-Thinking SFT、Thinking SFT |

> [!abstract] 因为「弱监督下RLVR的成败无法预判且跨模型家族不一致」，作者在「标准RLVR pipeline」基础上改了「引入pre-RL推理忠实度诊断与Thinking SFT干预」，在「Llama3.2-3B的三种弱监督设置（稀缺数据/噪声奖励/自监督代理奖励）」上取得「Base模型此前全部失败，CPT+Thinking SFT全部实现泛化」

- **关键性能1**: CPT + Thinking SFT 在三种弱监督设置下均实现泛化，Base模型此前全部失败（Figure 6）
- **关键性能2**: Thinking SFT 是Llama弱监督下实质性学习的必要条件，Non-Thinking SFT无效或性能下降
- **关键性能3**: CPT（52B token）放大Thinking SFT效果但不能替代它，compute-matched对比排除计算量解释

## 背景与动机

大语言模型的推理能力通常通过RLVR（Reinforcement Learning with Verifiable Rewards）来强化——例如用最终答案的正确性作为奖励信号来训练模型解数学题。然而现实场景中，高质量的可验证奖励往往难以获得：标注数据可能稀缺，奖励信号可能带噪声，或者只能依赖模型自监督生成的代理奖励。这些弱监督条件下的RLVR何时能成功、何时会失败，现有文献缺乏系统性理解。

现有研究的主要做法可分为三类。**第一类**聚焦算法改进，如改进GRPO等RL算法的探索效率或稳定性，但假设模型已具备基本推理能力。**第二类**研究SFT初始化对RL的影响，但多使用Non-Thinking SFT（仅在最终答案上微调），未区分推理轨迹的作用。**第三类**工作报告跨模型家族的不一致现象——Qwen系列上有效的方案在Llama上可能完全失效——却未解释深层原因。

这些方法的共同短板在于：**将RL视为独立优化阶段，忽视pre-RL模型属性对弱监督学习成败的决定性作用**。具体而言，现有工作无法回答两个关键实践问题：（1）诊断层面——在投入大量RL计算之前，如何判断给定模型和弱监督条件是否值得训练？（2）干预层面——当预判RL会失败时，应在哪个阶段、以何种低成本方式介入？

本文通过系统性实证研究，识别出「推理忠实度」（reasoning faithfulness）作为pre-RL的核心预测属性，并据此设计具体的pre-RL干预方案，将RLVR的成败判断从「黑箱试错」转化为「可诊断、可干预」的工程问题。

## 核心创新

核心洞察：弱监督下RLVR的成败不取决于RL算法本身，而取决于模型在RL开始前是否具备「推理忠实度」——即中间推理步骤在逻辑上真正支持最终答案的能力，因为缺乏忠实度的模型会走捷径找到表面正确的答案模式导致奖励饱和但不泛化，从而使「通过pre-RL干预提升忠实度」成为恢复泛化的关键杠杆。

| 维度 | Baseline（标准RLVR） | 本文 |
|:---|:---|:---|
| 诊断指标 | 仅看最终奖励/测试性能 | 训练奖励饱和动态（预饱和阶段长度）作为实时诊断信号 |
| 预测属性 | 输出多样性 | 推理忠实度（aligned-response rate），多样性单独不具预测力 |
| 干预阶段 | RL算法调参或延长训练 | pre-RL阶段：Thinking SFT注入忠实推理先验，CPT放大效果 |
| 核心假设 | RL可以教模型推理 | RL只能强化已有能力；pre-RL决定「有没有东西可强化」 |
| 实践逻辑 | 「踩油门」 | 「先装好引擎，再踩油门」 |

## 整体框架


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99ca482f-3196-4413-a1b9-902ff7ea8815/figures/Figure_6.png)
*Figure 6: Figure 6. RL training dynamics and generalization on MATH for Llama3.2-3B Base, CPT, and Instruct variants under differentSFT initializations across three weak supervision settings: scarce data (N = 8*



本文的实证框架是一个「诊断-干预」两阶段pipeline，数据流如下：

**输入**：目标模型（如Llama3.2-3B-Base）+ 弱监督设置（三选一：8-sample稀缺数据、γ比例噪声奖励、自监督代理奖励）+ 下游评估任务（MATH-500、AMC域内、SCP-Hard域外）

**Pre-RL诊断阶段**：
- **模块A：推理忠实度评估** —— 输入pre-RL模型，输出aligned-response rate（中间步骤与最终答案的逻辑一致性比例），用于预判泛化区间
- **模块B：饱和动态监测** —— 输入RL训练过程，输出训练奖励曲线与下游性能曲线的同步性，识别「预饱和阶段」vs「后饱和平台期」

**Pre-RL干预阶段**：
- **模块C：SFT配置选择** —— 输入Base/CPT/Instruct模型，输出Thinking SFT（显式推理轨迹微调）或Non-Thinking SFT（仅最终答案微调）的初始化模型
- **模块D：CPT增强（可选）** —— 输入Base模型，输出52B token持续预训练后的领域对齐模型

**RL训练阶段**：
- **模块E：标准RLVR** —— 输入pre-RL配置模型，输出训练后模型及奖励/性能曲线

**输出**：泛化与否的二元判断 + 最优pre-RL配置推荐

```
[弱监督设置 + 模型] 
    → [诊断: 推理忠实度评估] → [预判: 能否泛化?]
    → [干预: SFT配置选择 + CPT?] → [RLVR训练] 
    → [监测: 饱和动态] → [输出: 泛化结果]
```

## 核心模块与公式推导

### 模块1: 推理忠实度度量（对应框架图 模块A）

**直觉**: 模型可能生成看似合理的推理链却与答案无关（如「幻觉推理」），需要量化中间步骤对最终答案的实际支持力度。

**Baseline概念** (标准答案正确率): 仅衡量最终答案对错，不考察推理过程。

$$\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat{y}_i = y_i]$$

符号: $\hat{y}_i$ = 模型预测答案, $y_i$ = 标准答案, $\mathbb{1}[\cdot]$ = 指示函数。

**变化点**: 最终答案正确率无法检测「正确答案来自错误推理」的捷径行为。需要引入对推理链本身的忠实度评估。

**本文公式（推导）**:
$$\text{Step 1: 响应对齐判定} \quad A_i = \mathbb{1}[\text{reasoning}_i \text{xrightarrow}{\text{逻辑蕴含}} \hat{y}_i] \quad \text{引入逻辑一致性检验以检测虚假推理}$$
$$\text{Step 2: 忠实度聚合} \quad \text{Faithfulness} = \frac{1}{N}\sum_{i=1}^{N} A_i \cdot \mathbb{1}[\hat{y}_i = y_i] \quad \text{仅统计正确样本中的忠实推理，分离「幸运猜对」}$$
$$\text{最终: aligned-response rate} = \frac{\sum_{i}\mathbb{1}[\text{reasoning}_i \text{ supports } \hat{y}_i \land \hat{y}_i = y_i]}{\sum_{i}\mathbb{1}[\hat{y}_i = y_i]}$$

**对应消融**: Figure 5显示推理忠实度与下游泛化高度相关，而输出多样性（diversity）在饱和后仍高却不预示泛化（Figure 4）。

---

### 模块2: 训练动态诊断——预饱和阶段识别（对应框架图 模块B）

**直觉**: 奖励曲线的平台期出现时机揭示了模型是在「学习推理」还是「记忆模式」。

**Baseline公式** (标准RL训练目标): 
$$L_{\text{RL}} = -\mathbb{E}_{(x,y)\sim\pi_\theta}\left[R(x,y)\right]$$

符号: $\pi_\theta$ = 策略模型, $R(x,y)$ = 可验证奖励（如答案正确性1/0）, 目标为最大化期望奖励。

**变化点**: 标准目标仅优化奖励期望，不监测奖励与下游性能的时序关系。快速最大化奖励可能导致对弱监督信号的过拟合。

**本文公式（推导）**:
$$\text{Step 1: 奖励动态分解} \quad R_{\text{train}}(t) = f(t; \theta_t, \mathcal{D}_{\text{weak}}) \quad \text{训练奖励作为时间函数，依赖当前策略和弱监督数据}$$
$$\text{Step 2: 泛化同步检验} \quad G_{\text{test}}(t) = \text{Performance}(\theta_t; \mathcal{D}_{\text{test}}) \quad \text{同步监测下游任务性能，分离「记忆」与「学习」}$$
$$\text{Step 3: 预饱和阶段定义} \quad T_{\text{pre-sat}} = \max\{t : \frac{dR_{\text{train}}}{dt} > 0 \land \frac{dG_{\text{test}}}{dt} > 0\} \quad \text{两曲线同步上升阶段的持续时间}$$
$$\text{最终诊断规则}: \text{若 } T_{\text{pre-sat}} \text{ 短或} \exists t > T_{\text{pre-sat}}: R_{\text{train}}\text{uparrow}, G_{\text{test}}\text{downarrow} \Rightarrow \text{停止RL，转向pre-RL干预}$$

**对应消融**: Figure 6显示Llama3.2-3B-Base在三种弱监督下均快速饱和（$T_{\text{pre-sat}}$极短）且测试性能停滞，而CPT+Thinking SFT延长预饱和阶段并实现泛化。

---

### 模块3: Pre-RL干预——Thinking SFT vs Non-Thinking SFT（对应框架图 模块C）

**直觉**: SFT初始化方式决定了RL开始时模型「知道如何推理」还是「只知道答案」。

**Baseline公式** (Non-Thinking SFT): 
$$L_{\text{non-think}} = -\mathbb{E}_{(x,y)}\left[\log \pi_\theta(y|x)\right]$$

符号: 仅优化最终答案$y$的条件概率，不建模推理过程$r$。

**变化点**: Non-Thinking SFT使模型缺乏显式推理先验，在弱监督下倾向于学习「答案-模式捷径」而非可泛化的推理策略。需要引入推理链作为显式监督信号。

**本文公式（推导）**:
$$\text{Step 1: 推理链条件化} \quad L_{\text{think}}^{(1)} = -\mathbb{E}_{(x,r,y)}\left[\log \pi_\theta(r,y|x)\right] = -\mathbb{E}\left[\log \pi_\theta(r|x) + \log \pi_\theta(y|x,r)\right] \quad \text{显式建模推理链生成}$$
$$\text{Step 2: 忠实度约束强化} \quad L_{\text{think}}^{(2)} = L_{\text{think}}^{(1)} + \lambda \cdot \mathbb{E}\left[\text{Faithfulness}(r,y)\right] \quad \text{加入推理-答案对齐正则项（若数据含标注）}$$
$$\text{最终}: L_{\text{Thinking SFT}} = -\mathbb{E}_{(x,r,y)\sim\mathcal{D}_{\text{reasoning}}}\left[\sum_{t=1}^{|r|+|y|}\log \pi_\theta(o_t|o_{<t},x)\right]$$

其中$o_t$为推理链$r$和答案$y$的联合token序列，$\mathcal{D}_{\text{reasoning}}$为含显式推理轨迹的数据。

**对应消融**: Figure 6中CPT+Non-Thinking SFT vs CPT+Thinking SFT的compute-matched对比显示，相同计算量下前者仍失败，后者成功，证明Thinking SFT的关键作用不可替代。

## 实验与分析

| Pre-RL配置 | 稀缺数据 (8-sample) | 噪声奖励 (γ corrupt) | 自监督代理奖励 | 综合 |
|:---|:---|:---|:---|:---|
| Base | ❌ 失败（快速饱和） | ❌ 失败 | ❌ 失败 | 全部失败 |
| Base + Non-Thinking SFT | ❌ 无效/下降 | ❌ 无效 | ❌ 无效 | 无效或更差 |
| Base + Thinking SFT | ⚠️ 部分改善 | ⚠️ 部分改善 | ⚠️ 部分改善 | 有改善但不充分 |
| CPT (52B) + Non-Thinking SFT | ❌ 仍失败 | ⚠️ 噪声设置小幅提升 | ❌ 失败 | 不能替代Thinking SFT |
| **CPT + Thinking SFT** | ✅ **泛化** | ✅ **泛化** | ✅ **泛化** | **全部成功** |
| Instruct + Thinking SFT |  |  |  | 作为额外验证 |


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99ca482f-3196-4413-a1b9-902ff7ea8815/figures/Figure_1.png)
*Figure 1: Figure 2. Effect of reward label corruption on training dynam-ics and generalization. γ denotes the fraction of training promptswith corrupted labels, ranging from clean (γ = 0) to mostly in-correct (*



**核心数字解读**：Figure 6的五条曲线对比是最关键的证据。Llama3.2-3B-Base在三种弱监督设置下均呈现「奖励快速饱和但测试性能停滞」的典型失败模式（avg@16指标无提升）。CPT+Thinking SFT联合配置打破了这一模式：预饱和阶段显著延长，训练奖励与MATH-500/AMC/SCP-Hard性能同步上升，最终实现域内域外双泛化。

**消融分析**：
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99ca482f-3196-4413-a1b9-902ff7ea8815/figures/Figure_5.png)
*Figure 5: Figure 5. Evolution of reasoning faithfulness (on correct sam-ples) and faithful diversity on models throughout RL using 8samples from a variety of datasets. Llama models in the MATHdomain exhibit sig*


- **Thinking SFT的必要性**：Figure 6中CPT+Non-Thinking SFT曲线（绿色）与CPT+Thinking SFT（红色）的compute-matched对比至关重要——两者消耗相同预训练计算量，但前者仍快速饱和，证明CPT不能弥补Non-Thinking SFT的缺陷。
- **CPT的放大作用**：Base+Thinking SFT有一定改善但不如CPT+Thinking SFT稳定，说明52B token的持续预训练通过领域对齐放大了忠实推理先验的效果。
- **多样性的迷思**：Figure 4显示Llama在饱和后保持较高语义多样性，Figure 5显示此时推理忠实度已崩溃——直接反驳「多样性即泛化」的朴素假设。

**公平性检查**：
- **基线强度**：未与DAPO、Dr.GRPO等最新RLVR算法直接对比，基线完整性受限；但本文核心主张是pre-RL属性决定成败，RL算法为固定控制变量，此设计合理。
- **计算成本**：CPT 52B token + Thinking SFT的pre-RL投入 vs 延长RL训练的成本权衡，论文明确建议优先前者。
- **失败案例**：Non-Thinking SFT在所有设置下的失效构成「负面结果」，增强结论稳健性；但跨模型家族（Qwen等）的系统性证据在正文中呈现不足，外推风险存在。
- **度量局限**：aligned-response rate的自动评估可能误判某些推理变体，饱和阶段的视觉判断缺乏形式化阈值。

## 方法谱系与知识库定位

**方法家族**：RLVR（Reinforcement Learning with Verifiable Rewards）→ 弱监督学习 → pre-RL诊断与干预

**父方法**：标准RLVR pipeline（SFT初始化 → RL训练），本文将其扩展为「pre-RL属性诊断 + 针对性干预」的两阶段框架。

**改动插槽**：
- **training_recipe**: 核心改动——将SFT细分为Thinking/Non-Thinking两种配置，引入CPT作为pre-RL增强
- **data_curation**: 使用含显式推理轨迹的数据进行Thinking SFT，区别于答案-only SFT
- **architecture**: 未改动（保持Llama3.2-3B等标准架构）
- **objective**: RL阶段目标未变，但pre-RL阶段引入忠实度约束
- **inference**: 未改动

**直接基线对比**：
- **vs 标准RLVR（如DeepSeekMath-GRPO）**: 本文不改进RL算法，而改进「RL之前的准备」；GRPO等方法假设模型已具备可强化能力，本文解决「何时不具备」的诊断与修复
- **vs SFT-only方法（如拒绝采样微调RFT）**: 本文将SFT作为RL的初始化手段而非终点，强调RL的必要性但前提是pre-RL正确配置
- **vs CPT相关研究**: 已有CPT工作关注通用能力提升，本文首次将其与Thinking SFT联合，作为弱监督RLVR的特定干预

**后续方向**：
1. **跨模型家族验证**：将诊断框架系统应用于Qwen、Gemma等系列，检验推理忠实度-泛化关系的普适性
2. **自动化饱和检测**：将预饱和阶段的视觉判断转化为可计算的early stopping准则
3. **Scaling law探索**：更大规模CPT或Thinking SFT数据是否能进一步降低对弱监督质量的敏感度

**知识库标签**：
- modality: text / reasoning
- paradigm: reinforcement_learning, supervised_finetuning, continual_pretraining
- scenario: low_resource, noisy_supervision, weak_supervision
- mechanism: faithfulness, saturation_dynamics, generalization_diagnosis
- constraint: compute_limited, label_scarce, reward_noisy

