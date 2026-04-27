---
title: 'SilentStriker: Toward Stealthy Bit-Flip Attacks on Large Language Models'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 面向LLM的隐蔽位翻转攻击SilentStriker
- SilentStriker
- SilentStriker is the first stealthy
acceptance: Poster
cited_by: 2
method: SilentStriker
modalities:
- Text
---

# SilentStriker: Toward Stealthy Bit-Flip Attacks on Large Language Models

**Topics**: [[T__Adversarial_Robustness]] | **Method**: [[M__SilentStriker]] | **Datasets**: [[D__DROP]], [[D__GSM8K]] (其他: TRIVIA, GPT-Naturalness DROP, WikiText Perplexity)

> [!tip] 核心洞察
> SilentStriker is the first stealthy bit-flip attack against LLMs that achieves significant task performance degradation while maintaining output naturalness by targeting key output tokens for suppression rather than using perplexity-based objectives.

| 中文题名 | 面向LLM的隐蔽位翻转攻击SilentStriker |
| 英文题名 | SilentStriker: Toward Stealthy Bit-Flip Attacks on Large Language Models |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2509.17371) · [DOI](https://doi.org/10.48550/arxiv.2509.17371) |
| 主要任务 | 对抗鲁棒性 / LLM位翻转攻击（BFA） |
| 主要 baseline | GenBFA, PrisonBreak, Bit-flip attack with progressive bit search |

> [!abstract] 因为「现有位翻转攻击要么产生乱码易检测、要么攻击效果弱」，作者在「GenBFA/PrisonBreak」基础上改了「关键输出token抑制 + 迭代渐进搜索 + 联合优化攻击有效性与隐蔽性」，在「DROP/GSM8K/TRIVIA」上取得「准确率降至0-12.6%同时GPT自然度保持51-68」

- **攻击效果**：GSM8K准确率从65.7%降至7.6%（INT8）/ 4.2%（FP4），DROP降至5.1% / 0.0%
- **隐蔽性优势**：WikiText困惑度仅60.4-152.9，远低于GenBFA的5.5×10⁵-6.1×10⁵
- **模型规模**：覆盖3B至32B参数模型，包括LLaMA2、Qwen3等

## 背景与动机

大语言模型（LLM）的量化部署使其权重以低精度格式（INT8/FP4）存储，这为位翻转攻击（Bit-Flip Attack, BFA）提供了物理层面的可乘之机——攻击者可通过Rowhammer等硬件漏洞或供应链篡改，翻转模型权重中的个别比特，从而破坏模型行为。然而，现有攻击面临一个根本困境：要么攻击效果强但极易被发现，要么隐蔽性好但几乎无损模型性能。

具体而言，GenBFA采用进化优化直接最小化输出困惑度（perplexity），虽能将准确率压至零，但输出沦为完全乱码（gibberish），PPL飙升至10⁵量级，任何基于困惑度的检测器都能轻易识别。PrisonBreak则走另一极端，通过少于25个精准位翻转实现jailbreak，但对常规任务准确率影响微弱（DROP仅降约4个百分点），缺乏实质性的性能破坏能力。两者均未解决「有效破坏」与「隐蔽自然」的联合优化问题。

这一缺口的根源在于目标函数设计：GenBFA的全局困惑度最小化导致语言模型整体崩溃，PrisonBreak的任务特定目标又过于局限。SilentStriker提出关键洞察——不必摧毁整个语言分布，只需精准抑制答案中的关键token即可误导输出，同时保持语句流畅。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3e56cf6e-c87f-48e7-9a8a-b2876bcaf7e8/figures/Figure_1.png)
*Figure 1 (motivation): The goal of our SilentStriker. Unlike previous methods, our method can compromise the model outputs in a stealthy manner.*



## 核心创新

核心洞察：攻击应针对关键输出token进行精准抑制而非全局困惑度破坏，因为LLM的生成质量对少数关键决策token高度敏感，从而使「高攻击有效性 + 高输出自然度」的联合优化成为可能。

| 维度 | Baseline (GenBFA/PrisonBreak) | 本文 (SilentStriker) |
|:---|:---|:---|
| 攻击目标 | 全局困惑度最小化 / Jailbreak特定提示 | 关键输出token抑制 |
| 优化策略 | 单目标（效果或隐蔽） | 联合优化：效果 + 自然度 |
| 搜索方式 | 直接位选择 / 固定目标位 | 迭代渐进搜索，topK=10位/步 |
| 输出特征 | 乱码（PPL~10⁵）/ 几乎无损 | 自然流畅但错误（PPL~10²） |
| 攻击数据 | 未显式构建 / 大规模 | Nq=2条GPT-4o生成简单问题 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3e56cf6e-c87f-48e7-9a8a-b2876bcaf7e8/figures/Figure_2.png)
*Figure 2 (architecture): The overview of our SilentStriker framework.*



SilentStriker的攻击流程包含五个串联模块：

1. **Attack Dataset Generation（攻击数据集生成）**：输入为GPT-4o提示「请生成Nq个跨领域的简单问题」，输出Nq=2条攻击问题。极小数据集降低攻击准备成本。

2. **Key Output Token Identification（关键输出token识别）**：输入攻击问题与victim model，通过前向传播识别对正确答案贡献最大的关键输出token集合𝒯_key，作为后续抑制目标。

3. **Iterative Progressive Bit Search（迭代渐进位搜索）**：输入目标token集合与模型权重，采用渐进式搜索识别脆弱位位置。每步选取topK=10个最优位，逐步逼近最优攻击位集合。

4. **In-Module Attack Execution（模块内攻击执行）**：输入选定位置与topK约束，执行实际位翻转操作，修改量化后权重。

5. **Joint Effectiveness-Stealthiness Evaluation（联合评估）**：输入攻击后模型输出，同步计算任务准确率、GPT-4o自然度分数（0-100）、WikiText困惑度，反馈至搜索过程指导下一步迭代。

```
GPT-4o提示 → [Attack Dataset] → Nq=2问题
                                    ↓
Victim Model → [Key Token ID] → 𝒯_key（目标token集）
                                    ↓
𝒯_key + 权重 → [Progressive Search] → 脆弱位候选
                                    ↓（迭代，topK=10/步）
[Bit Flip Execution] → 篡改后量化权重
                                    ↓
[Joint Eval] → 准确率 ↓ + 自然度 ↑（GPT-NAT + PPL）
```

## 核心模块与公式推导

### 模块 1: 关键输出token抑制损失（对应框架图 Key Token ID → Progressive Search）

**直觉**：不必让整个模型「失语」，只需让模型「说错话」——精准压制答案中的关键token，使模型转向错误但自然的替代输出。

**Baseline 公式** (GenBFA): $$\mathcal{L}_{\text{PPL}} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i}, \theta \oplus \Delta)$$
符号: $\theta$ = 原始模型参数, $\Delta$ = 位翻转掩码, $\theta \oplus \Delta$ = 篡改后参数, $N$ = 序列长度。该损失最小化全局困惑度，导致语言模型整体崩溃。

**变化点**：全局困惑度最小化使所有token概率分布扭曲，产生乱码；应改为仅抑制特定关键token，保持其余分布自然。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{T}_{\text{key}} = \text{TopK}_t \left\{ \frac{\partial \text{Acc}(x_{\text{attack}}, \theta)}{\partial P(t|x_{\text{attack}}, \theta)} \right\} \quad \text{识别对准确率影响最大的关键token}$$
$$\text{Step 2}: \mathcal{L}_{\text{suppress}} = -\sum_{t \in \mathcal{T}_{\text{key}}} \log P(t | x_{\text{attack}}, \theta \oplus \Delta) \quad \text{仅抑制这些token的生成概率}$$
$$\text{最终}: \mathcal{L}_{\text{attack}} = \mathcal{L}_{\text{suppress}} + \lambda \cdot \mathcal{L}_{\text{constraint}} \quad \text{加入约束防止过度扰动}$$
**对应消融**：Table 4显示损失函数组件的消融效果，验证关键token抑制项的必要性。

### 模块 2: 联合攻击-隐蔽性优化（对应框架图 Joint Eval → Progressive Search反馈）

**直觉**：单目标优化必然牺牲一方，需显式引入隐蔽性约束，将攻击重构为带约束的多目标问题。

**Baseline 公式** (GenBFA/PrisonBreak): $$\Delta^* = \text{arg}\max_{\Delta: \|\Delta\|_0 \leq N_{\text{bits}}} \mathcal{L}_{\text{attack}}(\theta \oplus \Delta)$$
符号: $\|\Delta\|_0$ = 翻转位数量（L0范数）, $N_{\text{bits}}$ = 预算上限。基线仅最大化攻击效果，无隐蔽性控制。

**变化点**：缺乏自然度约束导致GenBFA输出PPL达10⁵量级；需引入GPT-4o自然度评分与困惑度联合约束。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}_{\text{stealth}} = -\alpha \cdot \text{GPT-NAT}(\theta \oplus \Delta) + \beta \cdot \text{PPL}(\theta \oplus \Delta) \quad \text{GPT-4o自然度与困惑度联合惩罚}$$
$$\text{Step 2}: \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{attack}} + \lambda_1 \mathcal{L}_{\text{stealth}} + \lambda_2 \|\Delta\|_0 \quad \text{总损失融合攻击、隐蔽、稀疏三项}$$
$$\text{最终}: \Delta^* = \text{arg}\max_{\Delta} \mathcal{L}_{\text{attack}}(\theta \oplus \Delta) - \lambda \cdot \mathcal{L}_{\text{stealth}}(\theta \oplus \Delta) \quad \text{s.t. } \|\Delta\|_0 \leq N_{\text{bits}}$$
**对应消融**：Table 4显示去掉自然度约束项后，GPT-NAT分数显著下降，验证联合优化的必要性。

### 模块 3: 迭代渐进搜索策略（对应框架图 Progressive Search → Bit Flip Execution）

**直觉**：一次性搜索全部脆弱位易陷入局部最优，分步渐进可动态适应已翻转位的累积效应。

**Baseline 公式** (传统Progressive Bit Search): $$\Delta_{\text{all}} = \text{PBS}(\theta, \mathcal{L}, N_{\text{bits}}) \quad \text{一次性搜索全部位}$$

**变化点**：LLM参数量巨大（3B-32B），一次性搜索效率低且忽略位间交互；需改为分模块迭代，每步固定topK位。

**本文公式（推导）**:
$$\text{Step 1}: \text{for } i = 1, 2, ..., N_{\text{iter}}: \quad \text{外层迭代循环}$$
$$\text{Step 2}: \quad \Delta_i = \text{TopK}_{\Delta} \left\{ \nabla_{\Delta} \mathcal{L}_{\text{total}}(\theta \oplus \Delta_{<i}) \right\}, \quad |\Delta_i| = 10 \quad \text{每步选梯度最大的10位}$$
$$\text{Step 3}: \quad \theta \leftarrow \theta \oplus \Delta_i; \quad \Delta_{<i+1} = \Delta_{<i} \cup \Delta_i \quad \text{更新权重并累积翻转集合}$$
$$\text{最终}: \Delta^* = \text{bigcup}_{i=1}^{N_{\text{iter}}} \Delta_i, \quad |\Delta^*| \leq N_{\text{bits}}$$
**对应消融**：Figure 3显示topK=10设置下，准确率随位翻转数量渐进下降而自然度保持稳定，验证迭代策略有效性。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3e56cf6e-c87f-48e7-9a8a-b2876bcaf7e8/figures/Table_2.png)
*Table 2 (comparison): Evaluation results after three different BFAs. We evaluated five models under quantization bit widths (W4 and W8) and diverse tasks. The baseline Q-BFA simply flips the bits with the largest absolute gradients. The best results in each block are in bold. Except LLaMA2-7B, all models have INT-quantized weights, and our SilentStriker also runs W4-A16.*



本文在DROP、GSM8K、TRIVIA三个问答基准上评估攻击效果，覆盖LLaMA2-7B/13B、Qwen3-8B/32B等模型，量化设置包括INT8（W8）和FP4（W4）。核心结果来自Table 2：SilentStriker在INT8设置下将GSM8K准确率从clean的65.7%压至7.6%，DROP从49.3%降至5.1%，TRIVIA从74.8%降至12.6%；FP4设置下效果更强，GSM8K仅余4.2%，DROP归零（0.0%）。与此同时，GPT自然度分数（GPT-NAT）保持于51-68区间，WikiText困惑度仅60.4-152.9，与clean模型（19.5-20.7）同数量级。

对比基线呈现鲜明分化：GenBFA虽能将准确率压至零，但GPT-NAT归零、PPL达5.5×10⁵-6.1×10⁵，输出完全不可读；PrisonBreak自然度维持高位（83.6-84.5），但准确率几乎无损（DROP 42.2-45.6%），缺乏实质破坏力。SilentStriker首次同时实现「近零准确率」与「可接受自然度」的帕累托前沿。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3e56cf6e-c87f-48e7-9a8a-b2876bcaf7e8/figures/Table_4.png)
*Table 4 (ablation): Effect of two loss function components. Evaluations on GSM8K using INT4-quantized LLaMA2-7B. LMP denotes language modeling perplexity.*



消融实验（Table 4）聚焦损失函数组件：在GSM8K/INT8/LLaMA2-7B设置下，完整损失函数取得最优平衡；移除关键token抑制项导致攻击效果显著衰减，移除自然度约束项则GPT-NAT大幅下滑。Figure 3进一步展示超参数敏感性：topK=10为效率与效果的平衡点，过小则收敛慢，过大则隐蔽性受损。Figure 4验证W4激活量化的影响，确认group size=128的GPTQ设置下攻击稳定。

公平性审视：基线选择存在结构性差异——PrisonBreak目标为jailbreak而非任务降级，直接对比准确率可能误导；评估依赖GPT-4o自然度判断，存在模型偏见风险；攻击数据集仅Nq=2条，泛化性未充分验证；未测试防御方法（如SmoothLLM）的抵抗效果。此外，实验局限于问答任务，开放域生成、代码生成等场景未覆盖。

## 方法谱系与知识库定位

SilentStriker属于**位翻转攻击（BFA）**方法谱系，直接继承自Rakin等提出的**Progressive Bit Search**（PBS）算法框架，核心位搜索机制沿用其梯度驱动的渐进式定位思想。与父方法相比，本文在三个关键slot上完成改造：**objective**（全局困惑度→关键token抑制）、**inference_strategy**（一次性搜索→迭代topK=10）、**reward_design**（单目标→联合效果-隐蔽优化）。

直接基线差异：
- **GenBFA**：同为LLM-BFA，但采用进化算法+困惑度最小化，输出乱码；SilentStriker替换为目标抑制+联合优化，实现隐蔽攻击
- **PrisonBreak**：同为LLM-BFA，但聚焦jailbreak且位翻转<25，任务破坏弱；SilentStriker扩展为通用任务降级，位预算更灵活
- **PBS（Rakin et al.）**：DNN通用BFA奠基工作，无LLM特定设计；SilentStriker引入token级语义目标与LLM自然度评估

后续方向：（1）扩展至编码/推理等更广任务类型；（2）结合防御方法评估攻击鲁棒性；（3）探索黑盒场景下的迁移攻击，降低白盒权重访问假设。

标签：modality=text | paradigm=adversarial_attack | scenario=LLM_deployment_security | mechanism=bit_flip + token_suppression + progressive_search | constraint=quantized_weights + white_box_access + limited_bit_budget

