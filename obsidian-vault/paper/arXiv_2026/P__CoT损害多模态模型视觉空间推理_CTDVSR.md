---
title: Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.16060
aliases:
- CoT损害多模态模型视觉空间推理
- CTDVSR
modalities:
- Image
---

# Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs

[Paper](https://arxiv.org/abs/2604.16060)

**Topics**: [[T__Agent]], [[T__Visual_Reasoning]], [[T__Benchmark_-_Evaluation]] | **Datasets**: 13 spatial benchmarks aggregate, BLINK, OmniSpatial, MMVP, VSR

| 中文题名 | CoT损害多模态模型视觉空间推理 |
| 英文题名 | Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16060) · [Code] · [Project] |
| 主要任务 | visual spatial reasoning（2D空间关系、3D几何推理、动态/时序空间理解） |
| 主要 baseline | standard CoT prompting, Non-CoT direct answering, No-Image ablation |

> [!abstract] 因为「Chain-of-Thought (CoT) 提示被广泛认为能提升推理能力，但在视觉空间推理中可能适得其反」，作者在「standard CoT prompting」基础上改了「引入No-Image++消融与系统化的CoT/Non-CoT对比评估」，在「13个空间推理基准（Qwen3-VL-8B-Thinking）」上取得「Non-CoT平均66.07% vs CoT 65.43%，但5/13数据集CoT更优」

- **关键性能1**: Qwen3-VL-8B-Thinking上Non-CoT平均准确率66.07%，CoT 65.43%，差距+0.64%（Table 9）
- **关键性能2**: 多个MLMs使用CoT后平均下降约3%（Figure 1）
- **关键性能3**: BLINK数据集上Non-CoT领先CoT达+7.57绝对百分点（66.70 vs 59.13）

## 背景与动机

视觉空间推理是多模态大模型的核心能力之一——判断图中A物体在B的左侧、估算三维场景中物体的相对深度、或理解视频中的运动轨迹。然而，当前社区默认采用Chain-of-Thought (CoT) 逐步推理作为标准评估设置，这一做法在数学和逻辑推理中被证实有效，却未经系统在视觉空间任务中验证。

现有方法如何处理这一问题？**Standard CoT prompting** 要求模型生成中间推理步骤再给出答案，是MRMs（Multimodal Reasoning Models）如GThinker-7B、ViGoRL-7B、Vision-R1的默认配置；**Non-CoT direct answering** 直接输出答案，被视为"弱基线"；**No-Image ablation** 则通过移除图像输入检测模型是否依赖文本捷径，但检测能力有限。

这些方法的共同缺陷在于：**默认假设CoT对视觉推理有益**。具体而言：(1) GThinker-7B、Vision-R1等模型盲目采用CoT-SFT+RL训练，未质疑CoT对空间任务的适用性；(2) 标准No-Image消融提示设计不足，难以精准捕捉模型从问题文本中"幻觉"视觉细节的行为；(3) 缺乏跨2D/3D/动态时空三类推理的统一评估。本文通过系统诊断揭示：CoT反而可能损害视觉空间推理，并提出了增强的检测工具No-Image++。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cede99d7-ba7a-4649-bee5-4e93da41346d/figures/Figure_1.png)
*Figure 1: Figure 1: (Left) CoT vs Non-CoT performance of open-source MRMs. (Right) Bar chart showing the averageaccuracy of various families of MLMs over 13 benchmark datasets. For each model, the left bar show*



## 核心创新

核心洞察：CoT的文本中心化推理机制与视觉空间推理存在结构性冲突，因为语言化的中间步骤会激活模型的文本先验（textual priors），从而使模型在无视觉证据时仍能"合理"地幻觉空间关系成为可能。

与 baseline 的差异：

| 维度 | Baseline | 本文 |
|:---|:---|:---|
| 推理策略 | CoT with image present（默认） | CoT/Non-CoT 系统对比 + No-Image++ 消融 |
| 捷径检测 | 标准No-Image ablation（提示简单） | No-Image++（增强文本提示，精准隔离文本先验影响） |
| 评估范围 | 单一/少量基准 | 13个基准覆盖2D/3D/动态时空三类任务 |
| 研究性质 | 方法改进（训练新模型） | 诊断评估（纯分析，不训练） |

## 整体框架



本文采用四阶段诊断流程，无训练环节，纯评估驱动：

1. **benchmark_collection（输入→标准化问题）**：整合13个空间推理数据集——3DSRBench、BLINK、CV-Bench2D/3D、MindCube、MMSIBench、MMVP、OmniSpatial、RealWorldQA、SAT、SpatialBench、VSR、V*Bench，统一为多选题格式输出。

2. **No-Image++_ablation（图像移除+增强提示→无视觉响应）**：接收原始问题，移除图像输入，替换为增强的文本-only提示（比标准No-Image更精细），输出模型在无视觉条件下的推理结果，用于检测shortcut learning。

3. **CoT_vs_Non-CoT_comparison（同问题双模式→准确率对比）**：对同一问题分别施加CoT（要求逐步推理）和Non-CoT（直接回答）提示，输出pass@1准确率差异。

4. **shortcut_learning_detection（响应分析→诊断结论）**：对比No-Image++响应与ground truth，量化模型依赖文本先验幻觉视觉细节的程度。

```
[13个空间基准] ──→ [统一多选题格式] ──→ [CoT/Non-CoT双路评估]
                                      │
                                      ↓
                    [No-Image++消融: 图像移除 + 增强文本提示]
                                      │
                                      ↓
                    [shortcut learning检测: 文本先验幻觉量化]
```

## 核心模块与公式推导

### 模块 1: 标准化评估指标 pass@1（对应框架图 评估层）

**直觉**: 空间推理任务需要统一、可比的准确率度量，确保CoT与Non-CoT的结果可直接对比。

**Baseline 公式** (标准多模态评估):
$$\text{pass@1} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat{y}_i = y_i]$$

符号: $N$ = 样本总数, $\hat{y}_i$ = 模型第$i$个样本的首次输出答案, $y_i$ = 真实标签, $\mathbb{1}[\cdot]$ = 指示函数。

**变化点**: 本文沿用该指标，但关键创新在于**控制变量设计**——同一模型、同一问题、仅改变提示方式（CoT vs Non-CoT），使pass@1差异纯粹反映推理策略影响。

**本文公式（推导）**:
$$\text{Step 1}: \Delta_{\text{CoT}} = \text{pass@1}_{\text{Non-CoT}} - \text{pass@1}_{\text{CoT}} \quad \text{（计算单数据集策略差异）}$$
$$\text{Step 2}: \bar{\Delta} = \frac{1}{13}\sum_{j=1}^{13}\Delta_{\text{CoT}}^{(j)} \quad \text{（13个基准平均，如Qwen3-VL的+0.64%）}$$
$$\text{最终}: \text{Report} = \{(\Delta^{(j)}, \text{direction}_j)\}_{j=1}^{13} \quad \text{（保留方向信息，揭示不一致性）}$$

**对应消融**: Table 9显示13个数据集中有8个$\Delta_{\text{CoT}} > 0$（Non-CoT优），5个$\Delta_{\text{CoT}} < 0$（CoT优），平均$\bar{\Delta} = +0.64\%$。

### 模块 2: No-Image++ 消融诊断（对应框架图 诊断层）

**直觉**: 标准No-Image仅简单移除图像，模型可能因提示模糊而随机猜测；增强文本提示可迫使模型明确依赖文本先验，从而暴露shortcut learning的严重程度。

**Baseline 公式** (标准No-Image ablation):
$$\text{Response}_{\text{No-Image}} = f_{\theta}(\text{Question}_{\text{minimal}}; \text{no image})$$

符号: $f_{\theta}$ = 多模态模型, $\text{Question}_{\text{minimal}}$ = 最小化文本提示（通常仅保留问题）。

**变化点**: 标准No-Image的提示过于简陋，模型低准确率可能源于"不知道"而非"幻觉"；No-Image++通过**增强文本提示**（如补充选项细节、空间描述线索），使模型有足够文本信息来"编造"视觉答案——若此时准确率显著高于随机，则证明存在严重的文本先验幻觉。

**本文公式（推导）**:
$$\text{Step 1}: \text{Prompt}_{\text{No-Image++}} = \text{Enhance}(\text{Question}, \text{Options}) \quad \text{（注入更多文本线索）}$$
$$\text{Step 2}: \text{Response}_{\text{No-Image++}} = f_{\theta}(\text{Prompt}_{\text{No-Image++}}; \text{no image}) \quad \text{（强制文本先验激活）}$$
$$\text{Step 3}: \text{Hallucination\_Score} = \frac{\text{pass@1}_{\text{No-Image++}} - \text{Random\_Baseline}}{1 - \text{Random\_Baseline}} \quad \text{（归一化幻觉程度）}$$
$$\text{最终}: \text{Shortcut\_Severity} = \text{Hallucination\_Score} \times \mathbb{1}[\text{pass@1}_{\text{No-Image++}} \gg \text{pass@1}_{\text{Random}}]$$

**对应消融**: Table 3和Table 6显示No-Image++比标准No-Image更有效检测shortcut learning，但具体数值在提供上下文中被截断。

### 模块 3: 多选题标准化提示模板（对应框架图 输入层）

**直觉**: 消除提示格式差异对结果的干扰，确保CoT与Non-CoT的唯一变量是"是否要求逐步推理"。

**Baseline 公式** (各基准原始格式):
各数据集原始提示格式不一，部分含选项字母、部分仅含文本。

**变化点**: 本文统一为强制选择格式，要求模型输出"字母+选项文本"，便于自动评估。

**本文公式（推导）**:
$$\text{Step 1}: \text{Template} = \text{Question} \oplus \text{Options\_Enumerated} \oplus \text{Instruction}_{\text{select}}$$
$$\text{Step 2}: \text{CoT\_Variant} = \text{Template} \oplus \text{Instruction}_{\text{think\_step\_by\_step}}$$
$$\text{Step 3}: \text{Non-CoT\_Variant} = \text{Template} \oplus \text{Instruction}_{\text{answer\_directly}}$$
$$\text{最终}: L_{\text{final}} = \text{pass@1}(f_{\theta}(\text{CoT\_Variant})) \;\text{vs}\; \text{pass@1}(f_{\theta}(\text{Non-CoT\_Variant}))$$

符号: $\oplus$ = 字符串拼接, $\text{Instruction}_{\text{select}}$ = "Please select the correct answer (letter and option text) from the options above."

## 实验与分析

**Qwen3-VL-8B-Thinking 主结果（Table 9）**：

| Benchmark | Non-CoT | CoT | Δ (Non-CoT − CoT) |
|:---|:---|:---|:---|
| BLINK | **66.70** | 59.13 | **+7.57** |
| OmniSpatial | **45.73** | 40.90 | **+4.83** |
| MMVP | **79.33** | 76.67 | **+2.66** |
| V*Bench | **81.68** | 79.58 | **+2.10** |
| MMSIBench | **30.40** | 28.70 | **+1.70** |
| VSR | **84.21** | 82.82 | **+1.39** |
| CV-Bench2D | **79.21** | 78.65 | **+0.56** |
| MindCube | 35.14 | 35.14 | 0.00 |
| 3DSRBench | 59.69 | **60.67** | −0.98 |
| CV-Bench3D | 92.67 | **92.75** | −0.08 |
| RealWorldQA | 70.98 | **73.73** | −2.75 |
| SAT | 70.33 | **74.00** | −3.67 |
| SpatialBench | 62.87 | **67.91** | −5.04 |
| **Average** | **66.07** | **65.43** | **+0.64** |



**核心发现分析**：支持主claim的数据集（8/13）中，BLINK (+7.57)、OmniSpatial (+4.83) 提升显著，表明在**细粒度视觉感知**和**综合空间理解**任务上CoT损害最大。然而，SpatialBench (−5.04)、SAT (−3.67)、RealWorldQA (−2.75) 显示CoT更优，这些任务可能涉及**更复杂的逻辑组合**或**需要显式推理链**的空间问题。



**消融实验**：No-Image++在Table 3和Table 6中展示了对shortcut learning的检测能力，优于标准No-Image。Figure 1显示多个MLMs家族平均约3%的CoT性能下降。

**公平性检查**：
- **Baselines强度**：Non-CoT作为"弱基线"意外成为更强方案，但作者未与专门的视觉推理架构对比；缺失Vision-centric CoT变体、空间专用奖励设计等强基线。
- **成本**：纯评估研究，无训练开销，但推理需覆盖17个模型×13个基准×2种模式。
- **统计显著性**：平均+0.64%幅度极小，无置信区间或假设检验，证据强度0.65。
- **失败案例/矛盾**：5/13数据集CoT更优，主claim的普适性受限；Qwen3-VL-8B-Thinking的详细结果未必推广至其他模型（Figure 1为聚合展示）。

## 方法谱系与知识库定位

**方法家族**：评估诊断类元分析（非方法改进）。无单一父方法，属于对现有MRM训练范式的系统性反思。

**修改槽位**：
| 槽位 | 继承 | 修改 |
|:---|:---|:---|
| architecture | — | 无（不训练新模型） |
| objective | pass@1准确率 | 沿用 |
| training_recipe | — | **移除/不适用**（纯评估） |
| data_curation | 标准基准评估 | **修改**为13基准统一覆盖2D/3D/动态 |
| inference | CoT with image | **新增**No-Image++消融 + CoT/Non-CoT系统对比 |

**直接基线与差异**：
- **Standard CoT prompting**：本文证明其在8/13空间基准上劣于Non-CoT
- **No-Image ablation（先前工作）**：本文升级为No-Image++，增强提示设计以精准检测文本先验幻觉
- **GThinker-7B / ViGoRL-7B / Vision-R1 / Vision-G1**：被评估的MRM训练范式，共同假设CoT有益，本文挑战该假设

**后续方向**：(1) 设计强制引用图像区域的Vision-centric CoT变体；(2) 开发空间专用RL奖励函数替代通用CoT-SFT；(3) 在No-Image++框架下系统比较不同训练范式的shortcut learning程度。

**标签**：modality=multimodal (vision+language) | paradigm=diagnostic evaluation / prompting analysis | scenario=spatial reasoning (2D/3D/dynamic) | mechanism=chain-of-thought ablation / shortcut learning detection | constraint=no training, existing-model-only evaluation

