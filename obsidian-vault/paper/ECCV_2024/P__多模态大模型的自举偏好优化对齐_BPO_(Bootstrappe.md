---
title: Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization
type: paper
paper_level: C
venue: ECCV
year: 2024
paper_link: null
aliases:
- 多模态大模型的自举偏好优化对齐
- BPO (Bootstrappe
- BPO (Bootstrapped Preference Optimization)
acceptance: Oral
cited_by: 86
method: BPO (Bootstrapped Preference Optimization)
followups:
- 多模态模型评判器LLaVA-Cr_LLaVA-Critic
---

# Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization

**Topics**: [[T__Visual_Question_Answering]], [[T__Retrieval]], [[T__Reasoning]] | **Method**: [[M__BPO]] | **Datasets**: [[D__MM-Vet]], [[D__LLaVA-Bench]] (其他: Object HalBench)

| 中文题名 | 多模态大模型的自举偏好优化对齐 |
| 英文题名 | Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization |
| 会议/期刊 | ECCV 2024 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2403.08730) · [Code](https://github.com/pi-research/BPO) · [Project] |
| 主要任务 | 多模态大语言模型（MLLM）的视觉-语言对齐，减少幻觉、提升有用性与真实性 |
| 主要 baseline | LLaVA1.5（基础模型）、DPO（优化框架）、SFT（消融基线） |

> [!abstract] 因为「MLLM 存在预训练语言先验导致的视觉幻觉和不对齐问题」，作者在「DPO + LLaVA1.5」基础上改了「通过图像弱化提示自举生成负样本的偏好数据构造方式」，在「MM-Vet / LLaVA-Bench / Object HalBench」上取得「7B-BPO 36.8 超越 13B 基线 36.8；13B-BPO 达 41.4（+4.6/+12.5%）」

- **MM-Vet**: 7B-BPO 36.8 vs LLaVA1.5-7B 31.7 (+5.1)，追平 13B 基线；13B-BPO 41.4 vs LLaVA1.5-13B 36.8 (+4.6)
- **SFT 消融**: 仅正样本 SFT 得 33.3（7B）/ 38.3（13B），比 BPO 低 3.5/3.1 分，证明负样本不可或缺
- **训练成本**: 8×A40 (48GB)，7B 模型 17 小时，13B 模型 28 小时

## 背景与动机

当前多模态大语言模型（MLLM）如 LLaVA1.5 虽然能回答视觉问题，但常产生「幻觉」——即生成与图像内容不符、依赖语言先验的响应。例如，面对一张没有香蕉的图片，模型可能因预训练语料中「水果篮常含香蕉」的偏见而错误回答「有香蕉」。这种视觉-语言不对齐严重损害了模型的有用性（helpfulness）和真实性（truthfulness）。

现有方法主要从三个角度应对此问题：
- **SFT（监督微调）**：在高质量指令数据上微调，但仅学习「正确回答」而缺乏对「错误模式」的显式区分，难以抑制深层偏见；
- **RLHF（基于人类反馈的强化学习）**：需训练显式奖励模型，流程复杂且对多模态场景标注成本高；
- **DPO（直接偏好优化）**：省去奖励模型，直接优化偏好对，但其标准实现依赖人工标注或外部模型排序的偏好数据，未针对 MLLM 的视觉特性定制负样本。

这些方法的根本局限在于：**负样本来源与 MLLM 自身的失败模式脱节**。人工标注的负样本难以系统覆盖模型的预训练偏见；外部模型生成的负样本又未必反映目标模型的真实弱点。因此，需要一种能**主动暴露并针对性纠正模型自身偏见**的数据构造机制。

本文提出 BPO（Bootstrapped Preference Optimization），核心思想是利用 MLLM 自身生成负样本——通过故意弱化视觉输入来「诱骗」模型暴露其语言先验偏见，从而构建针对性的偏好对进行优化。

## 核心创新

核心洞察：**MLLM 的预训练语言偏见可以通过故意弱化视觉输入来主动暴露**，因为当视觉信号减弱时，模型被迫依赖语言先验生成响应，这些「脱锚」响应恰好构成了最有教学价值的负样本，从而使无需人工标注的针对性偏好优化成为可能。

| 维度 | Baseline (DPO/LLaVA1.5) | 本文 (BPO) |
|:---|:---|:---|
| 负样本来源 | 人工标注、外部模型排序、或简单采样 | **自举生成**：用目标 MLLM 自身在图像弱化条件下的输出 |
| 负样本特性 | 通用错误，不一定反映目标模型弱点 | **针对性暴露预训练偏见**，与视觉输入 explicitly ungrounded |
| 数据构造成本 | 需额外标注或调用外部模型 | **零额外标注**，利用模型自身生成 |
| 优化目标 | 通用偏好对齐 | **强化视觉输入的偏好权重**，抑制语言先验主导 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caae98d-d298-4195-a4d9-1e74c093397f/figures/Figure_1.png)
*Figure 1 (pipeline): Figure 1: Illustration of the proposed SFT-BPO framework.*



BPO 整体流程以 LLaVA1.5 检查点为起点，通过「双路响应生成 → 偏好数据集构建 → DPO 优化」三步完成增强：

1. **正样本生成（Positive Response Generation）**：输入为完整图像 + 文本问题，输出为基于视觉信息的 grounded 正确响应 $y^+$；
2. **图像弱化负样本生成（Image-weakened Negative Generation）**：输入为经 $g_{\text{weaken}}$ 处理后的弱化/移除图像 + 相同问题，输出为暴露语言先验的 ungrounded 错误响应 $y^-$；
3. **LLM 错误注入（可选, LLM Error Injection）**：输入为正样本 $y^+$，经 LLM 组件注入特定错误类型，输出多样化负样本（Table 1 展示具体提示模板）；
4. **偏好对构造（Preference Pair Construction）**：将 $(x_{\text{visual}}, x_{\text{text}}, y^+, y^-)$ 组合为偏好数据集 $\mathcal{D}_{\text{BPO}}$（Table 2 展示数据来源分布）；
5. **DPO 风格优化（DPO Optimization）**：在偏好数据上执行 DPO 训练，输出 BPO 增强模型。

```
LLaVA1.5 Checkpoint
    ├── 完整图像 + 问题 ──→ y⁺ (正样本, grounded)
    └── 弱化图像 + 问题 ──→ y⁻ (负样本, 暴露偏见)
              ↓
    ┌─────────────────┐
    │  Preference Pair │  + 可选 LLM 错误注入
    │  (x_v, x_t, y⁺, y⁻) │
    └─────────────────┘
              ↓
    DPO-style Optimization (LoRA r=64, lr=2e-6, 2 epochs)
              ↓
         BPO Model
```

## 核心模块与公式推导

### 模块 1: DPO 基础损失（对应框架图「DPO Optimization」模块）

**直觉**: 直接偏好优化绕过显式奖励模型，将偏好学习转化为分类问题，但标准 DPO 未解决「偏好数据从何而来」的问题。

**Baseline 公式** (DPO): 
$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

符号: $\pi_\theta$ = 策略模型, $\pi_{ref}$ = 参考模型（冻结）, $\beta$ = 温度系数控制与参考模型的偏离程度, $y_w, y_l$ = 人类标注的优胜/失败响应。

**变化点**: 标准 DPO 的 $\mathcal{D}$ 依赖人工标注或外部模型排序，成本高且与目标模型弱点不对齐。BPO 完全替换数据来源。

**本文公式**:
$$\text{Step 1: 数据替换} \quad \mathcal{D}_{BPO} = \{(x_{visual}, x_{text}, y^+, y^-)\} \quad \text{用自举偏好对替代人工标注}$$
$$\text{Step 2: 保持优化形式} \quad \mathcal{L}_{BPO} = -\mathbb{E}_{(x_v, x_t, y^+, y^-) \sim \mathcal{D}_{BPO}} \left[ \log \sigma \left( \beta \Delta^+ - \beta \Delta^- \right) \right]$$
其中 $\Delta^+ = \log \frac{\pi_\theta(y^+|x_v, x_t)}{\pi_{ref}(y^+|x_v, x_t)}$, $\Delta^- = \log \frac{\pi_\theta(y^-|x_v, x_t)}{\pi_{ref}(y^-|x_v, x_t)}$

**对应消融**: Table 4 显示，若仅保留正样本做 SFT（即移除偏好学习），7B 模型从 36.8 降至 33.3，下降 3.5 分；13B 从 41.4 降至 38.3，下降 3.1 分。

---

### 模块 2: 图像弱化负样本生成（对应框架图「Image-weakened Negative Generation」模块）

**直觉**: MLLM 的视觉-语言不对齐根源在于预训练语言模型的强大先验；故意剥夺视觉信息会迫使模型「裸奔」，其输出恰好标记了需要抑制的失败模式。

**Baseline 公式** (标准负采样):
$$y^- = f_{\text{external}}(x_{visual}, x_{text}) \quad \text{或} \quad y^- \sim \text{human annotations}$$

符号: $f_{\text{external}}$ = 外部模型（如 GPT-4、其他 MLLM）或人工标注。

**变化点**: 外部负样本未必反映目标模型的真实弱点；BPO 让模型「自我暴露」偏见，实现针对性教学。

**本文公式（推导）**:
$$\text{Step 1: 图像弱化} \quad \tilde{x}_{visual} = g_{\text{weaken}}(x_{visual}) \quad \text{其中 } g_{\text{weaken}} \in \{\text{mask}, \text{blur}, \text{remove}, \text{corrupt}\}$$
$$\text{Step 2: 自举生成} \quad y^- = f_{MLLM}(\tilde{x}_{visual}, x_{text}; \theta_{ref}) \quad \text{用参考模型自身生成}$$
$$\text{关键性质: } \mathbb{P}(y^- \text{ contains hallucination}) \gg \mathbb{P}(y^+ \text{ contains hallucination})$$

**最终**: 负样本 $y^-$ 显式 ungrounded in visual content，与正样本 $y^+$ 形成「视觉依赖度」的鲜明对比。

**对应消融**: Table 5 显示，若用自生成响应但不进行图像弱化（即标准自举），性能弱于图像弱化版本，验证了「purposefully exposing pretraining bias」的必要性（具体数值待补充，原文仅描述趋势）。

---

### 模块 3: LLM 错误注入（可选扩展，对应 Table 1）

**直觉**: 图像弱化主要产生「遗漏型」错误（忽略视觉信息）；LLM 错误注入可补充「编造型」错误（主动虚构不存在的信息），增加负样本多样性。

**Baseline**: 无此组件，仅依赖单一负样本来源。

**本文实现**: 将正样本 $y^+$ 输入 LLM，通过特定提示模板（Table 1 展示）指令其注入各类错误：对象替换、属性篡改、关系错误、数量错误等。

**公式**:
$$y^-_{\text{inject}} = f_{LLM}(y^+; p_{\text{error}}) \quad \text{其中 } p_{\text{error}} \text{ 为 Table 1 中的错误注入提示}$$

该组件与图像弱化负样本共同构成多样化的偏好数据集（Table 2 展示数据来源的均匀采样策略）。

## 实验与分析


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caae98d-d298-4195-a4d9-1e74c093397f/figures/Table_3.png)
*Table 3 (quantitative): Table 3: Results on MMBench and SEEDBench benchmarks.*



本文在三个核心基准上评估 BPO：MM-Vet（综合视觉能力）、LLaVA-Bench/LLaVA-Wild（视觉有用性）、Object HalBench（幻觉检测）。Table 3 汇总了主要结果。

**MM-Vet  headline 结果**：BPO-7B 取得 36.8 的总分，不仅超越同规模 LLaVA1.5-7B 的 31.7（+5.1，+16.1%），更追平甚至超越 LLaVA1.5-13B 的 36.8——这意味着通过偏好优化，小模型可匹敌大模型基线。BPO-13B 进一步达到 41.4，较 13B 基线提升 +4.6（+12.5%）。这一差距覆盖了识别（Rec）、OCR、知识（Know）、生成（Gen）、空间（Spat）、数学（Math）六个维度，表明视觉偏好的强化具有广泛迁移性。

**幻觉与真实性**：在 Object HalBench 上，BPO 在响应级和对象级幻觉指标上均较 LLaVA1.5 基线降低，验证了「暴露偏见 → 抑制偏见」机制的有效性。LLaVA-Bench 的有用性评分也显示一致提升。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caae98d-d298-4195-a4d9-1e74c093397f/figures/Table_4.png)
*Table 4 (comparison): Table 4: Comparison with SFT baselines.*



**SFT 对比消融**（Table 4）：仅使用正样本进行监督微调（SFT）的 7B 模型得 33.3，较 BPO 的 36.8 低 3.5 分；13B SFT 得 38.3，较 BPO 的 41.4 低 3.1 分。作者指出这「demonstrates the indispensability of negative responses and preference learning」——负样本的存在使模型学会区分 grounded 与 ungrounded 响应，而非简单记忆正确答案。

**图像弱化消融**（Table 5）：对比「图像弱化自举」与「无图像弱化的自举」（直接用 MLLM 正常生成作为负样本），前者性能更优，证明「purposefully exposing pretraining bias」优于被动采样。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caae98d-d298-4195-a4d9-1e74c093397f/figures/Figure_5.png)
*Figure 5 (ablation): Fig. 5: The effect of image-informed prompting.*



**公平性检查**：
- **基线强度**：主要对比限于 LLaVA1.5 同一家族，未与 InstructBLIP、Qwen-VL、GPT-4V 等跨模型比较；也未对比 IPO、KTO、RLHF-PPO 等其他偏好优化变体。
- **计算预算**：8×A40 训练配置对学术界可及，但未报告与全量微调的对比。
- **评估偏差**：LLaVA-Bench 使用 GPT-4 作为评判，可能存在固有偏见；Object HalBench 的自动度量也可能漏检细微幻觉。
- **缺失消融**：Table 1 的 LLM 错误注入组件未在消融中单独量化贡献，其必要性证据较弱。

## 方法谱系与知识库定位

**方法家族**: 偏好优化（Preference Optimization）→ 多模态对齐（Multimodal Alignment）

**父方法**: DPO (Direct Preference Optimization, Rafailov et al., 2023)
- BPO 继承 DPO 的「无需奖励模型、直接优化偏好对」框架，但彻底改造了**数据构造机制**（data_pipeline）和**优化目标的针对性**（objective 从通用偏好转为视觉-语言对齐）。

**直接基线与差异**:
| 方法 | 与 BPO 的差异 |
|:---|:---|
| LLaVA1.5 | BPO 在其检查点上进行 DPO 微调而非 SFT，引入负样本暴露偏见 |
| DPO | BPO 替换其人工/外部偏好数据来源，改为自举图像弱化负样本 |
| SFT | BPO 证明仅正样本微调远不足，必须引入针对性负样本的偏好学习 |

**修改槽位**: data_pipeline（自举负样本生成）、objective（视觉偏好强化）、training_recipe（LoRA r=64, 2 epochs, lr=2e-6）

**后续方向**:
1. **跨模型验证**：将 BPO 数据机制迁移至 InstructBLIP、Qwen-VL 等架构，检验通用性；
2. **错误注入量化**：系统消融 LLM 错误注入 vs 图像弱化的互补贡献，设计更精细的错误类型控制；
3. **动态弱化策略**：当前 $g_{\text{weaken}}$ 为固定操作，可探索自适应弱化强度（如根据模型不确定性调整）。

**标签**: 模态-multimodal (vision-language) | 范式-preference optimization, self-bootstrapping | 场景-visual question answering, hallucination reduction | 机制-image weakening, bias exposure | 约束-parameter-efficient (LoRA), no human annotation

## 引用网络

### 后续工作（建立在本文之上）

- [[P__多模态模型评判器LLaVA-Cr_LLaVA-Critic]]: Bootstrapped preference optimization for MLLMs; direct methodological competitor

