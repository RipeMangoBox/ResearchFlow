---
title: 'PersonaVLM: Long-Term Personalized Multimodal LLMs'
type: paper
paper_level: B
venue: CVPR
year: 2026
paper_link: https://arxiv.org/abs/2604.13074
aliases:
- 长期个性化多模态LLM记忆框架
- PersonaVLM
- 核心直觉是：将个性化能力从模型内部参数中解耦
acceptance: accepted
code_url: https://github.com/MiG-NJU/PersonaVLM
method: PersonaVLM
modalities:
- Text
- Image
---

# PersonaVLM: Long-Term Personalized Multimodal LLMs

[Paper](https://arxiv.org/abs/2604.13074) | [Code](https://github.com/MiG-NJU/PersonaVLM)

**Topics**: [[T__Retrieval]] | **Method**: [[M__PersonaVLM]]

> [!tip] 核心洞察
> 核心直觉是：将个性化能力从模型内部参数中解耦，转移到外部结构化记忆系统与人格画像中。通过四类记忆的分层管理（事件/语义/程序/核心）和Big Five人格的持续推断，系统能够在不修改模型权重的前提下，随交互轮次积累用户画像并动态更新。有效性来源于两点：一是记忆检索将长上下文压缩为精准相关片段，大幅降低计算开销；二是人格画像注入使生成风格与用户个性持续对齐，而非依赖单次输入的静态提示。

| 中文题名 | 长期个性化多模态LLM记忆框架 |
| 英文题名 | PersonaVLM: Long-Term Personalized Multimodal LLMs |
| 会议/期刊 | CVPR 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.13074) · [Code](https://github.com/MiG-NJU/PersonaVLM) · [Project](待补充) |
| 主要任务 | 长期个性化多模态对话、动态用户画像记忆管理、人格对齐生成 |
| 主要 baseline | Qwen2.5-VL-7B (128k-Full, 128k-RAG), MyVLM, Yo'LLaVA, A-Mem, MemoryOS, InternVL3-8B/38B |

> [!abstract] 因为「现有多模态大语言模型只能实现静态单轮个性化，无法追踪用户随时间演变的偏好与人格」，作者在「Qwen2.5-VL-7B」基础上改了「外挂结构化四类记忆+Big Five人格推断模块，替代模型参数微调」，在「Persona-MME基准」上取得「相对128k-RAG基线提升20.4%（59.01%→71.05%），token消耗减少95%（43530→2170）」

- **效率**: 平均token消耗从43530降至2170，减少95.0%，推理加速4.8倍
- **性能**: Persona-MME上PersonaVLM-RL达71.05%，较128k-Full基线（54.48%）提升16.57pp
- **人格对齐**: P-SOUPS Style维度达44.00，开放式生成对比GPT-4o胜率79.0%（Gemini-2.5-Pro评判）

## 背景与动机

想象一位用户与AI助手持续交互数百轮：初期偏好简约风格，中期培养出摄影爱好，后期形成稳定的社交习惯。现有多模态大语言模型（MLLM）无法捕捉这种动态演变——它们要么每次对话从零开始，要么依赖固定的静态提示。

现有方法分为三类，各有致命局限：

**适应类方法**（如MyVLM、Yo'LLaVA）通过微调将用户知识编码进模型参数。每新增用户概念都需重新训练，扩展性差，且参数一旦固化便无法追踪偏好演变。

**增强类方法**通过外部数据库检索用户记忆，但依赖手动预定义数据库，缺乏主动管理与更新机制。现有通用记忆架构（如A-Mem、MemoryOS）主要面向纯文本，无法处理真正的多模态输入。

**对齐类方法**（如DPO、PAS）将优化目标从通用标准重定向为用户特定标准，但依赖每用户独立训练，扩展性受限，且无法动态更新用户画像。

核心瓶颈在于双重矛盾：一是**动态性矛盾**——用户偏好、行为、人格特质随时间演变，而现有方法假设画像静态；二是**效率矛盾**——长上下文（128k token）下直接输入全部历史对话导致平均43530 token/请求，计算开销极高，而简单RAG在短上下文下甚至会损害性能（偏好理解任务下降9.33%）。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e8a1fc86-0c9a-4f2e-8337-0c2b81deea35/figures/Figure_1.png)
*Figure 1: Figure 1. Illustration of PersonaVLM’s three core capabilities for long-term personalization. PersonaVLM proactively remembers userpreference shifts, performs multi-turn reasoning with retrieval, and*



本文提出将个性化能力从模型内部解耦至外部结构化记忆系统，通过异步记忆更新与人格持续推断，在不修改模型权重的前提下实现跨数百轮交互的动态个性化。

## 核心创新

**核心洞察**：将个性化能力从模型参数中外挂为结构化记忆+人格画像系统，因为用户偏好的动态演变本质上是时间序列上的记忆积累与人格漂移，从而使不修改模型权重、仅通过记忆检索与人格注入即可实现长期个性化成为可能。

| 维度 | Baseline (Qwen2.5-VL-7B 128k-Full/RAG) | 本文 (PersonaVLM) |
|:---|:---|:---|
| 个性化存储位置 | 模型参数 / 原始对话上下文 | 外部结构化记忆数据库（四类记忆） |
| 用户画像维度 | 无显式画像，依赖上下文隐式推断 | Big Five人格模型（OCEAN五维）持续更新 |
| 记忆更新机制 | 静态，无更新 / 手动数据库 | 每轮交互后异步自动提取与更新 |
| 推理效率 | 43530 token/请求（Full）/ top-5简单RAG | 2170 token/请求（检索精准记忆片段） |
| 多模态支持 | 文本记忆为主 | 真正多模态记忆（图像+文本事件） |

关键差异在于：baseline将个性化视为"一次性知识注入"或"上下文压缩问题"，本文将其重构为"动态记忆管理+人格演化"的系统工程问题。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e8a1fc86-0c9a-4f2e-8337-0c2b81deea35/figures/Figure_2.png)
*Figure 2: Figure 2. Overview of the PersonaVLM Framework. It leverages a personalized memory architecture and operates in two collaborativestages to achieve long-term personalization. In the Response Stage (blu*



PersonaVLM以Qwen2.5-VL-7B为骨干，外挂一套完整的个性化记忆与人格推理系统，运行两个协作阶段：

**Memory Stage（记忆阶段）**：每轮交互完成后异步执行，不阻塞用户响应。
- **输入**：当前轮次的用户查询、AI回复、多模态内容（图像+文本）
- **模块：记忆提取器** → 从交互内容中抽取出结构化信息，分类写入四类记忆
- **模块：人格推断器（PEM）** → 基于Big Five模型更新用户人格画像
- **输出**：更新后的个性化记忆数据库 + 演化后的人格向量

**Response Stage（回复阶段）**：处理新查询时同步执行。
- **输入**：用户新查询 + 当前对话上下文
- **模块：记忆检索器** → 从数据库中检索与查询相关的记忆条目（非简单RAG的top-k，而是语义关联匹配）
- **模块：记忆-上下文整合器** → 将检索记忆与当前上下文融合，构建精简的个性化提示
- **模块：人格注入生成器** → 将人格特征注入生成过程，调整输出风格
- **输出**：符合用户个性与历史偏好的多模态回复

四类结构化记忆的分层职责：
- **Core Memory**：用户基本信息（姓名、年龄、职业等静态属性）
- **Episodic Memory**：具体交互事件（"上周三用户提到喜欢莫奈画作"）
- **Semantic Memory**：抽象知识与偏好（"用户偏好印象派风格、厌恶写实主义"）
- **Procedural Memory**：行为习惯与操作模式（"用户习惯先浏览缩略图再放大查看"）

```
用户交互 → [Memory Stage] → 记忆数据库更新 + 人格画像演化
              ↓（异步，每轮触发）
新查询 → [Response Stage] → 记忆检索 → 整合 → 人格注入生成 → 个性化回复
              ↑（同步，按需触发）
```

## 核心模块与公式推导

### 模块 1: 记忆检索与上下文压缩（对应框架图 Response Stage 左侧）

**直觉**：长上下文直接输入效率极低，但简单RAG会丢失个性化关联，需设计精准的记忆-查询匹配机制。

**Baseline 公式** (128k-RAG): 给定查询 $q$ 和历史对话 $H = \{h_1, h_2, ..., h_n\}$，标准RAG检索top-$k$相关片段：
$$\text{Context}_{\text{RAG}} = \text{TopK}_{\text{cos}}\left(\{e(h_i)\}_{i=1}^n, e(q), k=5\right)$$
其中 $e(\cdot)$ 为预训练编码器，输出上下文仅包含5个最相似历史片段。

**变化点**：top-5简单RAG在个性化任务上性能下降9.33%，因为用户偏好理解需要跨片段的语义关联，而非单片段相似度。本文改为基于四类记忆结构的语义关联检索。

**本文公式**：
$$\text{Step 1}: \quad M_{\text{relevant}} = \{m \in \mathcal{M}_{\text{core}} \cup \mathcal{M}_{\text{episodic}} \cup \mathcal{M}_{\text{semantic}} \cup \mathcal{M}_{\text{procedural}} \text{mid} \text{Rel}(q, m) > \tau\}$$
加入了跨记忆类型的关联评分 $\text{Rel}(q, m)$，综合查询与记忆的语义相似度、时间衰减因子、人格相关性权重。

$$\text{Step 2}: \quad \text{Context}_{\text{personalized}} = \text{Concat}\left(\text{Sort}_{\text{priority}}\left(M_{\text{relevant}}\right)\right), \quad |\text{Context}| \leq 2170 \text{ tokens}$$
重归一化以保证token预算严格受限，优先级排序融合记忆类型重要性（Semantic > Episodic > Procedural > Core）。

**最终**：
$$\text{Response} = \text{LLM}\left(q, \text{Context}_{\text{personalized}}, \phi_{\text{personality}}\right)$$

**对应消融**：

---

### 模块 2: 人格演化模块 PEM（对应框架图 Memory Stage 右侧）

**直觉**：用户回复风格随人格特质变化，需将心理学Big Five模型量化为可注入生成的连续向量。

**Baseline 公式** (标准提示工程): 无显式人格建模，依赖静态system prompt：
$$\text{Style}_{\text{base}} = \text{FixedPrompt}(\text{"You are a helpful assistant"})$$

**变化点**：静态提示无法适应用户个性，且不同用户需要不同风格。本文引入可演化的人格向量 $\phi \in \mathbb{R}^5$（对应OCEAN五维）。

**本文公式（推导）**：
$$\text{Step 1}: \quad \phi^{(t)} = \text{PEM}_{\text{infer}}\left(\phi^{(t-1)}, \{(q_i, r_i)\}_{i=t-w}^{t}\right) \quad \text{基于最近} w \text{轮交互推断人格更新}$$
加入了时间窗口 $w$ 内的对话历史作为观测，解决人格漂移追踪问题。

$$\text{Step 2}: \quad \phi^{(t)}_{\text{norm}} = \sigma\left(\phi^{(t)}\right) \in [0,1]^5 \quad \text{Sigmoid归一化到标准人格量表范围}$$
重归一化以保证与心理学量表可比，便于跨用户迁移。

$$\text{Step 3}: \quad P(r|q, \phi) = P_{\text{base}}(r|q) \cdot \exp\left(\lambda \cdot \text{Align}(r, \phi)\right) / Z$$
人格注入通过调整生成概率分布实现，$\text{Align}(r, \phi)$ 度量回复 $r$ 与目标人格 $\phi$ 的风格一致性，$\lambda$ 为注入强度。

**最终**：
$$\text{PEM Loss} = -\mathbb{E}_{(q,r^*)}\left[\log P(r^*|q, \phi^{(t)}_{\text{norm}})\right] + \beta \cdot \text{KL}\left(\phi^{(t)} \| \phi^{(t-1)}\right)$$
第二项约束人格演化平滑性，避免突变。

**对应消融**：P-SOUPS Style维度达44.00，论文未报告移除PEM的精确Δ值，但指出RL阶段相比SFT在128k-RAG设置下额外提升约3.87pp，PEM为Response Alignment核心组件。

---

### 模块 3: 两阶段训练目标（对应框架图整体训练流程）

**直觉**：先建立基础记忆操作能力，再通过强化学习优化长期个性化策略。

**Baseline 公式** (直接SFT或RL): 
$$\mathcal{L}_{\text{SFT}} = -\sum_{(x,y) \in \mathcal{D}} \log P_\theta(y|x), \quad \mathcal{L}_{\text{RL}} = \mathbb{E}[r(x,y)] - \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**变化点**：单一阶段训练难以同时掌握记忆格式生成与长期策略优化。本文设计课程式两阶段，并引入Persona-MME作为评估信号。

**本文公式**：
$$\text{Stage 1 (SFT)}: \quad \mathcal{L}_{\text{SFT}}^{\text{persona}} = -\sum_{(x,y, m^*, \phi^*)} \left[\log P_\theta(y|x) + \gamma_1 \log P_\theta(m^*|x,y) + \gamma_2 \log P_\theta(\phi^*|x,y)\right]$$
加入了记忆生成监督 $m^*$ 和人格标签监督 $\phi^*$，强制模型学习结构化输出格式。

$$\text{Stage 2 (RL)}: \quad \mathcal{L}_{\text{RL}}^{\text{persona}} = \mathbb{E}_{\pi_\theta}\left[R_{\text{Persona-MME}}(y) + \alpha \cdot R_{\text{efficiency}}(|\text{Context}|)\right] - \beta \text{KL}(\pi_\theta \| \pi_{\text{SFT}})$$
奖励函数 $R_{\text{Persona-MME}}$ 基于七个维度（记忆、意图、偏好、行为、关系、成长等）的自动评估，$R_{\text{efficiency}}$ 激励上下文压缩。

**最终**：两阶段合计在Persona-MME上平均提升5.35%。

**对应消融**：Table 1显示RL（71.05%）相比SFT（约67.18%，由"RL额外提升3.87pp"反推）在128k-RAG设置下的增益。

## 实验与分析


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e8a1fc86-0c9a-4f2e-8337-0c2b81deea35/figures/Figure_5.png)
*Figure 5: Figure 5. Qualitative comparison on open-ended generation, evalu-ated by Gemini-2.5-Pro. The evaluation assesses both the factualaccuracy and the personality alignment of the responses.*



**主结果（Persona-MME基准）**：

| Method | 设置 | Persona-MME (%) | 备注 |
|:---|:---|:---|:---|
| Qwen2.5-VL-7B | 128k-Full | 54.48 | 直接输入全部历史 |
| Qwen2.5-VL-7B | 128k-RAG (top-5) | 59.01 | 简单RAG基线 |
| PersonaVLM | SFT | ~67.18 | （推算值） |
| **PersonaVLM** | **RL (本文)** | **71.05** | **相对128k-RAG提升20.4%** |
| InternVL3-8B | 32k | 52.97 | 更强基线模型，短上下文 |
| InternVL3-38B | 32k | 57.93 | 更大模型，短上下文 |

核心数据解读：71.05% vs 59.01%的12.04pp绝对提升支撑了"结构化记忆优于简单RAG"的核心 claim。但需注意：若与128k-Full（54.48%）计算则相对提升30.4%，摘要声称的22.4%口径不透明。

**效率验证**：
- Token消耗：43530 → 2170（减少95.0%）
- 推理加速：4.8倍
- 样本量：仅100个样本，内存异步更新时延被排除


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e8a1fc86-0c9a-4f2e-8337-0c2b81deea35/figures/Figure_7.png)
*Figure 7: Figure 7. Data composition for the training of PersonaVLM*



**消融与细分**：
- RL相比SFT在128k-RAG下额外提升3.87pp，验证强化学习优化长期策略的有效性
- 开放式生成对比GPT-4o胜率79.0%，但评判模型为Gemini-2.5-Pro，存在跨厂商偏好偏差风险
- P-SOUPS Style维度44.00，为人格对齐的量化证据

**公平性检查**：
- **Baseline强度**：RAG仅使用top-5简单实现，未与更强RAG方法（如多跳检索、重排序）对比；未与A-Mem多模态变体直接对比
- **跨基准泛化**：PERSONAMEM基准上PersonaVLM仅47.28%，低于InternVL3-38B（57.93%）和InternVL3-8B（52.97%），说明方法受特定基准设计影响
- **计算成本**：两阶段训练（SFT+RL）需额外数据合成与RL训练资源，论文未报告总训练开销
- **失败案例**：不支持视频/音频中的人物识别与追踪；记忆系统为时间线性结构，不支持跨时间段情节关联合并；性能受底层Qwen2.5-VL-7B能力上限约束

## 方法谱系与知识库定位

**方法家族**：记忆增强型多模态大语言模型（Memory-Augmented MLLM）

**父方法**：Qwen2.5-VL-7B（骨干模型）+ 经典记忆架构理论（Episodic/Semantic/Procedural Memory区分源于认知心理学Tulving模型）

**改变的插槽**：
- **架构**：外挂记忆数据库 + PEM人格模块，不改基座权重
- **目标函数**：SFT增加记忆/人格生成监督；RL引入Persona-MME多维奖励
- **训练配方**：两阶段课程（SFT→RL），配合2000+案例的Persona-MME评估
- **数据策划**：自动化的用户画像构建与多轮对话合成流水线（Figure 3）
- **推理**：异步记忆更新 + 同步记忆检索-人格注入生成

**直接基线对比**：
| 方法 | 与本方法差异 |
|:---|:---|
| MyVLM / Yo'LLaVA | 参数微调适应，无外部记忆，无法动态更新 |
| A-Mem / MemoryOS | 纯文本记忆，无多模态支持，无主动管理 |
| DPO / PAS | 单用户独立训练，无记忆架构，扩展性差 |
| 128k-RAG (top-5) | 无结构化记忆类型区分，无人格建模，检索策略简陋 |

**后续方向**：
1. **跨模态扩展**：将记忆架构从图像-文本扩展到视频-音频的人物追踪与事件记忆
2. **图结构记忆**：将线性时间记忆升级为时序图网络，支持跨时间段情节关联与合并
3. **可解释人格**：将Big Five推断过程可视化，提供"你为什么这样回复"的用户透明解释

**标签**：
- **modality**: image+text
- **paradigm**: memory-augmented generation
- **scenario**: long-term personalized dialogue
- **mechanism**: structured episodic/semantic/procedural memory + personality evolving module
- **constraint**: async memory update latency excluded, linear temporal structure only

