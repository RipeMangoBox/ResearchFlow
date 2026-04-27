---
title: 'RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography'
type: paper
paper_level: B
venue: ERS
year: 2026
paper_link: https://arxiv.org/abs/2604.15231
aliases:
- 胸部CT工具调用智能体RadAgent
- RadAgent
- 核心变化是将端到端单次推理的3D VLM替换为RL训练的工具调用智能体
method: RadAgent
---

# RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography

[Paper](https://arxiv.org/abs/2604.15231)

**Topics**: [[T__Agent]], [[T__Medical_Imaging]], [[T__Reasoning]] | **Method**: [[M__RadAgent]]

> [!tip] 核心洞察
> 核心变化是将端到端单次推理的3D VLM替换为RL训练的工具调用智能体，通过ReAct迭代循环和持久化草稿板将报告生成分解为可追溯的多步骤决策序列。有效性来源于两点：一是迭代精炼机制允许智能体在初始草稿基础上逐步纠错和补充，而非依赖单次前向推理；二是RL训练使智能体自动发现最优工具调用策略，将专用诊断工具的精确性与通用LLM的灵活性结合。每个决策锚定于具体工具输出，从而提升鲁棒性和可解释性。

| 中文题名 | 胸部CT工具调用智能体RadAgent |
| 英文题名 | RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography |
| 会议/期刊 | The European Respiratory Society eBooks (2026) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15231) · [DOI](https://doi.org/10.1183/9781849841313.007319) |
| 主要任务 | 胸部CT报告生成、可解释医学影像分析、工具增强迭代推理 |
| 主要 baseline | CT-Chat（3D VLM，兼作内部工具） |

> [!abstract] 因为「现有3D VLM将CT解读压缩为单次黑盒推理，临床医生无法验证中间过程」，作者在「CT-Chat」基础上改了「以Qwen3-14B为策略模型、GRPO强化学习训练工具调用、ReAct迭代循环配合持久化草稿板」，在「CT-RATE测试集」上取得「宏平均F1提升6.0个百分点（相对36.4%），鲁棒性提升24.7个百分点（相对41.9%）」

- **宏平均F1**: 22.5% → 28.5%（+6.0pp，相对+36.4%），CT-RATE测试集
- **微平均F1**: 27.6% → 33.0%（+5.4pp，相对+19.6%），CT-RATE测试集
- **忠实度（Faithfulness）**: 37.0% vs CT-Chat 0.0%，CT-RATE测试集
- **鲁棒性（Robustness）**: 提升24.7个百分点（相对+41.9%），对抗注入提示干扰场景

## 背景与动机

胸部CT包含数百张横断面切片，放射科医生解读时需逐层浏览、调整窗宽窗位、比对历史影像、综合多器官发现，最终形成结构化报告——这是一个天然的多步骤、迭代性认知过程。然而，现有3D视觉语言模型（如CT-Chat）将这一复杂过程压缩为单次端到端前向推理：输入3D CT体积，直接输出完整报告。临床医生面对最终文本，无法追溯"这个结节结论来自哪张切片""该弥漫性病变判断依据何种窗宽设置"，系统成为不可验证的黑盒。

现有方法的处理方式及其局限：

**CT-Chat** 作为代表性3D VLM，通过统一编码器-解码器架构直接映射CT体积到报告文本，在标准基准上取得领先性能，但完全隐藏中间推理，不支持人机协作验证。

**CT-Agent / CTPA-Agent** 等早期agentic系统尝试引入多步骤推理，但依赖手工设计的固定工作流和预定义查询池，工具调用策略由人类专家硬编码，缺乏从数据中自动学习最优策略的能力，且同样未提供可追溯的推理轨迹。

核心缺口在于：**诊断准确性与推理可解释性、鲁棒性之间存在结构性张力**。单次前向推理虽简洁，但无法纠错——初始草稿若遗漏关键发现，系统无从知晓；面对误导性上下文（如错误提示"患者有肺癌史"），缺乏验证机制的系统易被带偏。临床场景要求AI不仅"答对"，更要"展示如何答对"并"能抵抗干扰"。

RadAgent的核心动机正是将CT报告生成重构为**可解释的、工具增强的迭代推理过程**：让智能体像人类放射科医生一样，调用专用工具逐步收集证据，在持久化工作区中累积和修正发现，最终生成可追溯来源的报告。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4f91d077-d80f-44f3-b5dd-8a624fd6e750/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of RadAgent.*



## 核心创新

核心洞察：**将端到端单次推理分解为RL训练的工具调用序列**，因为CT解读本身是工具依赖的迭代认知过程（调窗、切片、分类、验证），从而使智能体能够自主发现最优诊断策略、逐步纠错并生成可追溯证据链成为可能。

| 维度 | Baseline（CT-Chat） | 本文（RadAgent） |
|:---|:---|:---|
| 推理模式 | 单次端到端前向传播 | ReAct迭代循环，多轮工具调用 |
| 策略来源 | 无显式策略，隐式编码于模型参数 | GRPO强化学习自动学习工具调用策略 |
| 中间状态 | 无，直接输出最终报告 | 持久化草稿板，记录所有工具输出与修正历史 |
| 可验证性 | 黑盒，无法追溯结论来源 | 每句报告可锚定至具体工具调用结果 |
| 纠错能力 | 无，一次生成定终稿 | 迭代精炼，依据检查清单逐项验证补充 |
| 工具生态 | 单一模型，无外部工具 | 10种专用工具通过MCP协议标准化协同 |

## 整体框架



RadAgent的整体数据流遵循"初始生成→逐项验证→迭代修正→综合输出"的临床工作流，核心组件如下：

**输入层**：3D胸部CT体积（DICOM格式）+ 可选临床提示（如病史、检查目的）。

**策略模型（Policy Model）**：以Qwen3-14B指令微调模型为核心LLM，经GRPO算法强化学习训练，负责决策"当前步骤调用哪个工具、提出何种问题"。输入为当前草稿板状态+任务目标，输出为结构化工具调用指令（工具名+参数）。

**工具箱（Toolbox）**：通过MCP（Model Context Protocol）协议跨两节点八块GPU部署，含10种专用工具：
- `report_generation()`: 基于CT-Chat生成初始报告草稿
- `3d_ct_vqa()`: 基于CT-Chat回答3D CT整体问题
- `2d_slice_vqa()`: 基于Gemma-3-27B-IT（temperature=0.0）回答特定2D切片问题
- `disease_classification()`: 基于CT-CLIP覆盖18种胸部病理分类
- `organ_segmentation()`: 器官分割工具
- `ct_windowing()`: 调整CT窗宽窗位参数
- `2d_slice_extraction()`: 提取指定层面2D切片
- 其余3种工具用于辅助操作（具体功能原文未详述）

**持久化草稿板（Scratchpad）**：维护整个推理过程中的中间观察记录，包括每次工具调用的输入参数、原始输出、时间戳。草稿板内容随迭代逐步累积，支持读取-修改-写入操作。

**结构化检查清单（Checklist）**：9个诊断类别（肺实质、结节、弥漫性病变、气道、胸膜/胸壁、纵隔、心脏/大血管、膈肌/下腹部、其他），指导智能体系统性地验证和补充发现，避免遗漏。

**输出层**：综合草稿板中累积的证据，生成最终结构化报告，每个发现标注来源工具调用。

```
CT体积 ──→ [report_generation] ──→ 初始草稿
              ↓
        [策略模型: GRPO训练]
              ↓
    ┌─→ [检查清单项1] ──→ [工具选择] ──→ [草稿板更新] ──┐
    │      ↑_____________________________________________↓  (迭代N轮)
    └─→ [检查清单项9] ──→ [工具选择] ──→ [草稿板更新] ──┘
                                    ↓
                              [综合输出] ──→ 可追溯报告
```


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4f91d077-d80f-44f3-b5dd-8a624fd6e750/figures/Figure_4.png)
*Figure 4 (architecture): The RadAgent toolbox.*

 展示了工具箱的详细架构设计，包括MCP通信层与GPU分布策略。

## 核心模块与公式推导

### 模块 1: GRPO强化学习训练策略模型（对应框架图 策略模型层）

**直觉**: 放射科医生的诊断策略难以手工编码，但可通过群体相对奖励比较自动学习——同一CT用多种工具调用序列尝试，优胜策略获得强化。

**Baseline 公式**（标准PPO/RLHF）: 
$$L_{\text{PPO}} = \mathbb{E}_{(x,y)\sim \pi_{\theta_{\text{old}}}} \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
符号: $r_t(\theta) = \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})}$ 为策略比率, $\hat{A}_t$ 为优势估计, $\epsilon$ 为裁剪超参。

**变化点**: PPO需要额外的critic网络估计价值函数，参数量大且训练不稳定；对于工具调用这种离散决策空间，优势估计的方差高。GRPO（Group Relative Policy Optimization）摒弃critic网络，改为**组内相对奖励归一化**，降低内存开销并适配工具调用序列的稀疏奖励特性。

**本文公式（推导）**:
$$\text{Step 1}: \text{对每问题 } q, \text{从旧策略采样 } G \text{ 组输出 } \{o_1, o_2, ..., o_G\}$$
$$\text{Step 2}: \text{计算组内奖励 } \{r_1, r_2, ..., r_G\}, \quad \bar{r} = \frac{1}{G}\sum_{i=1}^G r_i, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G(r_i-\bar{r})^2}$$
$$\text{Step 3}: \text{优势归一化 } \hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r} \quad \text{（消除绝对奖励尺度，突出组内相对优劣）}$$
$$\text{最终}: L_{\text{GRPO}} = \mathbb{E}_{q\sim P(Q), \{o_i\}\sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G}\sum_{i=1}^G \left( \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]$$

**对应消融**: 原文未提供GRPO vs PPO的直接消融，但指出GRPO的组内归一化机制使策略模型能在8×GPU约束下稳定训练。KL散度项 $\beta \mathbb{D}_{\text{KL}}$ 约束策略不偏离参考模型过远，防止工具调用策略崩溃。

---

### 模块 2: ReAct迭代循环与草稿板更新（对应框架图 核心循环层）

**直觉**: 人类放射科医生不会一次性写完报告——先生成草稿，再逐项核对、补充、修正。草稿板是这一认知过程的外部化记忆。

**Baseline 公式**（标准ReAct，无持久化状态）: 
$$a_t = \pi_{\theta}(\text{thought}_t, \text{action}_t | x, a_{<t}, o_{<t}), \quad o_t = \text{Tool}(a_t)$$
符号: $a_t$ 为第t步动作（工具调用）, $o_t$ 为观察结果, 历史仅通过上下文窗口隐式传递。

**变化点**: 标准ReAct的"思考-行动-观察"三元组在上下文窗口中线性累积，长序列时早期证据被压缩或遗忘；且缺乏结构化组织，难以追溯特定发现的来源。RadAgent引入**显式持久化草稿板** $S_t$，作为可读写的外部记忆，支持对历史发现的主动修正而非仅追加。

**本文公式（推导）**:
$$\text{Step 1}: \text{初始化 } S_0 = \text{report_generation}(\text{CT}), \text{ 含初始发现集合 } \{f_1^{(0)}, ..., f_k^{(0)}\}$$
$$\text{Step 2}: \text{对检查清单每项 } c \in \{c_1, ..., c_9\}:$$
$$\quad \text{thought}_t = \text{LLM}("\text{检查 } c: \text{当前草稿 } S_t \text{ 是否充分?}")$$
$$\quad \text{action}_t = \text{select_tool}(\text{toolbox}, \text{thought}_t) \in \{t_1, ..., t_{10}\}$$
$$\quad o_t = \text{execute}(\text{action}_t, \text{CT})$$
$$\quad S_{t+1} = \text{update}(S_t, o_t, \text{thought}_t) \quad \text{（非仅追加，支持覆盖/修正/补充）}$$
$$\text{最终}: \text{Report} = \text{render}(S_T), \quad \text{faithfulness}(f) = \mathbb{1}[\exists t: f \text{ 直接源自 } o_t]$$

**对应消融**: 图3（Faithfulness and robustness）显示忠实度37.0% vs CT-Chat 0.0%，验证了草稿板机制使可追溯输出成为可能；但37%绝对值表明63%的发现仍无法精确锚定，存在归因模糊问题。

---

### 模块 3: 工具调用决策与MCP协同（对应框架图 工具箱层）

**直觉**: 不同诊断子任务需要不同"专家"——3D整体感知用CT-Chat，2D精细分析用Gemma-3-27B，病理分类用CT-CLIP。策略模型需学习"何时调用谁"。

**Baseline 公式**（固定工作流，如CT-Agent）: 
$$\text{action}_t = \text{Workflow}_{\text{handcrafted}}(t) \quad \text{（第t步动作为预定义，与输入无关）}$$

**变化点**: 固定工作流无法适应病例特异性需求——有的CT需重点分析结节，有的需评估弥漫性病变。RL训练使策略模型能根据当前草稿板状态动态选择工具，实现**输入自适应的工具编排**。

**本文公式（推导）**:
$$\text{Step 1}: \text{状态编码 } h_t = \text{Encoder}(S_t, c_{\text{current}}, q_{\text{history}})$$
$$\text{Step 2}: \text{工具选择分布 } p(t_j|h_t) = \text{softmax}(W_j \cdot h_t + b_j), \quad j \in \{1,...,10\}$$
$$\text{Step 3}: \text{MCP标准化调用 } \text{request}_t = \text{MCP}_{\text{format}}(t_j, \text{params}_t) \text{xrightarrow}{\text{跨节点}} \text{GPU}_{\text{assigned}}$$
$$\text{最终}: \text{响应 } o_t = \text{MCP}_{\text{parse}}(\text{response}_t) \text{ 回填至 } S_{t+1}$$

**关键设计**: temperature=0.0 的2D VQA（Gemma-3-27B-IT）确保切片级分析确定性；CT-CLIP的18类分类输出为策略模型提供结构化决策信号。MCP协议解耦工具实现与策略模型，支持工具热插拔。

**对应消融**: 原文未提供工具子集的正式消融，但指出工具集变化需重新运行RL流程——暗示工具组合对策略学习存在强耦合。

## 实验与分析

**主结果：CT-RATE测试集报告生成质量**

| Method | Macro-F1 | Micro-F1 | Faithfulness | Robustness |
|:---|:---|:---|:---|:---|
| CT-Chat | 22.5% | 27.6% | 0.0% | 基准值 |
| **RadAgent** | **28.5%** | **33.0%** | **37.0%** | **+24.7pp** |
| Δ | +6.0pp (+36.4%) | +5.4pp (+19.6%) | +37.0pp (新指标) | +24.7pp (+41.9%) |

*置信区间通过bootstrapping获得，统计显著性原文标注*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4f91d077-d80f-44f3-b5dd-8a624fd6e750/figures/Figure_2.png)
*Figure 2 (result): Report generation quality comparison between the trained RadAgent system and the CT-Chat*



**核心发现分析**：

- **诊断准确性提升**：宏平均F1的相对增益（36.4%）高于微平均（19.6%），说明RadAgent对**稀有病理的识别改善更显著**——单次推理的CT-Chat易被常见模式主导，而迭代验证机制能挖掘易被忽略的发现。

- **忠实度指标的结构性不对称**：CT-Chat得0.0%因其架构无法产生可追溯输出，非公平能力比较；37.0%的绝对值仍较低，论文自述"clearly leaves substantial room for future work"。该指标更宜解读为"机制可行性验证"而非"成熟度证明"。

- **鲁棒性提升41.9%**：在注入误导性提示干扰场景下，RadAgent通过工具验证抵抗错误先验。但对抗条件的具体构造方式（注入何种提示、干扰强度）在摘录中未详述，置信度略低于主指标。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4f91d077-d80f-44f3-b5dd-8a624fd6e750/figures/Figure_3.png)
*Figure 3 (result): Faithfulness and robustness of RadAgent under injected prompt hints.*



**外部验证**：RadChestCT数据集上同样优于CT-Chat，但评估仍使用相同18类病理标签体系，域外泛化存在局限——未验证标签体系外的表现。

**公平性检查**：
- **基线强度**：仅与CT-Chat单一基线定量对比，未与CT-Agent、CTPA-Agent等同类agentic系统比较，无法确定agentic范式内的相对优劣。
- **计算成本**：两节点八卡GPU部署，资源需求显著高于单模型推理。
- **失败模式**：(1) 37%忠实度意味着多数发现仍无法精确归因；(2) 工具集固定，新增工具需重训RL；(3) 2D VLM选择（Gemma-3-27B）基于探索性实验，未正式消融验证。

## 方法谱系与知识库定位

**方法家族**：工具增强型LLM智能体（Tool-Augmented LLM Agent）→ 医学影像专用分支

**父方法**：ReAct推理框架（Yao et al., 2022）+ GRPO强化学习（DeepSeek-R1技术路线）

**改变的插槽**：
| 插槽 | 父方法/常规做法 | 本文改变 |
|:---|:---|:---|
| 架构 | 单LLM端到端 | 策略模型+10种专用工具通过MCP协同 |
| 目标函数 | SFT/标准RLHF | GRPO组内相对奖励优化，免critic网络 |
| 训练配方 | 固定工作流/手工设计 | 从数据中自动学习工具调用策略 |
| 数据策划 | 无显式检查清单 | 9类结构化诊断检查清单驱动迭代验证 |
| 推理机制 | 单次前向/线性ReAct | 持久化草稿板支持读写修正的循环ReAct |

**直接基线与差异**：
- **CT-Chat**：RadAgent以其为内部工具（report_generation + 3D VQA），差异在于增加迭代精炼层与工具编排策略
- **CT-Agent / CTPA-Agent**：同为agentic医学影像系统，RadAgent差异在于RL自动学习策略替代手工工作流，以及持久化草稿板机制

**后续方向**：
1. **忠实度提升**：当前37%归因率过低，需探索更精细的证据锚定机制（如句子级-工具输出对齐）
2. **工具动态扩展**：解决工具集变化需重训RL的问题，朝向持续学习或元工具学习
3. **资源效率优化**：降低两节点八卡的部署门槛，探索单节点或边缘适配方案

**知识库标签**：
- **模态（modality）**: 3D医学影像 / 胸部CT
- **范式（paradigm）**: 工具增强智能体 / 强化学习 / ReAct迭代推理
- **场景（scenario）**: 放射科报告生成 / 临床决策支持 / 可解释AI
- **机制（mechanism）**: GRPO策略优化 / MCP工具协议 / 持久化记忆
- **约束（constraint）**: 多GPU部署 / 固定工具集 / 18类病理标签体系

