---
title: 'SMRS: advocating a unified reporting standard for surrogate models in the artificial intelligence era.'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 代理模型统一报告标准SMRS
- SMRS (Surrogate
- SMRS (Surrogate Model Reporting Standard)
- A unified
acceptance: Poster
cited_by: 1
method: SMRS (Surrogate Model Reporting Standard)
modalities:
- Text
paradigm: N/A
---

# SMRS: advocating a unified reporting standard for surrogate models in the artificial intelligence era.

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__SMRS]]

> [!tip] 核心洞察
> A unified, modular, and model-agnostic reporting standard (SMRS) is urgently needed to systematically capture essential design and evaluation choices in surrogate modeling while remaining implementation-agnostic.

| 中文题名 | 代理模型统一报告标准SMRS |
| 英文题名 | SMRS: advocating a unified reporting standard for surrogate models in the artificial intelligence era |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.06753) · [DOI](https://doi.org/10.48550/arxiv.2502.06753) |
| 主要任务 | Benchmark / Evaluation（代理模型评估标准化） |
| 主要 baseline | Model Cards; Ad-hoc surrogate modeling reporting practices |

> [!abstract] 因为「代理模型（surrogate model）文献在数据收集、采样决策、不确定性量化等关键维度上报告方式高度不一致，导致可复现性与跨领域比较困难」，作者在「Model Cards」基础上改了「针对代理建模领域特性的六维结构化报告清单与需求评估框架」，在「17项近期研究的文献回顾」中验证「当前文献存在显著报告异质性，SMRS提供了首个统一的领域专用标准」。

- 对17项近期研究的回顾显示：多数论文描述模型架构与预测性能，但很少一致报告数据来源、采样决策、不确定性量化及局限性处理
- SMRS覆盖从经典统计代理模型到现代AI生成式方法的全谱系，具有模型无关性
- 提出建模前的初步需求评估框架，明确代理模型适用的决策条件（计算成本、输入输出平滑性、替代方案可行性）

## 背景与动机

代理模型（surrogate model）是用计算成本较低的近似替代昂贵仿真或观测过程的核心技术，广泛应用于工程设计、气候模拟、流行病学、数字孪生等领域。例如，一个海洋冰盖模拟可能需要数天单次运行，研究者训练神经网络来毫秒级预测冰层厚度分布——但不同团队如何报告这个替代模型的训练数据、采样策略、不确定性边界，目前完全没有统一规范。

现有做法如何处理这一问题？**Model Cards** [34] 提供了机器学习模型的通用文档框架，报告预期用途、性能特征与局限性，但其面向的是观测数据的预测模型，而非针对已知昂贵过程的仿真替代。**Ad-hoc 文献实践**是当前代理建模领域的主流：各团队自行决定报告内容，多数论文详细描述模型架构与预测性能（如RMSE、R²），但对数据来源、生成方式、采样设计、训练细节、不确定性量化及模型局限性的报告参差不齐。

这些做法为何不足？核心局限在于**领域错配与结构性缺失**：Model Cards未涵盖代理建模特有的关键决策——如仿真数据与实验数据的区分、空间/时间采样设计、替代模型与原始仿真器的验证协议、以及科学可靠性所必需的不确定性量化。作者对17项近期研究的回顾发现，**少数论文一致报告了数据收集方式、采样决策依据、不确定性是否被量化、以及局限性如何处理**。这种碎片化严重限制了跨研究比较、方法复现、以及科学结论的可靠性，尤其在AI驱动的新型代理模型（如PINNs、GNN surrogates、生成式代理）快速涌现的当下，标准化需求更为紧迫。

本文提出SMRS，首个专为代理建模设计的统一、模块化、模型无关的报告标准，通过六维结构化清单系统捕获建模全流程的关键决策。

## 核心创新

核心洞察：代理建模的本质是"用近似替代已知昂贵过程"而非"从观测中学习预测"，因此其报告标准必须覆盖**数据生成机制、采样设计、与原始仿真器的验证关系**等仿真特有维度，从而使科学可靠性审查和跨领域方法比较成为可能。

| 维度 | Baseline (Model Cards / Ad-hoc) | 本文 (SMRS) |
|:---|:---|:---|
| 核心定位 | 通用ML模型文档；或完全无标准 | 代理建模专用结构化报告框架 |
| 数据报告 | 数据集来源与统计特征 | **数据生成模式**（仿真/实验/混合）、格式、**采样设计**及其假设 |
| 模型报告 | 架构、性能指标、预期用途 | **模型结构+显式假设**、确定性/概率性选择、**不确定性量化方法** |
| 训练报告 | 部分提及超参数 | **完整训练方法学**（损失函数、优化器、收敛标准、计算资源） |
| 评估报告 | 标准测试集性能 | **与原始仿真器的验证协议**、**评估标准的选择依据**、外推/插值区分 |
| 前置决策 | 无 | **初步需求评估框架**：判断代理模型是否必要（vs. 代码优化、直接数值方法） |
| 适用范围 | 单一模型实例 | 覆盖**经典统计代理**（Kriging、PCE）到**现代AI生成方法**的全谱系 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/48a851fd-cd66-4652-adfd-9c7b384f717d/figures/Figure_1.png)
*Figure 1 (pipeline): An Overview of the proposed Surrogate Model Reporting Standard (SMRS).*



SMRS框架由两大核心组件构成：**初步需求评估框架**（Preliminary Need Assessment Framework）与**六维报告清单**（Six-Dimension Reporting Checklist），贯穿代理模型开发的全生命周期。

数据流与模块功能：

1. **Need Assessment（需求评估）** → 输入：问题规格（计算成本、维度、非线性程度、现有替代方案）；输出：是否采用代理模型的二元决策及文档化依据。这是SMRS区别于所有现有报告标准的独特前置步骤，避免"为代理而代理"的方法误用。

2. **Data Sources and Formats（数据来源与格式）** → 输入：数据生成模式（仿真、实验、或混合）；输出：带明确假设与有效性条件的数据来源文档。强制区分合成数据与观测数据，记录数据格式与预处理决策。

3. **Sampling Design（采样设计）** → 输入：实验设计与采样策略选择；输出：采样决策的显式报告与合理性论证。包括空间/时间采样、自适应采样、多保真采样等设计的文档化。

4. **Model Structure and Assumptions（模型结构与假设）** → 输入：模型架构与推理选择；输出：结构决策与不确定性量化方法的显式文档。强制报告确定性vs概率性选择、平滑性假设、对称性/不变性假设等。

5. **Training Methodology（训练方法学）** → 输入：训练过程细节；输出：可复现的训练配方文档。涵盖损失函数、优化器、超参数、收敛标准、计算资源与随机种子。

6. **Evaluation Criteria（评估标准）** → 输入：评估指标与验证协议；输出：标准化的评估文档以实现跨研究比较。明确区分插值与外推评估、与原始仿真器的验证关系。

7. **Uncertainty Quantification（不确定性量化）** → 输入：不确定性估计方法；输出：不确定性量化方法及其局限性的文档。这是科学可靠性核心，但当前文献最常遗漏的维度。

整体流程可概括为：
```
问题规格 → [Need Assessment: 代理模型是否必要?] 
    → 是 → Data Sources → Sampling Design → Model Structure 
    → Training Methodology → Evaluation Criteria → Uncertainty Quantification
    → 全流程文档化输出（SMRS报告）
```

## 核心模块与公式推导

SMRS作为报告标准框架，不涉及传统意义上的训练损失函数或网络架构公式。其核心"推导"体现在从**通用报告原则**到**领域专用结构化清单**的逻辑展开。以下阐述两个最具方法论深度的模块：

### 模块 1: 初步需求评估框架（对应框架图最左端）

**直觉**：在投入代理模型开发前，先系统判断代理是否真正必要，避免方法滥用。

**Baseline 逻辑**（Model Cards / 常规ML实践）：直接假设模型开发必要，跳过前置可行性分析，进入数据收集与模型训练。

**变化点**：代理建模领域存在多种替代路径（直接优化仿真代码、采用更高效数值算法、使用降阶模型），盲目使用代理可能引入不必要的近似误差。SMRS将需求评估显性化为报告标准的必要组成部分。

**本文决策逻辑**：
$$
\text{代理模型适用性} = f(\underbrace{C_{\text{sim}}}_{\text{单次仿真成本}}, \underbrace{N_{\text{eval}}}_{\text{所需评估次数}}, \underbrace{S_{\text{io}}}_{\text{输入-输出映射平滑性}}, \underbrace{A_{\text{alt}}}_{\text{替代方案可行性}})
$$

具体判定准则（作者归纳）：
- **必要条件**：$C_{\text{sim}} \times N_{\text{eval}} \gg C_{\text{train}} + C_{\text{inference}}$（总仿真成本远大于代理模型训练+推理成本）
- **结构条件**：$S_{\text{io}}$ 足够高（输入-输出关系具有一定平滑性，否则代理近似困难）
- **排除条件**：$A_{\text{alt}}$ 不可行时（代码优化、直接数值方法、降阶模型均不适用或成本更高）

**对应消融**：移除该框架将导致代理模型在不适用的场景中被误用，但本文未提供定量消融（见实验分析部分的局限性讨论）。

### 模块 2: 六维报告清单的结构化展开（对应框架图主体六模块）

**直觉**：将隐性的"好实践"转化为显性的、可检查的、可比较的结构化文档。

**Baseline 公式**（Model Cards 的通用报告结构）：
$$\text{Report}_{\text{MC}} = \{\text{Model Overview}, \text{Intended Use}, \text{Factors}, \text{Metrics}, \text{Evaluation Data}, \text{Training Data}, \text{Ethical Considerations}, \text{Caveats}\}$$

符号：各元素为描述性文本段落，无强制结构或领域特化要求。

**变化点**：Model Cards面向通用ML预测模型，未针对代理建模的**仿真替代本质**进行定制。具体而言：
- 未区分"数据生成模式"（仿真vs实验vs混合）
- 未要求"采样设计"的显式报告（代理模型训练数据的采样直接影响近似质量）
- 未纳入"与原始仿真器的验证协议"（代理的核心科学问题是"近似是否忠实"）
- 未强制"不确定性量化"（科学决策需要误差边界，而非点预测）

**本文公式（结构化展开）**：
$$\text{SMRS Report} = \{\underbrace{D_{\text{data}}}_{\text{Data Sources \& Formats}}, \underbrace{D_{\text{sample}}}_{\text{Sampling Design}}, \underbrace{D_{\text{model}}}_{\text{Model Structure \& Assumptions}}, \underbrace{D_{\text{train}}}_{\text{Training Methodology}}, \underbrace{D_{\text{eval}}}_{\text{Evaluation Criteria}}, \underbrace{D_{\text{uq}}}_{\text{Uncertainty Quantification}}\}$$

其中每个维度 $D_i$ 进一步展开为强制检查项：

$$\text{Step 1: } D_{\text{data}} = \{\text{generation mode}, \text{source provenance}, \text{format specification}, \text{preprocessing}, \text{validity conditions}\} \quad \text{（加入数据生成模式以区分仿真/实验）}$$

$$\text{Step 2: } D_{\text{sample}} = \{\text{design type}, \text{space coverage}, \text{adaptive criteria}, \text{justification}\} \quad \text{（显式采样设计以支撑近似质量评估）}$$

$$\text{Step 3: } D_{\text{uq}} = \{\text{method type}, \text{epistemic/aleatoric decomposition}, \text{calibration}, \text{limitations}\} \quad \text{（重归一化科学可靠性要求）}$$

**最终**：
$$\text{Complete SMRS} = \text{Need Assessment} \rightarrow \text{bigotimes}_{i=1}^{6} D_i \text{ with cross-references and version control}$$

**对应消融**：对17项研究的回顾显示，当前文献在 $D_{\text{sample}}$（采样设计）和 $D_{\text{uq}}$（不确定性量化）上的报告率最低，间接证明移除这些维度会严重削弱报告完整性，但无定量Δ%指标。

## 实验与分析

本文属于**概念性/框架性论文**，不包含传统意义上的训练实验或性能基准测试。其"实验"体现为对现有文献的系统回顾与标准验证。



作者在**17项近期代理建模研究**的回顾中发现：当前文献在六个报告维度上存在显著异质性。多数论文（约80%以上）详细描述了模型架构与预测性能指标（如RMSE、MAE、R²），但**少数一致报告了数据收集方式**（仿真代码版本、实验条件、数据预处理链）、**采样决策的依据**（为何选择特定实验设计）、**不确定性是否被量化及如何量化**（频率法、贝叶斯方法、集成方法等）、以及**模型局限性的明确讨论**（外推能力、失效模式、与仿真器的系统性偏差）。这一发现构成了SMRS必要性的核心证据：若无结构化标准，关键科学信息将持续缺失。

值得注意的是，17项案例研究覆盖了**多元方法谱系**——从经典高斯过程（Kriging）、多项式混沌展开（PCE），到现代深度学习方法（PINNs、GNN surrogates、Bayesian neural networks、生成式代理模型），以及跨领域应用（结构健康监测、海冰模拟、油藏历史拟合、流行病建模）。这种广度支撑了SMRS声称的"模型无关性"，但**未经过形式化验证**。

**公平性审查**：
- **基线强度问题**：本文未与Model Cards进行定量或定性的系统对比（如：同一组研究用Model Cards vs SMRS报告，信息完整度差异），仅声明SMRS"受Model Cards启发"并"针对领域适配"。
- **证据强度问题**：17项研究的选择标准、编码方法、评分者间一致性等系统回顾方法论细节未充分披露，限制了结论的可推广性。
- **缺失基线**：未包含用户研究（研究者使用SMRS后报告质量/效率是否提升）、未包含跨领域验证（17项研究是否覆盖所有主要应用领域）、未与DOME（Data, Optimization, Model, Evaluation）等其他科学ML报告框架对比。
- **作者披露的局限**：标准本身为描述性/清单式，非强制性；采纳依赖社区自愿；无实证验证SMRS采纳确实改善可复现性或科学产出。

## 方法谱系与知识库定位

**方法家族**：报告标准 / 模型文档框架（Model Documentation Standards）

**父方法**：**Model Cards** [34] —— 通用机器学习模型报告框架。SMRS明确声明受其启发，但进行领域特化扩展：将通用描述转化为代理建模专用的六维结构化清单，并新增前置需求评估框架。

**直接基线与差异**：
- **Model Cards**：面向观测数据预测模型；SMRS面向仿真替代模型，增加数据生成模式、采样设计、仿真器验证协议、不确定性量化等维度。
- **Ad-hoc 代理建模文献实践**：当前主流，无统一标准；SMRS提供首个结构化替代，但自愿采纳、无强制执行机制。
- **DOME / 其他科学ML框架**（未在本文中对比）：可能覆盖Data-Optimization-Model-Evaluation结构，SMRS更聚焦代理建模全流程但未进行横向比较。

**变化槽位（changed slots）**：
- `reporting_framework`：新增（从无处到六维清单+需求评估）
- `data_pipeline`：修改（从隐式/不一致到显式强制报告数据来源、格式、采样设计）
- `inference_strategy`：修改（从确定性/概率性选择常不报告到显式文档化）

**后续方向**：
1. **工具化实现**：开发SMRS的自动化报告生成工具（如Python库、Jupyter插件），与SMT 2.0等代理建模工具箱集成，降低采纳门槛。
2. **实证验证**：设计用户研究或大规模文献复现实验，量化SMRS采纳对可复现性、跨研究比较效率、科学决策可靠性的实际影响。
3. **社区治理与演进**：建立SMRS的版本控制与领域扩展机制（如针对多保真代理、实时数字孪生、联邦仿真等新兴场景），从"建议性标准"向"期刊/会议投稿要求"转化。

**标签**：
- **modality**: text / documentation
- **paradigm**: 标准化框架 / 报告规范
- **scenario**: 科学计算、工程设计、气候模拟、流行病学、数字孪生
- **mechanism**: 结构化清单、领域适配、不确定性量化报告
- **constraint**: 自愿采纳、无强制执行力、描述性而非验证性

