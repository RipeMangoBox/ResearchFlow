---
title: 'LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models'
type: paper
paper_level: A
venue: ICLR
year: 2024
paper_link: null
aliases:
- LLM端到端碳足迹建模框架
- LLMCarbon
- LLMCarbon is the first end-to-end c
acceptance: Oral
cited_by: 125
method: LLMCarbon
modalities:
- Text
followups:
- AI碳足迹的概率化核算框架PCA_PCAM_(Probabilis
---

# LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__LLMCarbon]] | **Datasets**: Operational carbon footprint validation, Embodied carbon footprint validation, Data center efficiency

> [!tip] 核心洞察
> LLMCarbon is the first end-to-end carbon footprint projection model that accurately estimates both operational and embodied carbon emissions for both dense and MoE LLMs before physical training, significantly outperforming mlco2.

| 中文题名 | LLM端到端碳足迹建模框架 |
| 英文题名 | LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models |
| 会议/期刊 | ICLR 2024 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2309.14393) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 大语言模型碳足迹估算 / Benchmark & Evaluation |
| 主要 baseline | mlco2, Carbontracker, Measuring the carbon intensity of AI in cloud instances |

> [!abstract] 因为「现有工具（如 mlco2）无法支持现代 dense/MoE 架构、仅覆盖 GPU、忽略隐含碳」，作者在「mlco2」基础上改了「架构感知多项式回归 + 缩放定律预测 + 多硬件支持 + 运营/隐含碳全生命周期覆盖」，在「GPT-3, PaLM, PaLM 2, XLM, Wu Dao 2.0 运营碳验证」上取得「首个支持 dense 与 MoE 双架构的端到端碳足迹投影能力」。

- **关键性能 1**：MoE 模型因 ~80% 稀疏度因子，碳估计较忽略稀疏性的方法降低约 5 倍（即避免 5 倍高估）
- **关键性能 2**：隐含碳占现代 LLM 总碳足迹的 15–30%，此前工具完全缺失此项
- **关键性能 3**：验证覆盖 GPT-3、PaLM、PaLM 2、XLM（训练 20.4 天）、Wu Dao 2.0 等模型的运营碳足迹

## 背景与动机

训练一个 GPT-3 级别的 LLM 可能排放数百吨 CO₂，相当于多辆汽车终身排放。在模型实际训练前预估碳足迹，对科研规划、碳中和承诺、政策制定至关重要。然而，现有工具严重落后于模型架构的演进。

**mlco2** [3] 是当前最常用的深度学习碳追踪工具，它根据 GPU 功耗和训练时长估算运营碳排放，但仅支持简单参数量线性缩放，无法区分同参数量不同架构的模型。**Carbontracker** [3] 同样聚焦 GPU 训练能耗，缺乏对现代硬件生态的覆盖。**Measuring the carbon intensity of AI in cloud instances** [13] 和 **Towards the systematic reporting of the energy and carbon footprints of machine learning** [18] 奠定了系统性碳报告的基础，但仍局限于运营碳（operational carbon）和云实例场景。

这些工具的共同短板在 LLM 时代被急剧放大：第一，**架构盲区**——MoE（Mixture-of-Experts）模型通过稀疏激活大幅降低实际计算量，但现有工具按总参数量估算，导致高达约 5 倍的高估；第二，**硬件单一**——TPU、ASIC 等非 GPU 加速器已成为 LLM 训练主流，固定 GPU 假设失效；第三，**生命周期缺失**——芯片制造、数据中心基建的隐含碳（embodied carbon）占总足迹 15–30%，却被完全忽略；第四，**阶段残缺**——仅覆盖训练，忽略推理、实验迭代、存储等关键阶段。

本文提出 LLMCarbon，首个在训练前即可投影 dense 与 MoE LLM 全生命周期碳足迹的框架，将碳估算从「黑箱后验统计」推进为「架构感知先验预测」。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a7de3982-ca62-43f6-912a-a7503fb87be7/figures/Figure_1.png)



## 核心创新

**核心洞察**：LLM 的碳足迹本质上是「架构配置 → 计算需求 → 能耗 → 碳排放」的确定性映射，但传统工具将第一步简化为参数量查表，丢失了 MoE 稀疏性、层数/隐藏维度比、序列长度等关键结构信息；LLMCarbon 用多项式回归直接学习架构参数到碳指标的映射，并引入神经缩放定律预测训练规模，从而使训练前的端到端碳投影成为可能。

| 维度 | Baseline (mlco2) | 本文 (LLMCarbon) |
|:---|:---|:---|
| 架构支持 | 仅参数量线性缩放，无 MoE 支持 | Dense + MoE 双架构，含隐藏维度、层数、头数、专家数、top-k 路由 |
| 硬件覆盖 | GPU-only | GPU / TPU / ASIC 多硬件，设备专属 TPD 建模 |
| 碳类型 | 运营碳（训练能耗） | 运营碳 + 隐含碳（制造、基建分摊） |
| 生命周期阶段 | 训练 only | 训练 → 推理 → 实验 → 存储 全阶段聚合 |
| 预测机制 | 直接测量或线性外推 | 多项式回归 + 缩放定律（α=0.34, β=0.28）先验预测 |

## 整体框架



LLMCarbon 的输入为「LLM 架构配置 + 硬件规格 + 数据中心参数」，输出为「训练前可得的端到端碳足迹投影」。数据流历经六大模块：

1. **Architecture Parser（架构解析器）**：接收 dense 或 MoE 模型的关键参数——层数 L、隐藏维度 H、注意力头数 A、序列长度 D、专家数 E、top-k 路由系数——输出归一化架构特征向量，替代基线的简单参数量查表。

2. **Hardware Profiler（硬件分析器）**：输入设备类型（GPU/TPU/ASIC）及其 TPD（Thermal Design Power）、内存带宽等规格，输出各操作类型的能耗速率，突破基线 GPU-only 限制。

3. **Scaling Law Predictor（缩放定律预测器）**：基于 Broken Neural Scaling Laws [7]，输入模型规模与数据规模，输出预测测试损失、所需训练步数及推理延迟，为碳估算提供「训练多久、推理多快」的先验知识。

4. **Polynomial Regression Engine（多项式回归引擎）**：将归一化架构特征映射到 FLOPs、内存带宽需求、训练时长等中间指标，是连接「架构」与「能耗」的核心非线性拟合器。

5. **Operational Carbon Calculator（运营碳计算器）**：汇总各阶段能耗，乘以数据中心 PUE（Power Usage Effectiveness）与地区电网碳强度 CI_grid，得到电力消耗产生的 CO₂。

6. **Embodied Carbon Calculator（隐含碳计算器）**：输入芯片面积、制程节点、设备数量，按生命周期利用率分摊制造与基建碳排放。

7. **Phase Aggregator（阶段聚合器）**：将训练、推理、实验、存储四阶段碳排放求和，输出 C_total = C_operational + C_embodied。

```
[Architecture Config] ──┐
                        ▼
[Hardware Spec] ──→ [Architecture Parser] ──→ [Scaling Law Predictor]
                        │                           │
                        ▼                           ▼
              [Polynomial Regression Engine] ←── 预测训练规模
                        │
                        ▼
              [Operational Carbon Calc] ←── PUE × CI_grid
                        │
              [Embodied Carbon Calc] ←── 芯片制造分摊
                        │
                        ▼
              [Phase Aggregator] → C_total (kgCO₂e)
```

## 核心模块与公式推导

### 模块 1: 总碳足迹聚合（对应框架图 Phase Aggregator）

**直觉**：碳足迹不应只算电费账单，芯片制造和数据中心基建的「上游排放」同样真实且可观。

**Baseline 公式** (mlco2):
$$C_{total} = C_{operational}^{training} \text{ (GPU only)}$$
符号: $C_{operational}$ = 运营碳排放（电力消耗产生），上标 training 表示仅覆盖训练阶段。

**变化点**：基线完全缺失隐含碳，且阶段覆盖残缺；本文将碳核算扩展至全生命周期。

**本文公式（推导）**:
$$\text{Step 1}: C_{operational} = E_{compute} \times PUE \times CI_{grid} \quad \text{（能耗→碳排放，引入数据中心效率与电网清洁度）}$$
$$\text{Step 2}: C_{embodied} = \frac{A_{chip} \times CF_{fab} \times n_{devices}}{Lifetime_{utilization}} \quad \text{（芯片面积×晶圆厂系数×数量，按生命周期利用率分摊）}$$
$$\text{最终}: C_{total} = C_{operational} + C_{embodied} = \sum_{phases}(E_{phase} \times PUE \times CI_{grid}) + \frac{A_{chip} \times CF_{fab} \times n_{devices}}{\sum_{all\ tasks} t_{utilization}}$$

**对应消融**：去掉 Embodied Carbon Calculator 后，现代 LLM 总碳足迹缺失 15–30%。

---

### 模块 2: 多硬件能耗计算（对应框架图 Hardware Profiler + Operational Carbon Calc）

**直觉**：不同加速器（A100 vs TPU v4 vs 定制 ASIC）的功耗曲线、利用率特征差异巨大，统一 GPU 假设导致系统性偏差。

**Baseline 公式** (mlco2):
$$E_{compute} = P_{GPU} \times t_{training} \text{ (GPU only)}$$
符号: $P_{GPU}$ = GPU 热设计功率，$t_{training}$ = 训练时长。

**变化点**：基线硬编码 GPU 功耗；本文按设备类型分解各阶段能耗，支持异构硬件调度。

**本文公式（推导）**:
$$\text{Step 1}: E_{compute} = \sum_{phases} (P_{device} \times t_{utilization} \times u_{utilization}) \quad \text{（多设备、多阶段、实际利用率加权）}$$
$$\text{Step 2}: t_{train} = \frac{FLOPs}{Throughput_{device}} \times \frac{1}{u_{utilization}} \quad \text{（根据设备吞吐量反推实际训练时间）}$$
$$\text{最终}: C_{operational} = \left[\sum_{phases} (P_{device} \times t_{phase} \times u_{phase})\right] \times PUE \times CI_{grid}$$
符号: $P_{device}$ = 设备 TPD（W），$u_{utilization}$ = 实际利用率（非峰值），$Throughput_{device}$ = 设备特定算力（FLOPs/s）。

**对应消融**：Table 4 显示不同硬件（如 A100、V100、TPU）的 TPD 差异导致碳估计显著分化。

---

### 模块 3: 架构感知预测引擎（对应框架图 Architecture Parser + Polynomial Regression + Scaling Law Predictor）

**直觉**：同参数量模型（如 175B 的 dense GPT-3 vs 175B 总参数的 MoE）实际激活计算量差异可达数倍，必须从架构细节出发预测碳指标。

**Baseline 公式** (mlco2):
$$\text{Linear scaling: } E \propto N \times D \text{ (parameters \times data tokens)}$$
符号: $N$ = 总参数量，$D$ = 训练数据量。

**变化点**：线性缩放无法捕捉层数-维度比、MoE 稀疏性等结构效应；本文用多项式回归拟合归一化架构特征，并用缩放定律预测训练所需规模。

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{x} = \left[\frac{L}{L_{max}}, \frac{H}{H_{max}}, \frac{A}{A_{max}}, \frac{D}{D_{max}}, f_{MoE}(E, k)\right] \quad \text{（归一化特征，MoE 额外编码有效激活比例）}$$
$$\text{Step 2}: \hat{y} = \sum_{i=0}^{d} \beta_i \mathbf{x}^i + \epsilon \quad \text{（多项式回归预测 FLOPs、内存带宽、训练步数等中间指标）}$$
$$\text{Step 3}: L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha} + \left(\frac{D_c}{D}\right)^{\beta} \quad \text{（缩放定律预测测试损失，} \alpha=0.34, \beta=0.28 \text{ 为拟合常数 [7]）}$$
$$\text{Step 4}: C_{MoE} = C_{dense} \times \rho_{sparsity} \times k_{routing} \quad \text{（MoE 碳效率：引入 } \sim\!80\% \text{ 稀疏度因子，仅 20\% 参数激活）}$$
$$\text{最终}: \hat{C}_{total} = f_{poly}(\mathbf{x}; \boldsymbol{\beta}) \rightarrow E_{compute} \rightarrow C_{operational} + C_{embodied}$$

**对应消融**：去掉 MoE 稀疏度因子后，MoE 模型碳估计高估约 5 倍；仅用参数量替代完整架构特征时，同参数量不同架构模型估计误差显著增大。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a7de3982-ca62-43f6-912a-a7503fb87be7/figures/Table_1.png)
*Table 1 (comparison): The comparison of LLMCarbon against prior work.*




![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a7de3982-ca62-43f6-912a-a7503fb87be7/figures/Table_4.png)
*Table 4 (validation): The validation on the operational Carbon Footprint of various LLMs.*



本文在运营碳与隐含碳两条线上展开验证。**运营碳验证**（Table 4）覆盖 GPT-3、PaLM、PaLM 2、XLM、Wu Dao 2.0 等主流 LLM，LLMCarbon 的投影与这些模型公开报告值对齐，而 mlco2 因架构假设过于简化无法处理 MoE 或 dense LLM 的精细结构，显著低估或高估实际排放。**隐含碳验证**（Table 5）首次为 LLM 硬件建立制造碳排放估算，XLM 训练时长 20.4 天的数据被用于校准训练时间预测模块。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a7de3982-ca62-43f6-912a-a7503fb87be7/figures/Table_5.png)
*Table 5 (validation): The embodied carbon footprint validation.*



消融实验揭示三项关键发现：其一，**MoE 稀疏度因子**是最敏感的单一组件——移除后 MoE 模型碳估计从合理区间跳升至约 5 倍高估，印证了架构感知建模的必要性；其二，**隐含碳模块**贡献总足迹的 15–30%，对于短期大规模训练任务（如 GPT-3 级别预训练），制造分摊碳不可忽视；其三，**多项式回归 vs 线性缩放**的对比显示，同参数量下不同层数/维度配置的模型，能耗差异可达数倍，线性假设系统性失效。


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a7de3982-ca62-43f6-912a-a7503fb87be7/figures/Figure_7.png)
*Figure 7 (result): The carbon footprint of OPT trained utilizing different devices.*



Figure 7 展示 OPT 模型在不同硬件（A100、V100 等）上训练的碳足迹分化，验证多硬件 TPD 建模的实际影响。Table 2 的数据中心效率参数表明，PUE 从 1.1（先进液冷）到 1.6（传统风冷）的跨度，可使同模型碳排放差异达 45%。

**公平性检查**：本文主要对比 mlco2 [3] 作为基线，但未纳入 CodeCarbon、ML Emissions Calculator、Experiment-Impact-Tracker 等同期工具进行 head-to-head 比较。验证依赖「已发表论文的自报告值」而非独立测量的 ground-truth，存在报告偏差风险。多项式回归系数基于现有架构拟合，对未来超出分布的架构（如全新稀疏模式）泛化性存疑。MoE ~80% 稀疏度因子的精确数值源自截断的 Figure 5 caption，需结合原文确认。

## 方法谱系与知识库定位

LLMCarbon 属于 **AI 系统碳足迹估算** 方法族，直接继承自 **mlco2** / **Carbontracker** [3] 的能耗追踪范式，并沿 **Measuring the carbon intensity of AI in cloud instances** [13] 与 **Towards the systematic reporting of the energy and carbon footprints of machine learning** [18] 的系统性碳报告方向扩展。

**谱系定位**：mlco2（父方法）→ LLMCarbon（子方法）。关键插槽变更：
- **architecture**：参数量查表 → dense/MoE 架构感知（层数、维度、头数、专家数、top-k）
- **data_pipeline**：GPU-only → 多硬件（GPU/TPU/ASIC）TPD 建模
- **objective**：运营碳（训练） → 运营碳 + 隐含碳（全生命周期）
- **inference_strategy**：训练 only → 训练/推理/实验/存储 四阶段聚合
- **training_recipe**：线性缩放 → 多项式回归 + 神经缩放定律（α=0.34, β=0.28）

**直接基线差异**：
- **mlco2** [3]：LLMCarbon 的直系父方法，本文在架构支持、硬件覆盖、碳类型、阶段覆盖上全面扩展
- **Carbontracker** [3]：核心碳追踪 predecessor，LLMCarbon 将其方法论特化为 LLM 场景
- **Measuring the carbon intensity of AI in cloud instances** [13]：云实例碳强度测量，LLMCarbon 引入预训练投影与架构参数敏感性
- **Broken Neural Scaling Laws** [7]：公式来源，提供 α, β 拟合常数用于训练规模预测

**后续方向**：(1) 动态工作负载下的在线碳预测（非静态架构配置）；(2) 更细粒度的芯片级能耗模型替代黑箱多项式回归；(3) 碳足迹与模型质量的联合优化（Pareto 前沿搜索）。

**标签**：modality:text | paradigm:carbon_estimation | scenario:pre_training_projection | mechanism:polynomial_regression + scaling_laws | constraint:lifecycle_completeness

## 引用网络

### 后续工作（建立在本文之上）

- [[P__AI碳足迹的概率化核算框架PCA_PCAM_(Probabilis]]: LLMCarbon is the most directly comparable prior work on end-to-end carbon footpr

