---
title: 'CarbonGlobe: A Global-Scale, Multi-Decade Dataset and Benchmark for Carbon Forecasting in Forest Ecosystems'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 全球森林碳预测ML基准数据集CarbonGlobe
- CarbonGlobe
- CarbonGlobe is the first global-sca
acceptance: Poster
code_url: https://github.com/zhwang0/carbon-globe
method: CarbonGlobe
modalities:
- tabular
- geospatial
- time series
paradigm: supervised
---

# CarbonGlobe: A Global-Scale, Multi-Decade Dataset and Benchmark for Carbon Forecasting in Forest Ecosystems

[Code](https://github.com/zhwang0/carbon-globe)

**Topics**: [[T__Time_Series_Forecasting]], [[T__Benchmark_-_Evaluation]] | **Method**: [[M__CarbonGlobe]] | **Datasets**: CarbonGlobe global carbon forecasting, Inference speed

> [!tip] 核心洞察
> CarbonGlobe is the first global-scale, multi-decade, ML-ready benchmark dataset that enables machine learning models to emulate expensive Ecosystem Demography models for carbon dynamics forecasting in forest ecosystems.

| 中文题名 | 全球森林碳预测ML基准数据集CarbonGlobe |
| 英文题名 | CarbonGlobe: A Global-Scale, Multi-Decade Dataset and Benchmark for Carbon Forecasting in Forest Ecosystems |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org) · [Code](https://github.com/zhwang0/carbon-globe) · [Project](https://github.com/zhwang0/carbon-globe) |
| 主要任务 | Time Series Forecasting, Benchmark / Evaluation |
| 主要 baseline | Ecosystem Demography (ED) v3.0, LSTM |

> [!abstract] 因为「Ecosystem Demography (ED) 物理模型在全球尺度森林碳动态预测中计算成本过高且缺乏公开ML-ready数据集」，作者在「ED v3.0」基础上改了「构建全球0.5°分辨率、40年多源遥感整合数据集并设计气候感知评估协议与政策对齐指标」，在「CarbonGlobe benchmark」上取得「ML emulator与ED可比精度但数量级推理加速」

- **空间覆盖**: 全球0.5°分辨率，40年时间跨度，整合100+变量
- **推理加速**: ML模型相比ED v3.0实现orders-of-magnitude速度提升（Figure 6）
- **评估创新**: 基于Köppen-Geiger气候分类的气候感知数据划分与政策导向指标（∆, CE）

## 背景与动机

森林生态系统是全球陆地碳循环的核心组成部分，准确预测森林碳动态对于气候变化政策制定至关重要。然而，当前主流的理论驱动方法——Ecosystem Demography (ED) 模型——面临根本性瓶颈：ED v3.0 作为基于个体演替和生理过程的物理模拟模型，在全球尺度、高分辨率运行时需要巨大的计算资源，单次完整模拟可能耗费数周甚至数月，无法满足实时碳政策决策的需求。

现有工作主要从三个方向应对这一挑战。其一，ED v3.0 等物理模型通过参数校准提升精度，但计算成本不可降低；其二，ClimART 等气候ML benchmark 为大气辐射传输提供了 emulator 框架，但未覆盖森林碳动态这一特定领域；其三，LSTM 等标准时间序列模型虽可用于碳变量预测，但缺乏针对全球空间异质性的专门设计。

这些方法的共同短板在于：**没有公开可用的ML-ready数据集**。研究者必须自行收集MERRA-2气象再分析、Hansen森林变化、Köppen-Geiger气候分类等多源数据，手动预处理并对齐到统一时空网格，再运行昂贵的ED模拟获取标签——这一流程重复且门槛极高，严重阻碍了ML社区对该问题的系统研究。此外，标准ML指标（RMSE, MAE）未能反映碳政策最关心的累积误差和趋势变化方向。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a7605cc9-452e-4874-9b8b-18093229d323/figures/Figure_2.png)
*Figure 2 (pipeline): Overview of CarbonGlobe.*



本文构建 CarbonGlobe，首次提供全球尺度、多十年、ML-ready的森林碳预测基准数据集，并配套气候感知的评估协议与政策对齐指标，使快速ML emulator的开发与公平比较成为可能。

## 核心创新

核心洞察：**将物理模型的领域知识嵌入数据工程而非模型架构**，因为 ED v3.0 经过充分校准已具备可靠的理论基础，从而使「数据集即方法」的范式成为可能——通过精心设计的输入整合、标签生成和评估协议，直接释放标准ML架构在全球碳预测中的潜力。

| 维度 | Baseline (ED v3.0 / 传统流程) | 本文 (CarbonGlobe) |
|:---|:---|:---|
| 数据可用性 | 无公开ML-ready数据集，研究者需自行预处理 | 全球0.5°分辨率、40年、100+变量的即用数据集 |
| 标签来源 | 原始ED输出，未明确校准声明 | 经全球/区域观测校准的ED输出作为训练标签 |
| 评估划分 | 随机或时序划分，忽略空间异质性 | 基于Köppen-Geiger气候分类的气候感知划分 |
| 评估指标 | 标准RMSE/MAE，与政策需求脱节 | 新增delta (∆)和cumulative error (CE)政策导向指标 |
| 推理效率 | 物理模拟，计算昂贵 | ML emulator，orders-of-magnitude加速 |

## 整体框架



CarbonGlobe 的完整数据流与评估流程包含五个核心阶段：

1. **多源原始数据接入**（输入：MERRA-2气象再分析、Hansen全球森林变化、Köppen-Geiger气候分类、Global Carbon Budget等；输出：多分辨率原始特征）—— 汇集100+变量的全球遥感产品，覆盖气象、植被、气候区划等维度。

2. **地理空间-时间对齐层**（输入：多源异分辨率特征；输出：统一0.5°全球网格）—— 将不同来源、不同分辨率的数据重采样并对齐到标准时空网格，替代传统手动预处理流程。

3. **ED v3.0 校准模拟**（输入：对齐后的环境强迫数据；输出：经观测校准的碳动态标签）—— 运行物理模型生成GPP（总初级生产力）等碳变量，并基于全球与区域观测数据进行校准，形成可靠的训练标签。

4. **ML-ready格式化与气候感知划分**（输入：对齐特征与ED输出；输出：结构化数据集）—— 构建标准化训练/验证/测试集，其中测试集按Köppen-Geiger气候区划分，确保跨气候泛化能力的严格评估。

5. **ML Emulator训练与政策对齐评估**（输入：训练集；输出：预测模型及RMSE/MAE/∆/CE指标）—— 支持LSTM、Spherical Fourier Neural Operators、Vision Transformer等多种架构的快速训练与公平比较。

```
多源遥感数据 → [地理空间-时间对齐] → 统一0.5°网格
                                    ↓
环境强迫数据 → [ED v3.0 校准模拟] → 碳动态标签
                                    ↓
                    [ML-ready格式化 + 气候感知划分]
                                    ↓
                        CarbonGlobe 数据集
                                    ↓
            [ML Emulator训练: LSTM / SFNO / ViT ...]
                                    ↓
                    快速推理 + 政策对齐评估 (∆, CE)
```

## 核心模块与公式推导

CarbonGlobe 作为数据集与基准框架，核心贡献在于数据工程与评估协议设计而非新损失函数。以下阐述其三个关键创新模块的数学形式与直觉。

### 模块 1: 气候感知评估划分（对应框架图第4阶段）

**直觉**: 随机划分会掩盖模型在稀有气候区的失败，而按Köppen-Geiger气候分类强制划分能暴露空间泛化盲区。

**Baseline 公式** (标准随机划分):
$$\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}} \sim \text{RandomSplit}(\mathcal{D}, \alpha_{\text{train}}, \alpha_{\text{val}}, \alpha_{\text{test}})$$
符号: $\mathcal{D}$ = 完整数据集, $\alpha$ = 划分比例。

**变化点**: 随机划分假设样本独立同分布，但全球碳动态具有强烈的空间自相关和气候区聚类效应；模型可能在训练气候区过拟合，在未见气候区失效。

**本文公式**:
$$\text{Step 1}: \quad c_i = \text{KöppenGeiger}(\text{lat}_i, \text{lon}_i) \quad \text{为每个网格点分配气候类别}$$
$$\text{Step 2}: \quad \mathcal{D}_{\text{test}}^{(k)} = \{(\mathbf{x}_i, y_i) \text{mid} c_i = k\}, \quad k \in \{1, ..., K\} \quad \text{按气候类别构建测试子集}$$
$$\text{Step 3}: \quad \mathcal{D}_{\text{train}} = \mathcal{D} \text{setminus} \text{bigcup}_k \mathcal{D}_{\text{test}}^{(k)} \quad \text{确保测试气候区不出现在训练集}$$
$$\text{最终}: \quad \text{Macro-F1}_{\text{climate}} = \frac{1}{K}\sum_{k=1}^{K} \text{Score}(f, \mathcal{D}_{\text{test}}^{(k)}) \quad \text{跨气候平均性能}$$

**对应消融**: Figure 4 显示不同气候区下的RMSE时间演变，揭示热带/寒带等区域误差分布差异。

### 模块 2: 政策对齐指标——Delta (∆) 与 Cumulative Error (CE)（对应框架图第5阶段）

**直觉**: 碳政策决策者关心的是"趋势方向是否正确"和"长期累积预算是否准确"，而非单点像素级误差。

**Baseline 公式** (标准ML指标):
$$\text{RMSE} = \sqrt{\frac{1}{NT}\sum_{n=1}^{N}\sum_{t=1}^{T}(\hat{y}_{n,t} - y_{n,t})^2}, \quad \text{MAE} = \frac{1}{NT}\sum_{n=1}^{N}\sum_{t=1}^{T}|\hat{y}_{n,t} - y_{n,t}|$$
符号: $n$ = 空间网格索引, $t$ = 时间步, $N$ = 网格总数, $T$ = 预测时长。

**变化点**: RMSE/MAE对短期波动敏感但无法反映：① 预测趋势与真实趋势的方向一致性（对碳汇/碳源判断至关重要）；② 长期累积碳收支的系统性偏差（影响国家排放配额计算）。

**本文公式**:
$$\text{Step 1 (Delta)}: \quad \Delta = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\left[\text{sign}(\hat{y}_{n,T} - \hat{y}_{n,1}) = \text{sign}(y_{n,T} - y_{n,1})\right] \quad \text{趋势方向一致率}$$
$$\text{Step 2 (Cumulative Error)}: \quad \text{CE}_n = \left|\sum_{t=1}^{T}\hat{y}_{n,t} - \sum_{t=1}^{T}y_{n,t}\right|, \quad \text{CE} = \frac{1}{N}\sum_{n=1}^{N}\text{CE}_n \quad \text{累积预算绝对误差}$$
$$\text{最终综合}: \quad \mathcal{L}_{\text{eval}} = \{\text{RMSE}, \text{MAE}, \Delta, \text{CE}\} \quad \text{四指标联合报告}$$

**对应消融**: Table 1 显示各ML模型在四项指标上的完整对比，∆ 和 CE 揭示 RMSE 无法捕捉的方向性失败案例。

### 模块 3: 多源数据对齐与不确定性量化（对应框架图第2阶段）

**直觉**: 不同遥感产品的分辨率、投影、时间频率各异，直接拼接会引入系统性空间偏差。

**Baseline 流程**: 研究者手动下载MERRA-2 (0.5°×0.625°)、Hansen森林变化 (30m)、Köppen-Geiger (1km) 等数据，各自重采样后假设空间对齐，无显式误差传播。

**本文设计**:
$$\text{Step 1}: \quad \mathbf{x}^{(m)}_{i,j,t} = \text{Resample}_m(\text{RawData}_m, \text{Grid}_{0.5°}) \quad \text{各数据源重采样到统一0.5°网格}$$
$$\text{Step 2}: \quad \mathbf{x}_{i,j,t} = \text{Concat}\left[\mathbf{x}^{(1)}_{i,j,t}, ..., \mathbf{x}^{(M)}_{i,j,t}\right] \in \mathbb{R}^{D} \quad \text{通道拼接，} D > 100$$
$$\text{Step 3 (ED校准)}: \quad y_{i,j,t} = \text{ED}_{\theta^*}\left(\mathbf{x}_{i,j,1:t}\right), \quad \theta^* = \text{arg}\min_{\theta}\|\text{ED}_{\theta} - \text{Obs}_{\text{in-situ}}\| \quad \text{观测校准参数}$$

**对应消融**: Figure 1 展示校准后ED模型与实地测量GPP的对比，验证标签可靠性；Figure 3 显示随机初始化与校准ED在第10年的全球差异分布，量化校准影响。

## 实验与分析



本文在 CarbonGlobe 数据集上系统评估了多种ML emulator与ED v3.0基线的性能。评估覆盖全球0.5°分辨率网格，时间跨度40年，预测任务为10年碳动态（以GPP为核心变量）。Table 1 汇总了各模型在RMSE、MAE、Delta (∆) 和 Cumulative Error (CE) 四项指标上的表现：ML方法（包括LSTM、基于Spherical Fourier Neural Operators的变体、以及Vision Transformer架构）在精度上与ED v3.0达到可比水平，同时实现推理速度的数量级提升。具体而言，Figure 6 的推理时间对比显示，ML emulator的执行时间相比ED物理模拟降低数个数量级，使得全球尺度实时碳预测在计算上成为可能。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a7605cc9-452e-4874-9b8b-18093229d323/figures/Figure_3.png)
*Figure 3 (result): Global distribution of the difference between the result of random and ED at Year 10.*





消融实验关注两个关键设计选择。其一，气候感知划分的必要性：Figure 4 沿时间维度展示10年预测期内不同气候区的RMSE演变，揭示热带湿润区与寒带大陆区的误差增长模式显著不同——若采用随机划分，寒带等样本稀少区域的失败将被全局平均掩盖。其二，ED校准对标签质量的影响：Figure 3 可视化随机初始化ED与校准ED在第10年预测结果的全球差异分布，显示未校准模型在亚马逊、刚果盆地等关键碳汇区域存在系统性偏差，经校准后这些偏差显著收敛。

公平性检验方面，作者明确承认若干局限：ED输出作为训练标签引入模型偏差，而非直接使用实地观测；Figure 4 和 Figure 5 均显示空间性能异质性，热带地区误差普遍高于温带；不确定性量化讨论有限。此外，基准对比中未包含CLM、ORCHIDEE、JULES等其他主流陆面模式，也未纳入Earthformer、AFNO等最新神经算子架构，存在进一步扩展空间。计算预算方面，ML模型的训练与推理成本显著低于ED模拟，但具体GPU小时数未在摘录中披露。

## 方法谱系与知识库定位

CarbonGlobe 属于 **气候-地球科学 × 机器学习交叉** 的方法谱系，核心范式为「物理信息数据集工程」——不修改模型架构，而是通过数据管道、评估协议和指标设计的系统性创新，释放现有ML方法的潜力。

**直接父方法与基线差异**：
- **Ecosystem Demography (ED) v3.0** [14]: 主要基线，CarbonGlobe 将其作为标签生成器而非竞争架构，关键差异在于将ED从"预测工具"重新定位为"可微分的数据增强源"
- **ClimART** [8]: 相关benchmark工作，CarbonGlobe 扩展其"气候ML emulator"理念到森林碳动态领域，并新增空间异质性评估
- **Spherical Fourier Neural Operators** [6] & **Vision Transformer** [11]: 架构灵感来源，CarbonGlobe 验证这些通用架构在特定地球科学任务上的适用性，但未提出架构变体
- **LSTM** [19]: 标准时间序列基线，用于证明即使简单序列模型在良好数据集上也能逼近复杂物理模型

**后续方向**：① 引入CLM/ORCHIDEE等多模型集成标签以降低单模型偏差；② 探索Graph Neural Network或最新神经算子（FNO, AFNO）显式建模空间依赖；③ 发展不确定性量化模块，为碳政策决策提供置信区间。

**知识库标签**: 模态=tabular/geospatial/time-series | 范式=supervised learning / benchmark | 场景=global climate policy / forest carbon monitoring | 机制=physics-informed data curation / climate-stratified evaluation | 约束=spatial heterogeneity / model bias from ED labels / computational efficiency

