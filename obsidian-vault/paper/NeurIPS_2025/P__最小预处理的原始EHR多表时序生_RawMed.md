---
title: Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 最小预处理的原始EHR多表时序生成框架
- RawMed
- RawMed is the first framework to sy
acceptance: Poster
method: RawMed
modalities:
- Text
- time-series
- tabular
paradigm: supervised
---

# Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing

**Topics**: [[T__Medical_Imaging]], [[T__Time_Series_Forecasting]] | **Method**: [[M__RawMed]] | **Datasets**: MIMIC-IV conditional, eICU conditional

> [!tip] 核心洞察
> RawMed is the first framework to synthesize multi-table, time-series EHR data that closely resembles raw EHRs using text-based representation and neural compression with minimal lossy preprocessing.

| 中文题名 | 最小预处理的原始EHR多表时序生成框架 |
| 英文题名 | Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2507.06996) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 多表时序电子健康记录（EHR）合成、文本序列化表示、离散潜在空间压缩、条件时序生成 |
| 主要 baseline | EHR-Safe、Hierarchical Autoregressive Language Model for EHR、Flexible Generative Model for Heterogeneous Tabular EHR、Diffusion models for EHR time series、EVA、Synteg、PromptEHR |

> [!abstract] 因为「现有EHR合成方法依赖专家特征选择与重度预处理，丢失原始多表时序数据的异构性与动态性」，作者在「VQ-VAE/RQ图像生成方法」基础上改了「文本序列化+1D CNN事件压缩+时序建模的两阶段架构」，在「MIMIC-IV和eICU」上取得「age prediction AUROC 0.77（MIMIC-IV, static+1/4 initial events）、gender prediction AUROC 1.0（MIMIC-IV，但存在数据泄漏）」

- **关键性能**: MIMIC-IV age prediction: static-only 0.75 → static+initial events 0.77 (+0.02)；eICU age prediction: static-only 0.65 → static+initial events 0.69 (+0.04)
- **关键性能**: MIMIC-IV gender prediction AUROC = 1.0（与真实数据持平），但作者承认系因gender-specific reference ranges泄漏所致，非模型能力
- **关键性能**: eICU gender prediction: static-only 0.52 → static+initial events 0.59 (+0.07)，但仍低于真实数据 0.67

## 背景与动机

电子健康记录（EHR）合成是医疗AI发展的关键瓶颈：真实患者数据受隐私法规（HIPAA等）严格限制，而合成数据需在统计保真度、时序动态性、跨表关系及隐私保护之间取得平衡。现有方法面临的核心困境是**预处理悖论**——为降低建模复杂度，研究者通常依赖专家筛选特征子集、离散化数值、归一化分布、删除罕见事件，最终将多表结构展平为固定维度向量。例如，某患者的多次实验室检查、用药记录、生命体征被压缩为"选定列的最近值"，丢失了事件间的时间间隔、异构类型（类别/数值/时间戳）及表间引用关系。

现有方法可分为三类：**GAN-based方法**（如EHR-Safe、MedGAN、CoR-GAN）通过对抗训练生成离散或连续特征，但依赖预处理的固定维度输入，难以处理可变长度的原始事件序列；**VAE-based方法**（如EVA）以重构损失优化连续潜在空间，但对高维异构时序数据的压缩效率有限；**自回归语言模型**（如Hierarchical Autoregressive Language Model for EHR）将患者轨迹展平为token序列，却面临长序列建模的指数级复杂度，且仍需预处理定义"词汇表"。**扩散模型**虽在图像生成中表现优异，应用于EHR时序时仍需将数据嵌入到固定网格，预处理负担未减。

这些方法的共同短板在于：**无法直接处理原始多表EHR的文本化异构事件**。一旦某列被专家排除或某张表被忽略，下游研究的复现性与泛化性即受损害；数值离散化的粒度选择引入信息损失；时间戳被简化为"距入院小时数"则丢失了精确的临床节律。本文的核心动机正是**消除预处理对专家知识的依赖**，通过文本序列化保留全部原始信息，再以神经压缩高效建模。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e8201b1a-edf8-495a-a764-0f6288e057e5/figures/Figure_1.png)
*Figure 1: Figure 1: Conceptual overview of the RawMed pipeline. Left: Real-world EHR data. Center: Datageneration process. Right: Resulting synthetic data. The bottom illustrates conventional approachfocused on*



本文提出RawMed，首次实现从原始多表时序EHR直接生成合成数据，仅需最小化有损预处理。

## 核心创新

核心洞察：**将原始EHR事件视为自然语言句子进行序列化**，因为医疗数据的表-列-值结构天然具备文本描述的语法性，从而使「端到端神经压缩+时序生成」成为可能——无需专家定义特征空间即可保留完整语义。

与 baseline 的差异：

| 维度 | Baseline（EHR-Safe/EVA/自回归模型） | 本文（RawMed） |
|:---|:---|:---|
| 数据表示 | 专家选择特征子集，离散化/归一化/展平为固定向量 | 文本序列化：「lab item Glucose value 95 uom mg/dL」，保留全部列与异构类型 |
| 压缩方式 | 直接对预处理后的特征向量建模，或字符级token嵌入 | 1D CNN编码器 + VQ-VAE/RQ离散量化，128×256 → 4×256 压缩比 |
| 条件生成 | 仅静态特征（年龄/性别/诊断）条件，或无条件生成 | 静态特征 + 初始1/4事件序列联合条件，增强时序上下文 |
| 架构范式 | 单阶段端到端（GAN/VAE/扩散直接生成特征） | 两阶段解耦：事件级压缩模块 + 时序事件间建模模块 |

## 整体框架



RawMed采用**两阶段解耦架构**，数据流如下：

**阶段一：事件级压缩（Event-level Compression）**
- **输入**：原始多表EHR中的单条事件（如lab表的一行：timestamp=2023-01-01 08:00, item=Glucose, value=95, uom=mg/dL, ...）
- **文本序列化模块**：将表名、列名、非空值拼接为文本字符串 `lab item Glucose value 95 uom mg/dL`，tokenize并padding至固定长度L=128，嵌入为 $x^p_i \in \mathbb{R}^{L \times F}$（F=256）
- **1D CNN编码器**：通过stride卷积将序列压缩为潜在表示 $\hat{z}^p_i = \text{Enc}(x^p_i) \in \mathbb{R}^{L_z \times F_z}$（Lz=4, Fz=256），实现32倍长度压缩
- **VQ/RQ量化器**：将连续潜在向量映射为离散码本索引，输出量化表示 $z^p_i$ 或离散索引序列 $k^p_i$
- **1D CNN解码器**：从量化表示重建文本嵌入 $\hat{x}^p_i$，与输入计算重构损失

**阶段二：时序事件间建模（Temporal Inter-Event Modeling）**
- **输入**：患者p的压缩事件序列 $S^p_{\text{quantized}} = [(t^p_i, k^p_i) \text{mid} i=1,...,n_p]$，包含时间戳与离散潜在码
- **时序建模模块**：学习事件间的动态时间关系（如用药后实验室指标的变化趋势、夜间/昼间的生理节律）
- **条件生成器**：推理时接受静态特征（年龄/性别/入院诊断）与可选的初始1/4事件，生成完整合成轨迹

```
原始多表EHR ──→ 文本序列化 ──→ Token嵌入 ──→ 1D CNN Encoder ──→ VQ/RQ量化 ──→ 离散潜在码
                                                                              ↓
静态特征 + 初始事件 ──→ 条件生成器 ←── 时序建模模块 ←── 压缩事件序列 [(t_i, k_i)]
                                                                              ↓
                                                                         合成患者轨迹
```

## 核心模块与公式推导

### 模块 1: 向量量化与残差量化（对应框架图「事件压缩模块」）

**直觉**: 单条EHR事件的文本嵌入维度高达128×256=32768，直接时序建模不可行；需离散化为紧凑码本索引，同时保证重建精度以保留临床语义。

**Baseline 公式** (VQ-VAE [van den Oord et al.]): 
$$\text{VQ}(\hat{z}; C) = \text{arg}\min_{k \in [K]} \|\hat{z} - \text{lut}(k)\|_2^2, \quad z = \text{lut}(\text{VQ}(\hat{z}; C))$$
符号: $\hat{z}$ = 编码器输出的连续潜在向量, $C$ = 码本（codebook）含K个嵌入向量, $\text{lut}(k)$ = 索引k对应的码本向量, $z$ = 量化后的离散表示

**变化点**: 标准VQ-VAE仅用单一代码本向量逼近$\hat{z}$，对复杂医疗事件的表达能力受限；本文引入**残差量化（Residual Quantization, RQ）**，通过D层迭代残差分解，用多个码本向量之和逼近原始向量，显著提升重建精度。

**本文公式（推导）**:
$$\text{Step 1}: \quad r_0 = \hat{z}; \quad k_d = \text{VQ}(r_{d-1}; C) \quad \text{（每层用VQ量化当前残差）}$$
$$\text{Step 2}: \quad r_d = r_{d-1} - \text{lut}(k_d), \quad z^{(d)} = \sum_{m=1}^{d} \text{lut}(k_m) \quad \text{（更新残差并累积部分和）}$$
$$\text{最终}: \quad \text{RQ}(\hat{z}; C, D) = (k_1, \ldots, k_D) \in [K]^D, \quad z = \sum_{d=1}^{D} \text{lut}(k_d)$$

当D=1时退化为标准VQ-VAE；D>1时，$z^{(d)}$逐步逼近$\hat{z}$，如图2所示，RQ重建的患者体重分布比VQ更接近真实数据。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e8201b1a-edf8-495a-a764-0f6288e057e5/figures/Figure_2.png)
*Figure 2: Figure 2: Comparison of VQ-VAE and RQ-VAE for the patientweight column in MIMIC-IV. Sub-figures: (a) real data, (b) VQ reconstruction, (c) VQ with less compression, (d) VQ with doubledcodebook, (e) RQ*



### 模块 2: 压缩映射与损失函数（对应框架图「编码器-解码器」）

**直觉**: 将文本嵌入的高维序列压缩为低维潜在序列，需保证时空维度的同时降采样，且量化过程需稳定训练。

**Baseline 公式** (标准VQ-VAE重建损失):
$$\mathcal{L}_{\text{base}} = \|x_i^p - \hat{x}_i^p\|_2^2$$

**变化点**: 仅重构损失导致编码器输出漂移，码本利用率低；本文加入**承诺损失（commitment loss）**约束编码器输出靠近量化向量，采用stop-gradient技巧分离梯度流。

**本文公式（推导）**:
$$\text{Step 1}: \quad x_i^p \in \mathbb{R}^{L \times F} \text{xrightarrow}{\text{Enc}} \hat{z}_i^p = \text{Enc}(x_i^p) \in \mathbb{R}^{L_z \times F_z} \quad \text{（1D CNN降维: } L=128 \to L_z=4\text{）}$$
$$\text{Step 2}: \quad z_i^p = \text{VQ/RQ}(\hat{z}_i^p) \in \mathbb{R}^{L_z \times F_z} \quad \text{（离散量化）}$$
$$\text{Step 3}: \quad \hat{x}_i^p = \text{Dec}(z_i^p) \in \mathbb{R}^{L \times F} \quad \text{（解码重建）}$$
$$\text{最终}: \quad \mathcal{L} = \|x_i^p - \hat{x}_i^p\|_2^2 + \beta \| \text{sg}[\text{Enc}(x_i^p)] - z_i^p \|_2^2 + \gamma \| \text{sg}[z_i^p] - \text{Enc}(x_i^p) \|_2^2$$
其中sg[·]为stop-gradient算子，β、γ为平衡系数（详见附录D.1和E）。

**对应消融**: Table 6显示移除RQ（Residual Quantization）改用单层VQ、移除Time Tok.（Time Tokenization）、移除Time Sep.（Time Separation）各组件的影响，但具体Δ%数值在提供片段中未完整显示。

### 模块 3: 条件生成策略（对应框架图「条件生成器」）

**直觉**: 纯静态特征条件无法捕捉患者个体的时序演变模式；提供初始事件作为"锚点"可引导生成更真实的后续轨迹。

**Baseline 公式** (静态特征条件):
$$p(S^p \text{mid} s^p) \quad \text{其中 } s^p = \{\text{age}, \text{gender}, \text{admission diagnosis}\}$$

**变化点**: 静态条件缺失时序上下文，生成轨迹可能与真实患者演变脱节；本文增加**初始1/4事件条件**，使生成过程具备"记忆"。

**本文公式（推导）**:
$$\text{Step 1}: \quad S^p_{\text{init}} = \{(t^p_i, k^p_i) \text{mid} i = 1, \ldots, \lfloor n_p/4 \rfloor\} \quad \text{（提取前1/4事件）}$$
$$\text{Step 2}: \quad c^p = [s^p; \text{Embed}(S^p_{\text{init}})] \quad \text{（静态特征与初始事件嵌入拼接）}$$
$$\text{最终}: \quad p(S^p_{\text{gen}} \text{mid} c^p) = \prod_{i=\lfloor n_p/4 \rfloor + 1}^{n_p} p(k^p_i \text{mid} k^p_{<i}, t^p_i, c^p) \quad \text{（自回归生成剩余轨迹）}$$

**对应消融**: Table 17（实验部分）显示，MIMIC-IV age prediction AUROC从static-only的0.75提升至static+initial events的0.77（+0.02），eICU从0.65提升至0.69（+0.04）；eICU gender prediction从0.52提升至0.59（+0.07）。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e8201b1a-edf8-495a-a764-0f6288e057e5/figures/Table_2.png)
*Table 2: Table 2: Results of single-table evaluation on MIMIC-IV and eICU datasets. Metrics are averagedacross columns and tables for each dataset. Lower values are better, with best in bold.*



本文在MIMIC-IV和eICU两个公开数据集上评估RawMed，涵盖**单表保真度**、**临床效用（下游任务TSTR）**、**时序保真度与隐私**、**条件生成**四个维度。Table 2展示单表评估结果（列级密度与相关性指标），Table 3展示临床效用（micro-averaged AUROC），Table 4展示时序保真度（Next Event Prediction F1）与隐私（Membership Inference Attack准确率），Table 5对比6小时观察窗口下的RealTabFormer与RawMed。由于这些表格的具体数值在提供片段中未完整显示，以下聚焦**条件生成性能**（Table 17）与**消融研究**（Table 6）。

**核心结果**：在MIMIC-IV条件生成任务中，RawMed（static+1/4 initial events）取得age prediction AUROC 0.77，较static-only的0.75提升+0.02，接近真实数据0.80（gap -0.03）；admission type AUROC 0.89，与static-only持平，真实数据0.92（gap -0.03）。值得注意的是，gender prediction AUROC达到1.0，与真实数据持平，但作者明确承认这是由于**条件生成无意中泄漏了gender-specific reference ranges**（如男性/女性不同的实验室正常值范围），这些临床先验成为直接的性别指示器，而非模型真正学到了鲁棒的性别表征。


![Table 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e8201b1a-edf8-495a-a764-0f6288e057e5/figures/Table_6.png)
*Table 6: Table 6: Ablation study on MIMIC-IV dataset.Variants exclude RQ (Residual Quantization),Time Tok. (Time Tokenization), and Time Sep.(Time Separation). Lower values are better, bestin bold.*



eICU数据集呈现不同模式：age prediction AUROC 0.69（static+initial）vs 0.65（static-only），+0.04接近真实0.72；但gender prediction仅0.59，较static-only的0.52提升+0.07，却显著低于真实数据0.67。作者将此列为局限，认为需超参数优化。admission diagnosis AUROC 0.88（static+initial）vs 0.86（static-only），+0.02接近真实0.89。



**消融分析**：Table 6在MIMIC-IV上系统评估了三个关键组件的贡献——移除RQ（Residual Quantization，回退到单层VQ）、移除Time Tok.（Time Tokenization，时间戳编码方式）、移除Time Sep.（Time Separation，时间分隔策略）。具体数值未在片段中完整显示，但结合Figure 2的定性对比，RQ对连续数值列（如患者体重）的重建保真度显著优于VQ，分布更接近真实数据的偏态特征。

**公平性检验**：本文比较基线涵盖GAN（EHR-Safe）、VAE（EVA）、扩散模型、自回归语言模型、提示学习（PromptEHR）等多种范式，覆盖面较广。但提供片段中**缺少RawMed与这些基线的直接头对头数值对比**，仅展示RawMed自身变体（static vs static+initial）及与"Real data"的比较。此外，gender prediction的perfect AUROC系数据泄漏所致，不应视为有效优势；eICU gender prediction的性能下降表明跨数据集泛化仍需优化。计算资源、训练时间、模型参数量等成本信息未披露。评估仅覆盖两个数据集，且均为英文ICU数据，多样性有限。

## 方法谱系与知识库定位

**方法族**: 离散潜在空间生成模型 → 医疗时序数据合成

**父方法**: VQ-VAE（Neural Discrete Representation Learning, van den Oord et al.）+ Residual Quantization（Autoregressive image generation using residual quantization, [16]）。RawMed将图像/视频领域的向量量化技术迁移至EHR领域，核心改动包括：（1）1D CNN替代2D CNN以处理文本序列；（2）文本序列化替代像素/图像块作为输入表示；（3）时序事件间建模替代空间自回归。

**直接基线与差异**:
- **EHR-Safe** [10]: GAN-based，依赖专家预处理特征 → RawMed改为文本原始表示+神经压缩
- **Hierarchical Autoregressive Language Model for EHR** [21][22]: 层级自回归直接建模token序列 → RawMed解耦为压缩+时序两阶段，降低长序列复杂度
- **Flexible Generative Model for Heterogeneous Tabular EHR** [11]: 处理缺失模态的异构表格 → RawMed扩展至多表时序，引入时间维度建模
- **Diffusion models for EHR time series** [12]: 扩散过程生成时序 → RawMed采用VQ/RQ离散空间，推理效率更高

**后续方向**:
1. **消除条件泄漏**：设计隐私约束的损失函数或后处理，防止gender/race等敏感属性通过临床参考范围泄漏
2. **跨数据集泛化**：在更多EHR系统（如Cerner、Epic导出格式）及非ICU场景（门诊、慢病管理）验证
3. **可解释压缩**：分析VQ/RQ码本向量的临床语义，实现"可解释的合成"——明确哪些码本对应实验室异常、用药组合等临床概念

**标签**: 模态=文本+时序+表格 | 范式=离散潜在空间生成+两阶段解耦 | 场景=医疗EHR合成 | 机制=向量量化/残差量化+1D CNN+条件自回归 | 约束=最小预处理+隐私保护

