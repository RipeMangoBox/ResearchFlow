---
title: 'Test-Time Adaptation for EEG Foundation Models: A Systematic Study under Real-World Distribution Shifts'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.16926
aliases:
- EEG基础模型测试时适应的系统诊断
- TAEFMA
- 本文的核心洞察是：EEG信号的高度非平稳性和复杂分布结构使得基于熵最小
paradigm: Reinforcement Learning
---

# Test-Time Adaptation for EEG Foundation Models: A Systematic Study under Real-World Distribution Shifts

[Paper](https://arxiv.org/abs/2604.16926)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]], [[T__Domain_Adaptation]], [[T__Medical_Imaging]]

> [!tip] 核心洞察
> 本文的核心洞察是：EEG信号的高度非平稳性和复杂分布结构使得基于熵最小化的梯度类TTA方法容易陷入退化解——当预测熵被强制压低时，模型可能坍缩至低熵但错误的预测模式。相比之下，无梯度的原型更新方法（T3A）通过利用置信预测来精化类别表征，避免了破坏性参数更新，因而更稳定。本质上，EEG的分布偏移结构与图像领域不同：EEG的偏移更多体现在信号统计特性和通道配置上，而非语义内容，这使得依赖语义一致性假设的梯度类方法失效。

| 中文题名 | EEG基础模型测试时适应的系统诊断 |
| 英文题名 | Test-Time Adaptation for EEG Foundation Models: A Systematic Study under Real-World Distribution Shifts |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16926) · [Code] · [Project] |
| 主要任务 | EEG基础模型在跨站点/跨设备/跨人群分布偏移下的测试时适应（TTA）评估 |
| 主要 baseline | Tent, SHOT, T3A（无TTA基线）; CBraMod, TFM-Tokenizer, REVE-Base, REVE-Large |

> [!abstract] 因为「EEG基础模型部署时面临跨临床环境、跨设备、跨人群的严重分布偏移，而现有TTA方法在EEG场景的有效性缺乏系统研究」，作者在「Tent/SHOT/T3A等标准TTA方法」基础上进行了「NeuroAdapt-Bench系统性评估框架」，在「TUEV/TUAB/SleepEDF-78/CHB-MIT/EESM23五个数据集」上取得「发现梯度类TTA方法在EEG场景普遍导致性能退化（最高降幅64%），而无梯度T3A相对稳定」的诊断性结论

- **关键性能1**: SHOT使REVE-Large在SleepEDF-78上Bal.Acc.从0.651降至0.233，降幅64%，Cohen's κ从0.678降至0.038
- **关键性能2**: T3A在CHB-MIT上为REVE-Large提供一致正向增益，在极端模态偏移Ear-EEG下使REVE-Large Bal.Acc.从0.400提升至0.422
- **关键性能3**: TFM-Tokenizer对TTA方法表现出相对更强鲁棒性，梯度类方法退化幅度明显小于其他模型

## 背景与动机

脑电图（EEG）基础模型在大规模神经数据预训练后展现出强大的表征学习能力，但临床部署时面临严峻的现实挑战：同一预训练模型从A医院部署到B医院，或从标准10-20头皮电极系统切换至耳戴式EEG设备，性能可能急剧下滑。这种跨站点、跨设备、跨人群的分布偏移在EEG领域尤为突出，因为EEG信号具有高度非平稳性、强受试者间差异，以及多样化的采集协议（电极配置、采样率、参考方式）。

现有方法如何应对？**测试时适应（TTA）** 作为无需访问源数据、仅利用推理阶段无标签目标数据进行模型适应的技术，在隐私敏感的医疗场景中具有天然优势。来自计算机视觉领域的**Tent**通过最小化预测熵更新归一化层参数；**SHOT**结合熵最小化、多样性正则与伪标签损失适应特征提取器；**T3A**则完全无梯度，通过在线维护类别原型更新分类器。这些方法在图像领域已取得显著成功。

然而，EEG的分布偏移结构与图像领域存在本质差异：偏移更多体现在信号统计特性和通道配置上，而非语义内容。现有文献几乎未系统检验这些标准TTA方法能否可靠迁移至EEG场景——预训练语料本身的异质性使得部分目标数据集可能已包含在预训练分布中（分布内），而另一些则完全未见（分布外），甚至涉及跨模态的极端偏移。这种多层次的分布偏移结构迫切需要系统性基准研究。本文正是针对这一空白，构建NeuroAdapt-Bench，首次对TTA方法在EEG基础模型上的有效性与局限性进行系统诊断。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b1e01c3f-6217-4aac-b34a-052e326e514f/figures/Figure_1.png)
*Figure 1: Figure 1: Distribution shift in EEGfoundation model deployment.Pre-trained EEG models often degradewhen applied to new sites and de-vices, motivating the need for labeland source-free test-time adapta*



## 核心创新

核心洞察：EEG信号的高度非平稳性和复杂分布结构使得基于熵最小化的梯度类TTA方法容易陷入退化解，因为当预测熵被强制压低时，模型可能坍缩至低熵但错误的预测模式；而无梯度的原型更新方法（T3A）通过利用置信预测来精化类别表征、避免了破坏性参数更新，因而在EEG场景更稳定。

| 维度 | Baseline（视觉/NLP TTA） | 本文（NeuroAdapt-Bench） |
|:---|:---|:---|
| 核心假设 | 分布偏移主要体现为语义内容不变下的风格/外观变化 | EEG偏移体现为信号统计特性、通道配置的根本性变化 |
| 评估对象 | 通用CNN/ViT/语言模型 | 四个专用EEG基础模型（CBraMod/TFM-Tokenizer/REVE系列） |
| 方法选择 | 熵最小化类方法（Tent/SHOT）为主流 | 系统对比梯度类（Tent/SHOT）与无梯度类（T3A） |
| 关键发现 | TTA通常带来正向增益 | **梯度类TTA在EEG场景普遍导致性能退化**，亟需EEG专用策略 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b1e01c3f-6217-4aac-b34a-052e326e514f/figures/Figure_2.png)
*Figure 2: Figure 2: Overview of NeuroAdapt-Bench. The benchmark consists of three stages: (1)supervised finetuning of an EEG foundation model (e.g., REVE, CBRaMod, or TFM-Tokenizer) with a classification head o*



NeuroAdapt-Bench的评估框架分为三阶段流水线：

**阶段一：监督微调**。输入为预训练EEG基础模型（CBraMod、TFM-Tokenizer、REVE-Base、REVE-Large）与源域标注数据，输出为针对特定下游任务（如病理检测、睡眠分期）微调后的模型。此阶段建立No-TTA基线性能。

**阶段二：测试时适应**。输入为目标域无标签批次数据，通过三种TTA策略之一进行在线适应：
- **Tent路径**：仅更新BatchNorm层的仿射参数（γ, β），通过梯度下降最小化批次预测熵；
- **SHOT路径**：固定分类头，适应特征提取器，联合优化熵最小化、多样性正则与伪标签损失；
- **T3A路径**：完全冻结特征提取器，在线维护每类目标特征支持集，以均值原型动态更新分类器权重。

**阶段三：性能评估**。在分布内（TUEV、TUAB）、分布外（SleepEDF-78、CHB-MIT）及极端模态偏移（EESM23 Ear-EEG）三类场景下，对比TTA后模型与No-TTA基线的Balanced Accuracy、Cohen's κ等指标变化（∆TTA）。

```
预训练EEG模型 → [监督微调] → 微调模型 → [No-TTA推理] → 基线性能
                                    ↓
                              [目标域无标签数据]
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
                 [Tent]         [SHOT]          [T3A]
               (梯度更新BN)   (梯度更新特征)   (无梯度更新原型)
                    └───────────────┴───────────────┘
                                    ↓
                    [∆TTA = Perf_TTA − Perf_No-TTA]
                                    ↓
              分布内 / 分布外 / 极端模态偏移 三类评估
```

## 核心模块与公式推导

### 模块 1: Tent——熵最小化BN适应（对应框架图阶段二左侧）

**直觉**: 利用目标域批次数据的预测不确定性作为自监督信号，通过最小化熵迫使模型对目标数据做出自信预测，仅更新最轻量的归一化层参数以避免过拟合。

**Baseline 公式** (Wang et al., ICLR 2021): $$L_{\text{Tent}} = \frac{1}{|B_t|} \sum_{x \in B_t} H\left(p_\theta(\cdot|x)\right) = -\frac{1}{|B_t|} \sum_{x \in B_t} \sum_{c} p_\theta(c|x) \log p_\theta(c|x)$$
符号: $\theta$ = 模型参数（实际仅更新BN层 $\gamma, \beta$），$B_t$ = 目标域批次，$H$ = 香农熵，$p_\theta(c|x)$ = softmax预测概率。

**变化点**: 视觉领域假设目标域与源域共享语义空间，偏移主要为低阶统计量（光照、纹理）；EEG场景中通道配置、采样协议的根本差异导致该假设失效——强制降低熵可能使模型坍缩至错误的确定性预测。

**本文公式（推导）**:
$$\text{Step 1}: \hat{\mu}_t, \hat{\sigma}_t^2 \leftarrow \text{BatchNorm stats on } B_t \quad \text{用目标域统计替换源域统计}$$
$$\text{Step 2}: \gamma^*, \beta^* \leftarrow \text{arg}\min_{\gamma, \beta} L_{\text{Tent}} \quad \text{仅优化仿射参数}$$
$$\text{最终}: \theta_{\text{updated}} = \{\gamma^*, \beta^*\} \cup \theta_{\text{frozen}}$$

**对应消融**: Table 3（Figure 3a）显示Tent使CBraMod在TUEV上∆TTA为负，在SleepEDF-78上使CBraMod Bal.Acc.从0.514降至0.202（降幅61%）。

### 模块 2: SHOT——特征提取器适应（对应框架图阶段二中间）

**直觉**: 比Tent更激进的适应策略，固定分类头以保持语义一致性，同时通过多目标优化使特征提取器适应目标域分布。

**Baseline 公式** (Liang et al., ICML 2020): $$L_{\text{SHOT}} = L_{\text{ent}} + L_{\text{div}} + \beta L_{\text{PL}}$$
其中 $L_{\text{ent}} = \frac{1}{|B_t|}\sum H(p_\theta(\cdot|x))$（熵最小化），$L_{\text{div}}$（多样性正则，防止所有样本预测同一类），$L_{\text{PL}}$（基于伪标签的交叉熵损失）。

**变化点**: SHOT假设目标域类别分布与源域一致且分类头已学到可迁移的语义。EEG中跨任务语义鸿沟（癫痫检测→睡眠分期）和极端通道差异使伪标签质量极低，$L_{\text{PL}}$ 引入错误放大；同时特征提取器的梯度更新破坏了预训练学到的EEG特异性表征。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{y} = \text{arg}\max_c p_{\theta}(c|x) \quad \text{生成伪标签（EEG场景噪声极高）}$$
$$\text{Step 2}: L_{\text{PL}} = -\frac{1}{|B_t|}\sum_{x \in B_t} \sum_c \mathbb{1}[\tilde{y}=c] \log p_\theta(c|x) \quad \text{错误伪标签驱动错误更新}$$
$$\text{Step 3}: L_{\text{div}} = -\sum_c \hat{p}(c) \log \hat{p}(c), \quad \hat{p}(c) = \frac{1}{|B_t|}\sum_{x} p_\theta(c|x) \quad \text{多样性约束}$$
$$\text{最终}: \theta_{\text{feat}} \leftarrow \theta_{\text{feat}} - \eta \nabla_{\theta_{\text{feat}}} (L_{\text{ent}} + L_{\text{div}} + \beta L_{\text{PL}})$$

**对应消融**: Figure 4a显示SHOT使REVE-Large在SleepEDF-78上Bal.Acc.从0.651骤降至0.233（降幅64%），Cohen's κ从0.678降至0.038，为所有方法中最严重的退化。

### 模块 3: T3A——无梯度原型更新（对应框架图阶段二右侧）

**直觉**: 完全规避梯度更新带来的表征破坏风险，通过在线维护目标域特征支持集并更新类别原型，实现"安全"的轻量适应。

**Baseline 公式** (Iwasawa & Matsuo, NeurIPS 2021): $$\hat{\omega}_k^{(t)} = \frac{1}{|S_k^{(t)}|} \sum_{z \in S_k^{(t)}} z, \quad p(c|x) = \frac{\exp(-\|z - \hat{\omega}_c\|^2)}{\sum_{c'}\exp(-\|z - \hat{\omega}_{c'}\|^2)}$$
符号: $z = f_\theta(x)$ = 冻结特征提取器输出的特征，$S_k^{(t)}$ = 时刻$t$类别$k$的目标域支持集，$\hat{\omega}_k$ = 类别原型。

**变化点**: T3A在视觉领域作为计算高效的替代方案；在EEG领域，其无梯度特性成为关键优势——避免了梯度类方法的退化解，但支持集构建依赖初始预测的可靠性，在严重偏移场景下仍可能累积错误。

**本文公式（推导）**:
$$\text{Step 1}: z = f_{\theta_{\text{frozen}}}(x), \quad c^* = \text{arg}\max_c p_{\theta_{\text{init}}}(c|x) \quad \text{冻结特征，获取置信预测}$$
$$\text{Step 2}: S_k^{(t)} \leftarrow S_k^{(t-1)} \cup \{z \text{mid} c^*=k, p(c^*|x) > \tau\} \quad \text{高置信样本加入支持集}$$
$$\text{Step 3}: \hat{\omega}_k^{(t)} = \frac{1}{|S_k^{(t)}|} \sum_{z \in S_k^{(t)}} z \quad \text{在线更新原型（指数移动平均变体）}$$
$$\text{最终}: p(c|x) \propto \exp\left(-\frac{\|z - \hat{\omega}_c^{(t)}\|^2}{2\sigma^2}\right) \quad \text{基于更新原型的最近邻分类}$$

**对应消融**: Figure 4b显示T3A在CHB-MIT上为REVE-Large提供一致正向增益；在极端模态偏移Ear-EEG下，T3A是唯一在部分模型上带来改善的方法（REVE-Large Bal.Acc. 0.400→0.422）。但T3A在TUAB上同样退化，在SleepEDF-78上仅边际增益，说明其优势存在边界条件。

## 实验与分析

主结果汇总（∆TTA相对No-TTA基线）：

| Method | TUEV (ID) | TUAB (ID) | SleepEDF-78 (OOD) | CHB-MIT (OOD) | Ear-EEG (Extreme) |
|:---|:---|:---|:---|:---|:---|
| Tent | 退化 | 退化 | 显著退化 | 退化 | 退化 |
| SHOT | 显著退化 | 退化 | **严重退化** | 退化 | 退化 |
| T3A | **边际增益** | 退化 | 边际增益 | **正向增益** | **部分增益** |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b1e01c3f-6217-4aac-b34a-052e326e514f/figures/Figure_3.png)
*Figure 3: Figure 3: TTA relative performance on in-distribution datasets (TUEV and TUAB). (a)∆TTA on TUEV relative to the No-TTA baseline; (b) ∆TTA on TUAB relative to the No-TTA baseline.*



**核心发现分析**：

梯度类方法（Tent、SHOT）的系统性退化是最突出的结果。在分布内数据集TUEV上，SHOT使CBraMod Bal.Acc.从0.378降至0.168；在分布外SleepEDF-78上，退化幅度随偏移强度加剧——Tent使CBraMod降幅61%（0.514→0.202），SHOT使REVE-Large降幅64%（0.651→0.233）且Cohen's κ从0.678崩溃至0.038。这一模式在Figure 3和Figure 4中高度一致，置信度达0.95-0.97。

T3A的相对稳定性在CHB-MIT和极端模态偏移场景下得到验证，但存在明显边界条件：TUAB上同样退化，SleepEDF-78上增益边际。这暗示T3A的有效性依赖于目标域与源域在特征空间中仍存在可分离的类别结构——当偏移过于剧烈时，初始伪标签质量不足以致支持集构建失败。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b1e01c3f-6217-4aac-b34a-052e326e514f/figures/Figure_4.png)
*Figure 4: Figure 4: TTA relative performance on out-of-distribution datasets (SLEEPEDF-78 andCHB-MIT). (a) ∆TTA on SLEEPEDF-78 relative to the No-TTA baseline; (b) ∆TTA onCHB-MIT relative to the No-TTA baseline*



**架构敏感性**：TFM-Tokenizer对TTA方法表现出相对更强鲁棒性，在SleepEDF-78上梯度类方法退化幅度明显小于CBraMod和REVE系列。这提示模型架构设计（如基于tokenizer的离散表征 vs. 连续卷积特征）对TTA稳定性有重要影响，为后续EEG基础模型设计提供参考。

**公平性检查**：基线选择方面，本文仅评估三种经典TTA方法，未包含SAR、CoTTA、TENT++等更新进展，结论普适性受限。计算成本上，T3A无梯度特性使其推理开销最低，SHOT因需反向传播更新特征提取器成本最高。失败案例方面，所有TTA方法在TUAB上均未取得正向增益，提示该数据集的偏移特性可能超出当前TTA范式的处理能力。多seed标准差显示部分结论方差较大，尤其是梯度类方法的退化幅度存在波动。

## 方法谱系与知识库定位

本文属于**测试时适应（TTA）**方法家族的**诊断性/评估性研究**，而非新算法提出。其方法论谱系可追溯至：

- **父方法**: Tent（Wang et al., ICLR 2021）——熵最小化BN适应；SHOT（Liang et al., ICML 2020）——特征提取器适应+伪标签；T3A（Iwasawa & Matsuo, NeurIPS 2021）——无梯度原型更新。
- **领域迁移**: 将视觉/NLP TTA范式首次系统引入EEG基础模型部署场景。

**改动槽位**: 
- **评估对象**: 从通用视觉模型 → 专用EEG基础模型（CBraMod/TFM-Tokenizer/REVE）
- **数据场景**: 从单一分布偏移 → 三层结构（分布内/分布外/极端模态偏移）
- **核心结论**: 从"TTA有效" → "标准TTA在EEG失效，需领域专用策略"

**直接基线差异**：
- vs. 视觉TTA综述（如Niu et al., 2023）: 本文聚焦EEG领域特性，发现与视觉领域相反的结论
- vs. EEG领域自适应工作（如DANN-based方法）: 本文聚焦无源数据、仅测试时适应的更严格设定
- vs. EEG基础模型原始论文（CBraMod/REVE）: 本文首次系统评估其部署时的分布鲁棒性

**后续方向**:
1. **EEG专用TTA算法**: 设计考虑通道拓扑结构和信号非平稳性的自适应机制，如频域感知的原型更新
2. **模型架构-TTA协同设计**: 基于TFM-Tokenizer的鲁棒性发现，探索 inherently TTA-friendly 的EEG基础模型架构
3. **连续适应与灾难性遗忘**: 借鉴CoTTA/SAR的连续适应框架，解决EEG长期部署中的时序分布漂移

**知识库标签**: 模态:EEG/神经信号 | 范式:测试时适应/TTA | 场景:医疗AI部署/跨域泛化 | 机制:熵最小化/原型学习/无梯度适应 | 约束:无源数据/隐私保护/在线适应

