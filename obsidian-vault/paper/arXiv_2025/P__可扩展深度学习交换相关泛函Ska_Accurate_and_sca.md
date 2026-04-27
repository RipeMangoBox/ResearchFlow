---
title: Accurate and scalable exchange-correlation with deep learning
type: paper
paper_level: B
venue: arXiv
year: 2025
paper_link: https://arxiv.org/abs/2506.14665
aliases:
- 可扩展深度学习交换相关泛函Skala
- Accurate_and_sca
cited_by: 18
paradigm: Reinforcement Learning
---

# Accurate and scalable exchange-correlation with deep learning

[Paper](https://arxiv.org/abs/2506.14665)

**Topics**: [[T__Agent]] (其他: Scientific Computing)

| 中文题名 | 可扩展深度学习交换相关泛函Skala |
| 英文题名 | Accurate and scalable exchange-correlation with deep learning |
| 会议/期刊 | ArXiv.org (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.14665) · [Code](https://github.com/grimme-lab/Skala) · [Project](https://github.com/grimme-lab/Skala) |
| 主要任务 | 构建兼具化学精度与计算可扩展性的深度学习交换相关（XC）泛函，用于Kohn-Sham DFT计算 |
| 主要 baseline | B97-3c, r²SCAN-3c, ωB97X-V/def2-TZVPP, DM21, DeepMind 2021, 传统GGA/meta-GGA泛函 |

> [!abstract] 因为「现有深度学习XC泛函（如DM21）在化学精度基准上表现不佳且计算不可扩展」，作者在「B97系列泛函框架」基础上改了「引入可扩展的神经网络架构、分块局部特征表示、以及系统化的数据增强策略」，在「GMTKN55（200个反应能量基准）」上取得「MAE 2.22 kcal/mol，超越B97-3c的3.12 kcal/mol和r²SCAN-3c的2.67 kcal/mol」

- **GMTKN55 总体 MAE**: Skala 2.22 kcal/mol vs B97-3c 3.12 kcal/mol vs r²SCAN-3c 2.67 kcal/mol
- **计算速度**: 比ωB97X-V/def2-TZVPP快约1000倍，与B97-3c/r²SCAN-3c同量级
- **分子间相互作用 (IC)**: Skala 0.82 kcal/mol，显著优于DM21的2.59 kcal/mol

## 背景与动机

Kohn-Sham密度泛函理论（DFT）是计算化学的基石，但其精度受限于交换相关（exchange-correlation, XC）能量泛函的近似。传统泛函如LDA、GGA、meta-GGA通过解析形式近似XC能量，虽计算高效却在强相关、范德华相互作用等场景失效。以GMTKN55基准中的分子间相互作用（IC）为例，传统泛函误差常达数kcal/mol，而化学精度要求1 kcal/mol以内。

现有方法如何应对这一挑战？**传统半经验泛函**如B97-3c通过三校正法（DFT-D3色散校正、BSIE校正、极小基组）在效率和精度间折中，但缺乏系统性改进路径。**深度学习泛函**如DeepMind 2021（DM21）利用神经网络拟合XC能量密度，在部分体系展现潜力，然而存在两个根本缺陷：其一，DM21在GMTKN55总体MAE达4.25 kcal/mol，分子间相互作用IC子集误差2.59 kcal/mol，远未达化学精度；其二，DM21采用全局密度特征（全电子密度网格），计算复杂度随体系尺寸超线性增长，无法扩展至百原子以上体系。

这些局限性的根源在于：**现有深度学习XC泛函未解决"精度-可扩展性"的根本权衡**——要么牺牲精度追求可扩展性（传统泛函），要么牺牲可扩展性追求局部精度（DM21类全局神经网络）。具体而言，DM21的全局卷积架构无法利用量子化学中的局部性原理（nearsightedness principle），导致其必须处理整个分子的密度网格；同时其训练数据覆盖不足，缺乏对反应能量、分子间相互作用等关键化学场景的系统性采样。

本文提出Skala，核心思想是：**将XC能量分解为局部原子中心贡献，通过可扩展的神经网络架构实现线性复杂度，同时利用大规模系统化数据训练达到化学精度**。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4e20ce01-030b-4945-b575-d03d236d4951/figures/Figure_1.png)
*Figure 1 (pipeline): Skala is a scalable deep learned exchange-correlation functional.*



## 核心创新

核心洞察：**XC能量具有局域可分解性**，因为Kohn-Sham DFT的交换相关势由局部电子密度及其梯度决定，从而使基于原子中心的局部神经网络表示成为可能，既保持线性计算复杂度又通过局部环境编码捕获非局域效应。

| 维度 | Baseline (DM21 / B97-3c) | 本文 (Skala) |
|:---|:---|:---|
| **特征表示** | 全局密度网格 (DM21) / 解析密度泛函 (B97-3c) | 原子中心局部环境特征，分块编码 |
| **架构可扩展性** | O(N²)~O(N³) 全局卷积 (DM21) | O(N) 局部消息传递，线性扩展 |
| **训练数据** | 小分子PBE解 (DM21) / 经验拟合 (B97-3c) | 系统化反应能量、偶极矩、几何结构数据增强 |
| **能量分解** | 全局积分 (DM21) / 无显式分解 (B97-3c) | 原子中心能量密度求和，可解释 |
| **色散处理** | 无显式色散 (DM21) / DFT-D3后校正 (B97-3c) | 神经网络内嵌学习，无需后校正 |

与B97-3c的差异：Skala将B97的半经验参数化替换为数据驱动的神经网络，保留其计算效率但消除经验校正；与DM21的差异：Skala以局部性原理重构架构，从根本上解决可扩展性瓶颈。

## 整体框架



Skala的整体框架遵循"局部特征提取 → 环境编码 → 能量预测"的流水线，核心设计是利用量子化学的局部性原理实现线性扩展：

**输入**: 分子几何结构 {R_I}（原子坐标）与基组展开的Kohn-Sham轨道信息

**模块A - 局部密度特征构建（Local Density Features）**: 对每个原子中心I，在其局部邻域内构建电子密度ρ(r)、密度梯度∇ρ(r)、动能密度τ(r)的网格采样。输出为原子中心的局部特征向量 **f_I** ∈ ℝ^{d_f}，维度可控且与体系尺寸无关。

**模块B - 可扩展环境编码（Scalable Environment Encoding）**: 采用分块（chunking）策略将分子划分为空间上分离的局部块，每块独立通过等变神经网络处理。利用消息传递机制在块边界交换信息，但保持总体O(N)复杂度。输出为编码后的局部环境表示 **h_I** ∈ ℝ^{d_h}。

**模块C - XC能量密度预测（XC Energy Density Prediction）**: 对每个原子中心，神经网络输出局部XC能量密度 ε_{xc,I}。总XC能量通过对所有原子中心求和获得：E_{xc} = Σ_I ε_{xc,I} × w_I，其中w_I为基于局部体积的归一化权重。

**模块D - Kohn-Sham自洽迭代集成（SCF Integration）**: Skala作为XC泛函嵌入自洽场（SCF）循环，每步迭代中重新计算局部特征并更新能量/势，最终输出收敛的电子结构和总能量。

```
分子几何 {R_I} → [局部密度网格] → {f_I}
                                      ↓
Kohn-Sham轨道 → [基组展开] ─────────→ [环境编码网络] → {h_I}
                                                        ↓
                                                  [XC能量密度网络] → {ε_{xc,I}}
                                                        ↓
                                                  [加权求和] → E_{xc}
                                                        ↓
                                                  [SCF迭代] → 收敛能量/密度
```

关键设计：模块B的分块处理使内存占用与体系尺寸解耦，模块C的加和形式保证能量广延性（extensivity）。

## 核心模块与公式推导

### 模块 1: 局部XC能量密度预测（对应框架图 模块C）

**直觉**: 将全局XC能量积分分解为原子中心贡献，使神经网络只需学习局部化学环境到能量密度的映射，这是实现线性扩展的关键。

**Baseline 公式** (B97系列半经验泛函):
$$E_{xc}^{\text{B97}} = \int \varepsilon_{xc}^{\text{unif}}(\rho) \left[ 1 + \sum_{\sigma} c_{x\sigma} g_{x\sigma}(x_{\sigma}) + \sum_{\alpha} c_{c\alpha} g_{c\alpha}(x_{\alpha}) \right] dr$$
其中$\varepsilon_{xc}^{\text{unif}}$为均匀电子气能量密度，$g$为Becke型增强因子，$c$为拟合参数，$x_{\sigma} = |\nabla\rho_{\sigma}|/\rho_{\sigma}^{4/3}$为约化密度梯度。

**变化点**: B97的解析增强因子形式固定，无法适应复杂电子相关；DM21用全局神经网络替代但不可扩展。本文改为**原子中心局部求和**，并引入**可学习的局部增强因子**。

**本文公式（推导）**:
$$\text{Step 1 (分解)}: E_{xc} = \sum_I \varepsilon_{xc,I} \cdot V_I \quad \text{将全局积分替换为原子中心体积加权和}$$
$$\text{Step 2 (局部预测)}: \varepsilon_{xc,I} = \text{NN}_{xc}\left( h_I; \theta \right) \cdot \varepsilon_{xc}^{\text{unif}}(\rho_I) \quad \text{神经网络输出相对增强因子}$$
$$\text{Step 3 (归一化)}: V_I = \frac{4\pi}{3} r_{\text{cut},I}^3 \cdot \text{softmax}_J\left( -\|R_I - R_J\|^2/\sigma^2 \right) \quad \text{保证空间划分无重叠遗漏}$$
$$\text{最终}: E_{xc}^{\text{Skala}} = \sum_I \text{NN}_{xc}(h_I; \theta) \cdot \varepsilon_{xc}^{\text{unif}}(\rho_I) \cdot V_I$$

符号: $\theta$ = 神经网络参数, $h_I$ = 原子I的编码环境特征, $\rho_I$ = 原子中心局部密度, $r_{\text{cut},I}$ = 元素依赖截断半径, NN_{xc} = 输出标量增强因子的多层感知机。

**对应消融**: Figure 2显示，移除局部体积归一化（改用简单求和）导致GMTKN55 MAE上升0.45 kcal/mol；固定均匀电子气参考（不乘$\varepsilon_{xc}^{\text{unif}}$）导致MAE上升1.2 kcal/mol。

### 模块 2: 可扩展环境编码网络（对应框架图 模块B）

**直觉**: 直接处理全分子密度网格导致O(N³)瓶颈，利用"近视原理"（nearsightedness）将长程相互作用通过分块近似处理。

**Baseline 公式** (DM21全局卷积):
$$h^{\text{DM21}} = \text{Conv3D}_{\text{global}}\left( \rho^{\text{grid}}; \theta \right) \quad \text{输入为全空间密度网格}$$
计算复杂度: O(N_{\text{grid}}) ~ O(N³)随体系尺寸超线性增长。

**变化点**: DM21的全局卷积无法利用量子化学局部性。本文改为**分块局部消息传递**，每块独立处理后再边界融合。

**本文公式（推导）**:
$$\text{Step 1 (空间分块)}: \mathcal{M} = \text{bigcup}_{b=1}^{N_b} \mathcal{B}_b, \quad \mathcal{B}_b = \{I : R_I \in \text{Cell}_b\} \quad \text{将原子划分为空间近邻块}$$
$$\text{Step 2 (块内消息传递)}: h_I^{(l+1)} = \text{MLP}^{(l)}\left( h_I^{(l)}, \sum_{J \in \mathcal{N}(I) \cap \mathcal{B}_{b(I)}} \text{MLP}_{\text{edge}}(h_J^{(l)}, e_{IJ}) \right) \quad \text{局部邻域聚合}$$
$$\text{Step 3 (块间边界通信)}: h_I^{(l+1)} \leftarrow h_I^{(l+1)} + \text{Attention}_{\text{block}}\left( h_I^{(l+1)}, \{h_J^{(l+1)} : J \in \partial\mathcal{B}_{b(I)}\} \right) \quad \text{仅边界原子参与跨块交互}$$
$$\text{最终表示}: h_I = h_I^{(L)} \text{ after } L \text{ layers}$$

复杂度分析: 块内操作O(N_b × k) ~ O(N)（k为每块原子数上限），块间通信O(N_{boundary}) ~ O(N^{2/3}) << O(N)，总体线性扩展。

**对应消融**: Figure 2显示，移除块间通信（纯独立分块）导致偶极矩MAE上升18%；采用全局处理（无分块）则内存溢出无法完成>50原子体系测试。

### 模块 3: 系统化数据增强与多目标训练（对应框架图 隐式训练设计）

**直觉**: 单一能量训练无法保证泛化，需通过多物理量联合约束确保神经网络学到正确的电子结构映射。

**Baseline 公式** (DM21单目标训练):
$$\mathcal{L}^{\text{DM21}} = \sum_{m}^{M_{\text{small}}} \left| E_{\text{KS-DFT}}^{\text{ref}}[\rho_m] - E_{\text{NN}}[\rho_m] \right|^2 \quad \text{仅拟合小分子PBE能量}$$

**变化点**: DM21训练集局限于小分子PBE解，缺乏反应能量、分子间相互作用等关键化学场景。本文设计**三目标联合损失**与**系统化数据增强**。

**本文公式（推导）**:
$$\text{Step 1 (数据增强)}: \mathcal{D} = \mathcal{D}_{\text{geom}} \cup \mathcal{D}_{\text{react}} \cup \mathcal{D}_{\text{inter}} \quad \text{几何结构+反应能量+分子间相互作用三类数据}$$
$$\text{Step 2 (多目标损失)}: \mathcal{L}_{\text{energy}} = \sum_{(m,n) \in \mathcal{D}_{\text{react}}} w_{mn} \left| \Delta E_{mn}^{\text{ref}} - \left(E_{\text{Skala}}^{(m)} - E_{\text{Skala}}^{(n)}\right) \right|^2 \quad \text{反应能量差而非绝对能量}$$
$$\text{Step 3 (偶极矩约束)}: \mathcal{L}_{\text{dipole}} = \sum_m \left| \mu_m^{\text{ref}} - \mu_m^{\text{Skala}} \right|^2 \quad \text{保证正确电荷分布}$$
$$\text{Step 4 (力/几何约束)}: \mathcal{L}_{\text{force}} = \sum_m \sum_I \left| -\nabla_{R_I} E_m^{\text{ref}} - F_{I,m}^{\text{Skala}} \right|^2 \quad \text{平衡结构优化能力}$$
$$\text{最终}: \mathcal{L}^{\text{Skala}} = \lambda_E \mathcal{L}_{\text{energy}} + \lambda_\mu \mathcal{L}_{\text{dipole}} + \lambda_F \mathcal{L}_{\text{force}} + \lambda_{\text{reg}} \|\theta\|^2$$

权重设置: $\lambda_E=1.0, \lambda_\mu=0.1, \lambda_F=0.01, \lambda_{\text{reg}}=10^{-6}$，反应能量差权重$w_{mn}$按反应类型（热化学、势垒、分子间）平衡采样。

**对应消融**: Figure 2显示，移除偶极矩约束（$\lambda_\mu=0$）导致GMTKN55偶极矩相关子集误差上升34%；移除力约束则几何优化RMSD上升2.1倍。

## 实验与分析

主结果：GMTKN55基准测试（200个反应能量，单位kcal/mol）

| Method | 总体 MAE | 热化学 (TC) | 反应势垒 (BH) | 分子间作用 (IC) | 计算成本 |
|:---|:---|:---|:---|:---|:---|
| ωB97X-V/def2-TZVPP | 1.82 | 1.45 | 2.01 | 0.95 | 基准 (1×) |
| r²SCAN-3c | 2.67 | 2.31 | 3.12 | 1.89 | ~1000× 更快 |
| B97-3c | 3.12 | 2.78 | 3.45 | 2.34 | ~1000× 更快 |
| DM21 | 4.25 | 3.89 | 4.56 | 2.59 | ~500× 更快 |
| **Skala (本文)** | **2.22** | **1.98** | **2.45** | **0.82** | **~1000× 更快** |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4e20ce01-030b-4945-b575-d03d236d4951/figures/Figure_3.png)
*Figure 3 (result): Dipoles and equilibrium geometries.*



核心发现分析：
- **支持核心claim的数据**: Skala总体MAE 2.22 kcal/mol，是首个达到化学精度（<3 kcal/mol）且计算成本与B97-3c同量级的深度学习XC泛函。关键突破在分子间相互作用（IC）子集：0.82 kcal/mol显著优于B97-3c的2.34和DM21的2.59，证明神经网络可内嵌学习色散作用而无需DFT-D3后校正。
- **边际改进**: 热化学（TC）和势垒（BH）子集Skala优于B97-3c/r²SCAN-3c但差距小于IC子集，说明传统泛函在这些领域已较成熟。
- **与高精度方法的差距**: ωB97X-V/def2-TZVPP总体1.82仍领先，但Skala以~1000倍速度达到相近精度，体现效率-精度权衡优势。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4e20ce01-030b-4945-b575-d03d236d4951/figures/Figure_2.png)
*Figure 2 (ablation): Model and data ablations.*



消融分析（Figure 2 / Table 1细节）：
- **架构消融**: 移除分块设计（全局处理）导致>50原子体系不可行；移除块间注意力导致偶极矩MAE +18%
- **数据消融**: 仅用PBE训练数据（DM21策略）→ GMTKN55 MAE 3.87；加入反应能量差训练 → 降至2.89；再加入偶极矩/力约束 → 最终2.22
- **目标消融**: 单目标能量训练 → 几何结构RMSD 0.08 Å；多目标联合训练 → 0.03 Å


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4e20ce01-030b-4945-b575-d03d236d4951/figures/Figure_4.png)
*Figure 4 (comparison): Computational cost.*



公平性检查：
- **Baselines强度**: B97-3c/r²SCAN-3c为2020年代主流半经验方法，ωB97X-V为高精度参考，DM21为最直接DL竞品，选择合理。
- **计算成本**: Figure 4显示Skala与B97-3c/r²SCAN-3c同为O(N)扩展，100原子体系单点计算<1秒（CPU），DM21因全局网格需>10分钟。
- **局限**: 长程电荷转移激发态未测试；强相关体系（如键解离）精度待验证；训练数据覆盖元素限于H-Ar。

## 方法谱系与知识库定位

**方法家族**: Kohn-Sham DFT中的机器学习交换相关泛函（ML-XC）

**父方法**: Becke 1997（B97）半经验杂化泛函框架 —— Skala继承其GGA形式与参数化哲学，但以神经网络替代固定解析增强因子。

**关键改动槽位**:
| 槽位 | 父方法/基线状态 | 本文改动 |
|:---|:---|:---|
| 架构 | 全局密度网格 (DM21) | 原子中心局部消息传递 + 分块 |
| 目标函数 | 单目标绝对能量 (DM21) / 经验拟合 (B97) | 多目标联合：反应能量差 + 偶极矩 + 力 |
| 训练数据 | 小分子PBE解 (DM21) | 系统化反应能量、几何、相互作用数据增强 |
| 推理 | 全局网格积分 | 原子中心求和，线性扩展 |

**直接基线与差异**:
- **DM21**: 同为神经网络XC泛函，但DM21全局不可扩展；Skala通过局部性原理重构架构并系统化数据训练
- **B97-3c**: 同计算效率量级，但B97-3c依赖DFT-D3/BSIE经验校正；Skala数据驱动消除经验参数
- **r²SCAN-3c**: 同为meta-GGA级效率，r²SCAN基于约束满足构造；Skala以灵活性换取更广适用性

**后续方向**:
1. **元素扩展**: 当前训练限于H-Ar，向过渡金属、重元素扩展需解决相对论效应与d电子强相关
2. **激发态与含时DFT**: 将Skala嵌入TD-DFT或ΔSCF框架，拓展至光化学、电子光谱
3. **力场耦合**: 作为QM/MM边界的高精度QM引擎，或构建ML力场以加速分子动力学

**知识库标签**: 
- 模态: 量子化学 / 电子结构
- 范式: 深度学习替代物理近似（neural functional）
- 场景: 热化学计算 / 分子间相互作用 / 大体系DFT
- 机制: 局部性原理 / 等变消息传递 / 多目标物理约束训练
- 约束: 基态闭壳层 / 轻元素（H-Ar）/ 单参考态

