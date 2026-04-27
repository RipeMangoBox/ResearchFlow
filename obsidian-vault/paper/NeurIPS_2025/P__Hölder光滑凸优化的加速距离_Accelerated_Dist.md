---
title: Accelerated Distance-adaptive Methods for Hölder Smooth and Convex Optimization
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- Hölder光滑凸优化的加速距离自适应方法
- Accelerated Dist
- Accelerated Distance-adaptive Method (ADM)
- We propose parameter-free accelerat
acceptance: Poster
method: Accelerated Distance-adaptive Method (ADM)
modalities:
- Text
---

# Accelerated Distance-adaptive Methods for Hölder Smooth and Convex Optimization

**Topics**: [[T__Reasoning]] | **Method**: [[M__Accelerated_Distance-adaptive_Method]] | **Datasets**: Softmax regression, Matrix games, Lp norm minimization

> [!tip] 核心洞察
> We propose parameter-free accelerated distance-adaptive methods that achieve optimal anytime convergence rates for Hölder smooth convex optimization without prior knowledge of smoothness parameters, target accuracy specification, or line-search procedures.

| 中文题名 | Hölder光滑凸优化的加速距离自适应方法 |
| 英文题名 | Accelerated Distance-adaptive Methods for Hölder Smooth and Convex Optimization |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.0xxxx) · Code (未开源) · Project (未提供) |
| 主要任务 | 凸优化 / 无参数优化 (parameter-free optimization) |
| 主要 baseline | Distance-over-Gradient (DOG), Universal Fast Gradient Method (Nesterov, 2015), Fast Gradient Method (FGM), D-adaptation, DAda |

> [!abstract] 因为「一阶凸优化方法需要手动调节步长且通用梯度方法需预先指定目标精度 ε」，作者在「Distance-over-Gradient (DOG)」基础上改了「将距离自适应机制与 Nesterov 加速结合，消除目标精度预设与线搜索需求」，在「softmax 回归、矩阵博弈、Lp 范数最小化」上取得「最优任意时间收敛速率 O(1/k²)，自动适应 Hölder 光滑参数 ν∈[0,1]」

- **理论保证**：ADM 达到最优收敛速率 O(L_ν^{2/(1+ν)} D_0^{2/(1+ν)} / k²)，无需预先知道 ν, L_ν 或目标精度 ε
- **方法创新**：首个将 DOG 距离自适应与 Nesterov 加速结合，并扩展至 Hölder 光滑复合优化
- **实验验证**：在矩阵博弈、softmax 回归、Lp 范数最小化上验证了对非光滑与弱光滑结构的自适应能力

## 背景与动机

一阶凸优化方法的实际部署长期受困于步长调参：使用者必须预先估计光滑常数 L、强凸参数 μ，或至少设定目标精度 ε。以训练一个带 L1 正则化的 logistic 回归为例，若将步长设得过大则发散，过小则收敛缓慢，而问题本身的光滑性可能随数据分布未知变化。

现有三类方法试图缓解这一痛点：

- **Universal Fast Gradient Method (Nesterov, 2015)**：通过预设目标精度 ε 构造步长调度，理论上可处理 Hölder 光滑（ν∈[0,1]）的统一框架，但 ε 必须在迭代前指定，且实际收敛对 ε 的选择敏感。
- **Distance-over-Gradient (DOG)**：Ivgi et al. 提出的无参数 SGD 步长规则 η_k = r̄_k / √(∑‖g_i‖²)，利用到初始点的距离自动适应问题几何，但仅限随机梯度设定，缺乏加速机制与复合优化（proximal）扩展。
- **D-adaptation / DAda**：近期无参数方法通过估计初始距离 D_0 自适应学习率，但未与 Nesterov 加速结合，也未覆盖完整的 Hölder 光滑谱。

这些方法的共同缺口在于：**加速、无参数、Hölder 光滑自适应三者未能同时实现**。Nesterov 通用方法有最优速率但需 ε；DOG 无参数但无加速且仅限 SGD；DAda 距离自适应但未加速。本文旨在填补这一空白，提出 Accelerated Distance-adaptive Method (ADM)，在单一框架内同时实现最优加速收敛、完全无参数、以及对 ν∈[0,1] 全谱 Hölder 光滑的自动适应。

## 核心创新

核心洞察：距离自适应步长可替代目标精度 ε 作为加速算法的"尺度锚点"，因为 DOG 式距离估计 r̄_k 隐式编码了问题局部几何与梯度累积信息，从而使 Nesterov 估计序列技术无需预设 ε 即可保持最优收敛速率成为可能。

| 维度 | Baseline (Nesterov Universal / DOG) | 本文 ADM |
|:---|:---|:---|
| 步长调度 | 依赖预设 ε 的通用序列 / 纯 SGD 无加速 | 距离自适应 η_k = r̄_k / √(∑‖g_i‖²)，嵌入加速框架 |
| 光滑性假设 | 需显式知 ν 或分别处理 ν=0/1 | 隐式适应任意 ν∈[0,1]，统一光滑-非光滑 |
| 线搜索 | 通用方法常需线搜索确定局部 L_ν | 完全消除线搜索，每步闭式计算 |
| 收敛保证 | O(L_ν^{2/(1+ν)} D_0² / (ε^{(1-ν)/(1+ν)} k^{(1+3ν)/(1+ν)}) + ε) | O(L_ν^{2/(1+ν)} D_0^{2/(1+ν)} / k²)，任意时间最优 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d3251acd-d545-4cd8-80ac-202e5b1135a2/figures/Figure_1.png)
*Figure 1 (result): Performance of compared algorithms. Left: submodular test on simulated dataset. Middle: matrix game problem with α = 1. Right: Matrix game of case (III) with α = 0.5.*



ADM 的完整数据流遵循"初始化 → 梯度查询 → 距离估计更新 → 自适应步长计算 → 加速动量外推 → 邻近投影"的循环结构：

1. **初始化模块**：接收初始点 x_0 与邻近算子 prox_{ηg}(·)，维护初始距离估计 r̄_0 = ‖x_0 - x_*‖ 的乐观上界。

2. **梯度计算模块**：在第 k 步计算 f 的（随机）梯度 g_k = ∇f(y_k) 或次梯度，其中 y_k 为动量外推点。

3. **距离估计更新模块**：在线更新 r̄_{k+1} = max{r̄_k, ‖x_{k+1} - x_0‖}，保证对真实初始距离 D_0 = ‖x_0 - x_*‖ 的上界估计单调不减。

4. **自适应步长计算模块**（核心创新）：按 DOG 规则计算 η_k = r̄_k / √(∑_{i=0}^k ‖g_i‖²)，该步长自动缩放至问题局部几何，无需 L_ν, ν, ε。

5. **加速动量更新模块**：执行 Nesterov 风格的估计序列更新，将距离自适应步长嵌入三点格式：y_k = x_k + τ_k(x_k - x_{k-1})，保持动量累积的加速效应。

6. **邻近算子投影模块**：处理非光滑正则项 g(x)，执行 x_{k+1} = prox_{η_k g}(y_k - η_k g_k)。

迭代流程可概括为：
```
x_0 → [梯度 g_k] → [更新 r̄_k] → [计算 η_k] → [动量外推 y_k] → [prox 投影] → x_{k+1}
         ↑___________________________________________________________↓
```

该框架的关键在于：距离估计 r̄_k 同时充当（a）步长分子提供尺度感知，（b）Lyapunov 分析中的势能项保证收敛，（c）消除对 ε 的显式依赖。

## 核心模块与公式推导

### 模块 1: Hölder 光滑性建模与上界估计（对应框架"梯度计算"位置）

**直觉**：统一光滑（ν=1）与非光滑（ν=0）分析，需要一种连续插值的光滑性度量，使算法能自动感知问题处于谱系的哪个位置。

**Baseline 公式** (Nesterov Universal Method)：
$$\|\nabla f(x) - \nabla f(y)\|_* \leq L_1 \|x - y\|, \quad \text{或} \quad f \in C^{1,1}$$
符号: $L_1$ = Lipschitz 光滑常数, $\|\cdot\|_*$ = 对偶范数。

**变化点**：标准分析仅限 ν=1 或退化到 ν=0（有界次梯度），无法处理中间弱光滑情形（如 0<ν<1 的 Hölder 函数）。

**本文公式（推导）**：
$$\text{Step 1}: \|\nabla f(x) - \nabla f(y)\|_* \leq L_\nu \|x - y\|^\nu, \quad \nu \in [0,1] \quad \text{(Hölder 条件，统一全谱)}$$
$$\text{Step 2}: f(y) \leq f(x) + \langle \nabla f(x), y-x \rangle + \frac{L_\nu}{1+\nu}\|y-x\|^{1+\nu} \quad \text{(Taylor 型上界，积分余项)}$$
$$\text{最终}: \psi(x) := f(x) + g(x), \quad \min_{x \in \mathbb{R}^d} \psi(x) \quad \text{(复合目标，g 有易 prox)}$$

**对应消融**：去掉 Hölder 统一建模（退化为纯光滑或非光滑假设）会导致对 ν∈(0,1) 问题收敛速率次优。

---

### 模块 2: 距离自适应步长规则（对应框架"自适应步长计算"位置）

**直觉**：梯度累积范数反映问题"难度"，到初始点的距离反映"尺度"，二者之比构成无需问题参数的自然步长。

**Baseline 公式** (Nesterov Universal FGM)：
$$\eta_k = \eta(\varepsilon, L_\nu, \nu) = c \cdot \varepsilon^{\frac{1-\nu}{1+3\nu}} L_\nu^{-\frac{2}{1+3\nu}}$$
符号: $\varepsilon$ = 预设目标精度, $c$ = 与问题维度相关的常数。

**变化点**：Nesterov 通用方法的步长显式依赖 ε，且需知 L_ν 与 ν；实际中 ε 选错导致提前饱和或浪费迭代。

**本文公式（推导）**：
$$\text{Step 1}: \bar{r}_{k+1} = \max\{\bar{r}_k, \|x_{k+1} - x_0\|\} \quad \text{(在线距离估计，乐观上界更新)}$$
$$\text{Step 2}: \eta_k = \frac{\bar{r}_k}{\sqrt{\sum_{i=0}^k \|g_i\|^2}} \quad \text{(DOG 式步长，分子为尺度，分母为累积梯度"难度")}$$
$$\text{Step 3}: \text{结合加速}: y_k = x_k + \tau_k(x_k - x_{k-1}), \quad x_{k+1} = \text{prox}_{\eta_k g}(y_k - \eta_k \nabla f(y_k))$$
$$\text{最终}: \eta_k \text{ 自动适应 } L_\nu^{\frac{2}{1+\nu}} D_0^{\frac{2}{1+\nu}} \text{ 的隐式尺度，无需显式估计}$$

**对应消融**：Table 2 显示移除距离自适应（改用固定 ε-依赖步长）导致需预设 ε 且对 ε 敏感；Figure 1-3 中 ADM 对比固定步长方法显示更稳定的收敛轨迹。

---

### 模块 3: 任意时间最优收敛速率（对应框架"整体收敛保证"）

**直觉**：通过 Lyapunov 函数将距离估计、步长选择、动量累积耦合，telescoping 后消去 ε 依赖，恢复最优 O(1/k²)。

**Baseline 公式** (Nesterov 2015 Universal Method)：
$$O\left(\frac{L_\nu^{\frac{2}{1+\nu}} D_0^2}{\varepsilon^{\frac{1-\nu}{1+\nu}} k^{\frac{1+3\nu}{1+\nu}}} + \varepsilon\right)$$
符号: $D_0 = \|x_0 - x_*\|$ = 初始距离, $k$ = 迭代次数。

**变化点**：该速率中 ε 同时出现在分母（驱动收敛）与加性项（饱和误差），形成权衡：小 ε 使首项收敛慢，大 ε 使加性项残留大。

**本文公式（推导）**：
$$\text{Step 1}: \sum_{i=0}^k \eta_i (f(x_i) - f(x_*)) \leq \bar{r}_k \sqrt{\sum_{i=0}^k \|g_i\|^2} + \text{边界项} \quad \text{(加权平均的 Lyapunov 控制)}$$
$$\text{Step 2}: \text{利用 } \eta_k \propto \bar{r}_k / \sqrt{\sum\|g_i\|^2} \text{ 的特定结构，telescoping 消去 } \varepsilon \text{ 依赖}$$
$$\text{Step 3}: \text{通过 Hölder 插值 } \|g_i\|^2 \sim L_\nu^{\frac{2}{1+\nu}} \eta_k^{-\frac{2(1-\nu)}{1+\nu}} \text{ 重新平衡指数}$$
$$\text{最终}: O\left(\frac{L_\nu^{\frac{2}{1+\nu}} D_0^{\frac{2}{1+\nu}}}{k^{2}}\right) = O\left(\frac{L_\nu^{\frac{2}{1+\nu}} D_0^{\frac{2}{1+\nu}}}{k^{2}}\right) \quad \text{(任意时间最优，无 } \varepsilon \text{ 项)}$$

**对应消融**：去掉加速动量（退化为梯度下降）速率降为 O(1/k)；去掉距离自适应（固定步长）则无法达到与问题结构匹配的最优常数。

## 实验与分析


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d3251acd-d545-4cd8-80ac-202e5b1135a2/figures/Figure_2.png)
*Figure 2 (result): Performance of the compared problem. Left: submodular test on random dataset. Middle: Logistics loss on Boston housing dataset. Right: submodular test on the synthetic problem.*



本文在三类问题上进行了初步实证验证：softmax 回归（光滑复合优化）、矩阵博弈（非光滑鞍点问题，ν=0）、以及 Lp 范数最小化（弱光滑，0<ν<1）。实验环境为配备 16GB RAM 的个人电脑 CPU。Figure 1 展示了模拟数据集上的 submodular 测试、矩阵博弈（α=1）及网络运行场景的比较结果；Figure 2 扩展至随机数据集与 Boston housing 的 logistics loss；Figure 3 报告了网络运行中的训练 loss、验证 loss 与准确率曲线。



从 Figure 1-3 的收敛曲线可观察到，ADM 在矩阵博弈（非光滑）与 softmax（光滑）两端均保持竞争力，无需针对问题类型切换超参。相比 Nesterov 通用方法，ADM 避免了预设 ε 的试错成本；相比 DOG，ADM 在动量加速下收敛更陡峭。与 D-adaptation 相比，ADM 在部分曲线上显示出更快的初始下降，但由于缺乏精确数值表格，具体加速倍数无法从当前摘录中定量提取。



消融实验（隐含于 Figure 1-3 的多方法对比中）验证了：移除加速机制后收敛明显放缓；移除距离自适应（即采用固定步长或预设 ε）则导致对问题参数敏感。作者明确承认实验仅为 preliminary，测试问题相对简单（softmax、矩阵博弈、Lp 范数），未覆盖大规模深度学习训练。

**公平性检查**：主要对比基线包括 D-adaptation、DoG、Adam、Universal Fast Gradient Method，但缺少与最直接竞争者 DAda（2025 同期工作）的实验对比，也未与 Carmon et al. 的加速无参数随机优化进行 head-to-head 测试。实验规模限于 CPU 小规模问题，无代码开源，可复现性受限。作者未声明 SOTA，仅声称"最优任意时间收敛速率"的理论保证。

## 方法谱系与知识库定位

ADM 属于**无参数加速一阶优化**方法族，直接父方法为 **Distance-over-Gradient (DOG)**，同时深度融合了 **Nesterov Universal Fast Gradient Method (2015)** 的加速结构。

**谱系变更槽位**：
- **training_recipe**：DOG 的 SGD 步长 → 加速复合优化的距离自适应步长，消除目标精度 ε 预设
- **architecture**：纯梯度步 → Nesterov 估计序列 + 邻近算子分解的三点格式
- **objective**：单一光滑/非光滑假设 → Hölder 全谱 ν∈[0,1] 隐式自适应
- **exploration_strategy**：线搜索 / 网格搜索 → 完全闭式距离自适应，零额外函数查询

**直接基线差异**：
- vs **DOG**：增加 Nesterov 加速与 proximal 扩展，从 SGD 设定进入复合凸优化
- vs **Nesterov Universal**：移除 ε 依赖，以距离估计替代目标精度作为算法锚点
- vs **DAda (2025)**：同为距离自适应，但 ADM 聚焦加速而非对偶平均，理论速率更优
- vs **Carmon et al. 加速无参数随机优化**：从随机优化扩展至确定性复合优化与 Hölder 光滑

**后续方向**：(1) 大规模深度学习训练验证（当前仅 CPU 小规模实验）；(2) 与 DAda 的严格实验对比与可能融合；(3) 非凸或非欧几里得几何下的距离自适应扩展。

**标签**：modality=数值优化 / paradigm=无参数一阶方法 / scenario=凸复合优化 / mechanism=距离自适应步长 + Nesterov 加速 / constraint=无目标精度预设、无光滑参数知识、无线搜索

## 引用网络

### 直接 baseline（本文基于）

- DADA: Dual Averaging with Distance Adaptation _(ICLR 2026, 直接 baseline, 未深度分析)_: Very recent (2025) distance-adaptive dual averaging; most directly related work,

