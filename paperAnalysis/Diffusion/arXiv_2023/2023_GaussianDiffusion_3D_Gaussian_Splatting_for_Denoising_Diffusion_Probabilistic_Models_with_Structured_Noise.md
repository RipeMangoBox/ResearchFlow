---
title: "GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/text-to-3d
  - diffusion
  - structured-noise
  - variational-optimization
  - dataset/COLMAP
  - opensource/no
core_operator: 把共享3D噪声源投影为跨视角相关的结构噪声，并对高斯位置/尺度做与扩散噪声同步的变分抖动，以稳定文本到3D的多视图蒸馏。
primary_logic: |
  文本提示 → 通过语义码采样、Point-E 与深度条件初始化 3D Gaussian 场景 →
  对每个视角渲染结果注入来自同一3D噪声源的结构噪声，并借助2D扩散模型分数蒸馏反传 →
  同时对高斯位置/尺度施加噪声级别相关的变分扰动 → 输出渲染更快、几何更一致、伪影更少的 3D 对象
claims:
  - "GaussianDiffusion 在 50 个生成场景上的 COLMAP 位姿方差降至 0.021，优于 SJC 的 0.081、3DFuse 的 0.053 和 DreamGaussian 的 0.106 [evidence: comparison]"
  - "GaussianDiffusion 在单张 RTX 4090 上约 2000 次迭代、5.5 分钟即可收敛到稳定结果，而 3DFuse 需要约 10000 次迭代、20.3 分钟 [evidence: comparison]"
  - "去掉 structured noise 或 variational Gaussian Splatting 后，方差分别退化到 0.056 和 0.033，并伴随 multi-face 或 floater/burr 类伪影回归 [evidence: ablation]"
related_work_position:
  extends: "SJC (Wang et al. 2023)"
  competes_with: "3DFuse (Seo et al. 2023); DreamGaussian (Tang et al. 2023)"
  complementary_to: "ControlNet (Zhang et al. 2023); LoRA (Hu et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Diffusion/arXiv_2023/2023_GaussianDiffusion_3D_Gaussian_Splatting_for_Denoising_Diffusion_Probabilistic_Models_with_Structured_Noise.pdf
category: 3D_Gaussian_Splatting
---

# GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.11221)
> - **Summary**: 该文把 3D Gaussian Splatting 引入 text-to-3D 全流程，并通过“共享源结构噪声 + 变分高斯扰动”同时缓解多视图几何不一致与局部最优伪影问题。
> - **Key Performance**: COLMAP-based pose variance 0.021 vs 3DFuse 0.053 / SJC 0.081；约 2000 iter / 5.5 min vs 3DFuse 10000 iter / 20.3 min

> [!info] **Agent Summary**
> - **task_path**: 文本提示 + 冻结2D扩散先验 -> 多视图一致的 3D Gaussian 物体表示
> - **bottleneck**: 2D 扩散蒸馏到 3D 时各视角监督缺少耦合，且精确高斯参数在噪声梯度下容易陷入局部最优并产生浮点/毛刺
> - **mechanism_delta**: 将各视角独立噪声改为来自同一 3D 噪声源的投影噪声，并让高斯位置/尺度在与扩散噪声同步的方差下抖动优化
> - **evidence_signal**: 50 个生成场景上的 COLMAP 方差从 3DFuse 的 0.053 降到 0.021，且去掉结构噪声或变分高斯都会明显退化
> - **reusable_ops**: [shared-source view-projected noise, noise-level-synchronized Gaussian jitter]
> - **failure_modes**: [变分扰动会带来 blur/haze, 面对细薄结构或背面语义不稳的提示词时仍可能残留多视图不一致]
> - **open_questions**: [结构噪声能否扩展到复杂场景级生成, 如何保留逃离局部最优能力同时消除变分带来的模糊]

## Part I：问题与挑战

这篇论文解决的是一个很具体的 text-to-3D 瓶颈：**如何把强大的 2D 扩散先验稳定地蒸馏成一个多视图一致、渲染又足够快的 3D 表示**。

### 1. 真问题是什么
此前 DreamFusion / SJC / 3DFuse 这类方法，大多依赖 NeRF 或其变体做 3D 表示。问题在于：

1. **NeRF 渲染慢**  
   点采样 + ray marching 导致训练和下游渲染都重，工业可用性差。

2. **2D 扩散监督天然是“单视角合理”，不是“跨视角一致”**  
   如果每个视角都各自加噪、各自去噪，扩散模型可以在不同视角给出彼此不兼容但局部看起来都合理的细节，于是就出现 multi-face、几何漂移、背面乱长结构。

3. **vanilla 3D Gaussian 优化太“硬”**  
   3D Gaussian Splatting 虽然快，但如果直接把位置/尺度当作确定值去拟合 noisy 的 2D guidance，很容易被早期不稳定梯度推到坏的局部极小值，形成 floaters、burrs、proliferative elements。

### 2. 输入/输出接口
- **输入**：文本提示词
- **中间接口**：冻结的 2D diffusion teacher + Point-E 初始化点云 + 深度条件
- **输出**：对象中心化的 3D Gaussian 场景表示，可从任意视角快速渲染

### 3. 为什么现在值得做
因为 3D Gaussian Splatting 已经证明了自己在表示和渲染效率上的优势。  
所以现在的关键不是“再找一个更强 teacher”，而是**让 2D teacher 给 3D 的监督更一致、更不容易把优化带偏**。这正是本文想动的核心旋钮。

### 4. 边界条件
这篇工作更适合：
- 单物体、对象中心化的 text-to-3D
- 固定半径、半球相机采样
- 需要较快渲染和较好外观的场景

它不是在做：
- 大场景重建
- 动态/时序 3D
- 显式 mesh/topology 生成

---

## Part II：方法与洞察

论文的主线很清楚：**用 Gaussian Splatting 替代 NeRF 作为 3D 表示底座，再针对“跨视角监督不一致”和“高斯优化易陷局部最优”这两个瓶颈各打一个补丁。**

### 方法总览

作者的整体 pipeline 里，有一些来自已有方法的稳定器：
- 用 **Semantic Code Sampling** 约束单一语义身份
- 用 **Point-E** 生成稀疏点云做初始化
- 把点云投影成深度图，给 **ControlNet** 作为空间条件
- 用 **LoRA** 做额外适配

但真正的新意主要是两件事：

1. **Structured Noise**
2. **Variational Gaussian Splatting**

### 1. Structured Noise：把“每视角独立噪声”改成“共享源投影噪声”
SJC 的思路是对每个视角渲染图加噪，再让扩散模型给分数梯度。  
本文认为问题在于：**如果每个视角的噪声彼此独立，那 teacher 给不同视角的梯度就可能互相冲突。**

所以它改成：
- 先构造一个**共享的 3D 高斯噪声源**
- 再把这个噪声源通过 Gaussian Splatting 投影到不同视角
- 得到每个视角上看起来仍近似高斯、但**跨视角有内在关联**的噪声

这样做的效果是：
- 每个视角仍能喂给 2D diffusion 一个合法噪声分布
- 但不同视角不再“各说各话”
- 梯度开始偏向于一个共同的 3D 解释，而不是每个视角单独凑局部合理性

作者还保留了 SJC 风格噪声，并让 structured noise 的占比从 0.3 逐步降到 0.05，意思是：
- 早期更强调跨视角几何耦合
- 后期降低其强度，让外观细化更自由

### 2. Variational Gaussian Splatting：把“点估计优化”改成“分布优化”
作者认为 vanilla Gaussian Splatting 的另一个问题是：  
**位置 z 和尺度 s 过早被硬性确定，容易在不稳定梯度下收缩到错误局部极小值。**

于是他们只对关键几何参数做变分化：
- 对 **位置 z**
- 对 **尺度 s**

具体做法是：
- 不直接优化单点参数，而是在其附近加一个与 diffusion 噪声等级同步的扰动
- 高噪声阶段，允许更大的抖动，几何更“松”
- 低噪声阶段，抖动变小，几何逐步收紧

这相当于把优化目标从“找到一个立刻最精确的高斯配置”
改成“先在一个更宽的参数盆地里搜索，再逐步收敛到精细解”。

其直接作用是：
- 更容易逃离局部最优
- 减少 floaters / burrs / proliferative elements
- 形成更自然的 coarse-to-fine 收敛过程

### 3. 3D Gaussian Splatting 作为底座的意义
这不是单纯“换个表示”。  
它实际上改了整个系统的可用性：

- 训练中渲染更快
- 下游渲染能接近传统 rasterization pipeline
- 不会像 NeRF 一样随像素规模出现明显慢速瓶颈

所以本文的效率提升，一部分来自新监督设计，一部分来自表示本身。

### 核心直觉

真正的因果链可以概括成两条：

#### 直觉 A：改噪声耦合方式
**变化**：从“每视角独立加噪”变成“同一 3D 噪声源投影到各视角”  
**改变的瓶颈**：跨视角监督从互不约束，变成了共享来源、弱耦合一致  
**能力变化**：3D 优化时更容易形成统一几何解释，multi-face 和几何漂移减少

为什么这有效：  
因为 diffusion teacher 只能看 2D 图像，它并不知道不同视角本应来自同一个物体。共享源噪声相当于人为给 teacher 的输入建立了一个跨视角关联，让不同视角的梯度不再完全独立。

#### 直觉 B：改参数优化的“硬度”
**变化**：从“固定高斯参数点估计”变成“随噪声级别变化的分布式优化”  
**改变的瓶颈**：早期 noisy guidance 不再立刻把几何锁死  
**能力变化**：更容易从粗到细优化，减少 floaters/burrs，并扩大可收敛区域

为什么这有效：  
早期扩散梯度本来就不稳定，如果此时要求几何参数必须精确，优化很容易被错误信号“钉死”。加上同步扰动后，模型先学一个更宽松的几何区域，后期再靠低噪声阶段细化。

### 战略取舍表

| 设计 | 解决的核心瓶颈 | 直接收益 | 代价/风险 |
|---|---|---|---|
| 3D Gaussian Splatting 替代 NeRF | 渲染慢、下游难部署 | 训练/渲染更快，表示更贴近传统图形管线 | 输出是高斯表示，不是显式 mesh |
| Structured Noise | 各视角扩散监督彼此独立 | 提升多视图一致性，减少 multi-face | 需要额外噪声构造与调度超参 |
| Variational Gaussian Splatting | 精确参数早期易陷坏局部极小值 | 更易逃离局部最优，减少 floaters/burrs | 会带来 blur/haze，且需手调扰动系数 |
| Point-E + 深度条件初始化 | 文本到 3D 初始几何太弱 | 更稳定起步、更快进入有效优化区间 | 依赖额外预训练模块 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 比较信号：几何一致性明显更好
作者用 3DFuse 的 COLMAP 协议做量化：  
对每个生成结果采样 100 个视角渲染，再用 COLMAP 估计相邻视图位姿，统计位姿方差。方差越小，说明多视图几何越一致。

核心结果：
- **GaussianDiffusion**: 0.021
- **3DFuse**: 0.053
- **SJC**: 0.081
- **DreamGaussian**: 0.106

这个信号支持的结论很明确：  
**本文方法在“跨视角几何一致性”上优于主要基线。**

#### 2. 比较信号：效率相对 NeRF 系更有优势
训练代价上：
- **GaussianDiffusion**: 2000 iter / 5.5 min
- **3DFuse**: 10000 iter / 20.3 min
- **SJC**: 10000 iter / 13.8 min
- **DreamGaussian**: 600 iter / 3.1 min

这里要读得更细一点：
- 它**不是绝对最快**，DreamGaussian 更快
- 但它在**速度与一致性之间给出了更平衡的点**
- 相比 3DFuse / SJC 这类 NeRF 风格方案，它的效率改善是实打实的

#### 3. 消融信号：两个新算子都有效
去掉结构噪声：
- variance 从 **0.021 → 0.056**
- 可视化出现 multi-face、跨视角几何不一致

去掉变分 Gaussian Splatting：
- variance 从 **0.021 → 0.033**
- 可视化中 floaters / burrs / proliferative artifacts 增多

这说明：
- **Structured Noise** 主要在修正跨视角几何一致性
- **Variational Gaussian Splatting** 主要在改善优化稳定性和伪影控制

### 1-2 个最关键指标
- **COLMAP pose variance**：0.021（越低越好）
- **训练成本**：2000 iter / 5.5 min on RTX 4090

### 局限性

- **Fails when**: 目标包含细薄结构、复杂背面语义或高频局部纹理时，2D diffusion 本身的不稳定仍可能传导到 3D；另外变分扰动会引入作者自己也承认的 blur/haze。
- **Assumes**: 依赖冻结的 2D diffusion teacher、Point-E 初始化、ControlNet 深度条件、Semantic Code Sampling、LoRA 适配；structured noise 比例和变分扰动系数需要手工设定；相机分布是对象中心化的半球采样。
- **Not designed for**: 大场景/室内外复杂环境生成、动态 3D、显式 mesh 拓扑控制、严格标准化 benchmark 上的大规模泛化评测。

### 复现与可扩展性注意点
- 论文正文未给出代码链接，故可复现性受限
- 量化评估主要依赖 **COLMAP 方差协议**，严格说这更像一个评测工具/协议而非标准数据集，因此证据强度应保守看待
- 所有实验基于单张 **RTX 4090 24GB**，资源门槛不算极高，但依赖模块较多

### 可复用部件
这篇文章最值得复用的不是整套 pipeline，而是两个操作符：

1. **shared-source view-projected noise**  
   适合任何“从 2D teacher 蒸馏到 3D/多视图表示”的场景。

2. **noise-level-synchronized parameter jitter**  
   适合任何“早期 guidance 噪声大、后期再精细收敛”的 3D 优化任务。

一句话总结“所以怎样”：  
**相对 prior work，GaussianDiffusion 的能力跃迁不在于更强的 teacher，而在于它把 teacher 的多视角监督变得更一致、把 3D 参数优化变得更不容易卡死，因此在速度、几何一致性和伪影控制之间拿到了更均衡的解。**

![[paperPDFs/Diffusion/arXiv_2023/2023_GaussianDiffusion_3D_Gaussian_Splatting_for_Denoising_Diffusion_Probabilistic_Models_with_Structured_Noise.pdf]]