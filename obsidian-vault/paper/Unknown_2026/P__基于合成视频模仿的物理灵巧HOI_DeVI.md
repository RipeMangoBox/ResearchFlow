---
title: 'DeVI: Physics-based Dexterous Human-Object Interaction via Synthetic Video Imitation'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.20841
aliases:
- 基于合成视频模仿的物理灵巧HOI生成
- DeVI
- DeVI的核心直觉是：对人体和物体采用「不对称维度」的参考信号——人体
method: DeVI
modalities:
- Image
---

# DeVI: Physics-based Dexterous Human-Object Interaction via Synthetic Video Imitation

[Paper](https://arxiv.org/abs/2604.20841)

**Topics**: [[T__Imitation_Learning]], [[T__Robotics]], [[T__Video_Generation]] | **Method**: [[M__DeVI]]

> [!tip] 核心洞察
> DeVI的核心直觉是：对人体和物体采用「不对称维度」的参考信号——人体用3D（因为HMR技术相对成熟），物体用2D（因为6D位姿从单目视频中恢复极不可靠）。这一「混合」设计将一个病态的3D重建问题转化为一个更易优化的2D追踪问题，以牺牲深度方向精度换取整体鲁棒性。有效性的根本原因在于：2D物体轨迹虽然信息不完整，但足以约束物体的平面内运动，而物理仿真中的接触约束可以隐式补偿部分深度方向的不确定性；同时，2D投影奖励的优化景观比6D位姿奖励更平滑，策略在相同训练预算内收敛更好。

| 中文题名 | 基于合成视频模仿的物理灵巧HOI生成 |
| 英文题名 | DeVI: Physics-based Dexterous Human-Object Interaction via Synthetic Video Imitation |
| 会议/期刊 | 2026 (arXiv预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20841) · [Code](https://github.com/snuvclab/DeVI) · [Project](https://devi-hoi.github.io/) |
| 主要任务 | 物理仿真中的灵巧人-物交互（HOI）运动生成，支持零样本泛化到未见物体 |
| 主要 baseline | PhysHOI, SkillMimic, InterMimic |

> [!abstract] 因为「高质量3D HOI运动捕捉数据稀缺且视频生成模型的2D输出无法直接用于物理仿真」，作者在「InterMimic等标准RL模仿学习」基础上改了「混合维度模仿目标设计（3D人体+2D物体）与视觉HOI对齐模块」，在「GRAB数据集16个HOI动作」上取得「严格阈值下成功率0.500 vs. 最强基线0.250」

- **关键性能1**: 严格阈值（MPJPE<0.1m且Tobj<0.1m）下成功率0.500，InterMimic仅0.250，PhysHOI/SkillMimic接近0.000
- **关键性能2**: 消融实验去除2D物体追踪奖励后成功率骤降至0.188，验证2D奖励为核心组件
- **关键性能3**: 零样本泛化到未见物体（Figure 4定性展示），无需任何3D运动捕捉数据

## 背景与动机

物理仿真中的灵巧人-物交互（HOI）生成面临一个根本性瓶颈：高质量3D运动捕捉数据（如GRAB数据集）获取成本极高，覆盖的物体类别和交互类型极为有限。例如，让虚拟人形抓取一个未见过的马克杯并自然饮用，现有方法因缺乏对应训练数据而难以实现。

现有方法沿两条路径尝试解决这一问题。**PhysHOI** 采用基于物理仿真的强化学习，依赖精确的物体6D位姿作为奖励信号训练策略，但要求昂贵的动捕标注。**SkillMimic** 和 **InterMimic** 同样遵循模仿学习范式，通过追踪参考运动训练人形控制器，其核心假设是参考信号（人体姿态+物体位姿）必须精确且完整。

这些方法在合成视频场景下遭遇系统性失效：将2D视频提升为精确3D HOI运动存在根本性的病态问题。独立估计的身体姿态和手部姿态在统一为SMPL-X模型时产生严重空间错位，尤其在手-物接触区域；物体的6D位姿从单目视频中恢复精度极低，导致基于6D位姿的奖励信号几乎完全失效（成功率接近0）。此外，接触时序估计仅能依赖像素速度，无法感知深度方向运动，产生噪声标签。

视频生成模型虽能提供丰富的交互语义知识，但其2D输出与物理仿真所需的精确3D控制信号之间存在巨大模态鸿沟。DeVI的核心动机即在于：如何在不依赖昂贵3D标注的前提下，跨越这一鸿沟，利用视频生成模型的泛化能力驱动物理仿真中的灵巧交互。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/baa5ea00-a6c9-49c2-9632-b8dfac790a9e/figures/Figure_3.png)
*Figure 3: Figure 3: Challenges in 4D HOI Reconstruction. Reconstructing 4D HOI from the synthetic videois challenging due to (a) noisy 6D pose estimation and (b) HOI alignment issues. DeVI addressesthese via hy*



## 核心创新

核心洞察：对人体和物体采用「不对称维度」的参考信号——人体用3D（HMR技术相对成熟），物体用2D（6D位姿从单目视频恢复极不可靠），因为2D投影奖励的优化景观比6D位姿奖励更平滑，从而使物理仿真中的接触约束隐式补偿深度不确定性、策略在相同训练预算内收敛更好成为可能。

| 维度 | Baseline (InterMimic/PhysHOI) | 本文 (DeVI) |
|:---|:---|:---|
| 人体参考 | 3D SMPL-X参数（动捕数据） | 3D SMPL-X参数（世界坐标系HMR+视觉HOI对齐优化） |
| 物体参考 | 6D位姿（3D平移+旋转） | 1024个表面顶点的2D投影轨迹（视频追踪器） |
| 奖励设计 | 单一维度追踪奖励 | 混合追踪奖励（3D人体+2D物体） |
| 数据来源 | 真实GRAB动捕数据集 | 视频扩散模型生成的合成视频（零样本泛化） |
| 核心假设 | 参考信号必须完整精确 | 不完整但鲁棒的信号可通过物理仿真补偿 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/baa5ea00-a6c9-49c2-9632-b8dfac790a9e/figures/Figure_2.png)
*Figure 2: Figure 2: Overview. Given a scene with an SMPL-X [33] human and object, we replace it with adeformed textured mesh and render an HOI video. Then hybrid imitation targets extracted from thevideo are us*



DeVI的整体流程分为三个阶段，形成从文本/图像条件到物理仿真运动的完整管线：

**阶段一：视频生成与3D场景初始化**。输入为交互文本提示（如"drink from cup"）和初始场景（SMPL-X人体+物体）。利用图像到视频（image-to-video）扩散模型，以场景初始渲染图像为条件生成HOI视频，实现生成视频与3D场景的有效对齐。

**阶段二：混合模仿目标构造**。这是DeVI的核心创新模块，包含两个并行分支：（2a）**3D人体参考获取**——使用世界坐标系人体网格恢复算法（world-grounded HMR）结合手部姿态估计器获得粗粒度3D人体，再通过**视觉HOI对齐（Visual HOI Alignment）**模块优化，基于Chamfer距离的HOI损失在接触开始帧附近最小化手-物空间误差；（2b）**2D物体轨迹提取**——使用视频追踪器直接提取1024个物体表面顶点的2D投影轨迹，完全绕开6D位姿估计。

**阶段三：混合奖励RL训练**。将3D人体追踪奖励与2D物体追踪奖励组合，训练人形控制策略 $\pi_\theta(a_t | s_t, g_t)$，其中目标向量 $g_t = (\hat{g}^h_t, \hat{g}^o_t)$ 同时包含3D人体参考和2D物体轨迹。

```
文本提示 + 初始场景(SMPL-X+物体)
    ↓
[图像到视频扩散模型] → 合成HOI视频
    ↓
├─→ [World-grounded HMR + 手部估计] → 粗3D人体 → [视觉HOI对齐] → 精化3D人体参考 ĝ^h_t
└─→ [视频追踪器] → 1024顶点2D投影轨迹 → 2D物体参考 ĝ^o_t
    ↓
[混合奖励RL: r = r_3D_human + r_2D_object] → 训练策略 π_θ(a_t|s_t, g_t)
    ↓
物理仿真输出：灵巧HOI运动
```

## 核心模块与公式推导

### 模块 1: 视觉HOI对齐（对应框架图阶段2a）

**直觉**: 独立估计的身体姿态和手部姿态合并为SMPL-X时存在严重空间错位，需在接触区域强制手-物空间一致。

**Baseline 公式** (标准HMR后处理): 直接使用HMR输出的SMPL-X参数 $\theta_{hmr}$ 和全局平移 $t_{hmr}$ 作为参考，无额外优化。

符号: $\theta \in \mathbb{R}^{72}$ = SMPL-X姿态参数, $\beta \in \mathbb{R}^{10}$ = 形状参数, $t \in \mathbb{R}^3$ = 全局平移, $M(\cdot)$ = SMPL-X网格生成函数, $\mathcal{V}_h$ = 手部顶点集, $\mathcal{V}_o$ = 物体顶点集

**变化点**: HMR输出的3D人体与视频帧中的2D手-物接触视觉证据不对齐，导致物理仿真中接触失败。DeVI引入基于Chamfer距离的HOI损失，在接触关键帧附近优化人体姿态。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}_{chamfer}(\theta, t) = \sum_{v \in M(\theta, \beta, t)[\mathcal{V}_h]} \min_{u \in \mathcal{V}_o} \|v - u\|_2^2 + \sum_{u \in \mathcal{V}_o} \min_{v \in M(\theta, \beta, t)[\mathcal{V}_h]} \|u - v\|_2^2$$
$$\quad \text{加入Chamfer距离项以强制手-物空间接近}$$

$$\text{Step 2}: \theta^*, t^* = \text{arg}\min_{\theta, t} \lambda_{hoi} \mathcal{L}_{chamfer}(\theta, t) \cdot \mathbb{1}[t \in \mathcal{T}_{contact}] + \lambda_{2d} \mathcal{L}_{2d\_reproj}(\theta, t) + \lambda_{reg} \|\theta - \theta_{hmr}\|_2^2$$
$$\quad \text{时间加权（仅接触帧附近激活）+ 2D重投影约束 + 正则化保证稳定性}$$

$$\text{最终}: \hat{g}^h_t = M(\theta^*, \beta, t^*) \quad \text{（优化后的3D人体网格序列）}$$

**对应消融**: 未显式报告视觉HOI对齐的独立消融，但去除整体3D人体追踪后系统无法运行（基线设计）。

### 模块 2: 混合追踪奖励（对应框架图阶段3）

**直觉**: 6D位姿奖励在合成视频下优化景观崎岖，2D投影奖励牺牲深度精度换取平滑收敛。

**Baseline 公式** (PhysHOI/InterMimic标准设计):
$$r_{base} = r_{human}^{3D} + r_{object}^{6D} = -\|J_{sim} - J_{ref}\|_2^2 - \left(\|p_{sim} - p_{ref}\|_2^2 + \alpha\|R_{sim} \ominus R_{ref}\|_2^2\right)$$
符号: $J$ = 人体关节位置, $p$ = 物体3D平移, $R$ = 物体3D旋转, $\ominus$ = 旋转空间差异

**变化点**: 合成视频中6D位姿估计噪声极大，$r_{object}^{6D}$ 信号几乎为随机噪声，导致策略无法学习。DeVI将物体追踪维度从6D降至2D，利用视频追踪器的鲁棒2D投影。

**本文公式（推导）**:
$$\text{Step 1}: \hat{g}^o_t = \{(u_i, v_i)_t\}_{i=1}^{1024} \quad \text{其中 } (u_i, v_i)_t = \Pi(K, T_{cam}, v_i^{world})$$
$$\quad \text{视频追踪器输出1024个物体表面顶点的2D投影（相机内参K、外参T_cam投影）}$$

$$\text{Step 2}: r_{object}^{2D} = -\frac{1}{1024}\sum_{i=1}^{1024} \|\Pi(K, T_{cam}, v_i^{sim}) - (u_i, v_i)_t^{ref}\|_2^2$$
$$\quad \text{2D投影距离替代6D位姿距离，优化景观显著平滑}$$

$$\text{Step 3}: r_{final} = \lambda_h \cdot r_{human}^{3D} + \lambda_o \cdot r_{object}^{2D}$$
$$\quad \text{线性组合，无需深度方向显式约束（由物理接触隐式补偿）}$$

$$\text{最终}: L(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} r_{final}(s_t, a_t, g_t)\right]$$

**对应消融**: 去除2D物体奖励（仅保留3D人体奖励）后成功率固定在0.188，完整系统为0.500（严格阈值），Δ-31.2个百分点，证明2D奖励为关键组件。

### 模块 3: 图像到视频条件生成（对应框架图阶段1）

**直觉**: 文本到视频生成与具体3D场景脱节，需以初始渲染图像为条件保证几何一致性。

**Baseline**: 纯文本到视频扩散模型（如Sora类），生成视频与特定3D场景无显式关联。

**本文设计**: 采用图像到视频（I2V）扩散模型，条件为$c = \{I_{render}, text\}$，其中$I_{render}$为SMPL-X人体和物体在初始姿态下的渲染图像。损失函数为标准扩散训练目标：
$$\mathcal{L}_{diff} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|_2^2\right]$$
推理时从$x_T \sim \mathcal{N}(0, I)$去噪至$x_0$，生成与初始场景几何对齐的HOI视频序列。

## 实验与分析

| Method | MPJPE<0.2m & Tobj<0.2m | MPJPE<0.15m & Tobj<0.15m | MPJPE<0.1m & Tobj<0.1m |
|:---|:---:|:---:|:---:|
| PhysHOI | 0.125 | 0.000 | 0.000 |
| SkillMimic | 0.000 | 0.000 | 0.000 |
| InterMimic | 0.500 | 0.312 | 0.250 |
| **DeVI (Ours)** | **0.500** | **0.500** | **0.500** |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/baa5ea00-a6c9-49c2-9632-b8dfac790a9e/figures/Figure_4.png)
*Figure 4: Figure 4: Qualitative Results on Various Objects. DeVI leverages a video diffusion model as anHOI-aware motion planner, allowing simulation of HOI with diverse objects through text prompts.*



**核心结果分析**: 在最宽松阈值（0.2m）下，DeVI与InterMimic持平（0.500），优势尚未显现；但随着阈值严格化，InterMimic快速衰减至0.250，而DeVI维持0.500不变。这一「阈值鲁棒性」模式支持核心主张：混合模仿目标设计在精确控制场景下具有结构性优势。PhysHOI和SkillMimic在合成视频下几乎完全失效，验证了基于6D位姿的方法对此设定的不适应性。

**消融实验**: 去除2D物体追踪奖励（仅3D人体奖励）后成功率骤降至0.188，与完整系统差距达31.2个百分点，证明2D物体奖励是性能跃升的关键而非辅助组件。去除视觉HOI对齐模块的独立消融未报告。

**定性验证**: 
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/baa5ea00-a6c9-49c2-9632-b8dfac790a9e/figures/Figure_5.png)
*Figure 5: Figure 5: Target-Awareness and Text Controllability. As DeVI leverages a video diffusion modelas a motion planner, (a) we can model HOIs that require a specific target, and (b) plan differentmotions f*

 Figure 4展示DeVI对多种未见物体的零样本泛化（球、瓶子、杯子等），Figure 5展示目标感知和文本可控性。

**公平性检查与局限**:
- **评估偏差**: 定量评估仅在双方均成功的子集上进行，此设计对基线不利，可能高估DeVI相对优势
- **阈值边界**: 最宽松阈值下与InterMimic持平，领先优势主要体现在中间阈值
- **零样本声明**: 仅Figure 4定性支持，缺乏系统性定量验证
- **深度误差**: 视频扩散模型的透视失真导致深度方向误差，在精确放置任务（如棒球入小杯）中表现下降
- **接触噪声**: 基于像素速度的接触标签无法感知深度运动，产生快速抓取等不自然动作
- **竞争验证缺失**: 未与视频驱动机器人操作方法（如UniPi、RT-X视频变体）进行定量对比

## 方法谱系与知识库定位

**方法家族**: 物理仿真中的模仿学习（Imitation Learning for Physics-based Character Control）

**Parent method**: InterMimic（2024）—— 标准的人形HOI模仿学习框架，采用MDP+RL训练策略追踪3D参考运动。DeVI保留其核心训练管线，主要改动集中在**模仿目标构造**和**奖励设计**两个槽位。

**直接基线与差异**:
- **PhysHOI**: 依赖真实GRAB数据+精确6D位姿奖励 → DeVI替换为合成视频+混合维度奖励
- **SkillMimic**: 专注技能迁移，需动捕数据预训练 → DeVI零样本泛化，无需任何3D标注
- **InterMimic**: 同设定下最强基线，完整3D参考追踪 → DeVI提出不对称维度设计，物体降维至2D

**后续方向**:
1. **深度补偿机制**: 结合单目深度估计或神经辐射场显式建模深度不确定性，缓解2D投影的信息损失
2. **端到端可微分管线**: 将视频生成、追踪、RL训练联合优化，替代当前三阶段分离设计
3. **真实世界迁移**: 将合成视频训练的策略通过域随机化或系统辨识迁移至真实机器人平台

**知识库标签**: 
- modality: video-to-3D / physics-simulation
- paradigm: imitation-learning / reinforcement-learning / diffusion-model-as-planner
- scenario: dexterous-HOI / zero-shot-generalization
- mechanism: hybrid-imitation-target / asymmetric-dimension-reward / visual-HOI-alignment
- constraint: no-3D-annotation / noisy-2D-supervision-only

