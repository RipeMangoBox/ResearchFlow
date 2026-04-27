---
title: "Falcon: Fast Visuomotor Policies via Partial Denoising"
venue: ICML
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - partial-denoising
  - latent-buffer
  - dataset/RoboMimic
  - dataset/MetaWorld
  - dataset/ManiSkill2
  - opensource/no
core_operator: 用当前观测对历史部分去噪动作做一步兼容性估计并阈值筛选，把每个控制时刻的扩散采样起点从高斯噪声改为最匹配的历史中间态
primary_logic: |
  当前观测 + 上一时刻未执行动作尾部 + 历史部分去噪动作缓存
  → 用一步后验估计判断哪些历史中间态仍与当前动作分布一致，并按噪声级偏好采样起点
  → 从该中间态继续少步去噪，输出当前动作序列
claims:
  - "Claim 1: Falcon在48个模拟环境和2个真实机器人任务上实现2–7×动作采样加速，同时相对DDPM/DDIM/DPMSolver保持接近的任务成功率 [evidence: comparison]"
  - "Claim 2: 在与Falcon相同NFE的设置下，直接缩减DPMSolver步数会在Robomimic若干任务上出现最高35.6%的成功率下降，而DPMSolver+Falcon能维持更高成功率 [evidence: comparison]"
  - "Claim 3: Falcon的自适应选择机制、阈值ϵ与探索率δ对速度-性能折中是关键；固定噪声层复用或极端超参都会明显恶化表现 [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Streaming Diffusion Policy (Høeg et al. 2024); Consistency Policy (Prasad et al. 2024)"
  complementary_to: "DDIM (Song et al.); DPMSolver (Lu et al. 2022)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Falcon_Fast_Visuomotor_Policies_via_Partial_Denoising.pdf
category: Embodied_AI
---

# Falcon: Fast Visuomotor Policies via Partial Denoising

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.00339)
> - **Summary**: 这篇工作把扩散策略每个控制时刻“从纯高斯噪声重新开始采样”改成“从历史部分去噪动作继续采样”，在不重训模型的前提下显著降低在线控制延迟。
> - **Key Performance**: Robomimic中可达约7.8×加速且成功率基本不变；真实抓取任务运行时从0.43s降到0.14s，成功率仍为100%。

> [!info] **Agent Summary**
> - **task_path**: 最近观测序列 / 机器人控制上下文 -> 未来动作序列
> - **bottleneck**: 相邻控制时刻高度相关，但标准扩散策略每次都从高斯噪声重采样，重复计算大量相似去噪轨迹
> - **mechanism_delta**: 用当前观测对历史partial denoising latent做一步兼容性估计，并从buffer中挑选最合适的中间态作为warm start
> - **evidence_signal**: 多基准成功率基本不掉的同时NFE显著下降，且同NFE下优于直接减步的DPMSolver
> - **reusable_ops**: [history-conditioned warm start, one-step compatibility screening]
> - **failure_modes**: [动作转移突变时只能从高噪声层启动导致加速有限, epsilon过大或delta过小会复用错误历史动作并累积误差]
> - **open_questions**: [如何跨任务自动选择epsilon/delta/kmin, 在弱时序相关或纯视觉高噪声场景中是否仍稳定]

## Part I：问题与挑战

这篇论文要解决的不是“扩散策略能不能做机器人控制”，而是更实际的一个瓶颈：

**扩散策略在在线控制里太慢了。**

### 1. 真正难点是什么
标准 diffusion policy 在每个决策时刻都会：
- 读入最近几步观测；
- 从一个新的高斯噪声动作序列开始；
- 反复去噪 K 步后才得到当前要执行的动作块。

这在离线评测里可以接受，但在真实机器人上会直接碰到实时性问题。尤其是：
- 控制频率高；
- 动作序列需要连续滚动生成；
- 模型本身已经较大，或要叠加视觉编码器/3D表征时。

**真正的瓶颈不是“单步去噪太贵”，而是“相邻控制时刻明明高度相关，却每次都从头采样”。**

### 2. 现有路线为什么不够
作者把已有加速方法分成两类：

1. **ODE/Solver 类**（如 DDIM, DPMSolver）  
   核心是减少去噪步数。问题是当步数压得很低时，近似误差会上升，控制性能会掉。

2. **蒸馏/一致性类**（如 Consistency Policy）  
   核心是把多步采样压成一步或少步。问题是通常需要**任务特定重训练**，而且可能损失多模态表达能力。

另外，SDP 也使用部分去噪思路，但仍需要任务特定训练，并依赖较大的噪声时间缓存。

### 3. 输入/输出接口与边界
Falcon不改训练目标，面向的是**已经训练好的 diffusion visuomotor policy**：

- **输入**：最近 \(T_o\) 步观测、上一轮预测但尚未执行的动作尾部、历史 partial denoising buffer
- **输出**：未来 \(T_p\) 步动作序列，并执行前 \(T_a\) 步

它成立的关键边界条件是：
- 相邻决策时刻之间具有**时序连续性**；
- 当前动作窗口与上一窗口存在**重叠**；
- 过去的未执行动作尾部，对当前目标动作有近似参考价值。

所以它本质上更适合：
- 连续控制、
- 滚动预测、
- 局部动力学平滑的 manipulation / visuomotor 任务。

## Part II：方法与洞察

Falcon的设计哲学很直接：

**不要再只想着“把单次采样做得更快”，而是利用跨时间步的重复结构，让下一次采样根本不用从头开始。**

### 核心直觉

#### what changed
从：
- 每个控制时刻都从标准高斯噪声初始化

变成：
- 从**历史上已经部分去噪过的动作中间态**开始，再继续去噪到当前动作。

#### which bottleneck changed
这改变的是**采样起点分布**：

- 以前的起点是“全局无条件高噪声 prior”
- 现在的起点是“与当前观测更兼容的局部中间态 warm start”

也就是说，模型不再需要从“很远的地方”走完整条去噪路径，而是从一个已经接近当前动作流形的位置出发。

#### what capability changed
直接带来的能力变化是：
- **NFE显著下降**
- **低步数下性能更稳**
- **多模态性更容易保留**

原因在于 Falcon **没有改原始扩散网络，也没有把采样硬蒸馏成单模态一步映射**；它只是把初始化变聪明了。

#### why this works
因果链条可以概括为：

**相邻决策时刻动作高度相关**  
→ 上一时刻未执行动作尾部通常接近当前目标动作  
→ 历史 partial denoising 中存在一些中间态，继续在当前观测下去噪后会落入当前动作盆地  
→ 只要把这些“兼容中间态”找出来，就能少走很多步。

---

### 方法流程拆解

#### 1. 用上一时刻未执行动作作为 reference action
Falcon先观察一个经验事实：

- 上一时刻预测的“尚未执行动作尾部”
- 与当前真正需要生成的动作

在很多任务里欧式距离很小。

所以它把这个未执行尾部当成**参考动作**，作为当前时刻的“近似目标”。

这一步的意义是：  
它不需要知道当前最优动作的精确真值，只需要一个**足够好的局部参照物**。

#### 2. 对历史中间态做“一步兼容性测试”
buffer里存了历史时刻、不同噪声层级的 partial denoising actions。  
但不是所有历史中间态都还能用于当前时刻，所以 Falcon做了一个筛选：

- 对每个候选历史中间态，
- 用当前观测和预训练噪声预测器，
- 做一次“一步 clean action 估计”。

直观理解就是：

> “如果我现在从这个历史中间态继续去噪，它最终会不会靠近当前参考动作？”

若估计结果与 reference action 足够接近，就通过阈值筛选，成为候选起点。

这一步是 Falcon 的核心保险丝。  
它避免“盲目复用历史 latent”。

#### 3. 温度采样选择起点，并偏向更低噪声层
通过阈值的候选动作里，Falcon再按温度 softmax 采样起点，并偏向：
- 噪声更低的中间态，
- 因为这通常意味着离终态更近，后续所需去噪步更少。

所以 Falcon 不是“永远选最近的一个”，而是保留一定随机性，以避免死板复用。

#### 4. 保留探索分支，避免历史错误滚雪球
如果永远只复用历史中间态，会有两个问题：
- buffer会越来越“陈旧”
- 参考动作一旦有偏差，错误会跨时刻积累

因此作者加入了一个类似 \(\epsilon\)-greedy 的**探索率 \(\delta\)**：
- 以小概率仍然从高斯噪声开始采样

这让 Falcon 保留新样本注入能力，也避免长期 rollout 退化成“重复旧动作”。

#### 5. 工程化细节
为了实用，作者还加入了两个设计：
- **priority queue buffer**：优先移除更老、且噪声更高的历史条目
- **kmin过滤**：避免直接复用过于接近终态的动作，减少“动作发呆/重复执行”

---

### 战略性取舍

| 设计选择 | 改变了什么 | 收益 | 代价/风险 |
|---|---|---|---|
| 从历史中间态启动而非从高斯启动 | 采样起点从全局prior变为局部warm start | 大幅减少NFE | 依赖相邻时刻的动作连续性 |
| 一步估计 + 阈值筛选 | 过滤掉与当前观测不兼容的旧latent | 保住性能，不是盲目复用 | 需要调阈值ϵ，过大过小都不好 |
| 温度采样偏向低噪声候选 | 更激进地利用接近终态的历史动作 | 加速更明显 | 过度利用会更脆弱 |
| 探索率δ保留高斯重启 | 防止历史错误持续积累 | 稳定长时rollout，维持多样性 | δ过大时速度收益会回落 |
| 有界buffer + 优先队列 | 控制显存和筛选成本 | 额外内存很小、可并行化 | 仍需维护缓存和并行筛选逻辑 |

### 这套设计为什么比“直接减步”更强
论文里一个很重要的观点是：

**Falcon不是简单减少solver步数，而是减少“必须从高噪声开始的次数”。**

这与 DDIM / DPMSolver 的差别在于：
- 后者压缩的是单条去噪轨迹；
- Falcon缩短的是每个控制时刻离终态的“起始距离”。

所以在低步数 regime 下，Falcon更像是在**减轻近似误差的暴露程度**，而 ոչ单纯强迫solver更快。

## Part III：证据与局限

### 关键实验信号

#### 1. 比较信号：在多基准上“少很多步，但成功率基本不掉”
Robomimic上最直观：
- 例如 **Lift ph**，DDPM的 NFE 从 **100 降到 12.9**，约 **7.78×** 加速；
- 成功率则基本维持在 **1.00 → 1.00**。

更复杂任务如 Square / Transport / ToolHang：
- 加速通常下降到约 **2×** 左右；
- 但成功率仍大体贴近原始DDPM/DDIM/DPMSolver。

这说明 Falcon 的收益与任务平滑性强相关，但整体 speed-performance tradeoff 很稳。

#### 2. 因果对照信号：同样NFE下，Falcon优于“硬减步”的solver
论文最有说服力的控制实验之一是：

- 把 DPMSolver 直接减到与 Falcon 相同的采样步数（DPMSolver*）
- 再与 DPMSolver+Falcon 比较

结果在多个任务上，DPMSolver* 明显掉点：
- 如 Robomimic **Square mh** 最高有 **35.6%** 的性能下降

而 Falcon 保住了更高成功率。  
这直接支持论文核心论点：

**性能保持不是因为“步数碰巧够用”，而是因为“历史中间态 warm start 本身就是有效机制”。**

#### 3. 真实机器人信号：速度收益能转到实体系统
两项真实任务：
- **Dexterous Grasping**：0.43s → 0.14s，**3.07×** 加速，成功率 **100%**
- **Square Stick Insertion**：0.43s → 0.15s，**2.86×** 加速，成功率维持 **90%**

这说明 Falcon不只是模拟器 tricks，而是确实对真实控制回路有价值。

#### 4. 消融信号：超参与选择机制确实在起作用
三类消融都比较关键：
- **阈值ϵ**：过小则找不到可复用历史状态，过大则会复用错误状态
- **探索率δ**：过低会积累历史偏差，过高又会丢失加速收益
- **固定噪声层复用**：明显不如自适应选择机制

这表明 Falcon 的有效性不是“任何 partial reuse 都行”，而是依赖：
- 当前观测条件下的一步兼容性判断
- 合理的利用/探索平衡

### 能力跃迁到底在哪里
相较 prior work，Falcon的能力跃迁主要在三点：

1. **比DDIM/DPMSolver更进一步**  
   它不是只在单次采样里少走几步，而是跨时间步复用已经算过的中间态。

2. **比蒸馏类方法更灵活**  
   不需要为每个任务重训，也不改变原policy。

3. **比SDP更轻量**  
   仍用 partial denoising 思路，但强调 training-free、plug-and-play 和较低额外内存。

### 1-2个关键指标
- **Robomimic Lift ph**：NFE **100 → 12.9**，成功率保持 **1.00**
- **真实抓取**：运行时间 **0.43s → 0.14s**，成功率保持 **100%**

---

### 局限性

- **Fails when**: 相邻控制时刻动作变化剧烈、阶段切换突然或需要高精度重规划时，历史中间态往往只能在较高噪声层被复用，因此加速收益明显变小；论文中的 Transport、ToolHang 一类复杂任务就体现出这一点。
- **Assumes**: 已有一个训练好的 diffusion policy；任务存在动作时序连续性和滚动窗口重叠；Falcon还依赖任务级超参调节（如 ϵ、δ、kmin、buffer大小），论文自己也承认尚不能用一套参数覆盖所有任务。
- **Not designed for**: 本质上不适合相邻时刻弱相关、离散跳变很强、或根本没有历史轨迹可复用的决策问题；它也不是“替代训练”的方法，基础策略训练成本依然存在。

### 复现与可扩展性的现实约束
- **训练免费**只发生在 Falcon 插件层，本体 diffusion policy 仍需预训练。
- 仿真中不少结果基于**state-based observation**，对纯像素输入的大规模泛化还需更多证据。
- Falcon的重计算部分是对buffer中样本做并行一步估计；作者称其可并行化、显存额外开销很小（Robomimic约 +12MB），但这仍假设有合适的GPU并行环境。
- 论文中**未提供明确代码链接**，因此开源可复现性目前偏弱。

### 可复用组件
这篇工作最值得迁移的不是某个公式，而是以下几个操作模板：
1. **history-conditioned warm start**：把序列建模任务的下一次采样起点放到上一轮中间态附近  
2. **one-step compatibility test**：用当前条件快速判断历史latent是否仍可复用  
3. **exploration fallback**：少量保留从噪声重启，避免长期误差积累  
4. **priority-queue latent buffer**：在显存受限下维护“最有价值的中间态缓存”

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Falcon_Fast_Visuomotor_Policies_via_Partial_Denoising.pdf]]