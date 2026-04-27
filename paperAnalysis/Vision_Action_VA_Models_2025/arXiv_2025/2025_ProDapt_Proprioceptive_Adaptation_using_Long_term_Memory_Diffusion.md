---
title: "ProDapt: Proprioceptive Adaptation using Long-term Memory Diffusion"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-navigation
  - task/imitation-learning
  - diffusion
  - keypoint-memory
  - receding-horizon
  - dataset/Isaac-Sim
  - opensource/full
core_operator: 把稀疏接触历史压缩成去重后的 keypoints，并作为长期记忆条件注入扩散控制器。
primary_logic: |
  近期本体感觉观测 + 接触检测 → keypoint manager 写入/去重/筛选历史接触关键点 → 将最多 Nkp 个 keypoints 与短时观测共同条件化扩散策略 → 生成未来动作序列并滚动执行，实现无外感知绕障到达目标
claims:
  - "On the simulated elbow setup, ProDapt achieves 80% success while all memory-less diffusion baselines with Ho in {3,6,20,50} achieve 0%, showing explicit keypoints preserve task-relevant contact information beyond any tested fixed observation window [evidence: comparison]"
  - "On the simulated bucket setup, ProDapt reaches 85.7% success with Ho=3, while low-horizon baselines fail most trials and only Ho=20/50 baselines solve all trials, indicating sparse keypoint memory recovers much of the benefit of longer histories but is not uniformly best on every obstacle geometry [evidence: ablation]"
  - "In real UR10e experiments, ProDapt finishes all four setups faster than every baseline while keeping inference time close to Ho=3/6 baselines and below Ho=20/50, supporting a better control-compute trade-off for 10 Hz deployment [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "memory-less Diffusion Policy (Chi et al. 2023)"
  complementary_to: "3D Diffusion Policy (Ze et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ProDapt_Proprioceptive_Adaptation_using_Long_term_Memory_Diffusion.pdf
category: Embodied_AI
---

# ProDapt: Proprioceptive Adaptation using Long-term Memory Diffusion

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.00193), [Code](https://tinyurl.com/prodapt-code), [Video](https://tinyurl.com/prodapt-video)
> - **Summary**: 论文把“过去在哪些位置发生过接触、接触法向如何”保存为稀疏 keypoints，作为扩散策略的长期记忆，使机器人在没有相机等外感知时仍能仅靠本体感觉绕开未知障碍。
> - **Key Performance**: 仿真 Elbow 场景成功率 80%（所有无记忆基线均为 0%）；实机四类场景的完成时间都优于所有基线，且推理时延接近短观察窗基线。

> [!info] **Agent Summary**
> - **task_path**: 短窗本体感觉观测 + 历史接触 keypoints -> 未来末端执行器位置序列
> - **bottleneck**: 仅靠固定长度观测窗时，扩散策略会遗忘早先接触事件，导致在长障碍或凹形障碍前循环、漂移或依赖隐式启发式
> - **mechanism_delta**: 在扩散策略前加入 keypoint manager，把唯一且信息量高的接触事件写入固定长度记忆槽，再与短时观测一起作为条件输入
> - **evidence_signal**: Elbow 仿真中 ProDapt 80% 成功而所有 Ho≤50 的无记忆扩散基线 0%，且实机趋势一致
> - **reusable_ops**: [contact-triggered memory writing, distance/angle-based keypoint deduplication]
> - **failure_modes**: [out-of-distribution 障碍几何下在 elbow 顶部闲置或超调, concave bucket 中仍可能贴壁卡住]
> - **open_questions**: [能否把 keypoint 管理学习化并与扩散端到端联合训练, 能否扩展到动态障碍与三维接触场景]

## Part I：问题与挑战

这篇工作真正要解决的，不是“扩散模型能不能控制机器人”，而是：

**当机器人只能看到自己的本体感觉时，扩散策略如何记住很久以前发生过、但对当前决策仍然关键的接触事件？**

### 1. 问题场景
作者关注的是一类 **外感知可能失效** 的机器人任务：比如月面、海底、粉尘/泥沙遮挡环境。  
此时相机、LiDAR 等 exteroception 可能不可用，机器人只能依赖：

- 末端位置等本体状态
- 力/力矩等接触反馈

目标是在 **未知静态障碍** 中，从起点走到目标点。

### 2. 真正瓶颈
现有 diffusion policy 虽然很擅长从 demonstrations 学习多模态动作，但它通常依赖：

- 视觉输入，或
- 固定长度的历史窗口 `Ho`

而在本体感觉场景里，**最关键的信息往往是稀疏、事件驱动、且需要长期保留的**：

- “我之前在这里撞过”
- “这个接触的法向朝哪边”
- “这块区域已经尝试过，不该再从这边绕”

如果只看最近几步，策略会忘记旧碰撞；  
如果单纯把 `Ho` 拉长，又会遇到两个问题：

1. **计算成本线性/显著增加**，不利于实时控制  
2. **记忆仍然有限**，一旦任务依赖超过窗口长度，还是会忘

### 3. 输入/输出接口
这篇论文的控制接口很清楚：

- **输入**：
  - 最近 `Ho` 步本体感觉观测
  - 若干长期保存的接触 keypoints
- **输出**：
  - 未来 `Hp` 步动作序列
  - 实际只执行前 `Ha` 步，然后滚动重规划

在实验里，动作为 **末端执行器的 x-y 目标位置**。

### 4. 边界条件
这篇工作并不是一个通用 blind robotics 全解法，它的边界很明确：

- 主要验证 **2D 平面内** 的末端运动
- 障碍物是 **未知但静态**
- 依赖 **接触可检测** 的本体感觉
- 训练数据来自 **仿真 teleoperation demonstrations**

---

## Part II：方法与洞察

### 方法骨架

ProDapt 的核心不是改扩散损失，也不是换 backbone，而是在扩散控制器前面加了一个 **长期记忆层**。

#### 1. 用 keypoint manager 管“什么该记住”
当机器人发生接触时，系统判断：

- 这是不是一次有效接触
- 它和已有接触是否足够不同
- 它在当前是否值得保留

如果值得保留，就写成一个 **keypoint**。

在 UR10e 实验中，一个 keypoint 包含：

- 接触发生时的末端 `x-y` 位置
- 接触法向的方向信息（用 `sin/cos` 表示）

#### 2. 用“稀疏事件记忆”代替“长时间密集历史”
作者不是把整段历史都保留，而是只保留**改变未来决策的关键事件**。  
这是一种非常明确的信息压缩：

- 从“所有时刻”
- 压缩为“关键接触事件”

#### 3. 把 keypoints 作为 diffusion conditioning
扩散控制器的输入从：

- 仅最近 `Ho` 步观测

变为：

- 最近 `Ho` 步观测
- 最多 `Nkp` 个 keypoints

若 keypoints 数量不够，就 zero-padding 到固定长度。

#### 4. 用标准 receding-horizon diffusion 执行
作者延续 Diffusion Policy 风格的 receding-horizon 控制：

- 预测 `Hp=20` 步
- 执行 `Ha=10` 步
- 下一轮再结合最新观测 + memory 重生成

这让方法改动集中在 **记忆表示**，而不是整套控制框架。

### 核心直觉

**变化了什么？**  
从“依赖固定长度最近历史”改成“短期历史 + 长期事件记忆”。

**哪个约束/信息瓶颈被改变了？**  
把控制器面对的瓶颈，从：

- **recency bottleneck**：只能看最近、旧信息会过期

改成：

- **relevance bottleneck**：只保留对未来动作真正重要的接触事件

**能力因此怎么变？**  
策略不再需要从很长的原始时序里“自己推断哪里撞过”，而是直接获得一种压缩过的环境交互痕迹，因此能：

- 避免撞完就忘
- 避免在凹形/长障碍里来回循环
- 在实时预算内保持更长的“任务记忆”

更具体地说，**接触位置 + 接触法向** 提供了局部障碍边界的稀疏线索。  
对 diffusion policy 来说，这已经足够回答一个关键决策问题：

> “当前这一侧是不是已经试过/撞过了？是否应该换另一侧绕行？”

这也是为什么作者不做显式建图：  
**从稀疏接触点恢复完整地图太重，而且对控制未必必要。**

### 为什么这个设计有效
因果链条可以概括为：

**显式写入接触事件**  
→ 不再要求模型从长序列隐式记忆  
→ 关键障碍信息不会因 `Ho` 太短而丢失  
→ 策略能根据“已探索过的碰撞证据”切换绕障方向  
→ 在只有 proprioception 时仍能完成目标导向导航

### 战略取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 短观察窗 + keypoints | 长序列记忆与实时性冲突 | 推理快、仍保留长期信息 | 依赖 keypoint 质量 |
| 稀疏事件记忆而非显式建图 | 稀疏接触难以恢复完整环境 | 实现简单，不需地图模块 | 缺乏全局几何一致性 |
| 手工 keypoint manager | 端到端记忆难训练 | 可控、易调试、容易部署 | 阈值工程化、任务依赖强 |
| 固定 `Nkp` + padding | 变长记忆难接到标准网络 | 容易接入现有 diffusion backbone | 容量有限，旧关键点可能被挤掉 |
| 保留“唯一接触”而非全部接触 | 历史冗余导致计算浪费 | 提高信息密度 | 去重规则若不合理会漏掉关键事件 |

---

## Part III：证据与局限

### 关键证据信号

#### 信号 1：真正的能力增益出现在“超出固定窗口记忆”的任务
最强证据来自 **Elbow** 场景：  
机器人先被斜墙引导，再碰到一段更长的竖墙。想成功，必须记住“肘部底部已经试过，应该换策略”。

- **ProDapt（仿真）**：80% 成功
- **所有无记忆扩散基线（Ho=3/6/20/50）**：0%

这说明论文的增益不是简单“更稳一点”，而是**从失败模式上跨过了一道能力边界**：  
有限窗口历史无法覆盖的长期依赖，显式 keypoint memory 可以。

#### 信号 2：把 Ho 拉长并不能完全替代 keypoints
在 **Bucket** 凹形障碍中：

- ProDapt：85.7%
- Ho=20/50 基线：100%
- Ho=3/6 基线：大多失败

这很重要，因为它说明论文并没有证明“keypoints 永远最强”，而是证明：

1. **低成本短窗 + 稀疏长期记忆** 能显著优于短窗无记忆
2. 对某些几何，**更长连续上下文** 仍有价值
3. ProDapt 的优势主要在 **长期事件记忆/实时性折中**，不是在所有 obstacle type 上绝对碾压

这是一个比较诚实、也更可信的结论。

#### 信号 3：实机趋势复现，说明不是纯 simulator artifact
作者在真实 UR10e 上重复了四类实验，整体趋势与仿真一致：

- ProDapt 在所有 setup 上都比基线更快完成
- 长观察窗基线在实机上更容易受时延影响
- 低观察窗基线在 clear/wall 上都表现更差，说明短期记忆不足在真实执行里会被放大

这支持论文不是“只在仿真中靠巧合成立”。

#### 信号 4：计算-能力折中是方法的重要卖点
作者专门测了单次 diffusion inference 时间。结论是：

- ProDapt 的时延接近 `Ho=3/6` 的短窗基线
- 明显低于 `Ho=20/50` 的长窗基线
- 更大的 `Ho` 已经开始逼近或超过 10 Hz 实时控制预算

因此，这篇论文的核心价值不是单纯“把成功率做高”，而是：

> **用稀疏长期记忆替代密集长序列输入，在保持实时性的同时跨过部分长期依赖难题。**

#### 信号 5：行为层面的失败分析与方法假设吻合
作者观察到无记忆基线会学出一种奇怪偏置：

- 即便 clear 场景没障碍，也会先向右斜漂
- 碰到障碍后再沿墙走

这表明基线因为缺少长期接触记忆，只能用一种隐式启发式来“赌”绕障方向。  
而 ProDapt 没有这种明显偏置，更接近示教者的直接目标导向行为。

### 1-2 个关键指标
- **Elbow / 仿真**：ProDapt 80% 成功；所有无记忆基线 0%
- **Bucket / 仿真**：ProDapt 85.7%，显著优于低观察窗基线，但低于 Ho=20/50

### 局限性

- **Fails when**: 障碍几何超出训练分布时，策略会在 elbow 顶部附近闲置、慢移或 overshoot；在 concave bucket 中也可能贴壁卡住。若任务中接触很少、接触不稳定，keypoint 记忆本身也难以提供有效线索。
- **Assumes**: 需要可靠的接触检测与法向估计；keypoint manager 使用人工阈值与规则（如 torque 阈值、距离/角度去重），需要按任务调参；默认障碍基本静态；训练依赖 165 条仿真 teleop 轨迹；训练使用 A100 80GB 级 GPU，实机 10 Hz 推理依赖 NUC + RTX 3080 Ti 级硬件。
- **Not designed for**: 全 3D 操作、动态障碍、多机器人协作、显式全局建图/规划，也不是一个端到端学习的 memory architecture；它当前更像是在 diffusion policy 前加一个工程化的长期记忆模块。

### 可复用组件
这篇工作最值得复用的不是整套 UR10e 细节，而是下面几个操作：

- **contact-triggered memory writing**：只在关键事件发生时写记忆
- **distance/angle-based deduplication**：把冗余碰撞压缩成少量高信息量 keypoints
- **fixed-slot memory conditioning**：把变长历史转成固定长度条件输入，方便接入现有生成式策略
- **short-horizon control + sparse long-horizon memory**：把短期控制与长期任务证据分开建模

### 一句话判断
这篇 paper 的贡献很聚焦：  
**它不是发明了更强的 diffusion controller，而是把“长期接触记忆”做成一个可插拔、低时延、对 blind manipulation 真有帮助的条件模块。**  
最强证据来自 elbow 这类“必须记住很久以前碰撞”的场景；最大短板则是 memory manager 仍然是手工工程，而非学习得到。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ProDapt_Proprioceptive_Adaptation_using_Long_term_Memory_Diffusion.pdf]]