---
title: "Learning a High-quality Robotic Wiping Policy Using Systematic Reward Analysis and Visual-Language Model Based Curriculum"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - reinforcement-learning
  - curriculum-learning
  - reward-shaping
  - dataset/MuJoCo
  - dataset/robosuite
  - opensource/no
core_operator: "用同心检查点把接触/力控制奖励改成有界回报，并让VLM/LLM按训练进展动态重配奖励权重。"
primary_logic: |
  随机曲率/摩擦表面上的航点与本体/力觉观测 → 用检查点限制质量奖励累计上界以消除“无限擦拭”局部最优，再由VLM总结失败、LLM调整奖励权重 → 输出兼顾完成率、目标压力与效率的盲擦拭策略
claims:
  - "在随机化的 2-point 模拟擦拭环境上，用有界检查点奖励替代朴素奖励后，800k 步训练、5 个随机种子平均导航成功率从 58% 提升到 92% [evidence: comparison]"
  - "在有界奖励基础上再加入 VLM 课程后，成功率进一步提升到 98%，平均完成步数从 29 降到 25，力控制 IAE 从 333 降到 243 [evidence: comparison]"
  - "作者的收敛性分析表明，朴素的“逐步质量奖励 + 终止完成奖励”难以同时压制 lazy 与 forever wiping 两种次优策略；引入检查点有界奖励后可恢复可行的正终止奖励区间 [evidence: theoretical]"
related_work_position:
  extends: "EUREKA (Ma et al. 2023)"
  competes_with: "TEXT2REWARD (Xie et al. 2023); Robotic table wiping via reinforcement learning and whole-body trajectory optimization (Lew et al. 2023)"
  complementary_to: "Domain Randomization (Tobin et al. 2017); Learning generalizable surface cleaning actions from demonstration (Elliott et al. 2017)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_a_High_quality_Robotic_Wiping_Policy_Using_Systematic_Reward_Analysis_and_Visual_Language_Model_Based_Curriculum.pdf
category: Embodied_AI
---

# Learning a High-quality Robotic Wiping Policy Using Systematic Reward Analysis and Visual-Language Model Based Curriculum

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.12599), [Project/Demo](https://sites.google.com/view/highqualitywiping)
> - **Summary**: 这篇工作把“擦得好”和“擦得快”的冲突正式建模为一个质量关键型 RL 问题，并用“有界奖励 + VLM/LLM 自动调参课程”把原本极易失稳的训练过程变成可收敛、少人工调参的擦拭策略学习流程。
> - **Key Performance**: 800k steps 后在随机曲率/摩擦 2-point 任务上达到 **98% 导航成功率**；同时实现 **243 force IAE / 25 平均完成步数**。

> [!info] **Agent Summary**
> - **task_path**: 已知航点 + 本体/力觉观测、未知曲率/摩擦表面 -> 6DoF 末端位姿控制 -> 满足目标压力的擦拭轨迹
> - **bottleneck**: 固定奖励比例下，逐步质量奖励会诱导“无限擦拭”，而过强终止奖励又诱导草率完成，导致 reward engineering 极度敏感
> - **mechanism_delta**: 把按时间累计的质量奖励改成按空间进度触发的有界奖励，并周期性用 VLM/LLM 根据失败模式重配奖励权重
> - **evidence_signal**: 5-seed 对比中 success 由 58%→92%→98%，且 bounded reward 单独提升完成率、课程再把力误差从 333 降到 243
> - **reusable_ops**: [concentric-checkpoint reward gating, VLM-assisted reward-weight curriculum]
> - **failure_modes**: [unbounded rewards cause perpetual wiping, bad reward ratios cause no-contact or sloppy fast completion]
> - **open_questions**: [能否稳定迁移到真实机器人, 无给定航点时能否端到端完成擦拭]

## Part I：问题与挑战

这篇论文要解决的，不只是“让机器人沿着表面移动”这么简单，而是一个更麻烦的目标：**既要保持高质量接触擦拭，又要尽快完成任务**。作者把这类任务称为 **quality-critical MDP**。

### 真正的难点是什么？
在擦拭任务里，策略同时面对两类目标：

1. **过程质量**：持续接触、把法向力维持在目标值附近（文中是 60N）、动作不要太激进。
2. **任务完成**：尽快擦到 waypoint，并结束 episode。

问题在于，常见 RL 写法会把这两类目标放进同一个 reward：
- 过程质量是 **每步 dense reward**
- 任务完成是 **episode 末端 sparse reward**

这会带来一个结构性冲突：
- 如果 dense quality reward 太强，策略可能学会 **一直擦、不结束**，因为“拖时间”本身也能拿奖励。
- 如果 terminal completion reward 太强，策略会学会 **草率完成**，比如大动作跳 waypoint、质量变差。

所以瓶颈不是网络不够大，而是：**目标函数本身让最优策略空间不稳定**。这也是为什么现实机器人 RL 经常卡在繁琐的 reward engineering 上。

### 输入 / 输出接口
这项工作研究的是 **blind wiping policy**，即推理时不依赖视觉感知：
- **输入**：46 维观测，包括 waypoint 信息、关节位置/速度、末端位姿、力/力矩传感器值
- **输出**：6 维末端 pose control
- **场景边界**：无障碍表面、给定 waypoint、目标是学会接触与力控制，而不是做视觉检测或污渍定位

### 为什么现在值得做？
因为在不同曲率、不同摩擦的表面上，传统模型控制器很难统一设计；RL 理论上更适合吃下这种不确定性，但它在落地前通常先死在 reward tuning 上。作者的切入点很明确：**先把 reward landscape 变成“可学的”，再谈策略泛化。**

---

## Part II：方法与洞察

方法由两个互补部件组成：

1. **Bounded Reward Design**：通过同心圆 checkpoint，把“按时间累积的质量奖励”改成“按空间推进触发的质量奖励”
2. **VLM-based Curriculum**：用 LLM/VLM 周期性查看训练指标和失败回放，自动调 reward weights

### 方法主线

#### 1) 有界奖励：先解决“无限擦拭”局部最优
作者先分析了三种典型策略：
- **optimal**：高质量地擦，并在合理时间内完成
- **lazy**：质量差但尽快完成
- **forever**：一直维持看似不错的接触质量，但故意不完成任务

朴素 reward 的核心问题是：`forever` 策略可以持续收集每步质量奖励，因此很容易在折扣回报下变成强局部最优。

作者的改法很巧妙：在 waypoint 周围放置 **同心 checkpoint 区域**。  
只有在进入新的 checkpoint 区域时，接触/力控制奖励才触发；不是每一步都给。

这样一来：
- “擦得好”仍然有奖励
- 但奖励和**空间进展**绑定，而不是和“拖时间”绑定
- 当 checkpoint 都被走完后，继续磨蹭只剩动作惩罚，不再有正收益

于是 `forever wiping` 的回报上界被切断，目标函数才变得真正可优化。

#### 2) VLM 课程：再解决“固定权重仍然很敏感”
即便 reward 变得可行，训练仍然很依赖权重比例。作者因此又加了一个 **训练时元控制器**：

- 初始训练一段时间后（文中 300k steps）
- 每隔固定步数（文中 100k）做一次评估
- 如果指标没有满足维护条件，就调用：
  - **LLM** 看成功率、压力统计等日志，决定该调哪些 reward weights
  - **VLM** 在需要时查看失败回放图像，解释失败原因，如“还没接触桌面就结束了”或“接近终点但没真正完成”

关键点是：**VLM/LLM 不参与机器人在线控制，只参与训练阶段的奖励重加权**。  
这让它更像“自动化的人类 reward engineer”，而不是一个端到端策略网络。

### 核心直觉

**改了什么？**  
把“按时间付费的质量奖励”改成“按进度付费的质量奖励”；把“固定 reward ratio”改成“随训练阶段变化的 reward ratio”。

**改变了哪个瓶颈？**  
- 有界奖励改变的是 **回报分布的结构**：不再允许“无进展但持续高回报”
- 课程调权改变的是 **多目标学习的优化顺序**：先确保成功轨迹出现，再逐步压低力误差和落地冲击

**能力为什么会变强？**  
因为策略不再需要在训练一开始就同时完美兼顾“到达、接触、稳力、平滑”。  
作者实际上把一个难以同时满足的静态多目标问题，拆成了一个可被逐步塑形的动态优化过程。

### 战略取舍

| 设计选择 | 改变的约束/信息瓶颈 | 带来的能力 | 代价/风险 |
| --- | --- | --- | --- |
| 同心 checkpoint 有界奖励 | 把无上界的 dense quality return 变成与空间推进绑定的有限回报 | 大幅抑制 perpetual wiping，提高完成率 | 质量监督变得更稀疏，单独使用时可能牺牲力精度 |
| VLM/LLM 课程调权 | 把固定多目标权重改成阶段性自适应重平衡 | 能从“先学会到达”过渡到“再学会高质量接触” | 依赖闭源 GPT-4 / GPT-4V，且触发规则仍需人工先设 |
| Blind policy + 已知 waypoint | 把问题聚焦到接触控制，而非视觉感知 | 更清楚地研究 reward 与 force control | 不解决 waypoint 生成、污渍识别、障碍规避 |

---

## Part III：证据与局限

### 关键实验信号

#### 信号 1：bounded reward 直接验证了“无限擦拭”确实是主故障模式
在 MuJoCo/robosuite 随机表面环境中，作者比较三种方法：
- non-bounded-reward
- bounded-reward
- bounded-llm-curr

最关键的第一跳是：
- **成功率 58% → 92%**
- **平均步数 38 → 29**

这说明仅仅改 reward structure，就已经显著减少了“不结束 episode”的坏局部最优。  
这条证据最直接支撑论文的核心理论分析。

#### 信号 2：课程调权不是锦上添花，而是把“会完成”变回“高质量完成”
有趣的是，单独的 bounded reward 虽然让成功率更高，但力控制 IAE 反而变差：
- non-bounded: IAE 267
- bounded: IAE 333

这说明：**只解决完成率，不等于质量也变好**。

加入 VLM curriculum 之后：
- **成功率 92% → 98%**
- **IAE 333 → 243**
- **步数 29 → 25**

也就是：课程模块把“先学会完成”进一步推成了“又快又稳地完成”。

#### 信号 3：在坏初始化下，课程模块能救训练
作者还做了一个更有说服力的 stress test：把 completion reward 初始化成只有 quality reward 的 10%。

结果：
- bounded-reward 在 600k steps 后几乎仍然学不出来
- bounded-llm-curr 在 500k steps 时可恢复到 **40% success**

这说明课程模块不只是微调，而是真能在早期探索阶段纠偏 reward ratio。

#### 信号 4：VLM 的价值体现在“失败语义解释”
作者给出案例表明，VLM 能从失败回放中指出：
- 失败发生在很早阶段
- 还没真正接触桌面
- 或接近终点但没完成 wipe

这类信息很像研究者人工看日志后做的判断。  
因此 VLM 的作用不在于提高策略表达能力，而在于**降低 reward debugging 的人工成本**。

### 局限性

- **Fails when**: 任务需要视觉污渍定位、障碍规避、未知物体几何适配，或需要长时多阶段清洁流程时，这个“blind + given waypoints”的设定会失效；若接触事件更稀少、动力学更复杂，当前课程也未必足够稳定。
- **Assumes**: 已知 waypoint；固定目标力（60N）；MuJoCo/robosuite 模拟与域随机化能够覆盖关键现实变化；训练中可访问 GPT-4 与 gpt-4-vision-preview；维护条件和触发谓词由人工预定义。
- **Not designed for**: 端到端视觉到动作的擦拭、自动生成 waypoint、真实机器人部署闭环验证。

### 复现与资源依赖
这篇工作的可扩展性有一个现实前提：  
它依赖 **闭源大模型 API** 做课程调权，而不是纯本地可复现流程；同时实验只在模拟中完成，没有真实机器人结果。论文提供了 demo 网站，但未说明代码开源，因此工程复现成本不会低。

### 可复用组件
1. **Checkpoint-gated dense reward**：适合任何“过程质量重要，但又必须及时完成”的接触式任务，如打磨、抛光、表面巡检。
2. **训练期元控制课程**：用轻量统计 + 偶发 VLM 失败诊断来调 reward weights，而不是每次重写 reward function。
3. **先可行、后精修的多目标训练逻辑**：先让成功轨迹出现，再细化质量指标，比一开始同时优化所有目标更稳。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_a_High_quality_Robotic_Wiping_Policy_Using_Systematic_Reward_Analysis_and_Visual_Language_Model_Based_Curriculum.pdf]]