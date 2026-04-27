---
title: "SpikePingpong: High-Frequency Spike Vision-based Robot Learning for Precise Striking in Table Tennis Game"
venue: ICLR
year: 2026
tags:
  - Embodied_AI
  - task/robotic-table-tennis
  - spike-vision
  - imitation-learning
  - transformer
  - dataset/TT2
  - "dataset/Ping Pong Detection"
  - opensource/promised
core_operator: 先用RGB-D与物理模型快速给出可击点，再用脉冲视觉监督的残差校正器补偿真实世界偏差，最后以目标条件模仿学习直接输出击球关节修正。
primary_logic: |
  来球RGB-D轨迹与历史速度/位置 + 目标落点条件 → System 1快速物理预测可击点，System 2利用脉冲视觉学到的偏差校正补偿旋转/空气阻力/感知误差，IMPACT再输出关节角微调 → 机器人在指定区域完成高精度回击
claims:
  - "Claim 1: Fast-Slow感知将整体球拍接触误差从仅用System 1时的44.13 MAE / 50.62 RMSE降到12.34 / 13.85，并优于RNN残差预测器的22.80 / 23.73 [evidence: comparison]"
  - "Claim 2: 在真实机器人平台上，SpikePingpong在四个目标区域达到30cm精度92%平均成功率、20cm精度70%，明显超过人类平均水平(53%/33%)、ACT和Diffusion Policy基线 [evidence: comparison]"
  - "Claim 3: 去掉神经校正后单目标回球成功率从92%降到23%，而完整系统仍能在100球顺序任务中达到78%，并在未见发球机位置的OOD设置下保持74%的30cm命中率 [evidence: ablation]"
related_work_position:
  extends: "GoalsEye (Ding et al. 2022)"
  competes_with: "ACT (Zhao et al. 2023); Diffusion Policy (Chi et al. 2023)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_SpikePingpong_High_Frequency_Spike_Vision_based_Robot_Learning_for_Precise_Striking_in_Table_Tennis_Game.pdf
category: Embodied_AI
---

# SpikePingpong: High-Frequency Spike Vision-based Robot Learning for Precise Striking in Table Tennis Game

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.06690)
> - **Summary**: 这篇工作把“高速乒乓拦截”拆成快物理预测、慢脉冲视觉校正和目标条件模仿击球三步，从而在真实机器人上实现了低时延且高精度的指定落点回球。
> - **Key Performance**: 30cm目标区平均成功率 **92%**，20cm目标区 **70%**；动作生成推理时延 **0.407 ms**

> [!info] **Agent Summary**
> - **task_path**: RGB-D来球轨迹/速度历史 + 目标落点区域 -> 校正后的可击点与机器人击球关节修正 -> 指定区域回球
> - **bottleneck**: 高速球的真实轨迹与理论物理模型之间存在系统性偏差，且这类厘米级接触误差会在击球后被放大成明显落点误差
> - **mechanism_delta**: 用“物理先验粗预测 + 脉冲视觉监督残差校正 + 目标条件模仿学习”替代纯物理或高时延端到端策略
> - **evidence_signal**: 真实平台上同时给出接触误差下降、30cm/20cm目标命中率提升、0.407ms低时延和消融收益
> - **reusable_ops**: [物理先验+残差校正, 目标条件模仿学习]
> - **failure_modes**: [极端旋转导致near-miss, 未见人类击球风格下性能下降]
> - **open_questions**: [能否在线显式估计spin而不依赖离线spike监督, 能否扩展到长回合对抗与对手建模]

## Part I：问题与挑战

这篇论文解决的不是一般的“机器人会不会打到球”，而是**能否在几十毫秒窗口里，把高速来球打到指定小区域**。  
真正难点有两个：

1. **感知瓶颈**：普通RGB相机在高速小球上容易模糊，物理模型又无法完整描述空气阻力、旋转、弹跳差异。
2. **控制瓶颈**：乒乓球对接触时刻和拍面接触点极其敏感，哪怕是厘米级误差，也会放大成几十厘米的落点偏差。

论文的输入/输出接口很清楚：

- **输入**：RGB-D观测到的球轨迹历史、速度历史、System 1预测的可击点，以及目标落点区域。
- **输出**：校正后的可击位置，以及用于击球的关节角修正。

它的边界条件也很明确：  
这是一个**固定球台、固定机械臂、固定击球平面、四分区落点控制**的系统，主要面向发球机来球；对人类来球只做了有限适应与泛化测试，还不是完整意义上的开放式竞技对打系统。

**为什么现在值得做？**  
因为动态机器人正在从“静态抓取”转向“高速交互”，而 spike camera 这类高时间分辨率传感器提供了一个新可能：不一定把它放进部署链路里实时跑，但可以把它作为**高质量教师信号**，去监督一个轻量校正器。

## Part II：方法与洞察

论文把问题拆成两个阶段：**拦截** 和 **击球**。

### 1）System 1：快而粗的物理预测

System 1 用 RGB-D 相机检测球，并基于简化弹道模型估计球何时、何地会到达击球平面。  
它的价值不是“绝对准”，而是“**先足够快地给出一个能用的可击点**”。

这一步的好处是：

- 低延迟
- 易解释
- 能直接输出机器人逆运动学所需的中间量

但它天然会被真实世界因素破坏：旋转、空气效应、传感噪声、桌面反弹差异都会让理论可击点偏掉。

### 2）System 2：用 spike supervision 学一个“偏差校正器”

这一步是全篇最关键的设计。  
作者没有让 spike camera 在线参与决策，而是用它在训练时捕捉**球拍接触瞬间**，测出“球中心”和“拍中心”的偏移，把它当作监督信号。

于是，System 2 学的不是整条轨迹，而是一个更可学的问题：

- 输入：位置历史、速度历史、System 1 预测的可击点
- 输出：**理论可击点到真实最佳接触点的偏差**

这相当于把“难建模的真实世界动力学”变成了一个**残差学习问题**。

### 3）IMPACT：目标条件的模仿学习击球

有了更准的接触点，还需要决定**怎么打到指定区域**。  
IMPACT 的做法不是扩散采样、也不是强化学习在线试错，而是：

- 先用 Fast-Slow 系统把球接到一个稳定击球位置
- 再对 3 个关键关节做随机扰动
- 只保留成功回球样本
- 把“来球轨迹 + 当前关节 + 目标区域”映射到“关节修正量”

本质上，这是一个**目标条件的低时延策略回归器**。

### 核心直觉

这篇论文真正改变的，不是网络更大，而是**问题表述方式**：

- 从“直接预测真实轨迹/动作”  
  变成  
  “先用物理先验给粗解，再只学物理模型的系统误差”

- 从“慢速采样式动作生成”  
  变成  
  “一次前向的目标条件关节回归”

对应改变了两个瓶颈：

1. **信息瓶颈**：高速接触真值原本不可见，现在被 spike camera 离线采成高精度监督。
2. **时延瓶颈**：动作生成不再依赖慢采样或复杂规划，而是一次回归直接输出。

最终带来的能力变化是：

- 接触点更准
- 推理更快
- 可以稳定做“指定区域回球”，而不只是“把球打回去”

### 策略权衡

| 设计选择 | 带来的收益 | 代价/风险 |
|---|---|---|
| 物理预测 + 神经残差校正 | 兼顾低时延与真实世界精度 | 依赖高质量标定，且物理模型仍决定粗解范围 |
| spike camera只用于训练 | 部署阶段不必承担20kHz视觉计算负担 | 数据采集门槛高，需要专用高速硬件 |
| 目标条件模仿学习替代扩散/RL | 推理极快，适合时间敏感控制 | 泛化能力依赖示范覆盖，超出分布时会掉点 |
| 先拦截再击球的任务分解 | 工程上稳定，便于定位误差来源 | 不能端到端联合优化所有误差链条 |

## Part III：证据与局限

### 关键证据信号

- **比较信号：接触精度显著提升**  
  Fast-Slow 完整系统把整体接触误差降到 **12.34 MAE / 13.85 RMSE**，明显优于仅用物理预测的 **44.13 / 50.62**，也优于 RNN 校正的 **22.80 / 23.73**。  
  这直接说明：作者解决的核心不是“检测到球”，而是“把理论可击点修正到真实可打点”。

- **比较信号：落点控制能力跨越式提升**  
  在四个目标区域上，系统达到 **92% 的30cm命中率** 与 **70% 的20cm命中率**，显著高于人类平均 **53% / 33%** 以及 ACT、Diffusion Policy。  
  这说明它不仅能拦到球，还能把球**打到想要的位置**。

- **时延信号：策略足够快，适合高速任务**  
  SpikePingpong 动作推理仅 **0.407 ms**，对比 ACT 的 **7.15 ms** 和 Diffusion Policy 的 **25.18 ms**，说明作者的设计确实针对“高速任务中的控制时延”做了机制层面的优化。

- **泛化/鲁棒性信号：有一定外推，但仍受分布限制**  
  100球顺序目标任务达到 **78%**；发球机位置改变后的 OOD 测试仍有 **74%** 的30cm命中率。  
  但在人类对打上，即便对单个玩家用100条示范微调，seen player 也只有 **47%**，zero-shot 到 unseen player 下降到 **31%**。这说明其泛化仍主要停留在“受控变化”而非“开放人类风格”。

- **失败分析信号：主要是 precision issue，不是完全失控**  
  失败案例里 **79.1%** 是“落在正确象限，但超出30cm圈”的 near-miss。  
  这很重要：它表明系统主要短板是**细粒度精度**，而不是基本回球能力。

### 局限性

- **Fails when**: 遇到强旋转、复杂人类击球风格、明显偏离训练分布的来球时，未建模的 Magnus 效应与高时空敏感性会把小的接触误差放大成 near-miss。
- **Assumes**: 依赖高质量相机-机器人标定、固定击球平面、ABB IRB-120 + EGM 控制链路、20kHz spike 相机进行离线偏差标注、以及真实世界收集的1k校正样本和2k击球示范；代码在文中为“将公开”，当前复现门槛仍高。
- **Not designed for**: 显式旋转估计、长回合博弈中的在线对手建模、跨硬件零调参迁移、完全开放式人机竞技对打。

### 可复用组件

- **物理先验 + 高频教师监督的残差校正**：适合所有“快、但物理模型不完美”的动态拦截任务，如抛射物拦截、抓取飞行物。
- **自动化随机扰动 + 成功样本筛选的目标条件模仿学习**：适合需要低时延、目标可控的机器人击打/投掷/快速操作任务。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_SpikePingpong_High_Frequency_Spike_Vision_based_Robot_Learning_for_Precise_Striking_in_Table_Tennis_Game.pdf]]