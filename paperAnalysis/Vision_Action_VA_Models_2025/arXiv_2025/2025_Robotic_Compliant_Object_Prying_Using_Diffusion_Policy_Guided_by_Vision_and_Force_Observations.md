---
title: "Robotic Compliant Object Prying Using Diffusion Policy Guided by Vision and Force Observations"
venue: "IEEE Robotics and Automation Letters"
year: 2025
tags:
  - Embodied_AI
  - task/robotic-disassembly
  - task/robotic-manipulation
  - diffusion
  - cross-attention
  - sensor-fusion
  - dataset/BatteryPryingTask
  - opensource/full
core_operator: "以力信号作为查询、视觉特征作为键值做跨注意力融合，让扩散策略在接触丰富撬取任务中更可靠地识别接触状态并切换动作模式"
primary_logic: |
  腕部RGB图像 + 末端力 + 末端位姿 → 力信号同步与增强（缩放+噪声）→ 以力查询视觉的跨注意力联合嵌入 → 扩散策略生成并滚动执行6DoF delta动作序列 → 完成插入、撬动与抬升
claims:
  - "在12个真实物体、120次机器人测试中，所提 DP-CA 总体成功率达到 96%，相比视觉-only Diffusion Policy 的 39%高出 57 个百分点 [evidence: comparison]"
  - "将力与视觉做简单拼接或线性升维后再融合，仅达到 48% 和 57% 成功率；跨注意力联合嵌入提升到 96%，表明关键增益来自关系建模而非仅仅维度匹配 [evidence: ablation]"
  - "模型在未见条件下保持较强泛化：未训练的 AA 与 C 电池平均成功率分别为 97% 与 100%，串联电池和新颜色电池测试成功率分别为 95% 与 90% [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2024)"
  competes_with: "Diffusion Policy (Chi et al. 2024); TacDiffusion (Wu et al. 2024)"
  complementary_to: "Universal Manipulation Interface (Chi et al. 2024); ManiWav (Liu et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Robotic_Compliant_Object_Prying_Using_Diffusion_Policy_Guided_by_Vision_and_Force_Observations.pdf
category: Embodied_AI
---

# Robotic Compliant Object Prying Using Diffusion Policy Guided by Vision and Force Observations

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.03998), [Project](https://rros-lab.github.io/diffusion-with-force.github.io/)
> - **Summary**: 本文把“力作 query、图像作 key/value”的跨注意力引入扩散策略，使机器人在电池撬取这类高精度接触任务中更稳地完成插入-撬动-抬升，并显著提升对未见电池与物体的泛化。
> - **Key Performance**: 真实机器人总体成功率 **96%**；相对视觉-only 扩散策略提升 **57 个百分点**。

> [!info] **Agent Summary**
> - **task_path**: 腕部RGB + 末端力/位姿 / 电池撬取场景 → 6DoF delta动作序列
> - **bottleneck**: 低维力信号在高维视觉特征中被稀释，导致接触状态识别和“插入→撬动→抬升”模式切换不稳定
> - **mechanism_delta**: 将简单拼接改为“力查询视觉”的跨注意力联合嵌入，并对力做随机缩放与噪声增强
> - **evidence_signal**: 12物体、120次真实机器人对比中，成功率从 39%/48%/57% 提升到 96%
> - **reusable_ops**: [force-as-query cross-attention, force scaling+noise augmentation]
> - **failure_modes**: [AAA紧公差下工具与缝隙轻微错位, 深腔体或初始粗定位偏差较大时插入与抬升不稳]
> - **open_questions**: [如何在不牺牲成功率的前提下显式限制最大外力, 能否扩展到更深腔体与更广泛接触丰富拆解任务]

## Part I：问题与挑战

这篇论文解决的是一个很“硬”的机器人拆解问题：**顺应性物体的撬取**。具体到场景，就是从带弹簧或紧配合结构的电池仓里，把 AAA/AA/C/D 电池撬出来。任务表面上像“插进去再撬一下”，但实际是一个多阶段、强接触、低容错过程：

1. 接近电池端部  
2. 对准狭缝并插入工具  
3. 在保持接触的同时施力撬动  
4. 抬升并带出电池  

### 真正难点是什么？

真正的瓶颈不是“扩散策略会不会生成多模态动作”，而是：

- **视觉不足以判断接触是否成立**：从图像看，“工具已经进缝”与“工具只是顶在壳体上”可能非常像。
- **力虽然关键，但维度很低**：直接把 3 轴力或 4 维力特征拼到高维图像特征后面，网络容易主要依赖图像，导致力被“淹没”。
- **任务依赖模式切换**：插入、撬动、抬升对应不同动作模式；如果接触状态判断错，机器人会过早撬、力不足、或抬升时失去接触。

所以，这篇论文回答的核心问题其实是：

> 如何让扩散策略**真正使用**力反馈，而不是“形式上接了力传感器，实际上还是看图像在猜”。

### 为什么现在值得解决？

- 电池回收和产品拆解需求在上升，自动化拆解有现实工业价值。
- Diffusion Policy 已经证明自己适合学习低层机器人技能，特别是多模态动作分布。
- 但在**接触丰富任务**里，单纯视觉策略不够，如何把力融进策略仍是空缺。

### 输入 / 输出接口与边界条件

**输入观测：**
- 腕部 RGB 图像
- 末端 3 轴力（文中进一步拆成力的大小 + 方向，共 4 个量）
- 末端 6DoF 状态
- 最近 `n=2` 帧历史观测

**输出动作：**
- 长度为 16 的 6DoF delta action 序列
- 每次只执行前 6 个动作，然后重新规划

**边界条件：**
- 不是端到端“从整机到拆解完成”，而是先用 Aruco / 检测方法把机械臂放到电池一端附近，再启动 diffusion policy
- 初始误差被限制在约 `1cm × 1cm × 2cm`
- 主要验证的是常见单层电池仓深度范围
- 策略只控制负责撬取的机械臂，另一只手臂用于搬运，不在策略内

---

## Part II：方法与洞察

作者的方法可以概括成三个操作杆：

1. **力信号同步与增强**：保证力与图像/状态对齐，并通过随机缩放与噪声增强扩大训练时的受力分布  
2. **以力为 query 的跨注意力融合**：让力去“索引”图像中和当前接触状态最相关的局部视觉信息  
3. **扩散策略滚动控制**：预测多步 delta 动作，但只执行一部分，从而持续纠偏

### 方法骨架

- 图像先经过 ResNet-18 编码成空间特征
- 力信号不直接粗暴拼接，而是先表示成 **大小 + 方向**
- 再把力投影到与图像特征同维度的 token
- 在跨注意力里：
  - **Query = force**
  - **Key/Value = image features**
- 得到的联合嵌入再与机器人位姿拼接
- 作为条件输入到 diffusion policy 的 U-Net 噪声预测网络中
- 最终输出 6DoF delta action 序列

### 核心直觉

**改变了什么？**  
从“图像特征后面附加一个小力向量”改成“让力主动去查询视觉特征”。

**改变了哪个瓶颈？**  
这改变的是一个典型的**信息竞争瓶颈**：原来低维力在高维视觉面前很容易被忽略；现在，力不再是一个被动条件，而是决定“该看图像哪里、该把哪部分视觉上下文拿来决策”的主导信号。

**能力上发生了什么变化？**  
这种设计直接提升了两件事：

- **接触状态判别**：知道自己是“已入缝可撬”，还是“仍未完全插入”
- **动作模式切换**：更稳地从插入过渡到撬动，再过渡到抬升

**为什么这在因果上有效？**  
因为在这个任务里，视觉外观变化很大，但**受力趋势**往往更稳定。  
未见电池颜色、尺寸、外壳角度变化时，RGB 分布会飘，但“插入成功时的力变化、撬动时的 z 向峰值、保持接触时的力趋势”更像一个稳定的状态信号。跨注意力把这个稳定信号拿来选择视觉上下文，自然更利于 OOD 泛化。

此外，作者还加了一个很实用的操作：**force augmentation**。  
训练时把力随机缩放到 `[0.9, 1.2]` 并加入高斯噪声，等于告诉策略：  
“不同刚度的物体，只要力趋势合理，也都可能是成功轨迹。”  
这拓宽了训练分布，减轻对特定物体刚度的过拟合。

### 策略层面的设计取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 风险 |
| --- | --- | --- | --- |
| 力作 query 的跨注意力 | 低维力被视觉淹没 | 更稳的接触判别与模式切换 | 模型更复杂，需要多模态同步 |
| 力缩放 + 噪声增强 | 训练力分布过窄 | 对未见刚度/受力水平更鲁棒 | 增强过强可能削弱精细力幅值 |
| delta action + 预测16执行6 | 长时开环误差累积 | 接触任务中持续纠偏 | 推理时间增加 |
| 端到端训练视觉 backbone | 预训练特征不匹配任务几何 | 更贴合插缝/撬动局部几何 | 需要更多任务内示教数据 |

### 这篇论文最关键的机制增量

不是“把 force 也喂进去”，而是：

> 把力从一个弱条件变量，变成了**决定视觉关注和动作模式切换的主导条件变量**。

这正是它相对 vision-only diffusion 和 naive force fusion 的本质提升。

---

## Part III：证据与局限

### 关键证据信号

**信号 1｜真实机器人对比结果很直接**  
作者在 12 个物体上做了每个 10 次测试，共 120 次真实机器人实验。  
整体成功率：

- DP-B（vision-only）: **39%**
- DP-LF（低维力直接拼接）: **48%**
- DP-PF（线性投影力）: **57%**
- **DP-CA（本文）: 96%**

这说明两件事：

1. 力反馈确实有帮助  
2. 但真正大的提升来自**关系建模**，不是简单加一个力向量

**信号 2｜Ablation 支持因果解释**  
作者不是只和 vision-only 比，而是把“加力”的不同方式拆开做了对照。  
结果表明：

- 直接拼接：有提升，但有限
- 线性升维：比直接拼接更好，但仍远不够
- 跨注意力：大幅跃升到 96%

因此最强的证据不是“force useful”，而是“**force-image relational fusion** 才是决定性操作杆”。

**信号 3｜未见对象泛化很强**  
训练只用了 AAA 和 D 两类电池，但测试时：

- 未见 AA：**97%**
- 未见 C：**100%**
- 未见颜色电池：**90%**
- 串联电池配置：**95%**

这说明模型学到的并不只是固定视觉模板，而是更接近“接触过程的状态规律”。

**信号 4｜行为风格接近人类示教**  
作者还比较了：

- 机器人与人类示教的任务耗时
- 峰值 z 向力
- 整个撬取过程中的力趋势

结论是：机器人推理虽然略慢，但同量级；峰值力和力曲线趋势与人类示教相近。  
这支持一个重要判断：模型不是单纯记住路径，而是学到了相对合理的受力-动作耦合模式。

### 1-2 个最关键指标

- **总体成功率：96%**
- **相对 vision-only baseline 提升：57 个百分点**

### 局限性

- **Fails when:** AAA 这类紧公差、小接触面的电池最容易失败；即便已经接近成功，只要工具与缝隙有轻微错位，就可能在撬动或抬升阶段失去接触。更深腔体、超出实验范围的电池仓结构也未被充分验证。
- **Assumes:** 需要外部粗定位把机械臂带到电池端部附近；依赖腕部相机、末端力/力矩传感器、固定撬取工具几何；数据采集依赖 kinesthetic teaching 后的轨迹回放；实验在低速、相对静态场景中进行。
- **Not designed for:** 端到端全自动搜寻与拆解；带严格安全外力上限的控制；高速动态接触任务；对电池穿刺风险极敏感、需形式化安全保证的场景。

### 复现与扩展时要注意的依赖

- **硬件依赖**：KUKA IIWA14、ABB IRB120、RealSense D415、力/力矩传感器级别配置
- **数据代价**：419 条真实示教 episode
- **训练成本**：RTX 3080 上约 14 小时，计算不算极端，但真实机器人采集成本不低
- **安全问题**：论文明确承认尚未解决“如何限制最大外力而不破坏成功率”

### 可复用组件

- **force-as-query cross-attention**：适合任何“视觉看不清、但接触反馈能判态”的 manipulation 任务
- **force augmentation**：适合不同刚度、不同接触阈值的 OOD 泛化
- **delta-action diffusion + receding horizon**：适合高精度接触操作
- **示教后回放采集多模态数据**：适合 wrist camera 会被人手遮挡的场景

### 一句话总结 “So what”

这篇论文的能力跃迁，不是从“无力觉”到“有力觉”，而是从“力只是附加输入”到“力真正主导视觉解释和动作模式切换”。这正是它能把真实机器人撬取成功率从 39% 拉到 96% 的关键。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Robotic_Compliant_Object_Prying_Using_Diffusion_Policy_Guided_by_Vision_and_Force_Observations.pdf]]