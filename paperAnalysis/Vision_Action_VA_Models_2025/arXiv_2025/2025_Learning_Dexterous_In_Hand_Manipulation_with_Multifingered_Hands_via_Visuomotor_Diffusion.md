---
title: "Learning Dexterous In-Hand Manipulation with Multifingered Hands via Visuomotor Diffusion"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/in-hand-manipulation
  - diffusion
  - teleoperation
  - outlier-filtering
  - dataset/Custom-Unscrewing-300Demos
  - opensource/no
core_operator: "用 AR 手部重定向采集高质量示范，并以腕部视觉+关节位置/effort 条件化扩散策略预测 Allegro Hand 关节增量，再通过聚类式异常剔除稳定训练。"
primary_logic: |
  AR 手部跟踪与单手拧盖任务初始状态 → 手-机器人重定向采集示范并用 ConvNeXt+HDBSCAN/GLOSH 过滤异常轨迹，再以腕部相机和本体感觉条件化扩散策略闭环输出 Δq → Allegro Hand 在真实环境中完成瓶盖旋拧
claims:
  - "在 20 次真实机器人试验中，腕部相机+关节位置+关节 effort 的 πnt 达到 85% 成功率，高于使用双相机+全部输入的 πall 的 55% [evidence: ablation]"
  - "移除关节 effort 会将成功率从 55% 降至 30%，主要失败表现为瓶身滑脱或抓持位置错误 [evidence: ablation]"
  - "仅删除 outlier 分数最高的 10% 示范不会低于 πall，但继续删除到 30%/50% 会使成功率降至 25%/10% [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Learning Dexterous In-Hand Manipulation (Andrychowicz et al. 2020); DexMV (Qin et al. 2022)"
  complementary_to: "Mobile ALOHA (Fu et al. 2024); DexCap (Wang et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_Dexterous_In_Hand_Manipulation_with_Multifingered_Hands_via_Visuomotor_Diffusion.pdf"
category: Embodied_AI
---

# Learning Dexterous In-Hand Manipulation with Multifingered Hands via Visuomotor Diffusion

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.02587), [Project](https://dex-manip.github.io/)
> - **Summary**: 这篇工作把现有的 visuomotor diffusion policy 落到真实四指 Allegro Hand 的单手拧瓶盖场景中，关键不在改扩散网络，而在于用 AR 重定向采集高质量示范、过滤异常示范，并找到最有效的观测组合。
> - **Key Performance**: 最佳配置在 20 次真实机器人测试中达到 **85%** 成功率；双相机全输入基线为 **55%**，去掉 effort 后仅 **30%**。

> [!info] **Agent Summary**
> - **task_path**: 腕部相机/关节位置/关节 effort + 近期历史的单手掌内瓶盖操作 -> Allegro Hand 关节增量序列
> - **bottleneck**: 精细掌内操作对局部接触状态极敏感，但真实示范容易因手-机器人运动学不匹配、示范噪声和抓持状态不可观测而失真
> - **mechanism_delta**: 用 AR 手部重定向提高示范质量，再用视觉聚类剔除异常轨迹，并用腕部视角+effort 条件化扩散策略替代“信息越多越好”的全输入方案
> - **evidence_signal**: 7 个真实机器人策略对照中，腕部相机+q+τ 从 55% 提升到 85%，且去掉 effort 会掉到 30%
> - **reusable_ops**: [AR手到机器人重定向采集, HDBSCAN+GLOSH示范过滤]
> - **failure_modes**: [瓶身被推离稳定抓持后不可恢复, 早期姿态错位导致拇指无法形成有效接近角]
> - **open_questions**: [能否泛化到不同瓶体/盖子尺寸, 仅靠joint effort是否足以替代触觉]

## Part I：问题与挑战

这篇论文解决的是一个比常见桌面抓取更难的问题：**单手、多指、掌内的瓶盖旋拧**。任务不仅要“抓住物体”，还要在抓持过程中持续完成物体重定位、稳定接触和旋拧动作。

### 真正的问题是什么
真正瓶颈不是“再设计一个更复杂的策略网络”，而是三件更现实的事：

1. **示范采集难**：人手和 Allegro Hand 的运动学差异很大，直接模仿人手动作会失真。
2. **接触状态难观测**：掌内操作成败高度依赖“有没有抓稳、有没有滑、拇指是否卡到正确角度”，这些信息单靠远视角图像并不充分。
3. **离线示范容易掺杂坏轨迹**：对于 diffusion policy 这类离线模仿学习方法，少量低质量示范就可能把动作分布拉偏。

### 输入/输出接口
- **输入**：顶部相机 \(I_t\)、腕部相机 \(I_w\)、关节位置 \(q\)、关节 effort \(\tau\)，以及短历史窗口。
- **输出**：Allegro Hand 的下一步关节增量 \(\Delta q\)。

### 边界条件
这不是一个“通用灵巧手 benchmark”：
- 单任务：拧开瓶盖
- 单平台：四指 Allegro Hand
- 单主要对象族：瓶身/瓶盖
- 数据来自 **1 位专家**，共 **300 条真实示范**
- 成功标准：**2 分钟内**完成拧开

### Why now
因为 diffusion policy 已经在多种 manipulation 任务上证明了“小规模专家示范 + 闭环动作生成”的可行性，而 wrist camera、AR 头显、低延迟 teleoperation 也让这种方法开始具备真实部署条件，尤其是对未来移动操作平台更有现实意义。

## Part II：方法与洞察

这篇论文的方法增量本质上是一个**系统组合**，而非新的扩散模型结构。它把“可学性”问题拆成三步：先把示范采好，再把坏数据剔掉，最后让策略只看真正有用的观测。

### 方法拆解

#### 1. AR teleoperation：先解决“示范能不能采对”
作者用 Meta Quest 3 做手部跟踪，把人手骨架传到 ROS 节点，再通过：
- 去掉小拇指、
- 对齐手平面与手指根部、
- 按手指长度做缩放、
- 对指尖目标做额外位移修正、
- 分指做逆运动学，

把人手动作重定向到 Allegro Hand。  
这一步的价值不在“花哨交互”，而在于把**原本难以稳定采集的灵巧手示范**变成一个操作者可控、可重复的过程。

#### 2. 示范过滤：先减掉坏分布，再谈学策略
作者没有直接把 300 条示范全喂给策略，而是先做无监督异常检测：
- 用预训练 ConvNeXt-Tiny 提取顶部/腕部图像特征；
- 对特征做 HDBSCAN 聚类；
- 用 GLOSH 给每条示范打 outlier score；
- 将两路相机得分平均，按百分位裁剪异常示范。

这一步的关键点是：他们不是假设只有一个“正确示范模式”，而是允许多模态示范存在，只去掉最偏离密度结构的坏轨迹。

#### 3. 策略学习：用 diffusion policy 做闭环关节增量预测
策略仍采用 Diffusion Policy 的 CNN 版本，本质上是：
- 不直接回归单步动作；
- 而是在条件观测下，逐步去噪一个动作序列；
- 并在闭环里持续根据新观测更新动作。

这对掌内操作有意义，因为这类动作不是“单步抓取成功”，而是**持续修正的小步协调过程**。

### 核心直觉

作者真正改变的不是模型容量，而是**策略看到的数据分布和信息瓶颈**：

- **从什么变成什么**：  
  从“在带噪声的多视角真实示范中学习复杂接触控制”，  
  变成“在更干净的示范分布上，用最贴近手-物相对关系的腕部视角 + 反映抓持稳定性的 effort 来学习闭环修正”。

- **改变了哪个约束/瓶颈**：  
  - AR 重定向减少了示范采集误差；
  - outlier removal 缩窄了训练分布中的坏模式；
  - wrist camera 降低了与任务无关的视角变化；
  - joint effort 补上了“是否抓稳”的隐变量。

- **能力为何改变**：  
  拧盖成败取决于物体在手内的局部几何关系和抓持稳定性。  
  顶视角能看到全局，但未必最接近接触决策；双相机看似信息更多，却也引入更多冗余变化。腕部相机更贴近执行点，而 effort 则提供纯视觉难以稳定恢复的接触线索，所以最优配置反而更“窄”。

### 战略性取舍

| 设计选择 | 改变的瓶颈 | 收益 | 代价/风险 |
| --- | --- | --- | --- |
| AR 重定向 teleop | 人手与 Allegro 手运动学不一致 | 可稳定采集高质量灵巧手示范 | 依赖 Quest 3、IK 调参与硬件标定 |
| 腕部相机优先于双相机 | 减少与任务无关的视角冗余 | 成功率最高，且更适合移动平台部署 | 丢失部分全局上下文 |
| 加入 joint effort | 抓持稳定性不可直接从图像读出 | 明显减少滑脱和错位失败 | 需要可靠的本体/力矩读数 |
| 只裁掉最异常 10% 示范 | 去除坏轨迹但保留多样性 | 基本不伤性能 | 阈值过激会破坏覆盖度 |

## Part III：证据与局限

### 关键证据信号

- **模态消融信号**：  
  最强结果不是“全部输入”，而是 **腕部相机 + 关节位置 + effort**。  
  该配置在 20 次真实试验中达到 **85%**，高于双相机全输入的 **55%**。  
  这直接支持论文的核心判断：**对掌内操作，局部执行视角比堆更多视角更重要**。

- **接触观测信号**：  
  去掉 effort 后，成功率降到 **30%**。  
  失败主要表现为瓶体滑脱或抓持位置错误，说明模型确实需要某种“抓稳了没有”的反馈，而这类信息不能稳定地从图像单独恢复。

- **异常示范过滤信号**：  
  只去掉 outlier 最高的 **10%** 示范，并不会比原始全量数据更差；  
  但继续删到 **30%/50%** 时，成功率掉到 **25%/10%**。  
  这说明两点同时成立：  
  1) 数据里确实有害示范；  
  2) 但示范多样性本身也很重要，不能过度清洗。

- **案例分析信号**：  
  最佳模型有时能从早期错误中恢复，但一旦瓶子被推离稳定抓持区，或初始错位让拇指找不到有效接近角，失败通常不可逆。  
  最快成功样例为 **19 秒**，说明策略不仅能完成任务，也能在有利初态下较高效地完成。

### 局限性

- Fails when: 瓶体在掌内被推离稳定抓持区域、早期重定位失败导致拇指接近角错误、或需要更细粒度接触反馈而不仅是视觉+joint effort 时，策略容易进入不可恢复状态。
- Assumes: 依赖 Allegro Hand、Meta Quest 3、相机与关节 effort 传感、手工设计的重定向/IK 管线，以及单专家采集的 300 条真实示范；异常过滤主要基于视觉嵌入，因此对“视觉上正常但力学上错误”的示范不一定敏感。
- Not designed for: 跨对象泛化、不同瓶盖几何与材质的大范围迁移、双手协作、纯触觉驱动操作、或无示范/少标注自监督学习。

### 可复用组件

- **AR 手到机器人重定向采集管线**：适合其他多指手的 demonstration collection。
- **基于视觉嵌入的示范异常过滤**：可迁移到其它离线 imitation/diffusion 训练流程。
- **观测选择经验**：对精细手内操作，优先考虑**腕部视角 + 本体/接触代理信号**，而不是盲目增加相机数。

### 结论判断

这篇论文的价值不在于提出新 diffusion 架构，而在于证明：  
**只要把示范质量、观测选择和异常数据处理这三个系统级旋钮拧对，已有的 diffusion policy 也能在真实多指手上完成相当难的单手掌内拧盖任务。**

但证据仍然主要来自**单任务、单对象族、单平台**，因此更应把它看作一个很强的真实系统验证，而不是已经解决“通用灵巧手操作”的终局方案。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_Dexterous_In_Hand_Manipulation_with_Multifingered_Hands_via_Visuomotor_Diffusion.pdf]]