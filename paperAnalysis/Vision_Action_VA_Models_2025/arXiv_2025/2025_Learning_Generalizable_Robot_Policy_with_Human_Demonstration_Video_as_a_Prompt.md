---
title: "Learning Generalizable Robot Policy with Human Demonstration Video as a Prompt"
venue: CoRL
year: 2025
tags:
  - Embodied_AI
  - task/video-understanding
  - diffusion
  - contrastive-learning
  - cross-prediction
  - dataset/RH20T
  - dataset/HOI4D
  - dataset/Bridge
  - opensource/no
core_operator: 先用跨具身视频预测学习人类与机器人共享的任务表征，再把该表征通过统一动作空间和原型对比扩散策略映射为灵巧手控制。
primary_logic: |
  人类演示视频 + 目标机器人初始观测 → 跨具身视频生成中抽取任务/对象/场景共享表征 → 将人类重定向动作与机器人动作放入统一动作空间并联合训练扩散策略 → 输出可被未见人类视频提示的机器人动作序列
claims:
  - "用阶段一视频表征替代语言提示后，真实世界位置/场景/背景泛化均提升；例如背景泛化 SR 从 0.36 提升到 0.64，而加入 PDCP 后进一步到 0.73 [evidence: comparison]"
  - "在表征+机器人+人类数据设置上加入 PDCP 后，位置泛化 SR 从 0.74 提升到 0.79，背景泛化 SR 从 0.64 提升到 0.73，说明技能级表征分离有助于减少多任务混淆 [evidence: comparison]"
  - "学习到的表征在 t-SNE 中按技能、目标位置和时间阶段形成结构化簇，并在跨具身向量算术的视频生成案例中表现出可组合性 [evidence: analysis]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Vid2Robot (Jain et al. 2024); EgoMimic (Kareer et al. 2024)"
  complementary_to: "Octo (Open X-Embodiment Team et al. 2024); RDT-1B (Liu et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_Generalizable_Robot_Policy_with_Human_Demonstration_Video_as_a_Prompt.pdf
category: Embodied_AI
---

# Learning Generalizable Robot Policy with Human Demonstration Video as a Prompt

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.20795)
> - **Summary**: 这篇工作把“人类演示视频”当作机器人策略的 prompt：先通过跨具身视频生成学出可迁移的任务表征，再把该表征对齐到统一动作空间，使灵巧手在不新增 teleop 数据、也不再微调策略的前提下，能由人类视频驱动完成新任务。
> - **Key Performance**: 真实世界泛化测试中，位置/场景/背景泛化的 SR 达到 **0.79 / 0.69 / 0.73**；其中背景泛化显著优于 Language+R.+H. 的 **0.36**，也优于 Repr.+R.+H. 的 **0.64**。

> [!info] **Agent Summary**
> - **task_path**: 第三人称人类演示视频 + 机器人当前 RGB 观测/初始图像 -> 19 维灵巧手动作序列
> - **bottleneck**: 语言提示缺少操作所需的时空细节，而人类视频又存在 human-robot embodiment gap，导致新任务通常仍需新 teleop 数据
> - **mechanism_delta**: 用跨具身视频预测先提炼与 embodiment 解耦的任务表征，再用共享动作空间 + 原型对比扩散策略把该表征压到机器人动作空间
> - **evidence_signal**: 真实世界位置/场景/背景泛化比较中，Repr.+R.+H.+PDCP 全面优于语言提示和无 human data 的基线
> - **reusable_ops**: [cross-prediction video pretraining, shared human-robot action space]
> - **failure_modes**: [novel-skill transfer success remains under 10%, sensitive to hand-tracking/retargeting and observation alignment]
> - **open_questions**: [can scaling human videos make unseen-skill transfer reliable, can the method work without explicit retargeting]

## Part I：问题与挑战

这篇论文要解决的，不是普通的模仿学习，而是一个更难的问题：

**能不能让机器人像人一样，看一段别人做事的视频，就知道自己该怎么做，而且不需要为每个新任务再收一批遥操作数据？**

### 1. 问题接口

- **输入**：人类第三人称演示视频，外加机器人当前观察到的场景/初始图像
- **输出**：灵巧手控制动作序列（文中为 19 维动作）
- **目标**：对机器人训练集中没有出现过的任务或技能，也能用人类视频作为 prompt 直接执行

### 2. 真正瓶颈

作者认为过去方法卡在三个地方：

1. **语言条件太稀疏**  
   文本能说“倒水”“按按钮”，但说不清楚物体相对位置、手部轨迹、时序节奏、场景约束。  
   对 manipulation 来说，这些正是关键。

2. **human-to-robot 的具身差异太大**  
   人手、夹爪、灵巧手的外形、自由度、接触方式都不同。  
   所以“看懂人类视频”不等于“能转成机器人动作”。

3. **新任务还得收新 teleop 数据**  
   这是最现实的成本瓶颈。灵巧手遥操作昂贵、慢、且难规模化。

### 3. 为什么现在值得做

这件事现在变得可行，依赖三类成熟能力同时出现：

- 大规模**视频生成预训练模型**已经具备物理世界动态先验；
- **人类操作视频**极易获取，规模远大于机器人演示；
- 手部跟踪与**动作重定向**工具已经足够强，能把 RGB 人类视频近似投到机器人动作空间。

### 4. 边界条件

这篇论文不是“任意场景任意机器人”的通解，它有较明确实验边界：

- 主要验证平台是 **xArm7 + dexterous hand**
- 观测以**固定第三人称 RGB 相机**为主
- 需要一定量的机器人数据，外加少量高质量 dexhand 数据
- human video 不是直接监督动作，而是先被转成共享表征，再参与策略学习

---

## Part II：方法与洞察

这篇工作的关键设计不是“直接从 human video 回归 robot action”，而是先做一个中间层：

> **先学会跨具身地“翻译视频任务”，再把这个可迁移表征拿去驱动扩散策略。**

### 方法主线

#### Stage 1：VGCP（Augment Video Generation by Cross Prediction）

作者以 **Stable Video Diffusion** 为底座，喂入：

- 一个**源具身**的视频 prompt（比如人手在做任务）
- 一个**目标具身**的初始图像（比如灵巧手当前场景）

模型要生成：**目标具身完成同一任务的视频**。

训练时不总是做人→机器人，也会混入同具身预测；论文中用概率方式在 cross-prediction 和 normal-prediction 间切换。  
这样做的目的不是只为了生成视频，而是逼模型学会一个更有用的隐变量表征：

- 什么任务在发生
- 目标物体在哪里
- 场景上下文是什么
- 哪些信息与“人手/机器人手长什么样”无关

训练完后，这个视频模型在第二阶段被**冻结**，作为表征提取器。

#### Stage 2：人类视频-动作增强的扩散策略

第二阶段把 Stage 1 的表征真正接到机器人控制上。

核心步骤有两个：

1. **把 human action 变成可兼容的 robot-like action**
   - 用 WiLoR 从 RGB 视频恢复手部关键点/网格
   - 提取 wrist pose
   - 用 AnyTeleop 风格的 retargeting，把人手动作映射到机器人手
   - 最终和机器人动作一起归一化到统一动作空间

2. **训练 representation-conditioned diffusion policy**
   - 机器人当前观测编码成 state feature
   - Stage 1 提取的人类视频表征作为 goal/task condition
   - 扩散策略输出灵巧手动作

于是，策略学到的不再只是“某个机器人如何做某个任务”，而是：

> **给定一个跨具身共享的任务提示，如何在机器人动作空间中实现它。**

#### PDCP：ProtoDiffusion Contrastive Policy

只把表征接到扩散策略上还不够，因为多任务、多物体、近邻操作容易发生**技能混淆**。

作者加入 PDCP，本质上是在策略表征空间里做**原型式技能分离**：

- 同技能样本拉近
- 不同技能样本拉远
- 用 prototype/cluster 约束让表征更稳定

这一步的作用，不是增加新信息，而是**重塑表示几何**，让策略更容易分清“当前该抓哪个、推哪个、按哪个”。

### 核心直觉

这篇论文真正调的“因果旋钮”可以概括为三步：

1. **把学习目标从“预测未来视频”改成“跨具身复现同一任务”**  
   这会迫使模型丢掉只跟外观绑定的信息，保留任务本质、对象关系与时序结构。

2. **把监督分布从“仅机器人动作”改成“人类+机器人统一动作空间”**  
   这显著扩大了行为覆盖面，减少对昂贵 dex teleop 的依赖。

3. **把策略表征从“混在一起的多技能空间”改成“按技能聚团的表征空间”**  
   这降低了动作扩散时的歧义，尤其对近邻物体、相似操作更重要。

也就是说：

**改变了什么**  
从 language-conditioned / robot-only imitation，改成 video-prompted / cross-embodiment / skill-clustered policy learning。

**哪类瓶颈被改变了**  
目标条件的信息瓶颈、human-robot 具身错配、以及多技能策略的表征纠缠。

**带来了什么能力变化**  
机器人开始具备一种有限但真实的“看人做 -> 自己做”的泛化能力，尤其在位置、背景和对象变化下更强。

### 战略性取舍

| 设计 | 解决的瓶颈 | 带来的收益 | 代价 / 风险 |
| --- | --- | --- | --- |
| Cross-prediction 视频预训练 | human-robot 具身差异导致的视频条件不可直接用 | 学到更偏任务/对象/场景的共享表征 | 训练更重，且需要多源视频数据 |
| 统一 human-robot 动作空间 | dex data 稀缺 | 人类数据可直接进入策略学习 | 强依赖手部跟踪与 retargeting 质量 |
| PDCP 原型对比约束 | 多技能混淆、近邻物体误操作 | 更清晰的技能边界，提升泛化稳定性 | 需要任务类别信号/聚类质量足够好 |
| 冻结 Stage 1 表征模型 | 避免端到端训练不稳定 | 表征迁移更稳，训练更简单 | Stage 2 无法反向修正表征缺陷 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：视频表征明显强于语言提示

最强的实验信号来自真实世界三类泛化测试：

- **位置泛化**：Ours SR = **0.79**
- **场景泛化**：Ours SR = **0.69**
- **背景泛化**：Ours SR = **0.73**

而语言条件基线 Language+R.+H. 分别只有：

- 0.58 / 0.56 / 0.36

这说明作者的核心判断是成立的：  
**视频 prompt 比语言 prompt 更适合 manipulation，因为它提供了可执行的时空细节。**

#### 2. 比较信号：human data 和 PDCP 都有增益

从表 1 看，能力提升不是一步来的，而是两步叠加：

- **Repr.+R. → Repr.+R.+H.**：说明 human action data 确实补充了策略覆盖
- **Repr.+R.+H. → Repr.+R.+H.+PDCP**：说明技能级聚类/对比约束进一步缓解了任务混淆

最典型的是背景泛化：

- Language+R.+H.: **0.36**
- Repr.+R.: **0.55**
- Repr.+R.+H.: **0.64**
- Ours: **0.73**

这个趋势很清楚：  
**先靠表征解决“看懂”，再靠 human data 扩充“做过”，最后靠 PDCP 解决“别做混”。**

#### 3. 分析信号：表征确实学到了“任务结构”，不是黑箱巧合

论文还给了两个表征诊断：

- **t-SNE 可视化**：同类技能会聚在一起，还能区分目标位置与时间阶段
- **向量算术案例**：human/robot 表征差分叠加后，生成视频仍能保留任务语义

这类证据不如定量 benchmark 强，但它支持了作者的中心论点：  
Stage 1 学到的不是纯视觉外观特征，而是更接近**跨具身任务表征**。

#### 4. 案例信号：有“新技能迁移”的趋势，但证据还不够硬

作者展示了机器人训练集中没有出现、但人类数据里出现过的技能可以被迁移到机器人上。  
这是全文最吸引人的点，也是作者称之为某种“in-context learning”能力的来源。

但这里必须保守解读：

- 论文在局限性里明确说**新技能成功率仍低于 10%**
- 所以现阶段更准确的结论是：  
  **它展示了潜力，而不是已经稳定解决了 unseen skill transfer**

### 局限性

- **Fails when**: 需要从人类视频激活全新技能、但视觉观测与执行前提对不齐时；手部跟踪/重定向不准时；近邻物体或高精度灵巧操作场景下仍容易误抓。论文明确承认 novel skill acquisition success rate 目前仍低于 10%。
- **Assumes**: 有固定第三人称 RGB 视角；能用 WiLoR 恢复人手状态并通过 AnyTeleop 式 retargeting 映射到机器人；有少量但高质量 dexhand 数据和一定机器人数据；训练依赖较重算力，Stage 1 约需 5 天、8×A100，Stage 2 也需要 A100/A800 级 GPU。
- **Not designed for**: 纯文本条件控制；没有清晰手部可见性的互联网视频；强视角变化/移动相机设置；完全零机器人示例的灵巧操作；显著依赖触觉或力反馈的精细接触任务。

### 可复用组件

这篇论文里最值得复用的，不是整套系统，而是三类模块化操作：

1. **cross-prediction video pretraining**  
   适合任何需要跨具身/跨域 task representation 的 video-conditioned control。

2. **shared human-robot action space via retargeting**  
   把廉价 human data 引入机器人策略训练，是很实用的数据扩展路线。

3. **prototype-aware contrastive regularization for diffusion policy**  
   对多技能 diffusion policy 来说，这是一种可插拔的“降混淆”插件。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_Generalizable_Robot_Policy_with_Human_Demonstration_Video_as_a_Prompt.pdf]]