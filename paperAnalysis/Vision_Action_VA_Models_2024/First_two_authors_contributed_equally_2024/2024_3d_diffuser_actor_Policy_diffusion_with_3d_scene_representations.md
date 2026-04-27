---
title: "3D Diffuser Actor: Policy Diffusion with 3D Scene Representations"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - relative-attention
  - state-token
  - dataset/RLBench
  - dataset/CALVIN
  - opensource/promised
core_operator: 将多视角RGB-D提升为3D场景token，并在统一3D坐标系中用相对位置去噪Transformer对未来末端执行器轨迹做扩散去噪。
primary_logic: |
  多视角RGB-D/语言指令/本体状态/加噪未来轨迹 → 将图像lift到3D场景token、将未来位姿表示为3D轨迹token，并通过3D相对位置注意力联合语言进行迭代去噪 → 输出末端执行器未来关键位姿/轨迹与夹爪开合
claims:
  - "在 RLBench 多视角设定上，3D Diffuser Actor 的平均成功率达到 81.3%，比先前 SOTA Act3D 高 18.1 个百分点 [evidence: comparison]"
  - "在 CALVIN zero-shot long-horizon 设定上，3D Diffuser Actor 的平均连续完成任务数达到 3.35，高于 GR-1 的 3.06 和 SuSIE 的 2.69 [evidence: comparison]"
  - "把 3D 条件表示退化成 2D 表示，或移除相对 3D 注意力，RLBench 平均成功率会分别从 81.3% 降到 47.0% 和 71.3% [evidence: ablation]"
related_work_position:
  extends: "Act3D (Gervet et al. 2023)"
  competes_with: "Act3D (Gervet et al. 2023); 3D Diffusion Policy (Ze et al. 2024)"
  complementary_to: "BiRRT (Kuffner and LaValle 2000)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/First_two_authors_contributed_equally_2024/2024_3d_diffuser_actor_Policy_diffusion_with_3d_scene_representations.pdf
category: Embodied_AI
---

# 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2402.10885) · [Project/Video](https://sites.google.com/view/3d-diffuser-actor)
> - **Summary**: 这篇论文把“扩散策略”与“3D 场景表示”合在一起，让机器人直接在统一 3D 空间中对未来动作轨迹去噪，从而同时解决动作多模态与跨视角泛化问题。
> - **Key Performance**: RLBench 多视角平均成功率 81.3%（较 Act3D +18.1pp）；CALVIN 平均连续完成任务数 3.35（高于 GR-1 的 3.06）

> [!info] **Agent Summary**
> - **task_path**: 多视角/单视角 RGB-D + 语言指令 + 当前末端状态 -> 未来末端执行器 3D 轨迹/关键位姿 + 夹爪开合
> - **bottleneck**: 机器人操作既有动作多模态，又需要稳定的 3D 空间对齐；2D 策略要隐式学 2D->3D 映射，传统 3D 策略又常用回归/分类而压扁多峰动作分布
> - **mechanism_delta**: 把未来动作表示成带噪 3D 轨迹 token，并让它与 3D 场景 token 在同一坐标系里通过相对 3D 注意力做条件扩散去噪
> - **evidence_signal**: RLBench 多视角相对 Act3D 提升 +18.1pp，且 2D 替代与去掉相对注意力的消融均显著退化
> - **reusable_ops**: [RGB-D lifting 到 3D token, 相对位置 3D Transformer 去噪]
> - **failure_modes**: [高精度位姿任务上关键位姿不够准, 多相似物体场景中的语言目标混淆]
> - **open_questions**: [如何把扩散采样进一步加速到更高控制频率, 如何扩展到动态场景与速度控制]

## Part I：问题与挑战

这篇论文解决的不是“能不能从演示学机器人”，而是更具体的两个耦合瓶颈：

1. **动作分布是多峰的**  
   同一个操作任务往往有多种等价完成方式。比如抓哪个杯子、从哪条轨迹绕过去、先对齐再插入还是先抬高再下落。  
   如果策略用单点回归或分类近似，容易学成“平均动作”或只覆盖主模态。

2. **视觉到动作的几何映射本质上是 3D 的**  
   2D 图像策略需要隐式学会相机视角、透视与深度几何，再把这些映射到 3D 末端位姿；这对跨视角泛化很不友好。  
   3D 策略虽然解决了几何对齐，但此前大多仍不是扩散式分布建模。

### 真正的瓶颈是什么？
**真正瓶颈是：机器人策略需要同时建模“多模态动作分布”和“视角无关的 3D 空间关系”，而过去方法通常只解决其中一个。**

- 2D diffusion policy：会建模分布，但几何对齐弱
- 3D deterministic/classification policy：几何强，但对多模态动作覆盖弱

### 为什么现在值得做？
因为两条技术线都成熟了：

- **扩散策略**已经证明比回归、VAE、能量模型等更擅长拟合动作分布
- **3D token 化场景表示**已经证明比 2D 表示更稳健，尤其在跨视角和新视角测试时更强

这篇论文的价值在于把两者真正统一，而不是把 diffusion 当后处理或规划插件。

### 输入/输出接口与边界条件
- **输入**：一到多个校准好的 RGB-D 视角、语言指令、当前本体状态
- **输出**：未来一段末端执行器轨迹（3D 平移 + 6D 旋转）以及夹爪开合
- **边界条件**：
  - RLBench / 实机中，常只预测下一个 keypose，再交给 BiRRT/MoveIt 去执行
  - CALVIN 中没有运动规划器，因此直接预测轨迹
  - 任务主要是**quasi-static manipulation**
  - 方法依赖**深度、相机外参、keypose 分段**

---

## Part II：方法与洞察

论文的核心设计不是“把 diffusion 搬到机器人上”，而是：

> **把动作去噪过程放到统一的 3D 空间里做，让动作 token 和场景 token 在几何上直接相互作用。**

### 方法主线

#### 1. 用 RGB-D 把视觉输入 lift 到 3D scene tokens
每个图像 patch 经过 2D encoder 得到特征，再结合深度与相机参数反投影到 3D。  
多视角时，把所有视角的 3D token 聚合起来。

这样场景表示不再绑定某个相机平面，而是放到统一工作空间坐标系里。

#### 2. 把未来动作轨迹也表示成 3D trajectory tokens
不是直接一次性回归未来动作，而是先对未来轨迹加噪，然后把每个未来位姿表示成一个 token：

- token embedding：位姿内容
- token position：其带噪后的 3D 平移位置

于是，**场景 token**和**动作 token**都在同一 3D 空间中。

#### 3. 用 3D relative denoising transformer 做条件去噪
模型让 3D scene tokens、trajectory tokens、本体状态 token 做相对位置 self-attention；  
语言没有天然 3D 坐标，因此通过普通 attention/cross-attention 融入。

关键点在于这里不是绝对位置编码，而是**相对 3D 注意力**。  
这使模型更接近**平移等变**：整体平移场景时，token 之间的相对几何关系保持不变。

#### 4. 分别预测平移噪声、旋转噪声和夹爪状态
输出分三部分：

- 位置噪声
- 旋转噪声
- 开合状态

论文还发现**平移和旋转用不同 noise scheduler**更好，这说明 6D rotation 的扩散统计与位置不完全同构。

### 核心直觉

**改了什么？**  
从“基于 2D 图像或全局 3D embedding 的单步动作预测”，改成“基于 token 化 3D 场景的轨迹级条件扩散去噪”。

**哪个信息瓶颈被改变了？**
1. **几何瓶颈**：  
   过去模型要隐式学 2D 到 3D 的对应；现在视觉 token 本身就在 3D 空间里，动作 token 也在 3D 空间里，网络只需学“3D 关系到动作”的映射。
2. **分布瓶颈**：  
   过去回归/分类更像在学单一答案；现在 diffusion 直接拟合条件动作分布，可保留多模态解。
3. **泛化瓶颈**：  
   token 化的 3D 场景比 holistic pooled embedding 更局部化；当场景某一部分变化时，只影响对应 token，而不是污染整个全局表征。

**能力上带来了什么变化？**
- 更强的跨视角与场景变化泛化
- 更能处理多模态、高精度、长时序操作任务
- 在 CALVIN 中表现出一定“失败后重试”的行为倾向

### 为什么这个设计有效？
因为它把两个本来纠缠的问题拆开了：

- **3D 表示**负责把“看见什么、物体在哪里”显式化
- **diffusion**负责把“接下来怎么做可能有多种合法答案”分布化

也就是说，论文不是单纯把 backbone 换成 3D，而是把**动作生成的坐标系**和**场景理解的坐标系**统一了。

### 策略性 trade-off

| 设计选择 | 带来的能力 | 代价/风险 |
|---|---|---|
| 3D token 化场景表示 | 更强几何对齐与跨视角泛化 | 需要深度与相机标定 |
| 轨迹级 diffusion policy | 能覆盖多模态动作分布，而非单峰回归 | 推理比回归慢，需要多步去噪 |
| 相对 3D 注意力 | 平移等变，更利于泛化 | 依赖 3D 坐标质量，深度噪声会影响 |
| token 化而非全局 pooled 3D embedding | 局部场景变化不会污染整段表示 | token 数量大，计算更重 |
| keypose + motion planner（RLBench/实机） | 执行更稳，降低长轨迹控制难度 | 规划器失败会成为外部瓶颈，不是纯端到端 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 比较信号：RLBench 上的能力跳跃很大
最强证据来自 RLBench。

- **多视角**：81.3% 平均成功率，较 Act3D 提升 **+18.1pp**
- **单视角**：78.4%，较 Act3D 提升 **+13.1pp**

更重要的不是平均值本身，而是提升集中出现在：
- **长时程任务**
- **高精度任务**
- **多模态任务**

比如 stack blocks、stack cups、place cups 这类任务，正好是“需要空间理解 + 动作多峰”的交叉难点。

#### 2. 比较信号：CALVIN 的 zero-shot long-horizon 泛化也成立
在 CALVIN 上，3D Diffuser Actor 达到：

- **Avg. Len = 3.35**
- **5-task chain success = 41.2%**

它不仅超过 3D Diffusion Policy 和 ChainedDiffuser，也超过 GR-1 / SuSIE 这类强 2D baseline。  
这说明收益不只是来自 3D 几何，也来自**更好的动作分布建模**。

一个有意思的细节是：给模型更长的执行 horizon 会继续增益，作者据此判断模型学到了一定的**retry 行为**。

#### 3. 消融信号：真正起作用的是“3D + relative attention”
这篇论文最有说服力的地方在于消融很直接：

- 换成 **2D Diffuser Actor**：81.3 -> **47.0**
- 去掉 **relative attention**：81.3 -> **71.3**

这两组结果基本把因果链钉住了：

- 3D 表示不是可有可无
- 相对位置注意力带来的平移等变也不是装饰

#### 4. 分析/案例信号：深度噪声和实机结果
- **深度噪声鲁棒性**：在单视角 RLBench 上，轻度深度扰动下平均成功率从 78.4 降到 72.4，说明有一定容错，但强噪声下会明显退化
- **实机**：12 个任务、每任务 15 个演示即可学会，说明数据效率不错；但这部分主要是 case study，没有系统 baseline 对照

### 1-2 个最关键指标
- **RLBench multi-view**：81.3% avg success，**+18.1pp** over prior SOTA
- **CALVIN**：Avg. Len **3.35**

### 局限性
- **Fails when**: 高精度位姿要求很高的任务、多个相似目标导致语言歧义的场景、以及 motion planner 找不到可行路径时，性能会明显下降；强深度噪声也会破坏 3D 对齐。
- **Assumes**: 需要校准好的 RGB-D 输入、相机外参、keypose 分段启发式，以及 quasi-static manipulation 设定；在 RLBench/实机上还依赖外部规划器（BiRRT/MoveIt）完成落地执行。
- **Not designed for**: 动态场景、速度控制、纯 RGB 无深度设定、以及需要极高闭环频率的控制任务。

### 复现与资源依赖
- 推理是 diffusion，多步采样导致**延迟高于非扩散策略**
- 在 CALVIN 上报告约 **600ms 生成 6 个末端位姿**
- 代码在文中状态是**upon publication 公布**，因此当前开放性仍是 `promised`

### 可复用组件
- **RGB-D -> 3D token lifting**
- **3D relative denoising transformer**
- **动作轨迹 token 化扩散建模**
- **位置/旋转分离的 noise scheduling**
- **keypose discovery + planner 组合接口**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2024/First_two_authors_contributed_equally_2024/2024_3d_diffuser_actor_Policy_diffusion_with_3d_scene_representations.pdf]]