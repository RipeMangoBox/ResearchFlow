---
title: "Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/one-shot-manipulation
  - gaussian-splatting
  - differentiable-rendering
  - data-augmentation
  - dataset/COCO
  - opensource/no
core_operator: 以对齐到真实机器人坐标系的3D高斯场景为可编辑中间表示，把单条真实示范自动扩展为跨位姿、视角、光照、外观、物体与机器人本体的高保真新示范。
primary_logic: |
  单条遥操作示范 + 静态场景多视图图像 + 机器人URDF
  → 3DGS重建场景，并用ICP初始化 + 可微渲染细对齐把高斯场景锚定到真实坐标系
  → 分解机器人/物体/背景后，在3D高斯空间执行位姿、物体类型、视角、光照、外观与embodiment编辑
  → 渲染并规划出新演示轨迹，训练可直接部署的视觉运动策略
claims:
  - "RoboSplat在真实世界五个操作任务、六类泛化扰动下，用单条示范生成的数据训练策略可达到87.8%平均成功率，显著高于“数百条真实示范+2D增强”的57.2% [evidence: comparison]"
  - "在8进程RTX 4090上，RoboSplat生成单条演示平均仅需0.64秒，而人工真实采集平均需19.1秒，速度提升超过29倍 [evidence: comparison]"
  - "在相机视角泛化实验中，RoboSplat在novel view与moving view上的三任务平均成功率为80.6%，明显高于VISTA的40.6% [evidence: comparison]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "ROSIE (Yu et al. 2023); VISTA (Tian et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); EquiBot (Yang et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Novel_Demonstration_Generation_with_Gaussian_Splatting_Enables_Robust_One_Shot_Manipulation.pdf
category: Embodied_AI
---

# Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.13175), [Project](https://yangsizhe.github.io/robosplat/)
> - **Summary**: 这篇工作把单条真实示范先变成与机器人坐标对齐、可编辑的3D高斯场景，再在3D中统一做位姿/物体/视角/光照/背景/本体扩增，从而用一次示范训练出更稳健的真实机器人操作策略。
> - **Key Performance**: 真实五任务六类泛化平均成功率 87.8% vs 57.2%；数据生成 0.64 s/条 vs 人工采集 19.1 s/条。

> [!info] **Agent Summary**
> - **task_path**: 单条遥操作示范 + 静态多视图RGB + URDF -> 多样化演示生成 -> 相对末端位姿动作块策略
> - **bottleneck**: 现有2D增强缺少3D几何一致性，而Real-to-Sim流程又难以准确重建并模拟真实操作场景
> - **mechanism_delta**: 将增强载体从2D图像/模拟器切换为与真实机器人坐标系对齐的3D高斯场景，并直接在该空间编辑部署时会变化的因果变量
> - **evidence_signal**: 五个真实任务、六类泛化扰动下的跨基线比较，平均成功率明显高于2D增强与手工采集基线
> - **reusable_ops**: [differentiable-rendering alignment, object-relative keyframe retargeting]
> - **failure_modes**: [deformable objects, contact-rich dynamic manipulation]
> - **open_questions**: [how to inject physics consistency into Gaussian editing, how to extend the pipeline to dynamic/deformable scenes]

## Part I：问题与挑战

这篇论文抓得很准的一点是：**真正的瓶颈不是“策略模型不够复杂”，而是“训练数据覆盖不到部署时的视觉与几何变化”**。

在真实机器人模仿学习里，部署时经常会变的不是任务本身，而是：
- 物体位姿
- 物体类别
- 相机视角
- 场景外观
- 光照条件
- 机器人本体

如果每种变化都靠人工重采示范，成本极高。论文里人工采一条示范平均要 19.1 秒，而真实大规模收集还包含人力与失败重试成本。

现有方案各有硬伤：
1. **2D图像增强**：改得快，但只是在像素层做编辑，常常破坏空间一致性。看起来“像是新场景”，但动作标签未必仍然成立。
2. **Real-to-Sim-to-Real**：理论上能做更多变换，但真实场景重建、物理建模、sim-to-real gap 都很重，尤其在操作场景里几何误差会直接传导到策略学习。

所以这篇论文真正要解决的是：

> 如何找到一个中间表示，既保留真实世界的3D几何关系，又足够容易编辑，还能低成本大规模生成新示范？

作者的答案是：**用 3D Gaussian Splatting 把真实场景变成一个“可编辑的、可渲染的、与机器人坐标对齐的 3D 世界”。**

### 输入/输出接口与边界

- **输入**：
  - 1 条遥操作专家轨迹
  - 静态场景的多视图 RGB 图像
  - 机器人 URDF
- **输出**：
  - 大量自动生成的新示范
  - 基于这些示范训练的视觉运动策略
- **边界条件**：
  - 场景基本静态
  - 物体以刚体为主
  - 机器人运动学模型可得
  - 单条示范可以被关键帧化并通过规划补全

### 为什么是现在

这件事现在可行，核心是工具链成熟了：
- 3DGS 提供高保真、显式、可编辑、可微渲染的场景表示
- Grounded-SAM 能分物体
- Depth Anything 提供深度先验
- AnyGrasp 能补抓取位姿
- 3D AIGC 可以补新物体

也就是说，论文不是只靠一个新网络，而是把几个成熟组件串成了一个对机器人真正有用的“自动示范工厂”。

## Part II：方法与洞察

### 方法主线

RoboSplat 的主线可以概括成五步。

1. **把真实场景重建成 3D 高斯**
   - 用 COLMAP + Depth Anything + 3DGS 得到整场景高斯表示。
   - 这一步得到的是“可渲染的真实场景”，但坐标系还不一定能直接给机器人用。

2. **把 3DGS 对齐到真实机器人坐标系**
   - 先用 ICP 做粗对齐。
   - 再利用 3DGS 的可微渲染能力，让“高斯渲染出的机器人 mask”去贴合“URDF 渲染出的机器人 mask”，继续优化平移、旋转、尺度。
   - 这是全篇非常关键的一步：**如果3D场景没有被锚到真实机器人坐标系，后面的动作重定向、换视角、换本体都不可靠。**

3. **把场景拆成机器人/物体/背景**
   - 物体：Grounded-SAM 分割
   - 机器人：用 URDF link 点云分配高斯
   - 剩下的是背景

4. **从单条示范生成六类泛化数据**
   - **物体位姿**：对目标物体做刚体变换，并把关键帧末端位姿按物体变换做等变重定向，再用 motion planning 补出完整轨迹。
   - **物体类型**：用 GPT-4 生成可抓物体名称，3D 内容生成模型产出新物体高斯，AnyGrasp 生成抓取位姿。
   - **相机视角**：直接在 3DGS 上做 novel view synthesis。
   - **机器人本体**：替换机器人高斯，保留 embodiment-agnostic 的末端位姿关键帧，再对新机器人做规划。
   - **场景外观**：替换背景 3D 场景，或用 COCO 图像贴到桌面/背景平面。
   - **光照条件**：直接扰动高斯颜色的缩放、偏移和噪声。

5. **训练策略**
   - 策略本身并不激进：两路相机图像 + 机器人状态，经 ResNet/MLP/Transformer，再输出 action chunk。
   - 这点很重要：**论文的增益主要来自数据生成，而不是策略网络换代。**

### 核心直觉

这篇论文最有价值的地方，不是“又做了数据增强”，而是：

> 它把增强对象从“2D像素”升级成了“和机器人动作共用同一坐标系的3D因果变量”。

具体地说：

- **what changed**：从 2D image editing / simulator editing，换成 **aligned 3D Gaussian editing**
- **which bottleneck changed**：训练数据里的视觉变化与动作标签之间，终于有了统一的 3D 几何约束
- **what capability changed**：单条示范可以稳定扩展成跨六类部署扰动的训练分布

### 为什么这个设计在因果上有效

1. **动作不是重新猜，而是跟着物体一起变**
   - 对物体位姿的编辑，会同步改变关键帧末端位姿。
   - 所以“场景变了，动作也合理地变了”。

2. **视角变化来自同一个3D场景重渲染**
   - 不再是2D方法那种“看起来换了视角”，而是真的从新相机位姿观察同一场景。
   - 多视角一致性因此更强。

3. **本体迁移利用了末端位姿这一抽象动作接口**
   - 关键帧动作不绑死在 FR3 关节空间，而是保持在 end-effector pose 层。
   - 所以更容易迁到 UR5e 这类新机器人。

4. **可微对齐解决了“能看不能编”的问题**
   - 普通 3D 重建可能视觉上像，但坐标不准。
   - 论文用 URDF + differentiable rendering，把它变成了一个可操作、可规划、可编辑的世界模型。

### 战略取舍

| 设计选择 | 带来的能力 | 代价/风险 |
| --- | --- | --- |
| 用 3DGS 代替 2D 像素增强 | 多视角一致、空间关系更准、可统一处理多类扰动 | 需要静态多视图采集与重建 |
| 用 URDF + 可微渲染做精对齐 | 3D 编辑能落到真实机器人坐标，支持轨迹重定向与换本体 | 依赖 URDF 准确、ICP 初值与渲染质量 |
| 不走物理仿真，只做几何/外观级生成 | 生成快，避免显式 sim-to-real 物理误差 | 对强接触、动态、形变任务支持弱 |
| 用 AIGC 3D 对象 + AnyGrasp 扩充物体 | 不需人工采很多新物体示范 | 受生成几何质量与抓取估计误差影响 |

## Part III：证据与局限

### 关键证据信号

**信号 1：它不只是“能生成”，而是真的“生成得快”**
- 在 8 进程 RTX 4090 上，单条示范平均生成时间是 **0.64 秒**
- 人工采集平均是 **19.1 秒**
- 这说明方法具备现实中的扩展性，而不是只能做小规模离线展示

**信号 2：生成数据规模上去后，性能持续涨**
- 在五个真实任务上，只看物体位姿泛化时：
  - 800 条生成示范已经接近 200 条人工示范
  - 1800 条生成示范把平均成功率推到 **94.7%**
- 这支持了论文的核心论点：**高质量生成示范能部分替代昂贵的人工覆盖**

**信号 3：3D一致性在跨部署变化时特别有优势**
- **相机视角泛化**：平均 **80.6%**，明显高于 VISTA 的 **40.6%**
- **物体类型泛化**：**76.7%**，高于 ROSIE 的 **60.0%** 和仅真实数据的 **23.3%**
- **跨 embodiment**：在 UR5e 上接近 **100% / 96.7%**，明显强于 RoVi-Aug

**最关键的总体结论**
- 论文把“一条示范只能教一个场景实例”变成了“一条示范可以合成一个受控的部署分布”
- 这就是它相对 prior work 的能力跃迁

### 1-2 个最该记住的指标

- 真实世界六类泛化平均成功率：**87.8%**
- 单条示范生成成本：**0.64 s/条**

### 局限性

- **Fails when**: 场景包含可形变物体、显著动态交互、或需要精确接触/力学结果的任务；论文也明确指出 naive 3DGS 不适合 deformable 与 contact-rich dynamic tasks。
- **Assumes**: 有一条可用专家示范；能额外采集静态场景多视图图像；机器人 URDF 与运动学可靠；Grounded-SAM、AnyGrasp、3D 内容生成模型可用；部分流程依赖 GPT-4 和较强 GPU 资源（RTX 4090）。
- **Not designed for**: 需要真实物理后果建模的仿真式控制、复杂非刚体操作、开放世界长时动态场景重建。

### 可复用组件

- `3DGS ↔ URDF` 的可微渲染精对齐
- 物体相对关键帧的等变重定向
- 基于 link-wise Gaussian 的机器人姿态变换
- 用 3D 场景替换与 novel view synthesis 统一处理 appearance/view 泛化

### 总结判断

如果只看表面，这篇论文像是在做“机器人数据增强”；但从机制上看，它真正引入的是一个新的**因果操作旋钮**：

**把数据增强单元从像素级提升到坐标对齐的3D场景级。**

这让“视觉变化”和“动作标签”不再脱钩，因此一次示范也能长出可部署的泛化能力。  
不过，证据虽然亮眼，仍主要集中在作者自建真实平台，且缺少更系统的模块级消融，所以整体证据强度我会保守地定为 **moderate**。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Novel_Demonstration_Generation_with_Gaussian_Splatting_Enables_Robust_One_Shot_Manipulation.pdf]]