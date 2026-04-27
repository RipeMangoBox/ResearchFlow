---
title: "FUNCTO: Function-Centric One-Shot Imitation Learning for Tool Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/one-shot-imitation-learning
  - task/tool-manipulation
  - functional-keypoints
  - dense-correspondence
  - visual-prompting
  - dataset/Custom-RealRobot
  - repr/3D-functional-keypoints
  - opensource/no
core_operator: 用“函数点-抓取点-中心点”三元3D功能关键点替代形状相似性对齐，并以点/平面/轴约束把单次人类示范迁移到新工具。
primary_logic: |
  单个RGB-D人类演示视频+任务描述+测试场景RGB-D → 提取并跟踪演示工具的3D功能关键点 → 在测试工具上进行演示引导的关键点迁移并通过函数点/平面/轴对齐建立功能对应 → 约束优化生成并执行机器人末端轨迹
claims:
  - "在五类真实机器人工具任务上，FUNCTO 在 novel/unseen tools 泛化时取得 79.5% 总体成功率，高于最佳模块化 OSIL 基线 DINOBot 的 57.5% [evidence: comparison]"
  - "与使用 50 条示范训练的行为克隆基线相比，FUNCTO 仅用 1 条示范就在 unseen tools 上达到 79.5%，显著高于 ACT/DP/DP3 的 32.5%/26.5%/17.5% [evidence: comparison]"
  - "Demo+VLM+DSC 的函数点迁移策略将平均关键点距离降至 18.54 像素，并把 AP@30 提升到 85.78%，优于去掉演示参考或去掉稠密对应的变体 [evidence: ablation]"
related_work_position:
  extends: "KPAM (Manuelli et al. 2019)"
  competes_with: "DINOBOT (Di Palo and Johns 2024); DITTO (Heppert et al. 2024)"
  complementary_to: "FoundationPose (Wen et al. 2024); Diffusion Policy (Chi et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_FUNCTO_Function_Centric_One_Shot_Imitation_Learning_for_Tool_Manipulation.pdf"
category: Embodied_AI
---

# FUNCTO: Function-Centric One-Shot Imitation Learning for Tool Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.11744), [Project](https://sites.google.com/view/functo)
> - **Summary**: 这篇工作把“单次示范学工具使用”从外观/几何匹配改写为“功能关键点匹配”，让机器人能从一段人类视频把倒、切、舀、刷、敲等技能迁移到形状差异很大的新工具上。
> - **Key Performance**: unseen tools 总体成功率 **79.5%**；函数点转移 **AKD 18.54 px / AP@30 85.78%**

> [!info] **Agent Summary**
> - **task_path**: 单视角 RGB-D 人类示范视频 + 任务文本 + 测试场景 RGB-D → 机器人 6DoF 末端轨迹与工具操作执行
> - **bottleneck**: 同功能工具在形状、尺度、拓扑上差异很大时，传统基于几何/外观的对应关系无法保留“怎么用”这一不变量
> - **mechanism_delta**: 用 3D 功能关键点表示工具的可用性，并用函数点/平面/轴三级对齐替代浅层外观匹配
> - **evidence_signal**: 5 类真实机器人任务、250 次试验中，FUNCTO 在 unseen tools 上达到 79.5%，显著高于 OSIL 与 BC 基线
> - **reusable_ops**: [demonstration-guided keypoint transfer, function-point/plane/axis alignment]
> - **failure_modes**: [open-loop grasp slip or tool flip, contact-rich trajectory collision or depth-induced misalignment]
> - **open_questions**: [how to recover when functional keypoints are occluded or collinear, how to model multi-modal tool usages from few demonstrations]

## Part I：问题与挑战

这篇论文处理的不是一般的“看视频模仿动作”，而是更难的 **工具使用 one-shot imitation**：

- 人类只给机器人 **1 段单视角 RGB-D 示范视频**
- 没有动作标注
- 测试时工具换了，但**功能相同**
- 机器人要输出一条可执行的 6DoF 末端轨迹，完成等价任务

### 真正的难点是什么？

作者认为，真正瓶颈不是“学一条轨迹”，而是：

**如何在几何差异很大的同功能工具之间建立功能对应。**

例如：
- mug 和 teapot 都能 pour
- 但它们的 body、handle、spout、整体拓扑都不同
- 如果仍然沿用“外观像不像”“点云能不能配准上”的思路，迁移会失败

这也是很多以往 OSIL 方法的共同假设：
- 工具之间外观相近
- 形状相近
- 或至少能靠全局配准/语义对应找出稳定映射

而这篇论文要解决的，恰恰是**intra-function variation**：  
同一功能下，工具的外形和结构差得很大，但“使用方式”仍有某种不变量。

### 输入 / 输出接口

**输入：**
- 人类示范阶段：
  - 单视角 RGB-D 视频
  - 任务文本描述（工具、目标物、功能）
- 测试阶段：
  - 机器人当前 RGB-D 观测
  - 对应任务文本

**输出：**
- 机器人末端执行器的 6DoF 轨迹
- 配合抓取后完成工具与目标物的交互

### 这项问题的边界条件

论文明确假设：

1. 单视角、静态相机
2. 无动作标注
3. 无物体 3D 模型、无手工任务约束、无 in-domain pretraining
4. 工具是刚体，且可由平行夹爪抓取
5. 可以使用基础模型中的 commonsense

这意味着它追求的是：
- **极低示范成本**
- **较强跨工具泛化**
- 但不追求复杂闭环控制或丰富接触反馈

### 为什么现在值得做？

因为近两年出现了几个可拼起来的“模块”：
- VLM：能推断“工具哪个部位是功能部位”
- Grounded-SAM：能做开放词汇检测/分割
- CoTracker：能从视频里稳定跟踪点
- dense semantic correspondence：能做更细粒度跨图像点转移

FUNCTO 的贡献不在于发明每个模块，而在于把它们组织成一条**功能中心的因果链**。

**What/Why**：真正瓶颈是跨工具的功能不变量表示与对应，而不是再堆更多示范去硬学一个端到端策略。

## Part II：方法与洞察

FUNCTO 的整体思路可以概括成三步：

1. 从示范视频里提取 **3D 功能关键点**
2. 把这些关键点转移到测试工具上，并建立 **function-centric correspondence**
3. 用这些对应关系约束 **轨迹规划与执行**

---

### 1）功能关键点提取：把“怎么用”压缩成三个点

作者提出的 3D functional keypoint representation 由三个点组成：

- **function point**：工具对目标起作用的位置  
  例：杯口/壶嘴、刀刃前缘、勺头
- **grasp point**：适合抓持的位置  
  例：把手
- **center point**：工具几何中心

这三个点不是普通语义关键点，而是“与功能直接相关”的关键点。

#### 关键做法

- 用 Grounding-SAM 在首帧找出工具和目标
- 在工具 mask 内均匀采样点
- 用 CoTracker 跟踪其 3D 运动
- 把轨迹从相机坐标系转到**目标物坐标系**

这一步非常关键：  
它把“绝对空间中的动作”变成“相对目标物的动作”，因此天然支持空间泛化。

#### 为什么还要找 keyframe？

因为 function point 在真正交互时常常会被遮挡。  
所以论文专门找四个关键帧：

- 初始帧
- 抓取帧
- 功能发生帧
- 功能发生前一帧（pre-function）

其中 pre-function 帧是为了在严重遮挡之前定位 function point。

#### 三个点分别怎么来？

- **grasp point**：手部 mask 和工具 mask 的交集
- **center point**：工具 3D 中心
- **function point**：靠 VLM + mark-based visual prompting 推断

也就是说，作者把“哪一部分真正承担功能”交给了 VLM 的常识，而不是只靠几何接触。

---

### 2）功能对应建立：不是对齐形状，而是对齐“可用性”

这一部分是论文的核心。

#### 2.1 功能关键点转移

作者先把示范工具上的 function/grasp point 投影到图像上，作为参考，再把它们转移到测试工具。

转移分两步：

- **粗粒度**：VLM 根据示范参考，在测试工具上提出候选区域
- **细粒度**：dense semantic correspondence 在候选区域内做点级匹配

这比纯 VLM 更精确，也比纯 dense correspondence 更能跨类别。

一个很重要的观察是：

> 测试工具上可能有多个“也说得过去”的功能点，但不是每个都对应示范中那种用法。

例如杯子可以前倾倒，也能侧倾倒。  
如果没有示范参考，VLM 可能选到“能用但不是这次示范所表达的那一种”。

#### 2.2 Function-centric correspondence

拿到测试工具上的三个点之后，FUNCTO 用三层约束建立对应：

1. **函数点对齐**  
   确保工具真正起作用的位置对准目标交互位置

2. **函数平面对齐**  
   确保工具的大体朝向合理

3. **函数轴对齐**  
   确保关键操作方向合理，例如 pour 时需要合适倾角

这里最有意思的是第三步。  
作者发现：若仅刚性对齐 function axis，跨类别工具仍可能失败，因为三点相对布局不同。

典型例子：
- mug 和 teapot 都是 pour
- 但它们所需倾斜角并不一样

为此，作者额外让 VLM 在一组渲染出来的候选姿态里选“最利于成功”的那个，等于让 VLM 来做一次**姿态可行性常识修正**。

---

### 3）基于功能关键点的动作规划：把对应关系变成机器人轨迹

有了：
- 初始状态关键点
- 功能关键帧关键点
- 示范关键点轨迹

就可以把测试工具的姿态序列当成一个约束优化问题来解。

核心思想不是学习 policy，而是：

- 固定起始姿态
- 固定功能关键帧姿态
- 中间轨迹尽量跟随示范中“function point 相对目标的运动模式”和工具姿态变化

然后：
- 用 GraspGPT 在 grasp point 周围采样 task-oriented grasp
- 假设抓取后夹爪与工具刚性连接
- 执行规划出的末端轨迹

这使整个方法呈现出非常清晰的层次：
**示范理解 → 功能对应 → 轨迹优化 → 执行**

### 核心直觉

以前的方法把跨工具迁移近似成：

**“形状/外观相似 ⇒ 可以建立对应 ⇒ 可以迁移动作”**

FUNCTO 把这个链条改成：

**“功能使用模式相似 ⇒ 三个功能锚点相对关系可对齐 ⇒ 可以迁移关键交互状态与轨迹”**

更具体地说：

- **改变了什么**：从全局几何/视觉相似性，改成低维功能关键点表示
- **改变了哪个瓶颈**：去掉了大量与任务无关的形状细节，缓解了跨类别工具的分布偏移
- **带来了什么能力**：从“只能泛化到像示范那样的工具”，变成“能泛化到同功能但拓扑很不同的工具”

为什么这套设计有效？

1. **功能关键点更接近任务成功的充分统计量**  
   真正决定 pour/cut/scoop 成功的，不是整个表面，而是“哪里起作用、哪里抓、整体怎么朝向目标”。

2. **目标物坐标系表示消除了布局依赖**  
   这让 spatial generalization 成为结构上的自然结果，不需要再额外学。

3. **VLM 补上几何看不出的常识**  
   比如：
   - 哪个边缘更像功能边
   - 这个工具在当前任务里该怎么倾斜
   - 多个候选点里哪个更符合示范意图

#### 战略权衡

| 设计选择 | 带来的能力 | 代价 / 风险 |
|---|---|---|
| 3D 功能关键点表示 | 只保留任务相关信息，增强跨类泛化，可解释 | 依赖关键点可见性和深度精度；三点共线会失效 |
| 演示引导的 VLM+DSC 转移 | 同时兼顾常识与点级精度，保持示范意图一致 | 依赖 VLM 质量、prompt 设计和 correspondence 模型 |
| VLM 函数轴细化 | 处理不同工具所需倾角/姿态差异 | 需要额外候选渲染和推理成本 |
| 开环轨迹优化 | 数据需求低、结构清晰、易加碰撞/速度约束 | 缺少闭环纠错，抓取滑移和接触碰撞时脆弱 |

**How**：作者真正调的因果旋钮是“表示与对应方式”——从几何相似性对齐切换为功能关键点对齐，再让 VLM 负责几何之外的常识补全。

## Part III：证据与局限

### 关键证据

- **比较信号 1：对 OSIL 基线**
  - 评测覆盖 5 种功能：pour / cut / scoop / brush / pound
  - 每种功能 5 个任务，共 250 次真实机器人试验
  - 所有方法在 seen 的空间泛化上都不差，但一旦换成 novel instance/category tool，传统方法明显掉点
  - 论文正文指出，最佳模块化 OSIL 基线 **DINOBot** 在 novel tool 泛化上平均为 **57.5%**
  - **FUNCTO 达到 79.5%**
  - 这说明收益不只是“更会跟随轨迹”，而是确实更会处理跨工具功能对应

- **比较信号 2：对行为克隆基线**
  - ACT / DP / DP3 都使用 **50 条示范**训练
  - unseen tools 上分别只有：
    - ACT：**32.5%**
    - DP：**26.5%**
    - DP3：**17.5%**
  - FUNCTO 只用 **1 条示范**就到 **79.5%**
  - 这个对比非常强，说明在高 intra-function variation 下，**显式功能对应**比“把一切压进 policy”更数据高效

- **消融信号 1：功能点转移策略**
  - 作者把 function point transfer 单独拆出来测
  - 提出的 **Demo+VLM+DSC** 最好：
    - AKD = **18.54 px**
    - AP@30 = **85.78%**
  - 纯 zero-shot VLM 明显差很多（AKD 56.09）
  - 说明示范参考是必要的；而 dense correspondence 主要负责补足点级精度

- **消融信号 2：对应关系怎么对齐最有效**
  - function point alignment 比对齐其他点更稳
  - VLM 的 function-axis refinement 比刚性轴对齐更好
  - 这支持论文的核心判断：  
    **跨工具泛化的关键不是“强行刚性复制姿态”，而是让功能状态保持可行。**

- **失败分析信号**
  - 主要失败源来自：
    - 抓取不稳，工具打滑/翻转
    - 轨迹规划中的碰撞，尤其是 contact-rich 任务
  - 反而功能点转移和 correspondence 不是最大失败来源
  - 这很重要：说明 FUNCTO 已经把系统主瓶颈从“看懂工具”推进到了“闭环执行与接触控制”

**So what**：这篇工作真正实现的能力跃迁，是把 one-shot imitation 从“对相似工具有效”推进到“对同功能但形状差异大的工具也有效”，而且比 50-demo 的行为克隆更省数据。

### 局限性

- **Fails when**: function point 在当前视角不可见；三个功能关键点在 3D 中近乎共线；抓取后发生滑移/翻转；接触丰富任务中出现未建模碰撞或深度误差导致的姿态错配。
- **Assumes**: 单视角静态 RGB-D 相机、刚性工具、平行夹爪、较准确深度；依赖 Grounding-SAM、CoTracker、VLM、dense semantic correspondence、GraspGPT 等外部模块；当前是开环执行。论文未明确 VLM 的具体实现，若复现实验依赖专有 API，则结果会对模型版本和服务可用性敏感。
- **Not designed for**: 非刚性工具或显著动力学主导的交互；严重遮挡的第一视角视频；需要从单次示范推断多模态用法的场景；长时闭环纠错和高精度连续接触控制。

### 可复用组件

- **演示引导的关键点转移模板**  
  先用示范关键点作为 reference，让 VLM 提候选 region，再用 dense correspondence 精修到点级。

- **函数点 / 平面 / 轴三级约束**  
  这是一种很通用的可解释 manipulation constraint 模板，不只适用于本文任务。

- **目标物坐标系下的相对轨迹表示**  
  对任何“工具相对目标物运动模式比较稳定”的任务都很有价值。

- **关键点约束 + 轨迹优化**  
  便于插入速度、平滑、碰撞规避等约束，适合与更强的闭环控制器继续组合。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_FUNCTO_Function_Centric_One_Shot_Imitation_Learning_for_Tool_Manipulation.pdf]]