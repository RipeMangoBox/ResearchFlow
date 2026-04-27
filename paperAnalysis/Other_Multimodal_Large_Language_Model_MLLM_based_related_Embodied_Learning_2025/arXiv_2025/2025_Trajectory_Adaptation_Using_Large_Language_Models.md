---
title: "Trajectory Adaptation Using Large Language Models"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/trajectory-adaptation
  - code-as-policy
  - in-context-learning
  - closed-loop-feedback
  - dataset/LaTTe
  - opensource/no
core_operator: 让LLM同时生成可审阅的高层计划与可执行Python代码，对已有机器人轨迹的waypoint和速度进行按指令改写。
primary_logic: |
  初始轨迹（waypoint+速度）+ 自然语言指令 + 对象位置/环境描述
  → 通过提示词约束与对象/轨迹API让LLM生成高层适配计划和轨迹编辑代码，并在用户反馈闭环中迭代修正
  → 输出满足语义/数值约束且尽量保持原轨迹形状与平滑性的改写轨迹
claims:
  - "该方法无需任务特定训练，即可把自然语言指令转为对已有轨迹waypoint与速度的代码级编辑，并在机械臂、地面机器人、无人机三类仿真平台运行 [evidence: case-study]"
  - "在LaTTe子集之外，作者构造的含数值与复合约束的扩展指令集上，系统能够生成对应的高层计划与轨迹修改代码，处理如‘left by 20’和‘distance of at least 10’这类精确要求 [evidence: case-study]"
  - "通过先审阅高层计划再执行代码，并允许用户补充反馈，系统可以修正初次误解导致的轨迹改写偏差 [evidence: case-study]"
related_work_position:
  extends: "Code as Policies (Liang et al. 2023)"
  competes_with: "LaTTe (Bucker et al. 2023); EXTRACT (Yow et al. 2024)"
  complementary_to: "Elastic Dynamical System Motion Policies (Li and Figueroa 2023)"
evidence_strength: weak
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Trajectory_Adaptation_Using_Large_Language_Models.pdf
category: Embodied_AI
---

# Trajectory Adaptation Using Large Language Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv:2504.12755](https://arxiv.org/abs/2504.12755)
> - **Summary**: 这篇论文把“语言驱动的轨迹适配”重写成“LLM 生成高层计划 + 可执行代码来编辑已有轨迹”的问题，从而在无需任务特定训练的前提下支持数值化、复合式的人类指令。
> - **Key Performance**: 0 任务特定训练；在 LaTTe 子集、自建扩展指令集，以及机械臂/地面机器人/无人机 3 类仿真平台上展示了可工作的轨迹适配案例。

> [!info] **Agent Summary**
> - **task_path**: 自然语言指令 + 初始轨迹 + 对象位置/环境描述 -> 修改后的轨迹 waypoints 与速度
> - **bottleneck**: 如何把开放式、含数值的人类偏好稳定落到局部 waypoint/速度编辑上，同时尽量保留原轨迹形状与平滑性
> - **mechanism_delta**: 将“直接预测新轨迹”改为“先生成可审阅高层计划，再生成可执行代码编辑轨迹”，并用用户反馈闭环修正误解
> - **evidence_signal**: LaTTe 子集与扩展复杂指令集上的定性结果 + 3 类机器人仿真案例 + 反馈纠错示例
> - **reusable_ops**: [plan-then-code, human-in-the-loop-plan-validation]
> - **failure_modes**: [主观程度词指令歧义, 代码生成轨迹缺乏形式化平滑与动力学保证]
> - **open_questions**: [如何加入可验证安全约束模块, 如何建立统一定量评测与用户研究]

## Part I：问题与挑战

这篇论文解决的不是“从零生成机器人动作”，而是更具体也更实用的问题：**给定一条已经存在的轨迹，如何按照人类自然语言在新情境下做局部改写**。这类需求在协作机器人、人机共驾、示教后修正等场景里很常见，比如“离人远一点”“靠近箱子时慢一点”“向左 20 个单位并保持离障碍至少 10”。

### 真正的瓶颈是什么？

真正的难点不在于“听懂一句话”，而在于以下三件事要同时成立：

1. **语言要落到几何编辑上**：  
   指令最终必须变成对 waypoint 和速度的具体操作，而不是一句抽象解释。

2. **要保留原轨迹的结构先验**：  
   轨迹不是任意点集，起点/终点、形状、平滑性、局部连续性往往都很重要。  
   直接重规划太重，直接重生成又容易丢掉原轨迹的“可执行性”。

3. **开放词汇与数值约束要能同时处理**：  
   传统训练式模型常覆盖固定指令分布；一旦出现“left by 20”“at least 10 away from the box”这类数值或复合约束，泛化就会变难。

### 为什么现在值得做？

作者的判断是：**LLM 的代码生成能力已经足够强，可以把“轨迹适配”转成“程序化编辑已有轨迹”**。  
这比训练一个端到端 sequence-to-sequence 模型更灵活，因为：

- 数值运算、条件判断、对象相对距离这类逻辑更适合写成代码；
- 不必为每种改轨模式单独训练模型；
- 中间的高层计划可被人检查，降低黑盒误改的风险。

### 输入/输出接口与边界条件

- **输入**：
  - 初始轨迹 \( \tau \)：由 \((x,y,z,v)\) 组成的 waypoint 序列
  - 自然语言指令 \(L_{instruct}\)
  - 环境对象集合 \(O\)：对象标签和位置
  - 可选环境文字描述 \(E_d\)

- **输出**：
  - 适配后的轨迹 \( \tau_{mod} \)

- **边界条件**：
  - 假设已经有一条基本可执行的初始轨迹；
  - 假设对象位置可以通过接口拿到；
  - 假设坐标系定义明确；
  - 假设用户能看懂并修正高层计划。

换句话说，这不是一个完整的“从感知到控制”闭环系统，而是一个**基于已有轨迹的语言化编辑器**。

## Part II：方法与洞察

### 方法主线

作者把问题拆成两个由 LLM 共同生成的中间产物：

1. **High-Level Plan (HLP)**  
   先把用户意图转成逐步的轨迹修改方案，比如：
   - 是否保持起点不变
   - 是否移动终点
   - 哪些中间点需要改
   - 如何相对某个对象增大/减小距离
   - 哪些局部需要减速/加速

2. **Python Code**  
   再基于这个计划生成可执行代码，直接操作 waypoint 和速度。

### 提示词里显式提供了什么？

为了让 LLM 不必“猜环境”，论文把关键结构外显到 prompt 中：

- 坐标系定义（left/right/front/up 的含义）
- `detect_objects(obj_name)`：返回对象位置
- `get_trajectory()`：返回轨迹点与速度
- 若干规则：
  - 尽量渐进式修改，保证平滑
  - 需要判断起点/终点是否应保持不变
  - 允许增删 waypoint
  - 速度变化相对原速度进行，且应平滑

此外，作者只给了 **两个高层计划 in-context 示例**，用来把输出风格固定到“可供人审阅”的形式。

### 闭环反馈机制

流程不是“一次生成、直接执行”，而是：

1. 生成 HLP + code  
2. 用户先看 HLP  
3. 如果 HLP 有误解，用户补充反馈  
4. LLM 基于原始指令 + 反馈重新生成  
5. HLP 通过后再执行代码

这一步很关键：作者并没有试图证明 LLM 第一次一定对，而是通过**把中间思路显式化**，让人可以在代码运行前拦截错误。

### 核心直觉

这篇论文最核心的改变是：

> **把轨迹适配从“学习一个从语言到连续轨迹的黑盒映射”，改成“在显式轨迹与对象 API 上做程序合成”。**

对应的因果链是：

- **What changed**：  
  从端到端轨迹预测，改成 `高层计划 -> 代码编辑已有轨迹`。

- **Which bottleneck changed**：  
  语言中的数值约束、对象相对关系、局部条件修改，不再需要完全靠训练分布去隐式记住；  
  它们被转写成代码中的显式算术、条件和列表操作。

- **What capability changed**：  
  系统更容易支持：
  - 数值化指令
  - 复合指令
  - 只改轨迹某一段
  - 用户可读、可纠错的中间表示

### 为什么这套设计有效？

因为它把最难的部分拆对了：

- **几何先验来自原轨迹**：不用从零找一条新路；
- **逻辑精确性来自代码**：数值偏移、距离阈值、局部减速都更适合程序表达；
- **可解释性来自 HLP**：用户能在执行前发现“模型理解偏了”；
- **泛化性来自 LLM 的程序合成能力**：不用为每种改轨语义单独训练。

### 策略性权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
| --- | --- | --- | --- |
| HLP + 代码，而非直接输出轨迹 | 黑盒映射难审查 | 可解释、可调试、支持复合逻辑 | 增加了链路复杂度 |
| 显式对象 API + 坐标系 | 环境 grounding 含糊 | 能处理对象相对距离和方向指令 | 强依赖外部感知/坐标一致性 |
| 在原轨迹上做渐进式修改 | 从零生成轨迹不稳定 | 更容易保留原形状与局部连续性 | 受原轨迹质量上限限制 |
| 用户先审 HLP 再执行 | 一次性生成易误解 | 可闭环修正 | 需要人工参与，实时性下降 |
| 允许增删 waypoint 与调速度 | 固定长度输出太僵硬 | 能表达停止、延伸、局部速度变化 | 更难保证动力学可行性 |

## Part III：证据与局限

### 关键证据信号

这篇论文的证据核心不是“量化 SOTA”，而是**覆盖面与可操作性**。

1. **跨平台定性案例**  
   作者在三类机器人/仿真平台上测试：
   - Kuka 机械臂（PyBullet）
   - 地面机器人（Gazebo）
   - Crazyflie 无人机（PyBullet）  
   说明该框架对“机器人种类”本身并不强绑定，只要下游接受 waypoint 轨迹即可。

2. **指令覆盖从简单扩展到数值与复合约束**  
   除 LaTTe 子集外，作者还构造了扩展集合，加入：
   - “Go left by 20”
   - “Keep at least 10 distance from the box”
   - “Go left by 20 keeping a distance of at least 10 from the box”  
   这支持论文的核心主张：**代码式编辑比训练式固定映射更适合数值/复合指令**。

3. **反馈纠错案例**  
   Figure 5 展示了用户补充信息后，系统能修正原先 HLP 的理解偏差。  
   这支持论文另一个重要点：**HLP 不是装饰，而是把错误暴露到可纠正层**。

### 能力跃迁到底在哪里？

相对 LaTTe 这类训练式轨迹语言模型，这篇论文声称的能力跳跃主要不是更高分，而是三点：

- **不需要任务特定训练**
- **能更自然地处理数值和复合指令**
- **提供可审查的中间计划，便于人类纠错**

也就是说，它的优势更像是**接口层与工作流层的扩展**，而不是已经被充分证明的 benchmark-level SOTA。

### 关键指标情况

论文的实证有明显局限：

- 没有统一报告成功率、轨迹平滑度、延迟等标准化指标；
- 没有系统性 ablation；
- 没有与 LaTTe 等方法的严格定量对比表；
- 没有真实机器人实验；
- 没有人类用户研究。

因此，这篇论文更像一个**有说服力的原型验证**，而不是已经被全面定量证明的方法论文。

### 局限性

- **Fails when**: 指令包含“very slow”“relatively slower”这类强主观词、环境描述缺失、对象定位不准、或任务需要严格动力学/避障/平滑保证时，生成代码可能语义上对、但几何上不规整甚至不可执行。
- **Assumes**: 系统假设已有一条可编辑的初始轨迹；可通过接口获得对象标签与位置；用户能提供坐标系和必要环境语义；核心生成依赖闭源 GPT-4o；实验主要在 PyBullet/Gazebo 仿真中完成，未展示真实机器人部署。
- **Not designed for**: 从零做全局规划、提供形式化稳定性/安全/约束满足证明、替代低层控制/IK/碰撞检测模块，或覆盖所有开放词汇指令的统一约束求解。

### 可复用组件

这篇论文里最值得迁移的，不是某个具体 prompt，而是几个模式：

- **plan-then-code**：先生成人类可读计划，再生成可执行程序；
- **轨迹编辑 API 化**：把环境对象与轨迹访问封装为简单函数接口；
- **保形修改规则**：显式要求渐进式修改、起终点保持、相对速度调整；
- **human-in-the-loop 审核位**：在代码执行前插入一个低成本纠错层。

如果后续与安全过滤、动力学约束优化、或稳定性保证模块结合，这个框架会更完整。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Trajectory_Adaptation_Using_Large_Language_Models.pdf]]