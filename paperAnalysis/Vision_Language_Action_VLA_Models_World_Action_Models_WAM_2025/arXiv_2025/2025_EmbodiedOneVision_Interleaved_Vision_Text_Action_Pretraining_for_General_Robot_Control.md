---
title: "EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-control
  - flow-matching
  - interleaved-pretraining
  - state-token
  - dataset/EO-Data1.5M
  - dataset/EO-Bench
  - dataset/LIBERO
  - dataset/SimplerEnv
  - opensource/full
core_operator: 把视觉、文本、状态与动作按时间交错成单序列，在共享 decoder 上同时做文本自回归与连续动作 flow matching。
primary_logic: |
  多视角观测/语言指令/机器人状态/历史动作 + 网页多模态数据与机器人轨迹 →
  构造交错的 vision-text-action 训练序列，并在统一 decoder 上联合进行文本自回归与连续动作 flow matching，配合 rectifying sampling 保持因果一致性 →
  同一模型输出中间推理文本与连续动作 chunk，实现开放世界泛化的机器人控制
claims:
  - "EO-1 (3B) 在 RoboVQA 上取得 58.5 BLEU-4，超过 GPT-4o 的 47.2，并在 ERQA 上达到 45.5，略高于 InternVL2.5 8B 的 45.2 [evidence: comparison]"
  - "EO-1 在 LIBERO 上取得 98.2% overall success rate，高于 OpenVLA-OFT 的 97.1% 与 π0 的 94.2%，并在 SimplerEnv 的 WidowX/Google-VM/Google-VA 上分别达到 72.7%/76.5%/63.0% [evidence: comparison]"
  - "混合 AR+flow 的统一架构与交错数据都带来可测收益：EO-1(base) 在 LIBERO 上较纯 AR 的 EO-1(fast) 提升 10.2 个点，而 EO-1(interleaved) 在 WidowX generalization overall 上由 0.71 提升到 0.80 [evidence: ablation]"
related_work_position:
  extends: "π0 (Black et al. 2024)"
  competes_with: "π0 (Black et al. 2024); OpenVLA-OFT (Kim et al. 2025)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2025/arXiv_2025/2025_EmbodiedOneVision_Interleaved_Vision_Text_Action_Pretraining_for_General_Robot_Control.pdf
category: Embodied_AI
---

# EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2508.21112), [Project](https://eo-robotics.ai/eo-1), [Code](https://github.com/eo-robotics), [HuggingFace](https://huggingface.co/IPEC-COMMUNITY)
> - **Summary**: 论文正文公开模型名为 **EO-1**；其核心贡献是把“看-想-动”按时间交错进同一序列，并用统一 decoder 同时学习文本推理与连续动作生成，从而减少规划与控制之间的接口断裂。
> - **Key Performance**: LIBERO overall 98.2%；RoboVQA BLEU-4 58.5

> [!info] **Agent Summary**
> - **task_path**: 多视角图像/语言指令/机器人状态/历史上下文 -> 中间推理文本 + 连续动作 chunk
> - **bottleneck**: 现有 VLA 多把动作放在序列末尾或与推理分开训练，学不到 reasoning→action→new observation 的因果闭环
> - **mechanism_delta**: 用共享 decoder 在交错的 vision-text-action 序列上联合做文本 AR 和动作 flow matching，并用 rectifying sampling 修正 interleaved 训练时的因果污染
> - **evidence_signal**: 多基准比较 + 架构/数据消融同时成立；LIBERO 98.2%，而纯 AR 版本为 88.0%，交错数据又把 WidowX generalization 从 0.71 拉到 0.80
> - **reusable_ops**: [interleaved embodied data construction, rectifying sampling]
> - **failure_modes**: [小规模控制数据下的动作域外位置变化, 与物理控制不对齐的通用 VLM 数据会削弱 grounding]
> - **open_questions**: [如何实现同步推理与动作而非串行解码, 如何扩展到导航/避障/人机协作等更广场景]

## Part I：问题与挑战

这篇工作的真问题，不是“怎么把 VLM 接一个 action head”，而是：

**现有 VLA 还不会像人一样把推理和动作交错起来。**

### 1. 真正的瓶颈是什么？
已有 VLA 往往有两类问题：

1. **数据层面断裂**  
   网页视觉-语言数据有语义知识，但没有真实物理交互连续性；机器人数据有动作监督，但通常缺少丰富的时空推理标注。  
   结果是模型要么“懂图不太会动”，要么“会动但不太会理解开放世界指令”。

2. **序列层面断裂**  
   许多方法默认把动作放在输出末尾，等价于“先理解，再一次性出动作”。  
   这忽略了 embodied interaction 的真实结构：  
   **推理指导动作，动作结果又反过来更新推理。**

3. **训练层面断裂**  
   一旦动作用连续 denoising/flow matching 来学，若把“噪声动作”直接放在交错序列中，后续文本或视觉条件就会看到一个不真实的动作上下文，破坏因果关系。

### 2. 为什么现在值得解决？
因为三个条件同时成熟了：

- 开源 VLM backbone 已经足够强，能提供通用视觉语言先验；
- 大规模机器人 demonstrations 已经可得；
- 社区开始从“会做单任务操作”转向“开放世界泛化 + 推理控制一体化”。

所以这篇论文要解决的是：  
**如何让一个统一模型既保留 VLM 的语义能力，又真正学到机器人执行过程中的时序因果关系。**

### 3. 输入/输出接口与边界条件
**输入：**

- 多视角图像/视频观测
- 文本任务指令或中间 QA
- 机器人状态
- 历史动作/当前 noisy action

**输出：**

- 文本推理结果（如 planning、verification、grounding 回答）
- 连续动作 chunk（文中推理时是 16-step action chunk）

**边界条件：**

- 主要覆盖的是 manipulation 场景，不是通用移动机器人全栈问题；
- 依赖多视角视觉与任务指令；
- 更擅长“看图—推理—操作”闭环，而非纯导航或长期自主探索。

---

## Part II：方法与洞察

这篇工作的设计哲学很明确：

> **不要把推理模块和控制模块硬拼起来，而是让它们从训练数据到序列建模都处在同一个因果流里。**

### 方法骨架

#### 1. 统一 backbone：一个 decoder，同时做两件事
作者使用单一 decoder-only transformer，初始化自 **Qwen2.5-VL**，共享参数处理：

- 图像 token
- 文本 token
- 机器人状态 token
- 动作 token

然后在共享 backbone 之上接两个输出头：

- **LM head**：负责文本自回归
- **Flow head**：负责连续动作生成

这一步改变的不是“模型大不大”，而是**动作不再是外挂专家模块，而是统一上下文里的一等公民**。

#### 2. 交错数据：不是只有 robot episodes，而是“带 reasoning 的 robot episodes”
训练数据分三类：

- **web multimodal data**：补通用语义与视觉理解
- **robot control data**：补真实控制监督
- **interleaved embodied data**：补 reasoning 与 action 的时序耦合

其中最关键的是第三类。作者从机器人视频中构造：

- 时间推理 QA：task planning、process verification、failure detection、physical commonsense
- 空间推理 QA：trajectory、object referring、object pointing、multiview pointing
- 再把这些 QA 与动作按时间顺序拼成 interleaved vision-text-action 序列

一个重要细节：  
论文里 **EO-Data1.5M** 是一个更大的 embodied reasoning 语料概念；真正用于 mixed-modality generation 的 interleaved VTA 子集约 **122k**。  
也就是说，它不是 1.5M 条纯 action-interleaved trajectory，而是“推理数据 + 一部分真正交错的动作序列”的组合。

#### 3. Rectifying sampling：解决 interleaved 训练中的因果污染
这是论文里最关键、也最容易被忽视的机制点。

问题在于：  
如果中间某段动作还是 noisy action，那么后续文本/视觉/动作去 attend 它，就等于训练时看到了“不真实历史”。

作者的修正方法是：

- 对包含多个 action segment 的 interleaved sequence，切成若干训练子序列；
- 对于中间 action segment，不再保留 noisy action，而是替换成 clean action；
- 这样后续 token 学到的就是“基于真实已执行动作”的上下文。

这相当于给交错 denoising 序列做了一个**因果校正**。

### 核心直觉

**改变了什么？**  
从“把动作放在最后单独预测”改成“让动作和推理文本一起，按时间顺序交错出现”。

**哪个瓶颈被改变了？**  
原先模型看到的是分离的语义分布和控制分布；现在它看到的是  
**reasoning ↔ acting 的联合条件分布**。  
同时，flow matching 替代纯离散动作 token，缓解了控制精度和推理速度瓶颈；rectifying sampling 又避免了 noisy action 破坏因果上下文。

**能力为什么会变强？**  
因为模型学到的不再是“看到图 -> 输出动作”的静态映射，  
而是“看到图 -> 说出下一步 -> 执行动作 -> 根据新状态继续判断”的循环。

这使它在三件事上更强：

- 更稳的 instruction following
- 更强的空间/时序 reasoning
- 更一致的 planning-control alignment

### 策略性取舍

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价/风险 |
|---|---|---|---|
| 单一共享 decoder + 双头输出 | planner/controller 接口断裂 | 语义与动作表征对齐更直接 | 共享参数可能有梯度干扰 |
| 动作用 flow matching，而不是全离散 token | 动作精度与推理延迟瓶颈 | 连续控制更精细，长程控制更稳 | 训练时必须处理 noisy action 的因果问题 |
| 交错 vision-text-action 数据 | 缺少 reasoning-acting 联合分布 | 学到时序因果闭环 | 标注管线复杂、成本高 |
| 任务对齐的 embodied QA，而非泛化通用 instruct data | 语言知识与物理 grounding 脱钩 | 提升 spatial/task reasoning 与 instruction following | 若数据设计失衡，可能过拟合特定 embodied 格式 |

### 一个很重要的洞察
这篇论文其实还给出一个负面结果：

> **不是 multimodal data 越多越好，而是要和 embodied control 对齐。**

作者的分析表明，简单混入通用 LLaVA 风格指令数据，反而会损害 WidowX/Agibot 上的泛化。  
这说明它真正有用的不是“多模态”这三个字，而是**物理语义和动作监督是否共现**。

---

## Part III：证据与局限

### 关键证据信号

- **对比信号 1：embodied reasoning 确实更强**  
  EO-1 在 RoboVQA 上达到 **58.5 BLEU-4**，高于 GPT-4o 的 47.2；在 ERQA 上 **45.5**，也略高于 InternVL2.5 8B。  
  这说明交错训练不仅没伤害视觉语言理解，反而提升了 embodied 场景下的推理质量。

- **对比信号 2：robot control 不是“会说不会做”**  
  在 LIBERO 上，EO-1 取得 **98.2%** overall success rate，超过 OpenVLA-OFT 97.1% 和 π0 94.2%；  
  SimplerEnv 上也系统性超过 π0。  
  这支持了论文最核心的主张：**统一 reasoning + action 并不会牺牲控制精度，反而可能提高泛化控制。**

- **对比信号 3：真实世界 reasoning-control 一体化有增益**  
  在作者构建的 real-world 28 任务评测中，EO-1 平均完成度 **86%**，高于 GR00T-N1.5 的 71% 和 π0 的 68%。  
  尤其在 Tic-Tac-Toe、Visual Rearrangement、Make Sandwich 这类“需要边判断边操作”的任务上，优势更明显。

- **因果信号：机制与结果对得上**  
  论文不是只给榜单，还做了结构性分析：  
  - **纯 AR -> 混合 AR+flow**：LIBERO 从 88.0% 提到 98.2%  
  - **仅 robot data -> interleaved embodied data**：WidowX generalization overall 从 0.71 提到 0.80  
  - **加入不对齐的通用 instruct data**：反而显著下降  
  这组证据相对清楚地支持了作者的因果叙事：  
  **真正起作用的是“统一动作建模 + 任务对齐的交错 embodied 数据”。**

### 1-2 个最关键指标
- **LIBERO overall**: 98.2%
- **RoboVQA BLEU-4**: 58.5

### 局限性

- **Fails when**: 动作域外变化较大、且对应控制数据规模不足时，泛化会明显下降；论文明确提到在 Make Breakfast Sandwich 中，仅仅改变 bread 位置就会让成功率下滑。
- **Assumes**: 依赖大规模多源数据（1.2M 机器人 episode + 5.7M web multimodal + interleaved embodied data）、强 VLM/LLM 辅助标注（如 Qwen2.5-VL 72B、ChatGPT）和人工校验；同时假设有多视角视觉输入与较稳定的 manipulation 接口。
- **Not designed for**: 目前并非面向完整通用 embodied stack；导航、避障、失败恢复、人机协作、意图识别以及真正“同步推理-动作”的在线系统还不在其核心覆盖范围内。

### 复现与扩展上的现实约束
虽然作者强调：

- 权重开源
- 训练代码开源
- 数据组件开源

且推理阶段声称单张 4090、约 6GB 显存即可实时运行，

但真正复现其完整 recipe 仍然有门槛：

- 大规模训练 token 总量高达 **135B**
- 数据构造依赖强闭环标注管线
- 高质量 interleaved embodied annotation 本身很难低成本复制

所以它是一个**开放但不轻量**的 recipe。

### 可复用组件
- **interleaved VTA 数据构造模板**：planning / verification / trajectory / free chatting 与动作拼接
- **rectifying sampling**：适用于“连续动作 denoising + 因果序列建模”并存的场景
- **共享 backbone + 双头输出**：文本 AR 与连续动作 flow 的统一接口
- **EO-Bench 的分维度诊断思路**：把 spatial / temporal / commonsense / state estimation 分开评估，而不是混成一个笼统 VQA 分数

![[paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2025/arXiv_2025/2025_EmbodiedOneVision_Interleaved_Vision_Text_Action_Pretraining_for_General_Robot_Control.pdf]]