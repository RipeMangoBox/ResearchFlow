---
title: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"
venue: NeurIPS
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - execution-based-evaluation
  - virtual-machine
  - accessibility-tree
  - dataset/OSWorld
  - opensource/full
core_operator: "在真实操作系统虚拟机中，用任务初始状态配置与逐任务执行脚本，把开放式电脑操作转成可复现、可自动判分的评测。"
primary_logic: |
  开放式真实电脑任务评测目标 → 在虚拟机中构造跨应用初始状态并收集369个任务/134类评测函数 → 通过执行式终态检查而非单一步骤匹配进行评分 → 揭示当前多模态代理在GUI grounding、操作知识与长程工作流上的能力边界
claims:
  - "OSWORLD提供一个支持Ubuntu、Windows和macOS的真实计算机评测环境，并基于其构建了369个Ubuntu任务、43个Windows分析任务、302种初始状态和134个独立执行式评测函数 [evidence: analysis]"
  - "在OSWORLD上，人类完成率为72.36%，而最佳基线仅达到12.24%；多应用工作流任务成功率最高仍不到8%，显示当前LLM/VLM离可用电脑助手仍有明显差距 [evidence: comparison]"
  - "更高截图分辨率和更长文本历史可提升代理表现，但窗口位置/尺寸扰动和界面噪声会使成功率下降60%–80%，说明GUI grounding与鲁棒性是主要瓶颈 [evidence: analysis]"
related_work_position:
  extends: "WebArena (Zhou et al. 2023)"
  competes_with: "OmniAct (Kapoor et al. 2024); AssistGUI (Gao et al. 2023)"
  complementary_to: "Set-of-Mark Prompting (Yang et al. 2023); CogAgent (Hong et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Embodied_Agents/NeurIPS_2024/2024_OSWorld_Benchmarking_Multimodal_Agents_for_Open_Ended_Tasks_in_Real_Computer_Environments.pdf
category: Survey_Benchmark
---

# OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2404.07972), [Project](https://os-world.github.io)
> - **Summary**: 本文提出首个面向真实电脑环境的开放式多模态代理基准，用虚拟机、中间态初始化和逐任务执行式评测，把“电脑助手到底会不会做事”从静态轨迹匹配变成了可复现的真实任务检验。
> - **Key Performance**: 人类完成率 72.36%，最佳基线仅 12.24%；多应用工作流任务成功率最高仍不到 8%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言任务指令 + 真实电脑观测（截图/a11y tree/终端） -> 鼠标键盘动作序列与任务终止判定
> - **bottleneck**: 现有基准缺少可交互、跨应用、允许多正确解且可自动判分的真实电脑评测环境
> - **mechanism_delta**: 把桌面代理评测从“静态下一步动作匹配”改成“虚拟机中的终态执行式验证”，并加入中间态初始化与跨应用任务
> - **evidence_signal**: 369任务+134评测函数的人类/多模型对比显示 72.36% vs 12.24% 的巨大能力差距
> - **reusable_ops**: [vm-snapshot-reset, example-wise-execution-eval]
> - **failure_modes**: [像素级点击不准导致GUI grounding失败, 窗口噪声与多应用切换下任务崩溃]
> - **open_questions**: [如何提升纯视觉高分辨率GUI理解, 如何为长程电脑任务构建记忆与反思机制]

## Part I：问题与挑战

**What/Why：真正的瓶颈不是“模型会不会预测下一步”，而是我们一直没有一个足够真实的电脑环境来测它。**

现有数字代理 benchmark 主要有两类缺陷：

1. **只有静态演示，没有可执行环境**  
   这类评测通常把任务写成轨迹预测或下一步动作预测，默认存在唯一标准解。  
   但真实电脑任务往往有**多条正确路径**，静态标注会误罚替代解，也无法支持探索、试错和交互学习。

2. **环境过窄，只覆盖单一应用或单一域**  
   许多已有环境局限于 web、mobile 或 coding。  
   这无法反映真实电脑使用中的核心复杂性：**跨应用工作流、GUI+CLI混合交互、中间工作状态、系统文件I/O、窗口噪声**。

OSWorld要解决的就是这个评测缺口：  
把“开放式电脑操作”变成一个可重复、可扩展、可自动判分的真实环境问题。

### 输入/输出接口

- **输入**：自然语言任务指令 + 真实电脑观测  
  - 全屏截图
  - accessibility tree
  - 终端输出等附加流
- **输出**：原始电脑动作  
  - 鼠标移动/点击/拖拽
  - 键盘输入/快捷键
  - `WAIT / FAIL / DONE`

这比 app-specific action schema 更接近真实人机交互，也更难。

### 边界条件

- 主基准主要构建在 **Ubuntu**，并提供 **43 个 Windows 分析任务**。
- 任务覆盖网页、桌面软件、文件系统、多应用工作流，以及一部分**不可完成任务**。
- 论文中的 baseline 实验使用 **15 step 上限**，因此结果反映的是“当前代理在有限交互预算下”的能力，而不是环境本身的上限。

## Part II：方法与洞察

### 评测系统怎么搭起来

OSWorld的关键不是再加一个数据集，而是把**环境、初始化、评测**三件事同时做实。

1. **环境层：真实 OS + 虚拟机**
   - 运行在 Ubuntu / Windows / macOS 虚拟机中
   - 支持 snapshot reset、并行运行、headless
   - 避免代理对宿主机造成不可逆破坏，也让任务可复现

2. **任务层：中间态初始化**
   - 每个任务都不只是一句 instruction
   - 还包含一个 **initial state setup config**
   - 可下载文件、打开软件、调整窗口、构造“人做了一半来求助”的中间状态  
   这比“从空白桌面开始”更接近真实使用场景，也显著增加探索难度。

3. **评测层：逐任务执行式判分**
   - 每个任务配一个定制 evaluator script
   - 从 VM / 云端 / 软件内部状态中提取目标文件、cookie、a11y tree、运行结果等
   - 用任务相关函数判定是否完成  
   例如：对比表格、检查邮件收件人、确认 cookie 被删除、抓取实时网页信息后再比对。

### 核心直觉

**核心变化**：从“动作像不像标注轨迹”改成“执行后电脑状态是否满足任务目标”。

这带来三个直接后果：

- **评测约束变了**  
  不再要求唯一解，允许不同但正确的操作路径。
- **信息瓶颈变了**  
  模型必须面对真实截图、高分辨率 GUI、窗口遮挡、跨应用依赖，而不是简化后的网页DOM或模拟状态。
- **诊断能力变了**  
  以前只能知道“轨迹偏了”；现在能更具体地知道：  
  是 **GUI grounding** 不行、**软件操作知识** 不够、**多应用协作** 崩了，还是 **长程记忆/鲁棒性** 不足。

为什么这套设计有效：

- **VM snapshot** 解决复现与安全问题；
- **中间态配置** 提高现实性，覆盖真实求助分布；
- **例级 evaluator** 把异构软件状态转成可判定的成功信号；
- **统一鼠标键盘动作空间** 避免 benchmark 被特定 app API 绑死。

### 战略权衡

| 设计选择 | 带来的能力 | 代价 |
|---|---|---|
| 真实虚拟机而非模拟网页/单应用环境 | 能测跨应用、GUI+CLI、开放域任务 | 环境搭建和维护复杂 |
| 例级执行式评测而非单一 gold trajectory | 接受多正确解，真正测功能完成 | 标注成本极高，脚本维护重 |
| 中间态初始化 | 更接近真实用户工作流 | 状态空间更大，探索更难 |
| 截图 + a11y tree 双观测 | 可分离视觉 grounding 与结构化感知问题 | a11y tree 很长且质量依赖软件实现 |
| 原始鼠标键盘动作 | 通用性最强 | 对像素定位和操作常识要求极高 |

## Part III：证据与局限

### 关键证据

**1. 对比信号：当前模型离“电脑助手”还很远**  
最强 baseline 也只有 **12.24%**，而人类是 **72.36%**。  
这不是小幅欠拟合，而是数量级差距，说明现有 LLM/VLM 在真实电脑操作上远未到可用阶段。

**2. 诊断信号：真正难的是 GUI 和工作流，不是纯语言理解**  
- CLI/OS 类任务相对更容易；
- GUI 密集的 Office 类任务更难；
- **多应用 workflow 最高仍不到 8%**。  
这表明短板主要不是高层计划本身，而是**视觉定位、界面操作常识、跨窗口协同**。

**3. 鲁棒性信号：输入分辨率、历史记忆、界面扰动都显著影响表现**  
- 更高分辨率通常更好；
- 更长的**文本历史**能帮助代理，说明它确实受益于过去状态；
- 但**图像历史**帮助有限，说明现有 VLM 还不擅长从图片轨迹中提炼稳定上下文；
- 窗口位置/尺寸变化、无关窗口遮挡会造成 **60%–80%** 的成功率下滑。  
所以 benchmark 不只是“更难”，而是能暴露此前 benchmark 看不到的真实失败模式。

**4. 难度校准信号：这个 benchmark 本身比已有 web benchmark 更接近真实使用**  
人类在 OSWorld 上的中位完成时间是 **111.94s**，而 WebArena 样本是 **35.38s**。  
这说明 OSWorld 不只是把网页任务搬到桌面，而是在任务结构上显著更复杂。

### 局限性

- **Fails when**: 需要外推到未覆盖的软件生态、复杂闭源商业软件、超长时序任务或细粒度安全副作用评估时，当前 benchmark 不能保证充分覆盖；对未良好支持 a11y 的应用，诊断能力也会下降。
- **Assumes**: 依赖虚拟机基础设施、逐任务手工初始化与定制 evaluator；构建成本高（文中约 1800 人时）；某些软件评测还依赖额外权限、调试接口或逆向辅助；主基准主要仍是 Ubuntu 开源软件栈。
- **Not designed for**: 全面的 agent safety 审计、真实宿主机不可逆操作评估、所有硬件/分辨率/商业软件组合上的完备覆盖。

### 可复用组件

- **虚拟机任务初始化管线**：适合任何 computer-use agent 训练/评测
- **例级 execution-based evaluator 库**：适合把开放式任务变成程序判分
- **统一观测/动作接口**：截图、a11y tree、pyautogui 动作
- **跨应用与不可行任务设计范式**：适合做更强的 agent diagnosis

**So what：**  
这篇论文最重要的“能力跃迁”不在于提出了更强代理，而在于首次把**真实电脑环境中的开放式代理评测**做成了统一基础设施。它让社区终于能系统地回答：当前模型到底卡在规划、感知、执行，还是鲁棒性。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Embodied_Agents/NeurIPS_2024/2024_OSWorld_Benchmarking_Multimodal_Agents_for_Open_Ended_Tasks_in_Real_Computer_Environments.pdf]]