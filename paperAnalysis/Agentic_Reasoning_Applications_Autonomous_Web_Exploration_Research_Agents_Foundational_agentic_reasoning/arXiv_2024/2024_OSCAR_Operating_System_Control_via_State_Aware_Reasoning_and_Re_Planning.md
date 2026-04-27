---
title: "OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning"
venue: arXiv
year: 2024
tags:
  - Others
  - task/os-control
  - task/gui-automation
  - state-machine
  - visual-grounding
  - re-planning
  - dataset/GAIA
  - dataset/OSWorld
  - dataset/AndroidWorld
  - opensource/promised
core_operator: 将操作系统控制建模为带执行/验证反馈的状态机，并用 A11Y+SoM 双重 GUI 接地支持任务级重规划。
primary_logic: |
  用户指令 + 屏幕截图/A11Y树 → 构造视觉与语义双重 GUI 接地表示并分解任务 → 为当前子任务生成可执行 Python 鼠标键盘/触屏代码 → 根据执行错误与任务验证结果做局部重规划 → 完成跨应用 OS 控制
claims:
  - "OSCAR 在 GAIA 上取得 28.7% 的平均成功率，且在最难的 Level 3 上达到 13.5%，高于 MMAC 的 24.0% 和 6.1% [evidence: comparison]"
  - "OSCAR 在 OSWorld 上达到 24.5% 平均成功率，并在 AndroidWorld 上达到 61.6%，分别高于 FRIDAY 的 18.8% 和 AppAgent 的 59.9% [evidence: comparison]"
  - "去除 GUI-grounding 或任务级重规划后，OSCAR 在 OSWorld 上的表现分别降至完整系统约 70% 和 80%，说明两者都是关键组件 [evidence: ablation]"
related_work_position:
  extends: "Plan-and-Solve prompting (Wang et al. 2023)"
  competes_with: "FRIDAY (Wu et al. 2024c); UFO (Zhang et al. 2024a)"
  complementary_to: "SeeClick (Cheng et al. 2024); GUICourse (Chen et al. 2024b)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2024/2024_OSCAR_Operating_System_Control_via_State_Aware_Reasoning_and_Re_Planning.pdf
category: Others
---

# OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.18963)
> - **Summary**: 该工作把跨桌面/手机应用的 OS 控制从“线性动作生成”改成“带执行与验证反馈的状态机闭环”，并通过 A11Y+SoM 的双重 GUI 接地与任务级重规划，提升了动态环境中的鲁棒性与泛化性。
> - **Key Performance**: GAIA 平均 28.7%（Level 3 为 13.5%）；OSWorld 24.5%；AndroidWorld 61.6%

> [!info] **Agent Summary**
> - **task_path**: 用户自然语言指令 + 当前 GUI 截图/A11Y 树 -> 鼠标/键盘/触屏代码执行 -> 跨应用任务完成
> - **bottleneck**: 动态 OS 中不存在唯一正确轨迹，纯线性执行或整链重规划无法高效利用实时反馈，且易发生错误传播
> - **mechanism_delta**: 把 OS 控制改为状态机闭环，并在任务层而非整条动作链上做局部重规划，同时用 A11Y+SoM 把 GUI 变成可引用的语义状态
> - **evidence_signal**: 三个动态基准上的对比领先，外加 OSWorld 上的消融与重规划效率分析
> - **reusable_ops**: [A11Y树转 SoM 视觉提示, 基于验证反馈的任务级局部重规划]
> - **failure_modes**: [超长多应用任务触发步数/尝试上限, A11Y 缺失或高密度专业软件界面导致接地不稳]
> - **open_questions**: [离开 GPT-4o/GPT-4-turbo 后是否仍能保持稳定性, 没有专门验证脚本时如何自动判断任务是否完成]

## Part I：问题与挑战

这篇论文解决的不是“模型会不会点按钮”这么浅的问题，而是：**在真实、动态、跨应用的操作系统里，如何让 agent 像人一样根据当前屏幕状态持续修正计划，而不是死守一条固定动作轨迹。**

### 1. 任务定义
输入接口是：
- 用户自然语言指令
- 当前屏幕截图
- 来自操作系统的可访问性树（A11Y tree）与窗口信息
- 历史任务/动作记忆
- 执行或验证阶段返回的反馈

输出接口是：
- 可执行的 Python 控制代码
- 底层动作是标准 OS 控件：鼠标、键盘、触屏，而不是应用私有 API

这意味着 OSCAR 面向的是**通用 OS 控制**，而不是只针对浏览器、办公软件或某个沙盒环境定制的 agent。

### 2. 真正瓶颈是什么
作者识别了三个瓶颈，但核心可归结为两个更本质的问题：

1. **感知瓶颈：GUI 不是自然图像。**  
   纯截图里文本密、控件小、图标语义弱，直接让多模态模型从像素里找可交互元素，容易漏检和误解。

2. **闭环瓶颈：动态环境没有唯一正确轨迹。**  
   同一个任务往往有多条合法路径。过去很多方法默认固定序列，或者失败后从头重规划，导致：
   - 一旦偏离参考轨迹就失败
   - 早期错误会传染到后续步骤
   - 重规划粒度太粗，成本高且反复绕圈

所以，这篇论文真正处理的是：**如何把 GUI 状态变成可推理、可引用、可修正的中间状态，并据此做低成本的局部恢复。**

### 3. 为什么现在做有意义
因为两件事同时成熟了：
- **LMM/LLM 已具备基础多步推理与工具使用能力**
- **OSWorld / AndroidWorld 这类动态可执行环境出现了**，允许 agent 真正接收实时反馈，而不只是离线模仿静态轨迹

因此，问题从“能不能做演示式 GUI 模仿”升级成“能不能做真实 OS 闭环控制”。

### 4. 边界条件
OSCAR 的设定并不是无条件通用，它依赖：
- 可访问原生 A11Y tree / window API
- 可执行 Python 控制环境
- 可获取任务完成验证信号（benchmark 中由脚本完成）
- 实验里默认最多 4 次尝试

换言之，它针对的是**有可观测 GUI 状态、可注入标准输入、且能定义成功判据的 OS 自动化场景**。

## Part II：方法与洞察

OSCAR 的方法可理解为四层拼接：**状态机闭环 + 双重 GUI 接地 + 任务级重规划 + 代码式动作执行**。

### 1. 状态机闭环
论文把 agent 组织成显式状态机：
- Init
- Observe
- Plan
- Execute
- Verify
- Success / Fail / Error / Reset

关键不是“有很多状态”本身，而是它把 OS 控制流程明确拆成：
**观察 → 规划 → 执行 → 再观察 → 验证 → 必要时重规划**

这让错误处理不再是 prompt 里的一句“如果失败请重试”，而是系统层面的控制逻辑。

### 2. 双重 GUI 接地
作者认为“只看截图”不足以做稳定 GUI 控制，于是加入两种接地信息：

- **视觉接地**：基于 A11Y tree 提取元素坐标，生成 SoM（Set-of-Mark）标注框
- **语义接地**：为元素补充 ID、label、坐标等文本描述

这样模型不是面对一张高熵截图，而是面对：
- 原图上下文
- 明确被标出的可交互区域
- 可被代码引用的语义元素 ID

这一步直接把“看到按钮”变成“知道按钮是谁、在哪里、能否被引用”。

### 3. 任务驱动重规划
这是论文最关键的机制改动。

OSCAR 不是每一步都从头想完整路线，而是做两级规划：
- **Level 1**：把用户目标拆成 SOP 风格的任务列表
- **Level 2**：只围绕当前任务逐步生成动作

当验证失败时，它不是：
- 只修当前一步，也不是
- 整条工作流全部重来

而是**只重规划出问题的任务段**。这带来两个直接收益：
- 缩小搜索空间，提高效率
- 减少“前面已做对的都被重置”的错误传播

### 4. 代码中心动作层
OSCAR 不直接输出模糊自然语言动作，而是输出可执行 Python 代码，通过 PyAutoGUI 等接口操作鼠标键盘/触屏。

优点在于：
- 动作语义清晰
- 可以直接引用 GUI 元素 ID 或坐标
- 跨桌面/手机有统一控制抽象

这一步把“语言计划”真正落成“可执行控制”。

### 核心直觉

OSCAR 的核心不是单纯“规划更强”，而是**改变了 agent 看到环境、修复错误、执行动作的三个因果接口**：

1. **从原始像素到语义状态**  
   改变前：模型面对密集 GUI 像素，元素定位困难。  
   改变后：A11Y+SoM 把屏幕变成“可引用元素集合”。  
   结果：降低了 GUI 感知的不确定性，提升定位和引用精度。

2. **从整链规划到任务级局部修复**  
   改变前：失败后整条路线重来，成功前缀被浪费。  
   改变后：验证反馈只触发相关任务段的重规划。  
   结果：把恢复粒度从“全局搜索”缩成“局部搜索”，减少冗余步骤。

3. **从描述动作到执行代码**  
   改变前：动作文本到执行器之间仍有语义缝隙。  
   改变后：模型直接生成可运行控制代码。  
   结果：动作表达更精确，更适合真实 OS 操作。

所以它带来的能力变化是：**不是单步点击更聪明，而是长流程、多应用、动态反馈下更不容易崩。**

### 战略权衡

| 设计选择 | 带来的能力提升 | 代价 / 风险 |
|---|---|---|
| A11Y + SoM 双重接地 | 元素更易定位，适合高密度 GUI，代码可直接引用元素 | 依赖应用暴露 A11Y 信息；自定义控件可能缺失或语义差 |
| 任务级重规划 | 失败后只修局部，减少冗余探索与错误传播 | 任务分解若本身错误，局部修复可能被错误高层结构束缚 |
| 代码式动作输出 | 控制语义清晰、可执行、跨平台统一 | 会引入语法/运行时错误，需要解释器反馈兜底 |
| 状态机 + 验证环 | 能处理 invalid action、false completion、系统异常 | 交互回合更多，延迟更高，且依赖验证脚本或外部判定 |

## Part III：证据与局限

### 关键证据信号

1. **跨基准比较信号：它不是只在单一桌面任务上有效，而是桌面/手机都涨。**  
   - 在 **GAIA** 上平均成功率 28.7%，最难的 Level 3 达到 13.5%，相比 MMAC 的 6.1% 接近翻倍。  
   - 在 **OSWorld** 上达到 24.5%，超过 FRIDAY 的 18.8%。  
   - 在 **AndroidWorld** 上达到 61.6%，高于 AppAgent 的 59.9%，尤其在 medium / hard 子集更强。  

   这说明它的提升主要出现在**长流程和动态反馈更强的场景**，而不是只在简单任务上刷分。

2. **消融信号：性能提升确实来自“接地 + 重规划”，不是 prompt 偶然性。**  
   - 去掉 GUI-grounding，只喂原始截图，性能只剩完整系统约 70%。  
   - 去掉任务级重规划，改成直接 prompt，性能约剩 80%。  
   - Detection+OCR 也弱于 A11Y tree，尤其在专业软件这类元素密集界面上更明显。  

   这直接支持作者的因果主张：**观测表示和重规划粒度都是关键旋钮。**

3. **过程分析信号：OSCAR 的优势在“恢复效率”，不是只在终局结果。**  
   - 成功样本里，超过 80% 的 OSCAR 案例少于 3 次重规划；FRIDAY 超过一半需要 3–4 次。  
   - 失败类型上，OSCAR 基本消除了 false completion 和 invalid action，剩余主要是 step limit 耗尽。  
   - 冗余重规划比例只有 15.2%，明显低于 FRIDAY 的 52.8%。  

   这很重要：它证明 OSCAR 的提升并非随机命中，而是**更少兜圈子、更少重复犯错**。

4. **补充证据：附录中的 GUI-World / Mind2Web / AITW 结果说明其观测与动作表示本身也有迁移性。**  
   即使不强调动态重规划，OSCAR 的 GUI 接地与动作建模也已经具备较强通用性。

### 能力跃迁到底在哪里
这篇论文最值得记住的“能力跳变”不是绝对成功率数字本身，而是：

- 从**静态模仿式 GUI agent**  
  变成  
- **能利用实时执行/验证反馈做局部修复的 OS agent**

也就是说，它把 prior work 常见的“会走，但容易走偏”变成了“走偏后能自己纠回来一部分”。

### 局限性

- **Fails when**:  
  - 任务跨度很长、涉及多应用切换且步骤依赖强时，仍会触发 step limit / attempt limit。  
  - A11Y 信息缺失、控件密集或应用使用大量自定义渲染组件时，元素接地会变差。  

- **Assumes**:  
  - 依赖强基座模型 API（实验中主要是 GPT-4o / GPT-4-turbo），这对复现和成本都有影响。  
  - 依赖原生 A11Y tree / window API，以及可执行 Python 控制环境。  
  - 依赖任务验证脚本来判定成功；真实开放世界桌面任务未必总有这么明确的 verifier。  
  - 代码尚未发布，仅承诺 upon publication 开源；实验虽主要是 API 调用，但仍使用了 2×A100 做完整评测流程。  

- **Not designed for**:  
  - 完全无可访问性元数据的纯像素远程桌面/游戏式界面  
  - 需要高权限、强安全约束或不可逆系统操作的场景  
  - 没有明确成功判据、只能靠主观判断完成度的开放式任务  

### 可复用组件
这篇论文里最值得复用的不是整套系统，而是几个通用操作件：

- **A11Y tree → SoM + semantic labels**：把 GUI 从像素空间转成可引用状态
- **Observe-Plan-Execute-Verify 状态机**：给 agent 一个显式闭环
- **任务级局部重规划**：比“整条链重来”更省、更稳
- **代码式动作层**：把自然语言计划落成真实可执行控制

一句话总结：**OSCAR 的价值在于把“GUI agent”从单步感知问题，推进成了“动态系统控制问题”。**

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2024/2024_OSCAR_Operating_System_Control_via_State_Aware_Reasoning_and_Re_Planning.pdf]]