---
title: "WorldScore: A Unified Evaluation Benchmark for World Generation"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/world-generation
  - task/video-generation
  - next-scene-decomposition
  - trajectory-conditioned-evaluation
  - metric-aggregation
  - dataset/WorldScore
  - opensource/full
core_operator: "把世界生成拆成带相机轨迹约束的逐步 next-scene 任务，并在统一视频输出上聚合可控性、质量、动态性三类指标评分"
primary_logic: |
  世界生成统一评测需求 → 将任务拆成(当前场景, 下一场景, 相机布局)的 next-scene 样本并构建3000个静态/动态/多风格测试例 → 在统一视频输出上计算可控性、质量、动态性共10个指标并归一聚合 → 揭示3D、4D、I2V、T2V方法在相机控制、长序列与动态建模上的能力边界
claims:
  - "相较于VBench、EvalCrafter、ChronoMagic-Bench和WorldModelBench等已有基准，WorldScore是文中唯一同时覆盖多场景、统一3D/4D/I2V/T2V评测、长序列、图像条件、多风格、相机控制和3D一致性的基准，并包含3000个测试样例 [evidence: analysis]"
  - "在WorldScore-Static上，3D场景生成方法整体强于视频生成方法：WonderWorld得分72.69，而最佳视频模型CogVideoX-I2V为62.15 [evidence: analysis]"
  - "视频模型中更大的运动幅度并不稳定对应更高的运动放置准确率，且运动幅度与运动平滑性存在明显权衡 [evidence: analysis]"
related_work_position:
  extends: "VBench (Huang et al. 2024)"
  competes_with: "VBench (Huang et al. 2024); WorldModelBench (Li et al. 2025)"
  complementary_to: "CameraCtrl (He et al. 2024); MotionCtrl (Wang et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Evaluating_World_Models/arXiv_2025/2025_WorldScore_A_Unified_Evaluation_Benchmark_for_World_Generation.pdf
category: Survey_Benchmark
---

# WorldScore: A Unified Evaluation Benchmark for World Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.00983), [Project](https://haoyi-duan.github.io/WorldScore/)
> - **Summary**: 该工作把“世界生成”改写为可逐步评测的 next-scene 任务，并用统一视频输出上的 10 个指标，首次把 3D、4D、I2V、T2V 方法放到同一基准下比较。
> - **Key Performance**: WonderWorld 在 **WorldScore-Static = 72.69** 居首；最佳视频模型 CogVideoX-I2V 达到 **WorldScore-Static / Dynamic = 62.15 / 59.12**。

> [!info] **Agent Summary**
> - **task_path**: 当前场景图像/文本 + 下一场景描述 + 相机布局规格 -> 统一视频输出上的世界生成能力评分
> - **bottleneck**: 现有基准主要评单场景视频质量，缺少多场景布局控制与跨3D/4D/I2V/T2V的统一接口，导致“世界生成能力”无法被公平测量
> - **mechanism_delta**: 用带相机轨迹的 next-scene 分解替代整段世界的黑箱打分，并把所有方法投影到同一视频观测空间上做10维评测
> - **evidence_signal**: 3000样例、20个模型、400人偏好研究共同表明该协议既能拉开模型差距，也与人类感知较一致
> - **reusable_ops**: [next-scene任务分解, 统一视频输出评测]
> - **failure_modes**: [视频模型相机控制弱, 长序列与户外场景显著退化]
> - **open_questions**: [如何评测交互式闭环world model, 如何统一评测同时含大相机运动与复杂物体运动的长时程世界]

## Part I：问题与挑战

这篇论文的核心不是再提一个生成模型，而是解决**“世界生成到底该怎么公平评”**的问题。

### 真正的问题是什么
已有视频生成基准大多只看**单段、单场景**视频质量，典型如 VBench：它能看清晰度、审美、局部时序，但很难回答下面这些对“世界生成”更关键的问题：

1. 模型是否真的从当前场景走到了**新场景**，而不是原地抖动或视角微调？
2. 模型是否遵守了**明确的相机轨迹/布局控制**？
3. 3D/4D 方法和 I2V/T2V 方法输入接口不同，如何**统一比较**？
4. 动态世界里，运动是否发生在**该动的区域**，而不是靠错误相机运动或无关物体运动“糊弄过去”？

### 真瓶颈在哪里
真瓶颈是**评测接口不统一**，不是单一指标不够多。  
3D/4D 方法往往需要相机位姿，I2V 需要参考图，T2V 主要吃文本；如果 benchmark 只提供 prompt 或只看最终短视频，就会天然偏向某一类方法。

### 输入/输出边界
WorldScore 把一次世界生成写成一步 step 的 world specification：
- **当前场景**：图像 + 文本
- **下一场景**：文本描述
- **布局**：相机轨迹矩阵 + 相机运动文本

所有模型最后都被要求输出**视频**，这样比较的是“观测到的世界行为”，而不是各自内部表示。

此外，论文把任务拆成两类：
- **静态世界生成**：重点测新场景切换、相机控制、几何/纹理一致性
- **动态世界生成**：固定相机，只看场景内运动，避免把相机运动误判为物体运动

这回答了第一个问题：**现在为什么必须做这件事？**  
因为视频、3D、4D 方法都已经具备“像在生成世界”的能力，但评测仍停留在“单视频质量分”。

## Part II：方法与洞察

### 统一任务协议
论文的关键设计是把 world generation 拆成**一串 next-scene generation**。

每一步都由 `(当前场景, 下一场景, 布局)` 定义：
- 当前场景同时给**图像和文本**
- 布局同时给**相机矩阵和自然语言相机指令**
- 预处理模块 `wproc` 再按模型类型适配输入  
  - 3D/4D：吃相机位姿
  - I2V：吃参考图
  - T2V：忽略图像，只用文本与相机描述

这样一来，benchmark 不再被某一模型范式“绑架”。

### 数据与覆盖面
数据集共 **3000** 个测试样例：
- **2000 静态**
- **1000 动态**

覆盖：
- 室内/室外
- 写实/风格化
- 10 类静态场景
- 5 类动态类型
- 8 类相机运动
- 7 种风格化样式
- 含短序列与较长序列（large world）

这使 benchmark 从“单视频打分”升级为“多子域能力诊断”。

### 评分协议
WorldScore 分成三大维度、10 个指标：

- **Controllability**
  - 相机控制
  - 物体控制
  - 内容对齐

- **Quality**
  - 3D 一致性
  - Photometric 一致性
  - 风格一致性
  - 主观质量

- **Dynamics**
  - 运动放置准确率
  - 运动幅度
  - 运动平滑性

最终聚合成：
- **WorldScore-Static**
- **WorldScore-Dynamic**

### 核心直觉

**它真正改变的不是模型，而是测量瓶颈。**

- **以前**：把“世界生成”当成一个整体黑箱，只能看单段视频好不好看
- **现在**：把它拆成可监督的逐步迁移过程，每一步都有明确的语义目标和布局目标

于是发生了三个因果变化：

1. **从架构不兼容，变成观测可对齐**  
   不同范式虽然内部表示不同，但都可以被压到统一的视频输出空间。

2. **从整体感受，变成局部可诊断**  
   你可以单独问：没生成新场景？没跟相机轨迹？纹理闪烁？还是运动放错地方？

3. **从“视频好看”转向“世界可控”**  
   这也是 Figure 1 的核心：两个在 VBench 上相近的模型，在 WorldScore 上会因为“没走到新场景/没遵循相机运动”被明显区分。

| 设计选择 | 获得的能力 | 代价/权衡 |
|---|---|---|
| next-scene 分解 | 把长程世界生成拆成可测局部步骤 | 不能完整覆盖开放式、分支式交互世界 |
| 相机矩阵 + 文本双布局 | 同时兼容 pose-aware 与 text-only 模型 | 对不支持显式位姿的视频模型，控制信号仍偏弱 |
| 统一视频输出 | 可跨 3D/4D/I2V/T2V 直接比较 | 原生3D可编辑性与内部表示优势被折叠掉 |
| 静态/动态分拆 | 把相机运动与物体运动分开诊断 | 不擅长评估两者强耦合的复杂场景 |
| 经验边界归一化聚合 | 10 个异构指标可合成总分 | 总分对经验上界/下界和辅助模型有依赖 |

## Part III：证据与局限

### 关键证据信号

**信号 1：指标有效性不是拍脑袋。**  
作者做了 **400 人** 的 2AFC 人类偏好实验来校准主观质量指标，并验证其他指标与人类偏好的一致性；同时还检验了不同分辨率/纵横比下分数的稳定性。对 benchmark 论文来说，这是最关键的可信度来源。

**信号 2：WorldScore 确实揭示了 prior benchmark 看不清的结构差异。**  
在 **WorldScore-Static** 上，3D 模型明显领先视频模型：
- **WonderWorld：72.69**
- **LucidDreamer：70.40**
- **最佳视频模型 CogVideoX-I2V：62.15**

这说明当前 world generation 的静态能力上限，仍主要来自带显式几何/相机控制的 3D 方法。

**信号 3：视频模型的主瓶颈不是“画不好”，而是“控不住”。**  
最佳视频模型的相机控制分数仍显著落后于 3D/4D 方法；论文明确指出，视频模型尤其在：
- 相机 controllability
- 长序列生成
- 户外场景
上表现较弱。

**信号 4：动态世界的难点不是让画面动，而是让该动的地方动。**  
论文发现运动幅度与运动准确率相关性弱，而且运动幅度与平滑性存在 trade-off。也就是说，“动得大”不等于“动得对”。

### 局限性
- **Fails when**: 需要评测开放式、交互式、超长时程 world model，或相机大位移与复杂物体运动同时发生的场景时，当前逐步分解与静/动态拆分不够完整；另外 large world 也只是有限长度，并非真正长期记忆测试。
- **Assumes**: 所有方法都能被转换成统一视频输出；数据构建依赖 GPT-4o 与商业风格化 API，评分依赖 DROID-SLAM、SEA-RAFT、SAM2、插帧模型等辅助系统，这些外部组件的误差会传递到最终分数；对不支持动态任务的 3D 模型，WorldScore-Dynamic 直接记 0。
- **Not designed for**: 原生3D可编辑性/可交互性评测、严格物理规律一致性、训练数据污染检测、在线 agent-loop 仿真或游戏式闭环评测。

### Reusable components
这篇工作最值得复用的不是某个单项分数，而是整套评测操作符：
- **world specification**：`当前场景 + 下一场景 + 布局`
- **统一输出协议**：不同范式统一落到视频观测空间
- **静/动态解耦评测**
- **基于 SLAM 的相机/几何一致性度量**
- **基于 mask + 光流 的运动放置准确率**
- **多维指标聚合而非单一美学分**

**So what：能力跳变在哪里？**  
WorldScore 的价值在于，它把“世界生成”从一个模糊口号变成了一个可分解、可诊断、可跨范式比较的任务。与 prior work 相比，它最重要的跃迁不是更大数据，而是**终于能区分‘视频好看’和‘世界真的生成出来了’**。

![[paperPDFs/Evaluating_World_Models/arXiv_2025/2025_WorldScore_A_Unified_Evaluation_Benchmark_for_World_Generation.pdf]]