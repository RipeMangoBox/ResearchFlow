---
title: "Adapting Vision-Language Models for Evaluating World Models"
venue: arXiv
year: 2025
tags:
  - Evaluation
  - task/video-understanding
  - task/visual-question-answering
  - partial-finetuning
  - frame-sampling
  - multitask-learning
  - dataset/Skygarden
  - opensource/no
core_operator: "把世界模型rollout评测改写成时序视频问答，并用仅调投影头的VLM在稀疏多帧输入上判断动作对齐与角色一致性。"
primary_logic: |
  world model rollout视频片段 + 结构化问题 →
  对14帧片段做均匀多帧采样，并以AR/CR和多种问答格式的混合监督适配PaliGemma →
  输出动作对齐与角色一致性的文本答案，再用EM/ROUGE汇总为语义评测结果
claims:
  - "UNIVERSE只更新2.66M参数的投影头（约0.07%模型参数），但在AR/CR × {binary, multiple-choice, open-ended} 的统一评测上达到与任务专用检查点相当的整体表现 [evidence: comparison]"
  - "在Action Recognition中，仅用2帧输入时，uniform sampling将EM从84.42%提升到90.47%（binary）、65.53%提升到83.93%（multiple-choice）、65.38%提升到82.68%（open-ended） [evidence: ablation]"
  - "在人类标注的8个rollout设置中，UNIVERSE在setting 1–7的总体graded accuracy为79.72%–91.11%，但在低分辨率失配的setting 8降至54.46% [evidence: case-study]"
related_work_position:
  extends: "Prometheus-Vision (Lee et al. 2024)"
  competes_with: "Cosmos (Agarwal et al. 2025); VBench (Huang et al. 2024)"
  complementary_to: "FVD (Unterthiner et al. 2018); CLIPScore (Hessel et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Evaluating_World_Models/arXiv_2025/2025_Adapting_Vision_Language_Models_for_Evaluating_World_Models.pdf
category: Evaluation
---

# Adapting Vision-Language Models for Evaluating World Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.17967)
> - **Summary**: 论文把世界模型rollout评测从低层分布相似度改写为“动作是否按时间发生、角色身份是否持续一致”的视频问答，并提出一个只需极轻量适配的统一VLM评测器 UNIVERSE。
> - **Key Performance**: 1 epoch时，UNIVERSE在AR/CR上的ROUGE分别达 **85.8 / 84.1**，显著高于最佳零样本PaliGemma的 **29.7 / 17.2**；人评setting 1–7总体graded accuracy为 **79.72%–91.11%**。

> [!info] **Agent Summary**
> - **task_path**: world-model rollout短视频片段 + 自然语言问题 -> 动作/角色语义判断文本
> - **bottleneck**: 现有FVD/CLIP/VBench类指标缺少时间对齐与实体一致性诊断能力，而零样本VLM又不具备该交互环境的领域语义
> - **mechanism_delta**: 将评测重写为AR/CR×三种问答格式的视频VQA，并用“投影头微调 + 8帧均匀采样 + AR/OE偏置混合监督”适配单一VLM
> - **evidence_signal**: 零样本VLM明显失效，而轻量适配后AR超过所有基线、CR接近更重的专用模型，并在跨环境人评中保持稳定
> - **reusable_ops**: [uniform temporal sampling, projector-head-only tuning]
> - **failure_modes**: [resolution mismatch, long-horizon temporal grounding failure]
> - **open_questions**: [how to extend beyond 14-frame clips, how much labeled adaptation data is needed in a new domain]

## Part I：问题与挑战

这篇论文真正要解决的，不是“视频看起来像不像真的”，而是：

1. **动作是否在正确时间产生了正确后果**  
   世界模型rollout是 action-conditioned 的，评测必须能回答“给了这个动作后，这段视频里是否真的发生了对应行为”。

2. **角色/实体是否在时间上保持一致**  
   同一个角色是否跨帧保持身份、外观和语义一致，是 rollout 语义正确性的另一条主轴。

### 真正瓶颈是什么？

作者认为，世界模型评测的核心瓶颈是 **时序语义对齐**，不是像素保真度。

- **FID / FVD / KID**：更偏分布距离，无法回答“某个时刻动作是否被正确执行”
- **CLIPScore / frame-level multimodal metrics**：更像帧级语义相似，缺少时间因果约束
- **T2V benchmarks（如 VBench）**：面向开放式文本生成，不是面向 action-conditioned rollout
- **人类评测**：最可靠，但贵、慢、难扩展

### 为什么现在值得做？

因为世界模型已经开始进入更实用的场景：游戏、具身智能、自动驾驶。  
这些系统越来越强调“可控 rollout”，但评测工具还停留在“看起来像不像视频”的阶段，导致研究进展和部署验证都被卡住。

### 输入 / 输出接口与边界条件

**输入**：
- 一个短 rollout 片段（论文中是 14-frame clip）
- 一个自然语言问题

**输出**：
- 文本答案
- 通过 EM / ROUGE 聚合成评测分数

**评测维度**：
- Action Recognition (AR)
- Character Recognition (CR)

**问题形式**：
- Binary
- Multiple-choice
- Open-ended

**边界条件**：
- 主要在模拟环境中验证
- 数据来自 Bleeding Edge / Skygarden 域
- 当前只覆盖 AR/CR 两类基础语义任务
- 主要关注短时片段，不是长时规划一致性

---

## Part II：方法与洞察

论文有两个层面的贡献：

1. **协议层**：把 world model 评测定义成结构化视频问答任务  
2. **模型层**：提出 UNIVERSE，用轻量适配把通用VLM改造成 rollout evaluator

### 1) 评测协议：把“好不好”改成“能不能识别”

作者将 rollout 评测拆成两类识别任务：

- **AR**：视频是否体现了给定动作的效果
- **CR**：角色身份/外观是否跨时间保持一致

每个clip生成 6 个 QA 对：
- AR / CR 各一组
- 每组包含 binary、multiple-choice、open-ended 三种形式

这个设计的价值在于：  
它把原本模糊的“生成质量”变成了**可验证、可监督、可比较**的任务形式。

### 2) UNIVERSE：统一评测器而非一堆任务专用模型

UNIVERSE 基于 **PaliGemma 2 3B**，但最终不是全量微调，而是一个很克制的 recipe：

- **只调 projection head**
  - 只训练 2.66M 参数
  - 约占全模型 **0.07%**
- **输入 8 帧**
  - 从14帧clip中做 **uniform sampling**
- **混合监督**
  - 任务比重：AR 0.8, CR 0.2
  - 格式比重：OE 0.8, binary 0.15, MC 0.05

作者的发现是：  
在算力和标注都受限的情况下，这个组合比“更重的适配”更划算。

### 核心直觉

#### 直觉一：把评测目标从“相似度”换成“受控识别”
**改变了什么**：  
从无条件的视觉分布比较，改成带问题约束的时序识别任务。

**改变了哪个瓶颈**：  
原来的瓶颈是“指标不知该看什么”；现在问题本身就指定了要检查的语义维度。

**带来的能力变化**：  
评测器开始能明确地区分：
- 动作有没有被执行
- 角色有没有漂移
- 不同问答形式下是否真的理解，而不是只会匹配模板

#### 直觉二：在固定token预算下，优先扩大时间覆盖
**改变了什么**：  
不是盯着前几帧，而是从全clip均匀抽样。

**改变了哪个瓶颈**：  
AR的难点不是“看清一帧”，而是“看到动作后果在时间上的展开”。

**带来的能力变化**：  
模型更容易捕捉跨时间的动作因果关系，因此 AR 提升明显，尤其在低帧数输入时。

#### 直觉三：不是一定要重调 backbone，先把视觉-语言接口对齐
**改变了什么**：  
只微调多模态 projection head，而尽量保留预训练视觉/语言能力。

**改变了哪个瓶颈**：  
问题不一定出在模型不会看、不会说，而是**不会把该域的视觉证据映射到正确语言语义空间**。

**带来的能力变化**：  
在有限数据和算力下，能快速适配到 rollout 评测域，并保留统一模型跨任务泛化能力。

### 战略取舍

| 设计选择 | 主要解决的瓶颈 | 带来的收益 | 代价/牺牲 |
|---|---|---|---|
| AR/CR + 多格式VQA协议 | 评测目标过于模糊 | 可诊断具体语义维度 | 覆盖面仍局限于预定义任务 |
| 8帧 uniform sampling | token预算不足导致时间覆盖差 | 更好捕捉动作后果 | 连续细粒度运动仍可能丢失 |
| 只调 projection head | 全量微调太重、样本效率低 | 0.07%参数即可适配 | 对更强视觉域移位未必最优 |
| AR-heavy + OE-heavy 混合监督 | 统一模型容易偏向简单任务/格式 | 提升 harder task 与真实生成能力 | binary/MC 可能不是绝对最优 |
| 单一统一评测器 | 多任务、多格式维护成本高 | 一个模型覆盖多设置 | 某些子任务不如专用模型极致 |

---

## Part III：证据与局限

### 关键信号 1：零样本VLM不够，适配是必要条件

最直接的比较信号是：

- 零样本 VideoLLaMA3 / PaliGemma 在 AR、CR 上都很差
- 1 epoch 后的 UNIVERSE：
  - **AR ROUGE = 85.8**
  - **CR ROUGE = 84.1**
- 而最佳零样本 PaliGemma 只有：
  - **AR = 29.7**
  - **CR = 17.2**

这说明问题不在于“VLM规模不够大”，而在于：
**不做领域适配，通用VLM并不具备世界模型rollout所需的时间落点与环境语义。**

### 关键信号 2：AR 的真瓶颈是时间证据，不是更多监督本身

作者发现 CR 很快就收敛：
- 只用约 **12.5% epoch（约4K样本）**，EM 就能超过 **97%**

但 AR 不一样：
- 如果只有单帧输入，即使继续训练，提升也很有限
- 当 **更多帧 + 更多训练** 同时增加时，AR 才持续上升

这非常关键。它说明：
- **CR 更像身份匹配问题**
- **AR 更像时间因果理解问题**

所以这篇论文真正诊断出的难点，是 **动作语义需要时间跨度**。

### 关键信号 3：uniform sampling 不是小技巧，而是关键因果旋钮

在 2 帧输入时，uniform sampling 对 AR 的提升非常明显：

- Binary：84.42% → 90.47%
- MC：65.53% → 83.93%
- OE：65.38% → 82.68%

这说明：
- 如果只看前几帧，模型常常看不到动作效果
- 稀疏但覆盖全程的证据，比密集但局部的证据更有价值

### 关键信号 4：数据混合比例会直接改变统一评测器的能力边界

作者做了分层 ablation：

- 提高 **AR 比重**，AR 表现持续上升，CR 基本稳定
- 提高 **open-ended 比重**，AR-OE 提升明显
- 最终选出：
  - AR 0.8 / CR 0.2
  - OE 0.8 / binary 0.15 / MC 0.05

这背后的含义是：
统一评测器不是“把所有数据混在一起训练”就行，  
而是必须按任务难度和捷径风险做 **监督配方设计**。

### 关键信号 5：与人类判断总体对齐，但有明显失配边界

在人类研究中，作者评估了 8 个设置：
- 涵盖不同 world model 尺度
- 不同 rollout fidelity
- 多个未见环境

结果：
- **setting 1–7 总体 graded accuracy 为 79.72%–91.11%**
- **setting 8 下降到 54.46%**
- 人类标注一致性 Cohen’s κ 为 **0.59–0.91**

最重要的解释是：
setting 8 来自较低分辨率模型，说明 **分辨率/域失配** 会显著影响评测器可靠性。

### 局限性

- **Fails when**: rollout分辨率与适配域明显失配、视频过长导致关键证据被稀疏采样漏掉、或样本本身语义含糊时，模型判断会明显退化。
- **Assumes**: 需要短时clip级监督、动作/角色标签或可由元数据生成的QA模板、PaliGemma式开放权重骨干；此外，虽然最终recipe轻量，但整套研究搜索成本高达约 **5,153 GPU-days**，且依赖游戏日志/元数据。
- **Not designed for**: 真实世界开放域视频、长时规划一致性、反事实因果验证、以及超出AR/CR的高层推理评测。

### 可复用组件

这篇论文最值得复用的，不只是一个模型，而是一套评测构件：

1. **AR/CR × 三种问答格式的评测框架**  
   适合把“视频是否语义正确”拆成可核对任务。

2. **uniform sparse temporal sampling**  
   在固定token预算下提高时间覆盖，适合其他视频判别/评测问题。

3. **projector-head-only adaptation**  
   当算力有限、又想保留预训练VLM能力时，这是很强的实用基线。

4. **层级式 data-mix 搜索**  
   先调任务比，再调问题格式比，适用于统一多任务评测器训练。

---

## Local PDF reference

![[paperPDFs/Evaluating_World_Models/arXiv_2025/2025_Adapting_Vision_Language_Models_for_Evaluating_World_Models.pdf]]