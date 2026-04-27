---
title: "Retrieval-augmented GUI Agents with Generative Guidelines"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/gui-control
  - retrieval-augmented-generation
  - task-aware-summarization
  - rejection-sampling
  - dataset/AndroidWorld
  - dataset/Multimodal-Mind2Web
  - dataset/AndroidControl
  - opensource/no
core_operator: 在推理时先对检索到的 GUI 教程做相关性判别，再生成结合当前状态与历史动作的精炼指导，供冻结的 GUI agent 决策使用
primary_logic: |
  任务描述 + 当前GUI状态 + 历史动作 + 检索教程
  → 轻量VLM判断每篇教程是否与当前任务/步骤相关，并生成任务感知指导
  → 仅将相关指导注入固定的 GUI agent
  → 输出下一步 GUI 动作
claims:
  - "在 AndroidWorld 上，RAG-GUI 将 Qwen2.5-VL-7B/72B 的成功率分别从 22.0%/35.0% 提升到 35.3%/45.7% [evidence: comparison]"
  - "相较直接拼接教程的 RAG，RAG-GUI 在 AndroidWorld、AndroidControl 和 Multimodal-Mind2Web 上更稳定地提升表现，说明“相关性判别 + 任务感知摘要”优于原始教程注入 [evidence: comparison]"
  - "去掉自引导拒绝采样微调（RSF）会在两种 backbone 和多项任务指标上持续掉点，例如 AndroidWorld 上 7B 从 35.3% 降至 32.8%，72B 从 45.7% 降至 44.4% [evidence: ablation]"
related_work_position:
  extends: "SeeAct (Zheng et al. 2024)"
  competes_with: "RAG (Lewis et al. 2020); AgentTrek (Xu et al. 2025b)"
  complementary_to: "UGround (Gou et al. 2025); OmniParser (Lu et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_Retrieval_augmented_GUI_Agents_with_Generative_Guidelines.pdf
category: Embodied_AI
---

# Retrieval-augmented GUI Agents with Generative Guidelines

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2509.24183)
> - **Summary**: 论文提出一个可插拔的轻量指导生成器，把检索到的长篇 GUI 教程转换成“是否相关 + 当前步骤指导”，从而在不改主 agent 参数的前提下提升真实 GUI 任务成功率。
> - **Key Performance**: AndroidWorld：Qwen2.5-VL-7B 从 22.0% 提升到 35.3%，72B 从 35.0% 提升到 45.7%；整体在 3 个 benchmark 上带来 2.6%–13.3% 绝对提升

> [!info] **Agent Summary**
> - **task_path**: 任务描述 + 当前 GUI 状态/截图 + 历史动作 + 检索教程 -> 下一步 GUI 动作
> - **bottleneck**: 原始教程过长、含噪且与当前子步骤常不对齐，直接 RAG 很难把“程序性知识”变成当前动作
> - **mechanism_delta**: 在主 agent 前插入一个轻量 VLM 适配器，先做教程相关性判别，再生成面向当前状态与历史动作的执行指导，并用 RSF 按动作正确性对齐
> - **evidence_signal**: 3 个 benchmark、2 个 Qwen backbone 上均优于 direct inference 与 vanilla RAG，且 RSF 消融稳定掉点
> - **reusable_ops**: [教程检索, 相关性筛选后的任务感知摘要]
> - **failure_modes**: [检索教程不相关或覆盖缺失时增益有限, 检索教程过多会拉长上下文并引入决策噪声]
> - **open_questions**: [能否迁移到非 Qwen 系 VLM/agent, 能否与轨迹级微调方法联合训练]

## Part I：问题与挑战

这篇论文抓到的真实瓶颈，不是 GUI agent 完全“看不懂界面”，而是它们在真实场景里**缺少长尾程序性知识**：  
很多网页/APP 任务涉及罕见流程、产品特定入口、跨步骤操作习惯，这些信息通常不在训练集里，却广泛存在于 web tutorials 中。

### 这个问题为什么难
现有两条路径都不够理想：

1. **靠更多训练数据补齐**  
   例如把教程转成轨迹再训练 agent，但合成轨迹质量不稳定，且泛化到新任务时仍受限。

2. **直接把教程塞进 prompt 做 RAG**  
   这在 GUI 场景里不顺手，因为教程是“过程型知识”：
   - 固定 chunk 容易切断步骤依赖；
   - 不切 chunk 又会导致输入过长、噪声过多；
   - 检索到的教程可能只和整体任务相关，但**不一定和当前步骤相关**。

### 输入/输出接口
论文把 GUI 任务表述为顺序决策：

- **输入**：任务描述 `g` + 当前界面状态 `s_t` + 历史动作 `A_t` + 检索到的 top-k 教程
- **输出**：下一步 GUI 动作 `a_t`

### 边界条件
这篇工作有一个很明确的设定：

- **主 agent policy 固定，不更新参数**
- 只训练一个轻量指导生成器 `fθ`
- 依赖外部教程库和检索器
- 目标是**推理时增强**，而不是端到端重训 agent

### 为什么现在值得做
因为两个条件同时成熟了：

- VLM-based GUI agent 已经具备不错的视觉 grounding 与基础推理能力；
- 网络上存在海量 how-to / GUI tutorials，可作为非参数知识库即时调用。

所以现在的关键不是“再把 agent 训大一点”，而是**如何把外部教程变成当前步骤可消费的行动提示**。

## Part II：方法与洞察

RAG-GUI 的核心做法是：  
**不要把检索到的教程原文直接给 agent，而是先用一个轻量 VLM 把教程变成“任务感知指导”。**

### 方法框架

#### 1）先建一个可检索的 GUI 教程库
作者从 MINT、OmniCorpus、WikiHow 中做了三阶段筛选：

- FastText 过滤
- 去重
- LLM 精筛

最终得到约 **2.6M** 条 GUI 教程，作为推理时的非参数知识库。

#### 2）对每篇检索教程做“相关性判别 + 指导生成”
对每个检索到的教程 `τ_i`，轻量模型 `fθ` 接收：

- 任务描述
- 当前状态
- 历史动作
- 教程内容

输出两部分：

- `ℓ`：这篇教程是否和当前任务/步骤相关
- `σ`：若相关，则生成一段面向当前步骤的简洁指导

然后只把被判为相关的指导 `σ` 提供给固定 agent `π` 来决定下一步动作。

#### 3）两阶段训练指导生成器
**SFT warmup**：  
先用教师模型 GPT-4.1-mini 在开源 GUI 数据上合成高质量指导，给 `fθ` 做一个热启动。

**RSF（self-guided rejection sampling finetuning）**：  
再让 `fθ` 采样多个候选指导，把这些指导喂给固定 agent。  
如果某个指导能帮助 agent 预测出正确动作，就把它保留下来继续训练 `fθ`。

这一步的关键不是“生成得像老师”，而是“生成得对 agent 真有用”。

### 核心直觉

RAG-GUI 真正调的不是“检索更多文本”，而是三个因果旋钮：

1. **从原始教程 → 相关性过滤后的教程**
   - 改变的瓶颈：减少检索噪声分布
   - 带来的能力：避免无关教程污染当前动作决策

2. **从长教程原文 → 当前步骤指导**
   - 改变的瓶颈：把长篇程序性知识压缩成可消费的 action hint
   - 带来的能力：即便是 7B 级别模型，也更容易利用外部教程

3. **从“语言上像好摘要” → “对 agent 选对动作有帮助”**
   - 改变的瓶颈：把训练目标从文本模仿改成下游动作效用
   - 带来的能力：指导生成器和 agent 的消费方式对齐，而不是只追求 fluent summary

一句话概括其机制变化：

**原来是把教程当“外部文本”塞给模型；现在是把教程转成“面向当前 GUI 步骤的行动先验”。**

### 为什么这个设计有效
因为 GUI agent 的很多错误并不是像素级看不清，而是：

- 不知道当前流程下一步通常该去哪；
- 不知道某个 app/site 的常见操作路径；
- 容易被长文里无关步骤干扰。

RAG-GUI 用“相关性筛选 + 条件摘要”把这些知识从长文里提纯出来，再用 RSF 保证这些摘要真的能帮助下游动作选择，所以提升集中出现在**长尾、在线、真实流程更复杂**的场景中。

### 战略权衡

| 设计选择 | 改变的瓶颈 | 收益 | 代价/风险 |
| --- | --- | --- | --- |
| 检索后先做相关性判别 | 教程可能只部分相关甚至错检 | 减少噪声教程污染 | 误判会丢掉有用教程 |
| 生成任务感知摘要而非原文拼接 | 长教程超出 token/注意力预算 | 把过程知识变成当前动作提示 | 可能压缩掉细节 |
| 用 RSF 按动作正确性筛指导 | “文本好看”不等于“agent 可用” | 让指导与下游决策对齐 | 训练需要动作标签和可执行评估流程 |
| 冻结主 agent，仅训练轻量适配器 | 全量重训成本高 | 真正可插拔、接入成本低 | 性能上限受主 agent 本身能力限制 |

## Part III：证据与局限

### 关键证据信号

#### 1）比较信号：在线 OOD 场景提升最大
最强证据来自 **AndroidWorld**：

- Qwen2.5-VL-7B：**22.0% → 35.3%**
- Qwen2.5-VL-72B：**35.0% → 45.7%**

这说明该方法最擅长补的是**真实长尾流程知识**，而不是只在离线 benchmark 上做 prompt trick。

#### 2）比较信号：收益不是“多给点文本”这么简单
vanilla RAG 直接拼接教程，提升很有限；  
而 RAG-GUI 在 AndroidControl 与 Multimodal-Mind2Web 上更稳定提升。

含义很明确：  
**真正有效的不是“有教程”本身，而是“把教程变成当前步骤可用的指导”。**

#### 3）消融信号：RSF 确实是关键旋钮
去掉 RSF 后，两个 backbone 在多项指标上都下降。  
这说明性能提升不只是来自 SFT 或更长 prompt，而是来自**用动作正确性反向筛选有效指导**的训练机制。

#### 4）控制变量信号：不是任何文本指导都行
作者还比较了“冻结 Qwen 直接生成 textual guidance”：

- 7B：27.5 < 35.3
- 72B：37.9 < 45.7

说明多做一步“在线生成摘要”还不够，**必须是任务适配、agent 对齐过的指导生成器**。

#### 5）设计分析：教程数量存在最优点
`k=3` 最优；教程太多会拉长上下文并增加混淆，太少又会漏掉关键信息。  
这也侧面验证了作者的问题诊断：**教程利用的核心是过滤与压缩，不是盲目扩容上下文。**

### 关键指标
- **AndroidWorld 成功率**：7B 22.0% → 35.3%；72B 35.0% → 45.7%
- **mm-Mind2Web Cross-Task Step SR**：7B 45.3% → 51.5%；72B 51.8% → 56.8%

### 能力跃迁 vs. 以往方法
相较以往两类路线：

- **比 vanilla RAG 更强**：因为它解决了“教程长、噪、弱对齐”的使用问题
- **比 AgentTrek 这类教程转轨迹训练更灵活**：因为它不依赖把教程先变成高质量训练轨迹，且能在推理时直接适配当前任务

所以它的能力跃迁不在于“训练出更强的基础 agent”，而在于让现有 agent **首次更稳定地消费 web tutorial 这种长尾外部知识源**。

### 局限性
- **Fails when**: 检索不到与当前 app/site 流程匹配的教程时，适配器几乎无从发挥；当任务失败主要来自精细视觉 grounding 而非程序性知识缺失时，文本指导帮助也有限。
- **Assumes**: 依赖一个大规模教程库（约 2.6M）、检索器（E5）、用于 SFT warmup 的教师模型 GPT-4.1-mini，以及已有 GUI 状态-动作训练元组；推理时还要对每篇候选教程额外跑一次指导生成，存在延迟开销。
- **Not designed for**: 端到端 agent 训练、无检索资源场景、超低延迟部署，以及跨架构普适性证明；论文实验也仅在 Qwen-VL 系列上验证，尚不能充分支撑“任意 VLM agent 都可即插即用”的强结论。

### 可复用组件
- **教程清洗管线**：FastText 预筛 + 去重 + LLM 精标，适合构建 GUI how-to 知识库
- **相关性门控摘要器**：把长过程文本变成步骤级指导，适合任何“长程序性文档 → 动作决策”的场景
- **agent-in-the-loop RSF**：用“是否帮助 agent 选对动作”来筛训练样本，是一种很实用的对齐策略

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_Retrieval_augmented_GUI_Agents_with_Generative_Guidelines.pdf]]