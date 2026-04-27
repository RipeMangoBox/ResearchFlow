---
title: "GenRL: Multimodal foundation world models for generalist embodied agents"
venue: NeurIPS
year: 2024
tags:
  - Embodied_AI
  - task/embodied-control
  - task/offline-reinforcement-learning
  - world-model
  - representation-alignment
  - trajectory-matching
  - dataset/dm_control
  - dataset/Kitchen
  - opensource/full
core_operator: 用仅视觉数据训练连接器与对齐器，把视频/语言提示映射为世界模型中的潜在目标轨迹，并在想象中以轨迹匹配奖励学习行为
primary_logic: |
  无奖励视觉交互数据 + 文本/视频提示 → 训练任务无关世界模型，并把 foundation 视频-语言表征通过 connector/aligner 映射到环境潜变量 → 在 imagination 中按潜在轨迹匹配优化 actor-critic → 得到可由文本或视频提示驱动的通用 embodied policy
claims:
  - "在 35 个任务的语言到动作 in-distribution 离线 RL 基准上，GenRL 的 overall 分数达到 0.80，高于最强基线 WM-CLIP-V 的 0.70 和 TD3-V 的 0.41 [evidence: comparison]"
  - "在未显式包含于训练数据的 21 个任务上，GenRL 的泛化表现显著优于图像/视频 VLM 奖励基线，并在 quadruped 与 cheetah 域接近专用专家策略的水平 [evidence: comparison]"
  - "aligner 是关键模块：移除 aligner 后总体分数从 0.76 降到 0.17，说明若不先缩小跨模态表征间隙，语言提示难以稳定映射到世界模型潜空间 [evidence: ablation]"
related_work_position:
  extends: "DreamerV3 (Hafner et al. 2023)"
  competes_with: "Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning (Rocamonde et al. 2023); Vision-Language Models as a Source of Rewards (Baumli et al. 2023)"
  complementary_to: "Plan2Explore (Sekar et al. 2020); Choreographer (Mazzaglia et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/NeurIPS_2024/2024_GenRL_Multimodal_foundation_world_models_for_generalist_embodied_agents.pdf
category: Embodied_AI
---

# GenRL: Multimodal foundation world models for generalist embodied agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.18043), [Project/Code/Data](https://mazpie.github.io/genrl)
> - **Summary**: 这篇论文把 foundation 视频-语言模型的语义空间，用“连接器 + 对齐器”落到世界模型的可控潜空间里，从而只靠视觉交互数据，就能让 embodied agent 根据文本或视频提示在 imagination 中学会行为。
> - **Key Performance**: 语言→动作 in-distribution overall = **0.80**（WM-CLIP-V = 0.70，TD3-V = 0.41）；移除 aligner 后 overall 从 **0.76** 降到 **0.17**。

> [!info] **Agent Summary**
> - **task_path**: 文本/视频提示 + 无奖励视觉交互数据 -> 多任务 embodied policy
> - **bottleneck**: embodied 域缺少语言标注，且 foundation VLM 语义空间与环境可控状态空间不对齐，导致提示难转成稳定可优化的行为目标
> - **mechanism_delta**: 用 vision-only 训练的 connector+aligner 先把提示变成世界模型中的 latent target trajectory，再以潜在轨迹匹配奖励在 imagination 中学策略
> - **evidence_signal**: 35 任务跨 locomotion/manipulation 对比实验，外加 aligner 与 temporal alignment 消融
> - **reusable_ops**: [vision-only cross-modal alignment, latent-goal imagination RL]
> - **failure_modes**: [静态提示常被翻译成轻微运动序列, 复杂开放场景下世界模型重建模糊会让目标失真]
> - **open_questions**: [如何摆脱重建瓶颈扩展到更复杂视觉世界, 如何突破短时视频窗口支持长时程任务组合]

## Part I：问题与挑战

这篇论文要解决的，不只是“让 RL 听懂语言”，而是更具体的一个瓶颈：

**如何在没有语言标注的 embodied 数据里，把 foundation VLM 的通用语义，变成环境内可达、可优化、可执行的行为目标？**

### 1. 真实难点是什么

传统 RL 要扩到多任务 embodied agent，最大障碍通常不是策略网络本身，而是：

1. **每个任务都要单独设计 reward**，成本高且容易错。
2. **直接拿 VLM/CLIP 相似度当 reward 不稳定**：自然图像/视频分布和 embodied 环境差距很大。
3. **embodied 域通常没有成规模的图像-文本/视频-文本配对数据**，所以很难像常规多模态任务那样 fine-tune 一个域内 VLM。
4. **语义对了不等于可执行**：即使语言 embedding 能表达“run”或“open microwave”，也未必能对应到这个环境里 agent 真能到达的状态轨迹。

所以，真正的瓶颈不是“缺少文本接口”，而是：

> **foundation 模型的语义空间，没有被 grounded 到环境动力学的可控潜空间。**

### 2. 为什么现在值得解决

这件事现在可做，靠的是两个趋势开始成熟：

- **世界模型**已经能用纯视觉交互数据学出较强的 latent dynamics，并在 imagination 中做 policy learning。
- **视频-语言 foundation model**已经具备不错的动态语义先验，不再只擅长静态图像语义。

GenRL 的想法就是：  
**不要把 VLM 直接当 reward model，而是把它当 task prior；不要在像素空间上硬算相似度，而是先把提示翻译成世界模型里的目标潜轨迹。**

### 3. 输入/输出接口与边界条件

**训练输入：**
- 视觉观测序列
- 动作序列
- 无 reward
- 无语言标注

**任务输入：**
- 文本提示，或
- 短视频提示

**输出：**
- 能在环境中执行对应行为的策略

**边界条件：**
- 主要面向视觉控制任务
- 评测集中在 locomotion + manipulation
- 任务大多是短到中时程行为
- 使用 8 帧视频窗口的 InternVideo2 作为 VLM
- 核心设定是离线数据预训练，随后在 imagination 中学策略；进一步还探索了完全不访问数据的 data-free policy learning

---

## Part II：方法与洞察

### 方法总览

GenRL 包含三层关键结构：

1. **任务无关世界模型**：从视觉交互数据学习环境 latent dynamics  
2. **跨模态落地模块**：用 connector + aligner 把 VLM 语义映射到世界模型潜空间  
3. **潜在轨迹匹配式策略学习**：在 imagination 中追踪 prompt 对应的 latent target trajectory

这三步合起来，才让“文本/视频提示 -> 可执行行为”成为可能。

### 1. 世界模型：先学一个“环境内部语言”

作者先训练一个 generative world model。它把每一帧图像编码成离散 latent state，再用序列模型建模状态随动作演化的过程。

这里有个值得注意的设计：

- encoder / decoder **不依赖** sequence model 的 hidden state
- 这样每个 latent state 更像一个**单帧视觉 token**
- 时序信息主要放在 GRU hidden state 里

这个拆法的好处是：  
世界模型的 latent space 更像一个**环境专属、受动力学约束的视觉状态空间**。后面 connector 才有机会把 VLM 语义安全地接进来。

### 2. Connector：把 VLM 视觉语义映射成环境 latent 轨迹

connector 的作用是：

- 输入：VLM 的视觉 embedding
- 输出：世界模型里的 latent state sequence

也就是说，作者先不处理语言，而是先学会：

> “如果 foundation VLM 看到了某段视频语义，这在当前 embodied 环境里大概对应什么潜在状态轨迹？”

这一步非常关键，因为它把“通用视频语义”变成了“这个环境里可解释的状态序列”。

### 3. Aligner：不用语言标注，补跨模态 gap

问题来了：connector 是用**视觉 embedding**训练的，但测试时希望输入也可以是**语言 embedding**。  
然而多模态对比学习模型普遍有一个问题：**multimodality gap**。也就是文本 embedding 和对应视觉 embedding 在球面空间里并不完全重合。

以往一些做法会在训练时对视觉 embedding 加噪声，让 connector 学得更鲁棒。  
这篇论文换了个角度：

- 不去污染 connector 的输入分布
- 而是额外学一个 **aligner**
- 用仅视觉数据，在视觉 embedding 周围采样点，学习把附近点“拉回”视觉 embedding 邻域

于是推理时：

- 视频提示：visual embedding -> connector
- 文本提示：text embedding -> aligner -> connector

这等于把语言先投影回“connector 熟悉的视觉邻域”，再交给 connector 生成环境 latent target。

### 核心直觉

**变化了什么？**  
从“直接在观测空间里用 VLM 相似度做 reward”，改成“先把 prompt 翻译成世界模型中的目标潜轨迹，再在潜空间做策略优化”。

**改变了哪个瓶颈？**  
这一步改变的是两个核心约束：

1. **监督约束**：从需要语言标注，变成只需要视觉交互数据  
2. **优化空间**：从 foundation 语义空间中的模糊相似度，变成环境动力学约束下的可达 latent target

**能力为什么会变强？**  
因为 agent 不再追逐一个“看起来像 prompt”的外部相似度分数，而是在追逐一个：

- 已被环境 world model grounding 过的
- 与环境动力学兼容的
- 可在 imagination 中密集评估的

**潜在目标轨迹**。

一句话概括这条因果链：

> **prompt grounding 进 latent dynamics**  
> → **跨模态语义被压到可达状态流形上**  
> → **reward 更稳定、时序更自然、策略更易学**  
> → **最终带来多任务泛化与 data-free adaptation。**

### 4. 在 imagination 中学行为：目标不是“像一帧”，而是“像一段轨迹”

GenRL 的策略学习不是匹配单帧目标，而是匹配一段由 prompt 生成的 latent trajectory。

做法上，给定文本或视频提示后：

- VLM 产生 embedding
- 文本时先过 aligner
- connector 产生目标 latent sequence
- actor-critic 在世界模型 imagination 中 rollout
- 用 imagined latent state 和 target latent state 的投影余弦相似度作为 reward

这点很重要：  
**目标变成轨迹，而不是静态图像。**  
所以动态任务（如 run / walk / flipping）会比纯 image-language reward 更受益。

### 5. Temporal alignment：解决“起点不一样”的错位问题

轨迹匹配会遇到一个实际问题：

- target trajectory 可能默认从“任务已经开始的状态”出发
- 但当前 agent 可能还躺在地上，或者位置姿态完全不同

如果直接逐时刻比对，会出现奖励错位。

作者的办法是做一个 **best matching trajectory**：

- 先在 imagined trajectory 上沿时间滑动
- 找到和 target 起始片段最匹配的对齐点
- 再按这个对齐点计算奖励

这本质上是在缓解一个很现实的问题：  
**提示描述的是目标行为，不保证和 agent 当前状态同相位。**

### 6. Data-free policy learning：把“下游适配”也搬进模型内部

GenRL 最有意思的一个外延，是作者提出了 **data-free policy learning**。

传统 offline RL 即便用了 world model，策略学习往往还得从数据里取初始状态，或者依赖任务 reward 标注。  
GenRL 则进一步做到：

- 初始 latent state 在模型内部采样
- rollout 在 imagination 中完成
- reward 只依赖 prompt 和 latent target
- 下游学新任务时，不再访问原始数据集

这让它更像 foundation model 的使用方式：  
**预训练时吃大数据，下游适配时不需要再碰原始数据。**

### 策略性权衡

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价/风险 |
|---|---|---|---|
| 视频-语言 VLM 而非图像-语言 VLM | 动态任务语义不足 | 对 walk/run 等时序行为更强 | 计算更重，时间窗有限 |
| connector + aligner，而非直接 CLIP reward | 语义空间未 grounding 到动力学空间 | prompt 能转成环境内可达 latent target | 额外模块，依赖 VLM 表征质量 |
| 重建式世界模型 | 需要可解释的 latent grounding | 可解码 target，便于检查 prompt 被如何理解 | 复杂视觉场景下重建会模糊 |
| imagination-based RL | 下游任务 reward 缺失 | 训练更高效，可支持 data-free adaptation | 强依赖 world model 覆盖度 |
| 简单 GRU 架构 | 控制实现复杂度 | 结构对称、易训练 | 长时依赖和复杂场景建模受限 |

---

## Part III：证据与局限

### 关键证据链

**信号 1：跨域主结果对比（comparison）**  
在 35 个任务的 language-to-action in-distribution 评测里，GenRL overall = **0.80**，高于：
- WM-CLIP-V：0.70
- TD3-V：0.41
- IQL-V：0.29

这说明核心收益不是单纯“用了 world model”或“用了视频 VLM”，而是**把 VLM 表征 grounding 到 latent dynamics 的这一步**。

**信号 2：未见任务泛化（comparison）**  
在 21 个未显式出现在训练集里的任务上，GenRL 依然明显强于 model-free 的 VLM reward 方法；尤其在 quadruped 和 cheetah 域，表现接近专用专家策略。  
这支持作者的核心主张：GenRL 学到的不只是“从数据里检索现成片段”，而是**借助已对齐的世界模型表征做任务泛化**。

**信号 3：视频提示也能驱动行为（comparison / case-like evidence）**  
GenRL 不只支持文本，还支持短视频提示。论文展示了不同视觉风格、不同视角、不同 morphology 的 prompt 也能被 grounding 到目标环境。  
这说明它学到的是**跨模态任务规格化接口**，而不是只对单一文字模板有效。

**信号 4：aligner 消融非常强（ablation）**  
去掉 aligner 后 overall 从 **0.76** 掉到 **0.17**。  
这基本直接证明：  
**如果不先处理 multimodality gap，语言 embedding 很难作为 connector 的有效输入。**

**信号 5：temporal alignment 与数据分布分析（ablation / analysis）**  
- temporal alignment 消融显示：对齐 imaginary trajectory 与 target trajectory 的时序位置能带来更稳的学习信号。  
- 数据分布分析显示：**多样化探索数据**对泛化非常关键，完整数据最好，探索数据次之，狭窄任务数据泛化最差。

这意味着 GenRL 的能力并不神奇地“凭空产生”，它依然依赖**足够丰富的预训练行为覆盖**。

**信号 6：data-free adaptation 可行（comparison）**  
预训练完 MFWM 后，GenRL 可以不访问原始数据，直接在 imagination 中学习新任务；整体性能虽略降，但仍接近 offline GenRL，并优于其他离线基线。  
这是论文最接近“foundation policy”叙事的一点：  
**预训练一次，后续靠 prompt 和模型内部 rollout 做适配。**

### 1-2 个最值得记住的指标

- **主指标**：语言→动作 in-distribution overall，GenRL = **0.80**，WM-CLIP-V = **0.70**  
- **关键消融**：移除 aligner 后 overall，**0.76 -> 0.17**

### 局限性

- **Fails when**:  
  - 静态提示会被翻译成略带运动的目标序列，导致一些 kitchen 静态任务上不一定最优  
  - 复杂开放环境下，世界模型重建变模糊，prompt 解码目标不够清晰；论文在 Minecraft 上已经观察到这个问题  
  - 精细 OOD manipulation 若训练数据里从未出现对应动作，模型可能只能“接近目标区域”而无法真正完成操作

- **Assumes**:  
  - 需要一个强的预训练视频-语言模型（本文用 InternVideo2）  
  - 需要较大规模、分布多样的视觉交互数据；复杂行为通常仍需要一定 expert/task-specific 数据覆盖  
  - 依赖可重建的 world model 表征；如果世界模型本身在复杂观测下失真，后续 grounding 也会受影响  
  - 训练资源不低：论文使用 16GB V100 集群；MFWM 训练 500k steps 约 5 天，且预先缓存 VLM embedding

- **Not designed for**:  
  - 真正超出训练数据行为支持范围的全新技能发现  
  - 超长时程、需多技能组合和规划的任务  
  - 高精度开放世界交互中对细粒度视觉与动作对齐要求极高的场景

### 可复用部件

1. **vision-only cross-modal alignment**：在无语言标注 setting 下缩小 text/video embedding gap  
2. **connector to environment latent space**：把 foundation semantics 映射成环境可达目标  
3. **latent trajectory matching reward**：把 prompt 条件化行为学习改写成 imagination 中的序列匹配  
4. **prompt decoding for explainability**：可先解码 latent target，看 prompt 被系统如何理解  
5. **data-free policy adaptation**：预训练后不访问数据集做下游任务学习

### 一句话结论

GenRL 的真正贡献，不是“又把语言接到 RL 上”，而是提出了一条更稳的落地路径：

> **先把 foundation 多模态语义压进 world model 的可控潜空间，再在这个空间里学行为。**

这一步把“会看懂提示”变成了“能在环境中做出来”，也是它相比直接 VLM reward 方法出现能力跃迁的核心原因。

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/NeurIPS_2024/2024_GenRL_Multimodal_foundation_world_models_for_generalist_embodied_agents.pdf]]