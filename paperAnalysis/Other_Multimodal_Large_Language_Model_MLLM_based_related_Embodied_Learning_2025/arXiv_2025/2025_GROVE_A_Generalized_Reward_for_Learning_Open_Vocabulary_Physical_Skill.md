---
title: "GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/physical-skill-learning
  - task/text-conditioned-control
  - reinforcement-learning
  - multimodal-reward
  - pose-to-clip
  - dataset/AMASS
  - dataset/Motion-X
  - repr/SMPL
  - opensource/partial
core_operator: 用LLM生成可执行物理约束、用Pose2CLIP+CLIP给出语义/自然度奖励，并以VLM反馈触发奖励重写的通用奖励框架
primary_logic: |
  自然语言指令 + 世界模型/agent关节描述 → LLM生成任务相关的精确物理约束奖励，Pose2CLIP把当前姿态映射到CLIP语义空间得到VLM语义奖励，并在VLM fitness持续下降时重写LLM奖励 → 驱动RL或分层控制器学到符合指令且更自然的开放词汇物理技能
claims:
  - "On five open-vocabulary humanoid instructions, GROVE (+CALM) achieves the highest human-rated task completion score (7.924) and the highest CLIP similarity (28.998) among compared baselines while maintaining competitive naturalness [evidence: comparison]"
  - "On standard RL benchmarks, GROVE improves reward-distance learning efficiency over direct expert-reward optimization in 3 of 4 evaluated tasks across Ant, Cartpole, and ANYmal settings [evidence: comparison]"
  - "Ablations show the full Pose2CLIP + LLM + RDP configuration outperforms VLM-only, LLM-only, and no-RDP variants in completion and convergence speed, supporting the complementarity of semantic and constraint rewards [evidence: ablation]"
related_work_position:
  extends: "Eureka (Ma et al. 2024)"
  competes_with: "AnySkill (Cui et al. 2024); VLM-RM (Rocamonde et al. 2024)"
  complementary_to: "CALM (Tessler et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_GROVE_A_Generalized_Reward_for_Learning_Open_Vocabulary_Physical_Skill.pdf
category: Embodied_AI
---

# GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.04191), [Project](https://jiemingcui.github.io/grove/)
> - **Summary**: 该工作把LLM的“精确物理约束生成”与VLM的“动作语义/自然度判别”闭环结合，构造出无需手工奖励和任务特定示范的开放词汇物理技能学习框架。
> - **Key Performance**: 相对基线，任务完成度提升 **25.7%**，训练收敛速度提升 **8.4×**

> [!info] **Agent Summary**
> - **task_path**: 文本指令 + agent状态/世界模型描述 -> 连续控制策略或高层latent技能
> - **bottleneck**: 奖励函数无法同时表达精确时序物理约束与整体动作语义/自然度
> - **mechanism_delta**: 用Pose2CLIP把姿态直接接到CLIP语义空间，再让VLM奖励既做监督也做LLM奖励的在线质检与重写触发器
> - **evidence_signal**: 多基准比较 + 7组消融均显示完整模型在完成度和收敛效率上最好
> - **reusable_ops**: [LLM奖励程序生成, pose-to-CLIP语义映射]
> - **failure_modes**: [多成功条件动态任务上不一定优于专家手工奖励, OOD姿态或非固定状态空间会削弱语义评估]
> - **open_questions**: [能否迁移到真实机器人和接触丰富场景, 如何为更多非人形体构建统一的Pose2CLIP适配]

## Part I：问题与挑战

**What/Why**：这篇文章真正要解决的，不是“再做一个会动的角色”，而是**如何为任意自然语言技能自动构造一个可优化、可泛化、又不容易被钻空子的奖励函数**。

### 1. 难点到底在哪
开放词汇物理技能学习有两个长期瓶颈：

1. **手工奖励不具可扩展性**  
   每个新任务都要重新写速度、姿态、稳定性、能耗、接触等奖励项，成本高且强依赖专家经验。

2. **示范/文本-动作方法受分布限制**  
   它们可以生成“像样”的动作，但对超出训练语料分布的抽象指令、长尾指令、组合技能，泛化明显不足。

更细一点说，奖励设计卡在一个“双重缺口”上：

- **LLM式奖励**：能把文本翻成精确约束，但往往只抓到局部关节关系，容易出现“技术上满足、看起来不自然”的动作。
- **VLM式奖励**：能判断动作看起来像不像，但对时间一致性、精细物理约束、可执行时序结构不够强。

### 2. 为什么现在值得做
因为现在三个条件同时成立：

- **LLM** 已经足够强，可以把自然语言拆成关节、位置、速度、空间关系等约束；
- **VLM/CLIP** 已经能提供跨文本-视觉的语义对齐信号；
- **IsaacGym 等高速模拟器** 让大规模RL试错变得现实。

所以现在缺的不是“更大的模型”，而是一个**把文本语义、视觉语义和物理控制统一成训练信号的奖励接口**。

### 3. 输入/输出与边界条件
- **输入**：自然语言指令、环境/world model描述、agent关节/形体描述、当前状态/动作
- **输出**：可用于RL优化的通用奖励，以及最终学到的控制策略
- **边界条件**：
  - 主要工作在**模拟环境**
  - 依赖**固定状态空间表示**
  - 最强的人形结果结合了**预训练低层控制器 CALM**
  - 不是直接面向真实机器人部署的系统论文

---

## Part II：方法与洞察

GROVE 的核心不是单一模型，而是一个**广义奖励框架**。它把奖励拆成两个互补来源，再加一个闭环修正机制。

### 1. 方法结构

#### (a) LLM-based reward：负责“字面约束”
作者沿用 Eureka 的思路，让 LLM 根据：

- world model
- agent关节命名与索引
- 代码模板
- 文本指令

生成可执行奖励函数。  
但这里做了两个关键增强：

- **加入详细的agent关节描述**，让LLM能更准确地写到具体身体部位；
- **明确提示“稳定/走路等底层能力已由其他模块处理”**，让LLM只关注“任务本质”。

这一步改变的是：  
**文本指令不再只是描述，而变成了可优化的物理约束程序。**

#### (b) VLM-based reward：负责“整体语义与自然度”
作者不再直接把仿真画面送进CLIP，因为这样既慢又有明显sim-to-real gap。  
他们提出 **Pose2CLIP**：把agent姿态直接映射到CLIP特征空间，再与文本特征算相似度。

Pose2CLIP 的角色很关键：

- 避免在线渲染
- 缓解“仿真图像不像自然图像”导致的VLM失真
- 让VLM奖励可以真正在线参与RL

训练上，它使用：

- AMASS + Motion-X
- 训练过程中采样的policy rollout poses
- Blender 多视角渲染生成CLIP目标特征

最终是一个很轻量的两层MLP，但起到了“把控制状态接上语义空间”的桥梁作用。

#### (c) RDP闭环：让VLM监督LLM奖励是否跑偏
这是整篇论文最有“系统因果味”的设计。

作者观察到：LLM写出的奖励函数虽然精确，但可能发生**semantic drift**，即策略学会了奖励漏洞，却没有真正完成任务。

所以 GROVE 让 **VLM reward 反过来做 fitness evaluator**：

- 如果训练过程中 VLM 评分连续下降，且跌到阈值以下，
- 就重新调用 LLM 生成奖励函数。

这相当于给 LLM reward 加了一个**在线质检/拒绝采样机制**。

### 核心直觉

过去的方法大多是**单源奖励**：

- 只有LLM：奖励精确，但容易“只满足字面，不满足观感”
- 只有VLM：语义强，但对精细动作控制太松

GROVE 把它改成了**正交双约束 + 在线审计**：

- **LLM奖励** 收紧“物理与几何可行域”
- **VLM奖励** 收紧“语义与自然度可行域”
- **RDP重写** 避免LLM奖励把策略带向错误局部最优

所以真正改变的是这条因果链：

**单一、欠约束的奖励目标**  
→ **同时具备局部精确约束和全局语义约束的奖励目标**  
→ **RL搜索空间被更有效收缩**  
→ **更快收敛到既做对任务、又看起来合理的动作**

一句话概括：  
**LLM负责“做得对”，VLM负责“看起来也对”，RDP负责“别学歪”。**

### 2. 两种使用方式
论文还强调这个奖励框架不是只服务一种控制范式：

1. **人形技能合成**：接在 CALM 低层控制器之上，训练高层latent policy
2. **标准RL benchmark**：直接用 PPO / SAC 从头学策略

这说明它不是单纯的“动作生成技巧”，而更像一个**可插拔的奖励接口层**。

### 3. 战略权衡表

| 设计选择 | 改变了什么瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| LLM奖励程序生成 | 文本无法直接变成可优化物理目标 | 能表达关节、位置、速度、空间关系等精确约束 | 易受prompt质量影响，可能语义漂移 |
| VLM语义奖励 | 奖励只管几何、不管整体观感 | 提供自然度和语义一致性监督 | 时序精度有限，直接用仿真图像会有域差 |
| Pose2CLIP | 在线渲染慢且不适合CLIP分布 | 免渲染、语义评估更稳、更快 | 需要额外构建大规模姿态-特征训练集 |
| RDP奖励重写 | LLM奖励可能被策略“钻空子” | 用VLM做闭环质检，抑制reward hacking | 依赖阈值设定，也增加API调用成本 |
| 结合预训练控制器 | 复杂人形低层运动难以直接学稳 | 高层专注指令对齐，动作更自然 | 对预训练控制器质量有依赖 |

---

## Part III：证据与局限

**So what**：论文最重要的结论是，GROVE 不只是“能用LLM/VLM做奖励”，而是证明了**这两类奖励在物理技能学习中是互补的，而且闭环结合后会带来明显的完成度与效率提升**。

### 1. 关键证据信号

#### 信号A：开放词汇人形技能比较
**信号类型：comparison**  
在五个开放词汇指令上，GROVE（接 CALM）的人评任务完成度最高，达到 **7.924**，同时 CLIP 相似度也最高 **28.998**。

这说明它不仅生成“像人的动作”，还更能执行：

- 抽象指令（如把身体摆成 C）
- 风格化指令（如像模特一样走）
- 复合指令（如奔跑并跳栏）

而这恰好是纯文本-动作数据驱动方法最容易掉队的地方。

#### 信号B：标准RL基准上的泛化
**信号类型：comparison**  
在 Ant / Cartpole / ANYmal 等基准上，GROVE 在 **4个任务中的3个** 比直接优化专家奖励有更低的 reward distance，说明它不仅能做“开放词汇人形表演”，还能做更一般的控制学习。

一个很强的效率对比是和 VLM-RM 的 bow 任务：

- **GROVE**：47 min，完成度 6.276
- **VLM-RM**：411 min，完成度 1.655（原纹理）/ 3.483（改良纹理）

这说明：  
相比“为了适配VLM去改渲染/纹理”，**直接把姿态接入语义空间 + 加LLM约束**更有效。

#### 信号C：消融证明“互补性”不是口号
**信号类型：ablation**  
7组消融里，完整的 **Pose2CLIP + LLM + RDP** 最好；只保留单模态奖励、去掉 Pose2CLIP、或去掉 RDP 都会退化。

特别是三点很有说服力：

1. **Pose2CLIP 比直接CLIP更好更快**  
   说明问题不只是“有没有VLM”，而是“VLM能否在仿真域可靠工作”。

2. **LLM-only 与 VLM-only 各有偏科**  
   一个更精确，一个更语义化，说明二者确实是互补而非冗余。

3. **RDP有用**  
   不做动态重写时，组合奖励的收益不稳定，说明闭环质检是必要组件，不是装饰性技巧。

#### 信号D：Pose2CLIP 本身有效
**信号类型：analysis**  
在姿态-文本相似矩阵评估里：

- Blender+CLIP：0.49
- Pose2CLIP：0.48

几乎持平，但 Pose2CLIP 不需要在线渲染。  
这证明它确实把“高保真视觉语义”有效蒸馏到了一个轻量姿态映射器中。

### 2. 1-2个关键数字
- **任务完成度**：相对基线提升 **25.7%**
- **训练效率**：收敛速度提升 **8.4×**

### 3. 局限性
- **Fails when**: 任务包含多重隐式成功条件、显式物体交互或复杂接触语义时，通用奖励未必优于专家手工奖励；论文里 ANYmal jump-up 就没有赢过 direct reward。若姿态明显超出 Pose2CLIP 的训练覆盖，语义评估也可能失真。
- **Assumes**: 需要结构化的关节/状态访问接口；依赖冻结的 CLIP 和闭源的 GPT-o1-preview 生成奖励；Pose2CLIP 训练需要约 170 万帧的 Blender 渲染监督；最佳人形结果还依赖预训练 CALM 控制器，以及 A100/4090 级别计算资源。
- **Not designed for**: 真实机器人直接部署、纯像素端到端控制、可变拓扑身体、以及需要精确对象状态建模的复杂操作任务。

### 4. 可复用部件
这篇论文最值得复用的不是某个具体数值，而是三个操作原语：

- **LLM奖励程序模板化生成**：把 world model + joint schema + instruction 拼成可执行奖励
- **Pose2CLIP 适配层**：把 simulator state 接到现成VLM语义空间
- **VLM-as-fitness 的奖励重写回路**：用语义模型给程序化奖励做在线质检

如果你在做 embodied RL，这三个模块可以分别拿出来复用。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_GROVE_A_Generalized_Reward_for_Learning_Open_Vocabulary_Physical_Skill.pdf]]