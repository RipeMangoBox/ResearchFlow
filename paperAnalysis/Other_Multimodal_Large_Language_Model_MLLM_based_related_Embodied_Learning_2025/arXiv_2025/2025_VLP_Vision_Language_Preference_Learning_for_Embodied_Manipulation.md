---
title: "VLP: Vision-Language Preference Learning for Embodied Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/embodied-manipulation
  - preference-learning
  - cross-modal-attention
  - reinforcement-learning
  - dataset/MTVLP
  - dataset/Meta-World
  - opensource/no
core_operator: 用三类语言条件隐式偏好关系训练视频-语言偏好打分器，再把其作为下游操控策略学习的自动偏好标注器
primary_logic: |
  多任务轨迹视频与语言指令 → 预训练视觉/语言编码并经跨模态注意力提取任务相关表征 → 用任务内/跨语言/跨任务偏好对齐出轨迹级偏好分数 → 比较分数得到目标任务偏好标签并驱动离线RLHF
claims:
  - "在 Meta-World 5 个测试任务上，P-IQL+VLP 的平均成功率为 71.0，高于 scripted-label P-IQL 的 62.9；CPL+VLP 为 83.8，高于 scripted-label CPL 的 78.0 [evidence: comparison]"
  - "VLP 生成的偏好标签相对 scripted labels 的平均准确率达到 97.4%，并在未见任务与未见语言表达上保持约 95.8%-97.0% 的 ITP 准确率 [evidence: comparison]"
  - "移除跨任务视频偏好项（λ2=0）会把 IVP 准确率从 91.7 降到 63.0，表明跨任务匹配约束是语言条件泛化的关键 [evidence: ablation]"
related_work_position:
  extends: "PEARL (Liu et al. 2024b)"
  competes_with: "VLM-RM (Rocamonde et al. 2024); RoboCLIP (Sontakke et al. 2023)"
  complementary_to: "IPL (Hejna and Sadigh 2023); CPL (Hejna et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_VLP_Vision_Language_Preference_Learning_for_Embodied_Manipulation.pdf
category: Embodied_AI
---

# VLP: Vision-Language Preference Learning for Embodied Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.11918)
> - **Summary**: 这篇论文把 embodied manipulation 里的“昂贵人工偏好标注”替换成一个可迁移的视频-语言偏好模型：先自动构造三类语言条件偏好，再用跨模态打分器给新任务生成偏好标签，供离线 RLHF 训练策略。
> - **Key Performance**: VLP 标签相对 scripted labels 的平均准确率为 **97.4%**；在 5 个 Meta-World 测试任务上，**P-IQL+VLP = 71.0**，高于 scripted-label P-IQL 的 **62.9**，也高于 VLM-based 对照的最佳区间。

> [!info] **Agent Summary**
> - **task_path**: 多任务轨迹视频 + 自然语言任务指令 -> 轨迹级偏好分数/轨迹对偏好标签 -> 下游离线操控策略
> - **bottleneck**: 人工偏好查询昂贵，直接用 VLM 当奖励又高噪声且缺乏跨任务语言对齐
> - **mechanism_delta**: 把“单任务奖励回归”改成“语言条件下的单视频偏好打分”，并用 ITP/ILP/IVP 三类关系显式学习任务内排序、语言失配抑制和跨任务匹配
> - **evidence_signal**: VLP 标签平均准确率 97.4%，且用这些标签训练的 P-IQL/CPL 在多项任务上可比甚至优于 scripted labels
> - **reusable_ops**: [语言条件偏好构造, 单视频-单指令打分后用 Bradley-Terry 做成对比较]
> - **failure_modes**: [语言指令信息不足时容易错标, 难以由视频与语言完整表达的复杂装配任务可能失效]
> - **open_questions**: [能否迁移到真实机器人与感知噪声环境, 对长时程多阶段接触任务是否仍能稳定泛化]

## Part I：问题与挑战

这篇工作真正要解决的，不是“策略优化器不够强”，而是 **偏好监督的供给方式不可扩展**。

在 embodied manipulation 里，常见路径各有明显瓶颈：

1. **手工 reward engineering**  
   设计成本高，而且容易 reward hacking；更关键的是，环境 reward 不一定真的等价于人类想要的任务语义。

2. **人工偏好标注的 RLHF**  
   偏好学习很有吸引力，但需要大量在线查询或离线标注的轨迹对，成本高、速度慢。

3. **直接拿 VLM/VLM-RM 当奖励**  
   虽然省人工，但给出的通常是高方差、噪声较大的标量信号；而且很多方法默认能访问环境信息，或没有把“语言条件”真正学成稳定的偏好判别器。

### 这篇论文的输入/输出接口

- **输入**：轨迹视频 `v` + 语言指令 `l`
- **中间表示**：轨迹级偏好分数 `f(v|l)`
- **输出**：对任意视频对 `(v1, v2)` 在指令 `l` 下给出偏好标签
- **用途**：作为 P-IQL / IPL / CPL 等离线 RLHF 算法的自动偏好标注器

### 边界条件

VLP 的设定并不是通用世界模型，而是比较明确的：

- 任务主要是 **模拟环境中的机器人操作**
- 任务目标需要能被 **视频观测 + 自然语言指令** 表达
- 训练依赖 **多任务轨迹库**，并且这些轨迹要有可自动构造的优劣关系
- 论文主验证场景是 **Meta-World**，外加 ManiSkill2 的补充标签泛化测试

### 为什么是现在做这件事

因为现在正好有两类条件成熟：

- **预训练视觉-语言模型** 已经足够强，能提供稳定的跨模态表征底座
- **LLM/VLM 的语言改写能力** 足够好，可以低成本扩写任务指令，形成更丰富的语言条件空间

所以作者的核心判断是：  
与其每个新任务都重新收集人类偏好，不如先学一个 **“会看视频、会理解指令、会输出偏好”** 的通用 annotator。

## Part II：方法与洞察

### 方法骨架

VLP 由三部分组成：

1. **MTVLP 数据集构造**
2. **视频-语言偏好模型**
3. **语言条件偏好学习目标**

#### 1）MTVLP：自动造出“可学偏好”

作者在 Meta-World 上构建了多任务视频-语言偏好数据。关键不是人工标注，而是 **利用任务结构自动产生隐式偏好关系**。

每个任务收集三种轨迹：

- **expert**：脚本策略 + 小噪声
- **medium**：只完成大约一半阶段的子任务
- **random**：随机动作

这一步很关键，因为它给了模型一个天然的“优劣层次”。  
同时，每条轨迹再配多个语言表达，语言由 GPT-4V 生成同义改写和不同动词结构，从而提升语言泛化。

#### 2）偏好模型：单视频、单指令打分

模型不是直接把“两段视频 + 一句语言”一起塞进去比较，而是学习一个可复用的标量打分器：

- 视频编码器：CLIP ViT-B/16
- 语言编码器：CLIP 文本编码器
- 跨模态模块：cross-modal transformer
- 输出：轨迹级偏好分数 `f(v|l)`

这样做的好处是：

- **推理更高效**：每个视频可单独打分，再做成对比较
- **更像 annotator**：学的是“这段视频在该指令下有多值得偏好”，而不是某个固定 pair 的分类器

#### 3）三类语言条件偏好

这是整篇论文最关键的设计。

**(a) ITP: Intra-Task Preference**  
同一任务的视频，在同一任务语言条件下，按轨迹优劣排序。  
这相当于传统 preference learning 的锚点。

**(b) ILP: Inter-Language Preference**  
同一任务的视频，但语言来自另一个任务，此时两段视频应当 **等偏好**。  
这一步在教模型：**语言不匹配时，不要乱判高低。**

**(c) IVP: Inter-Video Preference**  
两个不同任务的视频，在某个任务语言条件下，与该语言匹配的任务视频应被偏好。  
这一步在教模型：**偏好不只看“动作质量”，还要看“是否完成了这句语言说的任务”。**

最终，VLP 用这三类关系联合训练，使 `f(v|l)` 既编码轨迹质量，也编码任务语义匹配。

### 核心直觉

VLP 的真正变化，不是“又做了一个多模态模型”，而是把偏好学习的对象从：

- **绝对奖励/单任务优劣**

改成了：

- **语言条件下的相对任务完成度**

也就是：

**从“视频本身好不好”变成“视频是否更符合这句指令”**。

这带来三个因果层面的变化：

1. **what changed**  
   从单任务 reward/preference，变成多任务、语言条件化的轨迹打分。

2. **which bottleneck changed**  
   过去模型容易把“动作平滑/接近目标”当成通用高分信号；  
   加入 ILP/IVP 后，模型被迫区分：
   - 动作质量
   - 任务语义匹配
   - 语言是否相关

3. **what capability changed**  
   学到的就不再是某个任务的 reward proxy，而是一个能迁移到 **未见任务、未见表达** 的偏好 annotator。

更具体地说：

- **ITP** 提供“同任务内谁更好”的排序锚点
- **ILP** 抑制“只按动作好坏乱排序”的假偏好
- **IVP** 强制建立“语言 ↔ 视频任务语义”的跨任务绑定

这也是为什么作者在消融里发现：  
一旦拿掉 IVP，模型很容易退化成“没有语言条件的普通偏好模型”。

作者还给出理论分析，说明在温和条件下，学习到的偏好分数与轨迹段的负 regret 具有对应关系。这个分析不是论文主卖点，但说明该分数不仅是经验上有用，也有一定决策解释性。

### 战略取舍表

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| expert / medium / random 三层轨迹 | 无人工标签时很难构造稳定偏好 | 自动形成任务内排序监督 | 依赖脚本策略与任务阶段划分启发式 |
| ILP（错配语言下等偏好） | 模型容易把一般“好动作”误判成目标完成 | 学会在语言无关时抑制虚假偏好 | 若任务间共享强语义/技能，可能过度压平差异 |
| IVP（跨任务视频偏好） | 跨任务泛化缺少显式对齐 | 学会任务语义匹配，支持 zero-shot 标注 | 需要多样任务覆盖，否则泛化受限 |
| 单视频-单指令打分 + 成对比较 | 直接 pairwise 输入计算成本高 | 可复用、推理高效、易插入下游 RLHF | 不显式建模 pair 级交互细节 |
| 预训练 CLIP + 轻量跨模态模块 | 从零学视频语言对齐成本大 | 训练成本低，单卡可完成 | 受限于 CLIP 域外偏移与视频理解上限 |

## Part III：证据与局限

### 关键证据信号

#### 1）最强信号：VLP 标签能真正替代 scripted labels
这不是一篇只看“标签准确率”的论文，最关键证据是：  
**VLP 产生的标签能驱动下游策略学到接近甚至更好的行为。**

- 在 5 个 Meta-World 测试任务上：
  - **P-IQL+VLP: 71.0**
  - scripted-label **P-IQL: 62.9**
  - **CPL+VLP: 83.8**
  - scripted-label **CPL: 78.0**

这说明 VLP 不只是“看起来像偏好”，而是作为监督信号对控制学习真的有用。

作者的解释也值得注意：  
某些环境里的 ground-truth reward 本身未必完全对齐任务语义，而 VLP 直接对齐“视频是否符合语言目标”，因此有时反而优于 scripted reward 派生出的标签。

#### 2）相对 VLM reward / VLM preference 基线，VLP 更稳
与 R3M / VIP / LIV / CLIP / VLM-RM / RoboCLIP 等相比：

- VLP 的平均成功率达到 **71.0**
- 明显高于 reward-based 对照中的最佳水平（如 VIP 的 **59.4**）
- 也高于 preference 化后的 VLM 对照最佳值（P-R3M 的 **60.4**）

更关键的是相关性：

- **VLP 标签与 scripted labels 的平均相关/一致性明显更高**
- Table 5 中平均相关性 **0.718**
- 高于 VLM reward 与 ground-truth reward 的最佳组别

这支持一个重要判断：  
**对 embodied manipulation 来说，稳定的“相对偏好”往往比 noisy 的“绝对奖励分数”更适合做下游监督。**

#### 3）泛化与消融支撑了机制解释
泛化结果表明 VLP 不只记住训练任务：

- 未见任务上，Seen / Phrase / Description / Color 变体下的 ITP 准确率大致仍在 **95.8%–97.4%**
- IVP 也维持在 **90%+**

而消融更重要：

- 去掉 IVP（`λ2=0`）后，IVP Acc 从 **91.7** 掉到 **63.0**
- 说明“跨任务视频-语言匹配约束”不是装饰，而是泛化能力的主要因果来源

补充实验里，VLP 在 ManiSkill2 测试任务上的平均标签准确率也达到 **97.9%**，但要注意这部分主要是标签层面的泛化，不是完整策略学习对比。

### 该怎么看这些结果

这篇论文的能力跳跃，不在于又多刷了几个点，而在于它把 prior work 的范式从：

- **VLM 直接出 reward**
- 或 **每个任务都重新收集 preference**

推进到：

- **先学一个跨任务、语言条件化的偏好 annotator，再复用到新任务**

这使得“偏好监督”本身成为可迁移资产，而不是一次性标注品。

### 局限性

- **Fails when**: 任务目标无法被视频和语言充分表达时，例如复杂装配、细粒度空间关系、隐藏状态强依赖的任务；以及语言指令过短、过模糊时，错标风险明显上升。
- **Assumes**: 能获得多任务 scripted policy 生成的 expert/medium/random 轨迹，并能用阶段划分启发式定义“medium”；还假设有语言扩写能力（文中主用 GPT-4V，但附录显示 GPT-3.5 / Llama-3.1-8B 也基本可行）。
- **Not designed for**: 真实机器人端到端部署验证、无文本任务定义场景、复杂长期规划与高精度装配控制；论文也没有把 VLP 作为完整控制架构，只把它作为偏好标注器。

### 资源与可复现性备注

- 训练 VLP 本身相对轻量：约 **单张 RTX 4090，6 小时**
- 但数据构造依赖：
  - 多任务模拟环境
  - scripted policies
  - 任务阶段划分
  - 外部 LLM/VLM 做语言改写
- 论文中**未报告代码/项目链接**，因此虽然方法思路清楚，可复现实操门槛仍不低

### 可复用部件

1. **三类语言条件偏好模板**：ITP / ILP / IVP  
2. **单视频-单语言偏好打分器**：适合做可复用 annotator  
3. **“expert / medium / random + 语言改写”数据配方**：适合低成本构造偏好数据  
4. **与现有离线 RLHF 算法解耦的接口**：可直接接 P-IQL / IPL / CPL

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_VLP_Vision_Language_Preference_Learning_for_Embodied_Manipulation.pdf]]