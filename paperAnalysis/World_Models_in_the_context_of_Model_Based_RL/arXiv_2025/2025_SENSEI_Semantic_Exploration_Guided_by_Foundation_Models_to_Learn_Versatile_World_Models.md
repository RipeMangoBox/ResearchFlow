---
title: "SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models"
venue: ICML
year: 2025
tags:
  - Others
  - task/task-free-exploration
  - task/model-based-reinforcement-learning
  - reinforcement-learning
  - world-model
  - preference-learning
  - dataset/MiniHack
  - dataset/Robodesk
  - "dataset/PokémonRed"
  - opensource/full
core_operator: "先用 VLM 对图像对的“有趣性”偏好进行蒸馏得到语义内在奖励，再让世界模型同时预测该语义奖励与不确定性，并以门控式 go/explore 策略把探索集中到更有意义的状态及其邻域。"
primary_logic: |
  自监督初始探索数据 + 环境简述 + 图像观测/低层动作
  → VLM 对观测对进行有趣性比较并蒸馏出语义奖励函数，RSSM 在潜空间内联合预测语义奖励与模型分歧
  → 策略先到达语义上有意义的状态，再从这些状态向外扩展探索，得到更丰富且更可迁移的世界模型
claims:
  - "Claim 1: On MiniHack KeyRoom-S15 and KeyChest, SENSEI discovers more task-relevant interactions and collects more sparse environment rewards during task-free exploration than Plan2Explore [evidence: comparison]"
  - "Claim 2: Removing the uncertainty/disagreement bonus causes semantic-only VLM-MOTIF exploration to get stuck in local optima, such as hovering near high-interest states without completing the task [evidence: ablation]"
  - "Claim 3: World models pretrained with SENSEI support faster downstream policy learning than Plan2Explore-pretrained world models and from-scratch baselines; in KeyRoom the paper reports roughly two orders-of-magnitude better sample efficiency than PPO [evidence: comparison]"
related_work_position:
  extends: "MOTIF (Klissarov et al. 2023)"
  competes_with: "Plan2Explore (Sekar et al. 2020); RL-VLM-F (Wang et al. 2024)"
  complementary_to: "TD-MPC2 (Hansen et al. 2024); DayDreamer (Wu et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_SENSEI_Semantic_Exploration_Guided_by_Foundation_Models_to_Learn_Versatile_World_Models.pdf
category: Others
---

# SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.01584), [Project](https://sites.google.com/view/sensei-paper), [Code](https://github.com/martius-lab/sensei)
> - **Summary**: 这篇工作把 VLM 的“人类觉得什么值得探索”先蒸馏成语义奖励，再让世界模型在想象中预测该奖励并与不确定性联动，从而在纯图像、低层动作环境里学到更有意义、对下游任务更有迁移性的 free-play 世界模型。
> - **Key Performance**: MiniHack KeyRoom 下游学习相对 PPO 约提升 2 个数量级样本效率；Pokémon Red 的 750k-step 探索中仅 SENSEI 到达首个 Gym，二次标注后还能击败 Brock 获得 Boulder Badge。

> [!info] **Agent Summary**
> - **task_path**: 图像观测 + 低层动作 + 环境简述/初始探索数据 -> 语义引导探索策略与可迁移世界模型
> - **bottleneck**: 纯新奇或信息增益会把探索预算花在低层但无任务价值的行为上，难以优先进入真正“有意义”的状态
> - **mechanism_delta**: 用 VLM 成对偏好蒸馏有趣性奖励，并把该奖励做成世界模型内可预测的量，再用分位数门控在“去往有趣状态”和“从那里向外探索”之间切换
> - **evidence_signal**: 跨 MiniHack、Robodesk、Pokémon 的比较实验，加上去掉不确定性奖励后的消融，共同支持“语义奖励 + 不确定性”优于任何单一信号
> - **reusable_ops**: [pairwise-vlm-preference-distillation, latent-semantic-reward-head, quantile-gated-go-explore-switch]
> - **failure_modes**: [semantic-reward-ood-drift, vlm-judgment-degrades-under-occlusion]
> - **open_questions**: [how-to-refresh-annotations-online, can-it-scale-to-real-robotics-without-multi-view-hacks]

## Part I：问题与挑战

这篇论文真正要解决的，不是“怎么让 agent 更广地乱逛”，而是：

**在没有手工任务奖励、只有像素观测和低层动作时，怎样让探索优先落到那些更可能包含任务结构的状态上？**

### 1) 真正瓶颈是什么

传统 intrinsic motivation，例如 novelty、surprise、information gain，确实能推动 agent 去“没见过”的地方，但它们容易发现的是：

- 低层可动性；
- 纯物理扰动；
- 与未来任务弱相关的细碎变化。

论文给出的核心判断是：  
**“新奇”不等于“有用”。**

对世界模型学习尤其如此。因为如果 free-play 阶段主要覆盖的是“无意义的新奇”，那么后面即便世界模型很会预测，也只是很会预测无关动态，而不是任务相关动态。

### 2) 为什么现在值得做

作者抓住了两个时间点红利：

1. **VLM/LLM 已经内化了大量人类常识偏好**  
   它们能大致判断“哪一帧更值得注意/更有意思”。

2. **模型式 RL 已经足够成熟**  
   例如 DreamerV3/RSSM 这类世界模型，可以把外部反馈内化为潜空间中的可预测信号，再在 imagination 里用于策略优化。

所以现在不只是“拿 foundation model 打标签”，而是可以把这种标签变成 agent 内部的探索偏置。

### 3) 现有 foundation-model-guided exploration 为什么还不够

作者明确批评了几类先前做法的假设过强：

- 需要**语言接地环境**；
- 需要**离线且覆盖充分的数据集**；
- 需要**高层动作接口**；
- 依赖 foundation model 持续在线指导，而**没有内部的 interestingness 模型**。

SENSEI 的目标，是把这些前提削弱到更贴近机器人/像素控制现实的版本：

- 输入是**图像观测**；
- 控制是**低层动作**；
- 初始数据来自**自监督探索**而非专家数据；
- 有趣性最终进入**世界模型内部**，而不是始终靠外部 VLM 在线判定。

### 4) 输入/输出接口与边界

**输入：**

- 初始探索数据集 \(Dinit\)，由自监督探索得到；
- 一个预训练 VLM；
- 一个简短环境描述（可人工给，也可由 VLM 自动生成）；
- 在线交互时的图像观测与低层动作。

**输出：**

- 一个语义引导的探索策略；
- 一个覆盖更丰富、对下游任务更有用的世界模型。

**边界条件：**

- 不要求语言化环境；
- 不要求高层 action；
- 不要求专家 demonstrations；
- 但要求图像里能较充分地显露“有意义事件”，否则 VLM 判断会退化。

---

## Part II：方法与洞察

SENSEI 的设计不是单纯“把 VLM 奖励加到 RL 里”，而是三步闭环：

1. **先把“有趣性”蒸馏出来；**
2. **再把“有趣性”内化进世界模型；**
3. **最后用“先去有趣处、再从那里扩展”的策略做探索。**

### 方法主线

#### A. 用 VLM 成对比较图像，蒸馏语义奖励

作者先用 Plan2Explore 收一批初始轨迹，然后从中采样观测对，问 VLM：

> 这两张图里，哪一张对这个环境来说更“有意思”？

这一步有两个关键点：

- 用的是**pairwise preference**，不是直接让 VLM 打绝对分数；
- 比较目标是**interestingness**，不是某个显式任务奖励。

接着把这些偏好蒸馏成一个奖励模型 \(R_\psi\)，使 agent 以后看到单张观测时就能得到语义奖励，而不必持续查询 VLM。

这一步相当于把人类常识偏好压缩成一个环境内可复用的 reward prior。

#### B. 在世界模型里学习“内部有趣性模型”

SENSEI 使用 DreamerV3 风格的 RSSM 作为世界模型。和普通世界模型不同的是，它不只预测：

- 观测；
- episode continuation；
- 外部奖励；

还额外预测：

- **语义奖励 \(rsem\)**。

这一步非常关键。因为一旦语义奖励进入潜空间预测头，agent 在 imagination/planning 时就能直接评估：

- 哪些未来状态**看起来有意义**；
- 哪些未来状态**模型还拿不准**。

这就是论文所谓的 **internal model of interestingness**。

#### C. 不是固定加权，而是门控式 Go-and-Explore

如果只最大化语义奖励，agent 容易卡在一些“看起来很有意思”的局部最优。  
如果只最大化不确定性，又会回到纯新奇探索。

所以作者没有用固定权重，而是做了一个**分位数门控切换**：

- 当状态还不够“有趣”时：主要追求**到达有趣状态**；
- 一旦到达高 interestingness 区域：更强调**从这里向外探索新行为**。

这和 Go-Explore 的思想一致：  
**先 go 到一个好地方，再 explore。**

---

### 核心直觉

**改了什么：**  
把探索目标从“在任何地方追求新奇”改成“先去 VLM 认为更有意义的状态，再在这些状态附近追求新奇”。

**改变了哪个瓶颈：**  
把探索预算从大量低层、弱任务相关的状态，重新分配到更可能包含对象交互、工具使用、战斗进展等结构化动态的区域；同时把外部 VLM 反馈变成了世界模型内部可想象、可规划的信号。

**带来了什么能力变化：**  
agent 不再只是学会“动起来”，而是更容易学到：

- 拿钥匙、开门、开箱子；
- 操作抽屉、按钮、积木；
- 在 Pokémon 里推进地图与战斗进度。

更重要的是，这些经历让世界模型覆盖了**更有下游价值的转移结构**。

### 为什么这个设计在因果上有效

1. **VLM 提供了语义先验，缩小搜索空间**  
   不是所有新奇都值得追。VLM 的偏好把搜索压到“人类觉得可能有后续价值”的区域。

2. **奖励蒸馏避免在线调用 foundation model**  
   语义评判先离线蒸馏，在线阶段只用世界模型预测，成本更低，也能在 imagined rollouts 中使用。

3. **不确定性奖励防止语义塌缩到局部最优**  
   语义奖励把 agent 带到“好地方”，disagreement 让 agent 别停在那儿摆拍，而是继续试探未见过的后续动态。

4. **世界模型中的语义预测提升迁移性**  
   最后学到的不只是策略，而是一个知道“哪里有意义”的 dynamics model，所以能更快服务下游任务学习。

### 策略权衡

| 设计选择 | 收益 | 代价 / 风险 |
|---|---|---|
| VLM 成对偏好蒸馏，而非手工写 reward | 引入人类常识偏置，减少任务工程 | 标注有噪声，依赖 prompt 与闭源 API |
| 在 RSSM 里直接预测语义奖励 | 语义信号可进入 imagination，提升样本效率 | 超出标注分布时会 reward drift |
| 分位数门控的 go/explore，而非固定加权 | 先到关键状态、再从关键状态扩展，更不易卡住 | 需要阈值和权重超参 |
| 初始数据来自自监督探索，而非专家数据 | 更现实，不依赖 demonstrations | 初始数据越贫乏，后续语义蒸馏越受限 |
| 自动生成环境描述（SENSEI GENERAL） | 减少人工先验注入 | 泛化仍受 VLM 认知质量限制 |

---

## Part III：证据与局限

### 关键证据

- **比较信号 / MiniHack**  
  在 KeyRoom-S15 和 KeyChest 上，SENSEI 比 Plan2Explore 发现了更多关键交互，如拿钥匙、开门、带钥匙到箱子处，并在 free-play 中收集到更多环境稀疏奖励。  
  **结论**：语义偏置确实把探索推向了任务相关状态，而不是仅仅增加状态覆盖。

- **消融信号 / 去掉不确定性奖励**  
  纯 VLM-MOTIF（只追语义奖励）会卡在局部最优：例如到达箱子附近但不去触发真正完成任务的动作。  
  **结论**：semantic usefulness 和 novelty 不能二选一，二者组合才构成持续推进的探索。

- **比较信号 / Robodesk**  
  SENSEI 在 1M 步探索中，对多数对象的交互次数高于 Plan2Explore 和 RND；而且 SENSEI GENERAL 在不人工写环境描述时也大体保留了效果。  
  **结论**：该方法不只适用于离散游戏，也能在连续控制、遮挡存在的机器人场景中工作。

- **比较信号 / 下游任务学习**  
  用 SENSEI free-play 学到的世界模型初始化 DreamerV3，在 MiniHack 和 Robodesk 的下游任务学习都更快；KeyRoom 中论文明确指出相对 PPO 约有两个数量级样本效率优势。  
  **结论**：SENSEI 的收益不止是“探索时更热闹”，而是确实学到了更有用的 world model。

- **比较信号 / Pokémon Red**  
  在 750k-step 的 task-based exploration 中，仅 SENSEI 到达首个 Gym；进一步做第二轮标注更新后，还能击败 Brock 获得 Boulder Badge。  
  **结论**：语义探索在超长链条、开放式环境里比纯外部奖励或纯不确定性更能推进目标相关进展。

### 两个最值得记住的结果

1. **下游学习不是小修小补，而是量级差异**  
   KeyRoom 上，PPO 需要超过 20M steps 才能稳定解题；SENSEI 预训练世界模型让 Dreamer 在 1M 量级内就显著推进，论文将其概括为约 **100x 级别** 的样本效率收益。

2. **复杂开放世界里，SENSEI 能走到 baseline 到不了的地方**  
   Pokémon Red 中，只有 SENSEI 到达第一个 Gym；这比单纯“分数更高”更说明它真的改变了探索轨迹。

### 局限性

- **Fails when:** 图像不能充分表达语义事件时会失败，尤其是遮挡、部分可观测、单帧难判断时序因果的场景；另外一旦 agent 进入远离初始标注分布的区域，语义奖励会出现 OOD 漂移，Pokémon 的二次标注实验正说明了这一点。

- **Assumes:** 依赖预训练 VLM/GPT-4 进行 100K–200K 级别的成对标注，依赖 500K–1M 级别的初始自监督探索数据，依赖 DreamerV3/RSSM 类世界模型与对应算力；Robodesk 还通过多视角相机缓解遮挡，因此可复现性会受到闭源 API、标注成本、高分辨率渲染和多视角设置的影响。

- **Not designed for:** 完全无法从视觉上判断“有意义事件”的环境；需要严格安全保障、不能容忍自由探索试错的真实系统；以及以长期语言交互/高层符号规划为主要接口的场景。

### 可复用组件

- **VLM pairwise preference distillation**  
  适合把“人类偏好/语义价值”从图像转成可训练的 reward model。

- **世界模型里的 semantic reward head**  
  很适合任何需要把外部反馈内化到 imagination/planning 的 model-based RL 系统。

- **分位数门控的 go/explore 机制**  
  比固定加权更适合“先到关键区域，再局部扩展”的探索任务。

- **迭代式再标注**  
  当 agent 走到初始数据外的区域时，可以把新数据回流到 VLM 再蒸馏，这是很自然的 active relabeling 闭环。

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_SENSEI_Semantic_Exploration_Guided_by_Foundation_Models_to_Learn_Versatile_World_Models.pdf]]