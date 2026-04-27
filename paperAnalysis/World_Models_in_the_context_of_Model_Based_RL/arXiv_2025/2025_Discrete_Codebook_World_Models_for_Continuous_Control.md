---
title: "Discrete Codebook World Models for Continuous Control"
venue: ICLR
year: 2025
tags:
  - Embodied_AI
  - task/continuous-control
  - task/model-based-reinforcement-learning
  - reinforcement-learning
  - quantization
  - model-predictive-control
  - dataset/DMControl
  - dataset/Meta-World
  - dataset/MyoSuite
  - opensource/full
core_operator: 用FSQ把连续状态量化为离散代码本，并以分类式随机潜在动力学替代连续回归式世界模型，再配合MPPI进行连续控制规划
primary_logic: |
  状态观测与动作 → 编码器+FSQ量化为离散代码 → 用交叉熵训练的随机潜在动力学进行多步预测并联合学习奖励 → 在离散潜空间中学习价值/策略并用MPPI规划输出连续动作
claims:
  - "在10个DMControl与10个Meta-World任务的潜空间消融中，离散代码本潜空间比连续潜空间具有更高样本效率，而最佳结果来自离散+随机+分类式动力学 [evidence: comparison]"
  - "相较于label编码与one-hot编码，代码本编码在连续控制中更稳定且更高效：label编码学习明显更差，one-hot虽有时接近样本效率但训练更慢 [evidence: comparison]"
  - "将DCWM替换进TD-MPC2后，aggregate表现进一步提升，说明收益主要来自离散随机代码本潜空间而不只是整体算法包装 [evidence: ablation]"
related_work_position:
  extends: "TD-MPC2 (Hansen et al. 2023)"
  competes_with: "TD-MPC2 (Hansen et al. 2023); DreamerV3 (Hafner et al. 2023)"
  complementary_to: "PETS (Chua et al. 2018); STORM (Zhang et al. 2023)"
evidence_strength: strong
pdf_ref: "paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_Discrete_Codebook_World_Models_for_Continuous_Control.pdf"
category: Embodied_AI
---

# Discrete Codebook World Models for Continuous Control

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.00653), [Project](https://www.aidanscannell.com/dcmpc), [Code](https://github.com/aidanscannell/dcmpc)
> - **Summary**: 这篇工作把连续控制 world model 的潜变量从“连续向量回归”改成“FSQ 代码本上的离散随机码分类”，在不做观测重建的前提下提升了多步动力学学习与规划的样本效率。
> - **Key Performance**: DMControl 30任务在 1M steps 的 aggregate normalized return 整体强于 TD-MPC2；Meta-World 45任务在 1M steps 的 episode success 整体与 TD-MPC2 持平并明显优于 DreamerV3。

> [!info] **Agent Summary**
> - **task_path**: 状态观测序列 + 连续动作条件的模型式RL -> 规划得到下一步连续控制动作
> - **bottleneck**: 连续控制里既要保留状态的局部顺序结构，又要稳定建模多步潜在转移不确定性；连续MSE潜空间和naive离散编码都难同时做到
> - **mechanism_delta**: 用FSQ代码本离散化潜状态，并把潜在动力学改成基于交叉熵训练的随机分类器，再在规划时使用期望代码降低方差
> - **evidence_signal**: 20任务潜空间消融 + “把DCWM装进TD-MPC2”的替换实验都显示离散随机代码本潜空间带来稳定收益
> - **reusable_ops**: [FSQ代码本量化, 类别式潜在动力学+ST Gumbel-softmax多步训练]
> - **failure_modes**: [加入观测重建会在Dog/Humanoid等难任务明显掉点, 用label或one-hot替换代码本编码会导致学习变差或计算变慢]
> - **open_questions**: [这一优势能否迁移到像素观测或Transformer world model, 如何去掉任务特定的噪声日程与N-step调参]

## Part I：问题与挑战

这篇论文研究的是**状态输入的连续控制 world model**：给定当前观测和候选动作序列，模型要在潜空间里预测未来转移与奖励，并据此做 decision-time planning，输出连续动作。

### 真正的难点是什么？

作者聚焦的核心不是“world model 是否有用”，而是：

1. **连续控制里，离散 latent 到底值不值得用？**  
   DreamerV2/V3 说明离散 latent 在一些场景有效，但在**状态型连续控制**上，TD-MPC2 这类**连续 latent + MSE 回归**方法更强。于是关键问题变成：离散 latent 的收益是否真的存在，还是只是某些特定架构的副产物？

2. **怎样离散化，才不会破坏连续状态的几何结构？**  
   连续控制的状态天然带有“邻近/顺序”关系。  
   - one-hot：所有类别彼此等距，几乎抹掉这种结构；  
   - label：只有单轴排序，容易施加错误的全局顺序；  
   - 真正需要的是一种**低维、稠密、保留局部顺序**的离散表示。

3. **多步潜在动力学为什么难学？**  
   连续 latent 的点估计回归很容易把下一状态“平均化”，多步 rollout 时误差会持续积累。作者怀疑，若把转移建模改成**分布预测**而不是点回归，world model 会更稳。

### 输入/输出接口与边界

- **输入**：状态观测、连续动作、经验回放中的转移样本。
- **中间表征**：离散代码本中的 latent code。
- **输出**：
  - 训练时：下一潜状态分布与奖励预测；
  - 控制时：通过 MPPI 规划得到下一步动作。
- **边界条件**：
  - 主要是**state-based continuous control**，不是像素重建主导的设定；
  - 评估集中在 DMControl、Meta-World、MyoSuite；
  - 环境本身多为确定性模拟器，但作者仍使用**随机潜在动力学**表示表征层不确定性。

### 为什么现在做这件事？

因为近来的 RL 和生成建模都给了动机：

- value learning 中，**classification 可能比 regression 更稳**；
- 生成模型里，**离散 codebook** 已被反复验证有效；
- 但这些 insight 还没有被系统验证到**连续控制 world model 的 latent design**上。

所以这篇论文实际上回答的是一个非常关键的“瓶颈拆解”问题：  
**连续控制 world model 的能力差异，究竟来自离散化本身、分类式训练、还是随机/多模态动力学？**

## Part II：方法与洞察

方法由两部分组成：

- **DCWM**：Discrete Codebook World Model
- **DC-MPC**：基于 DCWM 的 model predictive control 算法

### 方法主线

1. **观测编码**
   - 用 MLP 编码器把状态观测映射到连续 latent。

2. **FSQ 代码本量化**
   - 使用 finite scalar quantization, FSQ，把连续 latent 压成离散代码。
   - 这里不是 Dreamer 的 one-hot，也不是单个 label，而是**多维代码向量**。
   - 另外，FSQ 的代码本是固定的，不需要 VQ 那类额外的 codebook learning loss，因此早期训练更稳定。

3. **类别式随机潜在动力学**
   - 给定当前 code 和动作，动力学模型不再回归“下一个连续 latent 向量”，
   - 而是输出“下一个 code 属于代码本中各个候选 code 的概率分布”。
   - 损失是**交叉熵**，不是 MSE。

4. **多步训练时做随机采样**
   - 用 ST Gumbel-softmax 在 rollout 中采样下一 code；
   - 这样既能让不确定性沿时间传播，也能让梯度反传回编码器。

5. **奖励、价值、策略**
   - reward model 与 world model 联合学习；
   - actor/critic 在离散 latent 上单独用 TD3 + N-step + REDQ ensemble 学习；
   - 作者明确**不把 value prediction 混进表征学习**，避免表示被 critic 目标绑架。

6. **规划时降低随机性**
   - 训练时采样，规划时不采样；
   - 直接用**下一代码的期望值**做 rollout，降低 MPPI 评估方差。
   - 之所以能这么做，是因为 codebook 具有一定顺序结构，期望值可被理解为在相邻代码间插值。

### 核心直觉

这篇论文真正拧动的“因果旋钮”不是网络大小，而是**潜在动力学的建模对象**。

它把 world model 从：

- **连续 latent 上的单点回归**

改成：

- **有序离散代码本上的分布分类**

于是能力链条变成：

**连续状态观测**  
→ **被压缩到有结构的离散码**  
→ **下一状态从“回归一个点”变成“预测一个分布”**  
→ **多步 rollout 不再总被平均化误差牵着走**  
→ **规划与价值学习得到更稳定的潜空间**  
→ **样本效率提升，尤其在高维 locomotion 任务里更明显**

#### 为什么这个设计有效？

**1. codebook 解决了“离散但仍有几何结构”的矛盾。**
- one-hot 太稀疏、太高维，而且所有类别等距；
- label 过于粗暴，只保留一维排序；
- codebook 则能用低维稠密向量表达多维局部顺序，更符合连续控制状态的结构。

**2. 分类式随机动力学解决了“多步预测只学到平均值”的问题。**
- 交叉熵训练的是一个候选下一状态分布，而不是单点；
- ST Gumbel-softmax 让多步训练真正看到“采样后的潜在未来”；
- 即使外部环境是确定的，latent 压缩本身也带来不确定性，因此随机潜在转移仍然合理。

**3. 去掉观测重建，避免表示学到无关细节。**
- 对连续控制来说，很多观测细节并不影响任务；
- 如果逼模型去重建全部观测，表示容量会被这些无关因素消耗掉。

### 战略权衡

| 设计选择 | 带来的能力变化 | 代价 / 风险 |
|---|---|---|
| FSQ代码本替代连续latent | 表示变得离散、紧凑、可分类训练，并保留局部顺序 | 需要选择 codebook 大小与量化级别 |
| 类别式随机动力学替代 MSE 回归动力学 | 更好表达下一潜状态不确定性，减轻多步误差平均化 | 训练更复杂，依赖 ST Gumbel-softmax 稳定性 |
| 去掉观测重建 | 聚焦控制相关信息，而非还原全部观测细节 | 若任务真的依赖细粒度重建，此假设未必成立 |
| 规划时用期望 code 而非采样 | 降低 MPPI rollout 方差，规划更稳 | 期望值不一定落在离散代码流形上 |

### 相对 prior work 的位置

- **相对 DreamerV3**：保留“离散 latent 可能有益”的大方向，但放弃 one-hot + reconstruction，改成 codebook + decoder-free consistency。
- **相对 TD-MPC2**：保留 decoder-free 与 MPC 的强框架，但把 latent 从连续回归改成离散随机分类。
- **相对一般量化模块**：这里的 quantization 不是一个附属正则，而是整个 dynamics / reward / planning 接口的核心表示。

## Part III：证据与局限

### 关键实验信号

#### 1. 潜空间消融直接说明：最佳配置是“离散 + 随机 + 分类”
Fig. 3 与 Fig. 9 是最关键证据：

- **离散 latent** 明显比 **连续 latent** 更高样本效率；
- 单纯离散化还不够，最佳结果来自：
  - 离散 codebook
  - 随机潜在动力学
  - 交叉熵训练
- 连续 latent 上即使用 Gaussian 或 GMM 建模随机性，也不能稳定复现这一收益，尤其在 Meta-World 上明显不如离散方案。

这说明收益并不只是“加了 stochasticity”，而是**结构化离散表示 + 分类式转移学习**的组合起作用。

#### 2. 编码方式消融说明：不是任何离散编码都行
Fig. 4 的结论很清晰：

- **label encoding** 学习明显更差，特别是在 Humanoid Walk；
- **one-hot encoding** 某些任务样本效率接近，但训练时间明显更长；
- 作者还提到：如果让 dynamics 也直接吃 one-hot 或 label，训练曲线会几乎 flat-line，所以后续消融只在 reward/value/policy 侧替换编码。

这说明论文的有效点不是“把 latent 离散化”这么粗，而是**用 codebook 保住了连续控制需要的多维局部结构，同时避免 one-hot 的高维稀疏代价**。

#### 3. Benchmark 结果说明：方法级竞争力成立
在三套基准上：

- **DMControl 30 tasks**：整体 aggregate 很强，尤其 Dog / Humanoid 等高维 locomotion 任务突出；
- **Meta-World 45 tasks**：整体与 TD-MPC2 接近，并明显强于 DreamerV3、SAC、TD-MPC；
- **MyoSuite 5 tasks**：整体与 TD-MPC2 相当。

所以这篇论文的“so what”不是全面碾压所有 baseline，而是：  
**离散随机 codebook world model 终于在 state-based continuous control 上，和 TD-MPC2 这类强连续 latent 方法正面打平甚至局部超越。**

#### 4. 替换实验隔离了 latent 设计的贡献
Fig. 6 非常重要：

- 把 DCWM 的离散随机 latent 装进 TD-MPC2 后，aggregate 表现还能继续提升；
- 这说明收益不是“整套 DC-MPC 的工程偏置”，而是**latent design 本身有可迁移价值**。

#### 5. 反证实验说明：观测重建在这里是负担
Fig. 20 表明：

- 给 DC-MPC 加上 observation reconstruction，性能不升反降；
- 在 Dog Run、Humanoid Walk 这类难任务上甚至显著伤害学习。

这直接支持论文的判断：  
在这类连续控制任务里，真正瓶颈不是“把观测重建得更像”，而是“把控制相关的潜在转移学得更稳”。

### 1-2 个最值得记住的指标

- **DMControl @ 1M env steps**：aggregate 的 **IQM / median normalized return** 整体优于 TD-MPC2。
- **Meta-World @ 1M env steps**：aggregate **episode success** 整体与 TD-MPC2 持平，并显著优于 DreamerV3。

### 局限性

- **Fails when**: 强行加入观测重建、或把代码本编码替换成 label / one-hot 接入该自监督 world model 接口时，难任务上会明显掉点，甚至学习曲线近乎停滞。
- **Assumes**: 依赖 state-based continuous-control 设定、MPPI 决策时规划、MLP backbone、任务相关的 exploration noise schedule，以及部分任务上手工调过的 N-step returns；说明它还不是“单套超参 everywhere”的方案。
- **Not designed for**: 像素重建驱动的 world model、超大规模通用多 embodiment world model、以及严格实时约束下要求极低决策延迟的部署场景。

### 复现与扩展时的现实约束

- 论文已开源，但推理不是纯 actor-only：
  - 每个环境步都要做 MPPI；
  - 默认 population 512、迭代 6 次，在线决策成本明显高于纯策略方法。
- 训练使用单 GPU 完成，硬件为 A100/MI250X；不过性能仍对**噪声日程**和**N-step 选择**有一定依赖。
- 当前结论主要建立在**状态输入 + MLP world model**上，迁移到 Transformer、diffusion、像素输入时仍需额外验证。

### 可复用组件

- **FSQ latent quantization**：适合把连续状态压成低维离散代码本。
- **类别式潜在动力学 + ST Gumbel-softmax**：适合多步 latent rollout 的分布建模。
- **planning-time expected code**：把随机 world model 用于稳定 MPC 的实用技巧。
- **表征学习与价值学习解耦**：避免 critic 目标直接干扰 latent representation。

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_Discrete_Codebook_World_Models_for_Continuous_Control.pdf]]