---
title: "Learning To Explore With Predictive World Model Via Self-Supervised Learning"
venue: arXiv
year: 2025
tags:
  - Others
  - task/reinforcement-learning-exploration
  - task/intrinsic-reward-learning
  - world-model
  - attention
  - modular-rnn
  - dataset/Atari
  - opensource/no
core_operator: "以注意力选择的稀疏模块化循环世界模型同时编码当前状态并预测下一状态特征，用预期破裂误差生成内在奖励。"
primary_logic: |
  像素观测序列 → 卷积编码并用注意力激活少量模块生成当前/期望下一状态潜表示 → 以预期与真实特征差作为内在奖励并用PPO更新策略
claims:
  - "在18个Atari游戏中，该方法在16个游戏上的最终外在得分高于Burda et al. (2018)基线 [evidence: comparison]"
  - "在高反应性环境中，该方法的纯内在奖励探索可显著超过基线，例如Atlantis达到47540±7804，而基线为9800±4475 [evidence: comparison]"
  - "该方法在极稀疏长时序任务Pitfall上仍未取得正分，说明其探索改进不足以完全解决长期信用分配问题 [evidence: comparison]"
related_work_position:
  extends: "BRIMs (Mittal et al. 2020)"
  competes_with: "ICM (Pathak et al. 2017); Large-Scale Study of Curiosity-Driven Learning (Burda et al. 2018)"
  complementary_to: "PPO (Schulman et al. 2017)"
evidence_strength: moderate
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_Learning_To_Explore_With_Predictive_World_Model_Via_Self_Supervised_Learning.pdf
category: Others
---

# Learning To Explore With Predictive World Model Via Self-Supervised Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.13200)（文中代码链接为占位符，未纳入）
> - **Summary**: 该文把注意力门控的稀疏模块化世界模型接到纯内在奖励探索里，用“预测到的下一状态特征”和“真实看到的下一状态特征”之间的偏差来驱动 agent 主动探索。
> - **Key Performance**: 18 个 Atari 游戏中有 16 个最终得分高于 Burda/Pathak 基线；如 Atlantis 47,540±7,804 vs 9,800±4,475，Asterix 2,465±1,236 vs 110±49。

> [!info] **Agent Summary**
> - **task_path**: Atari像素观测序列 + 无外部奖励训练 -> 内在奖励驱动的动作策略
> - **bottleneck**: 单体 curiosity 表征难以在高维、非平稳环境中区分“真正值得探索的变化”和无关像素变化
> - **mechanism_delta**: 把单一预测器换成注意力门控的稀疏模块化循环世界模型，并显式区分当前状态表征与下一状态期望
> - **evidence_signal**: 18个Atari对比中16个游戏最终外在得分高于Burda et al.基线
> - **reusable_ops**: [模块竞争激活, 预期-现实潜特征差奖励]
> - **failure_modes**: [极稀疏长时序任务如Pitfall仍难学好, 单层模块设置不足以验证其层级建模优势]
> - **open_questions**: [模块是否稳定学到可解释子动力学, 多步想象与连续控制场景能否保持收益]

## Part I：问题与挑战

**What/Why：** 这篇论文真正想解决的问题，不是“怎么再造一个 curiosity bonus”，而是**怎样在没有人工奖励的情况下，让 agent 从高维像素里学到对探索真正有用的内部表征**。

传统内在奖励方法有两个常见瓶颈：

1. **直接做像素预测太难**：高维视觉输入里有大量与行动无关、又难以预测的变化。
2. **单体 latent 容易混杂因素**：即便转到特征空间，若所有环境因素都挤在一个隐状态里，敌人、障碍、路线、可交互目标会互相干扰，尤其当策略不断变化时，训练分布也在跟着漂移。

作者的核心判断是：  
**瓶颈不只是“预测误差怎么定义”，而是“世界表征怎么组织”。**  
如果 world model 本身没有模块化、稀疏激活和选择性注意，它生成的 surprise 往往会被无关变化污染，最后导致探索不稳定或和任务目标错位。

**输入/输出接口：**
- **输入**：84×84 灰度 Atari 图像，4 帧堆叠。
- **训练信号**：训练时只用内在奖励；外在奖励只用于评测。
- **输出**：策略网络动作；世界模型给出 intrinsic reward。

**边界条件：**
- 这篇论文虽然标题强调 predictive world model，但**实际并不是用 world model 做显式规划**。
- 更准确地说，它是**“用世界模型造内在奖励”的 PPO 探索方法**，而不是经典意义上的 model-based planning。

## Part II：方法与洞察

**How：** 作者引入的关键因果旋钮，是把 curiosity 模块从“单个前向预测器”改成了**注意力驱动的稀疏模块化循环世界模型**。

### 方法结构

1. **视觉编码器**
   - 当前状态 \(s_t\) 先经过卷积编码，得到特征 \(x_t\)。

2. **模块化世界模型**
   - 用 BRIM/RIM 风格的循环模块做世界建模。
   - 每个时间步只有少量模块被注意力选中并更新，其余模块保持休眠。
   - 论文实验里实际使用的是**单层、8 个模块、每步 4 个激活/期望模块**的配置。

3. **双状态流**
   - 一组模块生成**当前状态表征**，供策略网络选动作。
   - 另一组模块生成**对下一状态的期望表征**，相当于“我预计接下来会看到什么”。

4. **内在奖励**
   - 当真实下一状态到来后，比较“上一步的期望表征”和“当前真实表征”的差异。
   - 差异越大，说明 agent 触发了更大的“预期破裂”，内在奖励越高。

5. **联合训练**
   - 策略网络用 PPO 最大化内在奖励。
   - 世界模型用自监督回归去减小预测与真实特征之间的偏差。
   - 编码器同时接收策略与世界模型的梯度，让 latent 既能预测、又能支持控制。

### 核心直觉

真正的变化链条是：

**从单体预测器 → 到稀疏模块竞争 + 注意力选择 + 当前/期望双表征**  
→ **改变了信息瓶颈：只有相关模块更新，无关视觉变化被更强地过滤，不同环境因素更容易被分配给不同子模块**  
→ **内在奖励更接近“有结构、可控制、和任务相关的意外”**  
→ **探索行为更容易和最终外在得分对齐。**

为什么这设计可能有效：

- **稀疏激活降低干扰**：不是每个因素都要每步重写整个隐状态。
- **模块专门化更适合非平稳环境**：随着策略变化，激活集合可以变，而不是逼一个统一 latent 持续兼容所有阶段。
- **预期破裂比像素误差更“语义化”**：奖励来自结构化潜空间里的 surprise，而不是原始像素噪声。
- **共享编码器让探索与决策耦合**：policy 真正依赖的 latent，也正是被用来计算 novelty 的 latent。

但要注意一个很重要的分析点：  
论文动机反复强调**hierarchy/top-down/bottom-up**，但实验实现实际上只用了**单层模块化 recurrent model**。  
所以本文最扎实验证到的，不是“层级世界模型”，而是**稀疏模块化 + 注意力 + 预测误差奖励**。

### 战略权衡

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 稀疏模块竞争激活 | 降低所有环境因素混在一个隐状态里的干扰 | 更容易聚焦敌人、路径、目标等局部动力学 | 需要设定模块数/激活数，可能出现模块塌缩或分工不稳 |
| 当前态/期望态双流 | 把 novelty 从像素级噪声转成结构化预测误差 | surprise 更像“世界模型被打破”而非单纯视觉变化 | 仍然是一阶预测，长程规划能力有限 |
| 编码器共享给策略与世界模型 | 让表征同时服务控制与探索 | 探索奖励更可能和可执行行为对齐 | 也可能把 world model 过度拉向当前 policy 分布 |
| 纯内在奖励训练 | 不依赖人工 reward engineering | 可直接测试自主探索能力 | 在超稀疏环境里依然可能和真正目标错位 |

## Part III：证据与局限

**So what：** 相比 prior curiosity 方法，这篇论文声称的能力跃迁是：  
**即使完全不看外在奖励，也能学到更贴近任务得分的探索行为。**

### 关键证据信号

1. **标准对比信号：18 个 Atari，16 个胜出**
   - 这是论文最强证据。
   - 代表性结果：
     - Atlantis：47,540 vs 9,800
     - Asterix：2,465 vs 110
     - BattleZone：6,300 vs 1,900
     - MsPacman：1,358 vs 380
   - 说明该内在奖励设计在多个游戏中比 Burda/Pathak curiosity baseline 更能导向高得分行为。

2. **混合反应/规划任务上的收益**
   - MsPacman 被作者作为重点案例：早期偏 dense/reactive，后期更 sparse/deliberative。
   - 该方法学习曲线更高，表明它不仅对“看到新东西”敏感，也更能维持和得分相关的探索。

3. **反应性环境上的明显优势**
   - Asterix、RiverRaid、Centipede、Atlantis 这类环境中，方法表现很强。
   - 这支持了作者的直觉：当场景里充满需要快速反应的可预测/半可预测事件时，模块化注意力更容易抓住“活下来并持续接触新状态”的关键因素。

4. **边界证据：极稀疏长期任务仍然弱**
   - Freeway 有提升，但 Pitfall 仍然是负分。
   - 这说明一步潜空间 surprise 虽然改善了探索，但还不足以跨越最难的长期信用分配问题。

### 局限性

- **Fails when:** 回报极稀疏、需要长期计划和长期信用分配的环境，如 Pitfall；以及 Solaris、Gravitar 这类游戏上最终分数并未超过基线。
- **Assumes:** 依赖 Atari 像素输入、PPO 主干、一步特征预测误差、固定模块数/激活数、128 并行环境和最高 170M steps 训练；另外论文虽声称代码公开，但给出的 GitHub 链接是占位符，当前复现可得性存疑。
- **Not designed for:** 基于世界模型的显式规划、多步 imagination、连续控制机器人场景，或严格证明每个模块学到了可解释子技能。

### 证据强度判断

我会把这篇论文的证据强度定为 **moderate**，原因是：

- **有标准基准对比**，而且覆盖 18 个 Atari，结果不弱；
- 但**缺少关键消融**：无法分辨收益到底来自
  - 模块化，
  - 注意力，
  - 稀疏激活，
  - 编码器共享训练，
  - 还是 PPO/归一化/rollout 等训练细节；
- **只在 Atari 单一域验证**，还没有跨域到连续控制、3D 环境或真实机器人；
- 论文叙述中还存在一些发布层面的粗糙之处（如模板残留、代码链接占位符），会削弱可复现性信号。

### 可复用组件

- **模块竞争激活**：可作为任意 curiosity/world-model 模块的稀疏路由器。
- **预期-现实潜特征差奖励**：适合替代像素误差 novelty。
- **共享编码器联合优化**：让探索表征和控制表征对齐，而不是彼此脱节。

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_Learning_To_Explore_With_Predictive_World_Model_Via_Self_Supervised_Learning.pdf]]