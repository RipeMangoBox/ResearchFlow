---
title: 'DAPO: An Open-Source LLM Reinforcement Learning System at Scale'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- DAPO：大规模LLM强化学习开源系统
- DAPO
- The DAPO algorithm with four key te
acceptance: Poster
cited_by: 1591
method: DAPO
modalities:
- Text
paradigm: Reinforcement Learning
followups:
- 基于单次不确定性估计的高效RL数_UFO-RL
---

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**Topics**: [[T__Math_Reasoning]], [[T__Reinforcement_Learning]] | **Method**: [[M__DAPO]] | **Datasets**: [[D__AIME_2024]]

> [!tip] 核心洞察
> The DAPO algorithm with four key techniques (Overlong Filtering, Clip-Higher, Soft Overlong Punishment, Token-level Loss, and Dynamic Sampling) enables reproducible, state-of-the-art large-scale LLM RL training, achieving 50 points on AIME 2024 with Qwen2.5-32B using only 50% of DeepSeek-R1-Zero's training steps.

| 中文题名 | DAPO：大规模LLM强化学习开源系统 |
| 英文题名 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.14476) · [Code](待开源) · [Project](待补充) |
| 主要任务 | Math Reasoning, Reinforcement Learning |
| 主要 baseline | GRPO, DeepSeek-R1-Zero, PPO |

> [!abstract] 因为「开源社区难以复现DeepSeek-R1-Zero的长思维链RL训练效果，且naive GRPO在AIME 2024上仅得30分」，作者在「GRPO」基础上改了「四重机制：Overlong Filtering、Clip-Higher、Soft Overlong Punishment、Token-level Loss + Dynamic Sampling」，在「AIME 2024」上取得「50分（Qwen2.5-32B），超越DeepSeek-R1-Zero-Qwen-32B的47分，且仅用50%训练步数」

- **关键性能 1**: DAPO在AIME 2024上达到50分，相比naive GRPO的30分提升+20分（+66.7%）
- **关键性能 2**: 相比DeepSeek-R1-Zero-Qwen-32B的47分提升+3分，训练步数减少50%
- **关键性能 3**: 渐进消融显示Dynamic Sampling单组件贡献最大（42→50，+8分）

## 背景与动机

当前最先进的推理大模型（如OpenAI o1、DeepSeek-R1）依赖大规模强化学习（RL）来激发长思维链（long-CoT）能力，但关键训练细节被隐藏在技术报告中，导致开源社区难以复现。一个具体例子是：DeepSeek-R1-Zero在Qwen-32B上达到47分AIME 2024成绩，但当研究者用公开描述的GRPO算法复现时，只能得到30分——差距巨大且原因不明。

现有方法如何处理这一问题？**GRPO** [2] 通过组内奖励归一化（group reward normalization）估计优势，无需训练critic网络，简化了PPO的流程；**DeepSeek-R1-Zero** [2] 在此基础上通过大规模RL训练自发涌现长CoT能力，但未公开完整的超参数和训练技巧；**PPO** [21] 作为基础算法，使用对称裁剪的clipped surrogate objective约束策略更新幅度。

这些方法为何不足？作者识别出三个关键失败模式：（1）**熵崩溃（entropy collapse）**：长CoT生成中模型迅速降低探索，陷入局部最优；（2）**奖励噪声**：二值化正确/错误奖励过于稀疏，且超长响应无意义膨胀长度却不改善正确率；（3）**训练不稳定**：序列级梯度聚合掩盖了不同位置的学习信号，而零梯度样本浪费大量计算。这些问题的叠加使得naive GRPO在长CoT场景下性能远逊于报告值。

本文提出DAPO算法，通过系统级的四重机制重新设计GRPO的训练流程，在开源可复现的前提下实现超越DeepSeek-R1-Zero的性能。

## 核心创新

核心洞察：长CoT RL训练的瓶颈并非单一算法组件的缺陷，而是数据质量、探索-利用平衡、信用分配粒度、奖励塑形之间的系统级耦合失效；通过非对称裁剪释放探索空间、token级梯度保留细粒度信号、动态采样消除无效数据、软惩罚引导长度行为，四者协同才能使大规模RL训练稳定收敛到SOTA性能。

| 维度 | Baseline (GRPO) | 本文 (DAPO) |
|:---|:---|:---|
| 探索策略 | 对称裁剪 ε=0.2，限制概率增减同等幅度 | **Clip-Higher**: ε_low=0.2, ε_high=0.28，允许更大概率上升 |
| 信用分配 | 序列级梯度聚合，单一样本单一样本 | **Token-level Loss**: 逐token计算重要性采样比率和梯度 |
| 奖励设计 | 二值化正确/错误奖励 | **Soft Overlong Punishment**: 基于长度塑形奖励，配合硬过滤 |
| 数据管道 | 全部采样数据用于更新 | **Dynamic Sampling**: 过滤零梯度样本，重新采样维持有效batch |
| 训练稳定性 | 易出现熵崩溃、长度失控 | 四机制协同，长度健康增长，收敛步数减半 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/dff65c5b-f071-42e7-8bc2-eaeeb3acf5bf/figures/fig_001.png)
*Figure: Algorithm 1*



DAPO的训练流程基于GRPO框架，但在数据收集、奖励计算、策略更新三个阶段插入四个新组件，形成七步数据流：

1. **Prompt Batch输入**：512个数学问题提示进入系统
2. **Policy Model Rollout**：策略模型 π_θ 为每个提示采样16个响应（G=16）
3. **Overlong Filtering（新）**：移除超过20,480 token阈值的超长响应，防止无意义长度膨胀污染训练
4. **Group Reward Computation**：对过滤后的响应组计算正确性奖励，组内归一化得优势估计 Â_i,t
5. **Soft Overlong Punishment（新）**：对长度接近16,384 token的响应施加软惩罚，在16,384-20,480缓冲区渐变增强
6. **Dynamic Sampling Filter（新）**：检测并过滤零梯度样本（组内所有响应奖励相同导致优势为零），重新采样维持有效batch size
7. **Token-level Policy Gradient with Clip-Higher（新）**：逐token计算重要性采样比率 r_t(θ)，应用非对称裁剪 (ε_low=0.2, ε_high=0.28) 更新策略

```
Prompts (512) → Rollout (16× responses) → Overlong Filtering 
    → Group Reward Norm → Soft Overlong Punishment 
    → Dynamic Sampling → Token-level Loss + Clip-Higher → Updated π_θ
```

## 核心模块与公式推导

### 模块 1: Clip-Higher（对应框架图 策略更新阶段）

**直觉**: 标准PPO/GRPO的对称裁剪同等限制概率增减，在长CoT中过早抑制探索导致熵崩溃；放宽"增加概率"的上界可维持健康探索。

**Baseline 公式** (PPO/GRPO): 
$$L^{CLIP}_{base}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}\left(r_t(\theta), 1-\varepsilon, 1+\varepsilon\right)\hat{A}_t\right)\right], \quad \varepsilon=0.2$$

符号: $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为重要性采样比率；$\hat{A}_t$ 为优势估计；$\varepsilon$ 控制裁剪区间。

**变化点**: 对称裁剪使策略增加和减少概率的代价相同，但长CoT生成需要早期积极探索以发现有效推理模式；单一$\varepsilon$无法区分"鼓励好的探索"与"抑制过度偏离"。

**本文公式（推导）**:
$$\text{Step 1}: \text{将单参数拆分为非对称双参数 } \varepsilon_{low}=0.2, \varepsilon_{high}=0.28 \quad \text{（允许更大的概率上升幅度）}$$
$$\text{Step 2}: \text{保持下界约束防止策略崩溃，上界放宽促进探索}$$
$$\text{最终}: L^{CLIP}_{DAPO}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}\left(r_t(\theta), 1-\varepsilon_{low}, 1+\varepsilon_{high}\right)\hat{A}_t\right)\right]$$

**对应消融**: Table 1显示添加Clip-Higher后AIME 2024从36提升至38（+2分）。

---

### 模块 2: Token-level Loss（对应框架图 策略更新阶段）

**直觉**: 序列级梯度聚合将长序列中不同位置的学习信号混为一谈，某些token需要增强、某些需要抑制，统一处理导致训练不稳定。

**Baseline 公式** (GRPO序列级):
$$L^{PG}_{seq}(\theta) = \mathbb{E}\left[\sum_{t=1}^{T} \log \pi_\theta(a_t|s_t) \cdot \hat{A}_{seq} \right], \quad \text{其中 } \hat{A}_{seq} \text{ 为序列级统一优势}$$

符号: $T$ 为序列长度；$\hat{A}_{seq}$ 为整个序列共享的优势值；梯度在序列内隐式平均。

**变化点**: 序列级损失使长序列末端token的梯度信号被稀释，且不同推理步骤（如"验证"vs"猜测"）获得同等权重，无法精细调整。

**本文公式（推导）**:
$$\text{Step 1}: \text{保留每个token独立的重要性采样比率 } r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \quad \text{（不再聚合为序列级乘积）}$$
$$\text{Step 2}: \text{token级优势 } \hat{A}_{i,t} \text{ 继承GRPO组归一化，但应用于每个位置}$$
$$\text{最终}: L^{PG}_{token}(\theta) = \mathbb{E}\left[\sum_{t=1}^{T} \min\left(r_t(\theta)\hat{A}_{i,t}, \text{clip}(r_t(\theta), 1-\varepsilon_{low}, 1+\varepsilon_{high})\hat{A}_{i,t}\right)\right]$$

**对应消融**: Table 1显示Token-level Loss从41提升至42（+1分）；作者特别指出其主要贡献在于"增强训练稳定性并使长度增长更健康"，而非单纯精度提升。

---

### 模块 3: Soft Overlong Punishment + Dynamic Sampling（对应框架图 奖励塑形与数据过滤阶段）

**直觉**: 二值奖励无法区分"正确但冗长"和"正确且简洁"的响应；同时，组内全对或全错的样本产生零优势，浪费计算。

**Baseline 公式** (GRPO):
$$R_i \in \{0, 1\}, \quad \hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^{G})}{\text{std}(\{R_j\}_{j=1}^{G})}$$

符号: $R_i$ 为第$i$个响应的二值正确性奖励；$G=16$ 为组大小。

**变化点**: （1）二值奖励导致模型只能通过"正确率"学习，无法抑制无意义的长度膨胀；（2）当组内所有$R_j$相同时，分子为零，整个batch的梯度消失。

**本文公式（推导）**:
$$\text{Step 1（Soft Overlong Punishment）}: R_i^{shaped} = R_i - \lambda \cdot f(\text{length}_i; L_{expected}=16384, L_{max}=20480)$$
$$\text{其中 } f(\cdot) \text{ 在}[0, 16384]\text{为零，在}[16384, 20480]\text{渐变惩罚，配合硬过滤移除}>20480\text{的响应}$$
$$\text{Step 2（Dynamic Sampling）}: \text{若 } \text{std}(\{R_j\}_{j=1}^{G}) = 0, \text{ 则丢弃该组并重新采样，维持有效batch size}$$
$$\text{最终}: \text{过滤后的非零梯度样本进入Token-level Policy Gradient更新}$$

**对应消融**: Table 1显示Soft Overlong Punishment从38提升至41（+3分）；Dynamic Sampling从42提升至50（+8分），为最大单组件贡献。

## 实验与分析



本文在AIME 2024竞赛数学基准上评估DAPO，使用Qwen2.5-32B作为基础模型，采用avg@32指标（32次采样取平均）。核心结果显示于Table 1：DAPO最终达到50分，超越DeepSeek-R1-Zero-Qwen-32B的47分，且仅用50%训练步数。这一结果的关键意义在于——它首次在完全开源、可复现的系统中，验证了长CoT RL训练可以达到甚至超越封闭技术报告的性能，同时大幅缩短收敛时间。



消融实验采用渐进式叠加策略，从naive GRPO的30分基线出发：添加Overlong Filtering提升至36（+6分），这是去除极端异常样本的基础收益；叠加Clip-Higher至38（+2分），验证非对称裁剪的探索价值；叠加Soft Overlong Punishment至41（+3分），显示长度塑形的必要性；叠加Token-level Loss至42（+1分），精度增益有限但定性改善稳定性；最后叠加Dynamic Sampling至50（+8分），揭示零梯度样本过滤是效率瓶颈的关键。值得注意的是，各组件存在非线性交互——Dynamic Sampling的增益建立在前序组件基础上，单独应用可能效果不同。

关于公平性：对比的baselines中，DeepSeek-R1-Zero-Qwen-32B是直接可比的最强开源结果，但Kimi k1.5 [11]、Open-Reasoner-Zero [14]等同期工作未在提供文本中显示直接数值对比；REINFORCE++ [15]作为算法变体也未纳入对比。实验仅覆盖数学推理（AIME），代码生成等更广义推理域的泛化性未验证；且仅测试32B单一模型规模。作者披露的训练资源为prompt batch size 512、每提示16响应、mini-batch 512，但未给出总GPU小时数，大规模可复现性仍受计算资源门槛限制。Token-level Loss的"稳定性提升"主张缺乏定量指标支撑，主要依赖训练曲线观察。

## 方法谱系与知识库定位

DAPO属于**策略优化（Policy Optimization）**方法谱系，直接父方法为**GRPO** [2]，上溯至**PPO** [21]的clipped surrogate objective框架。GRPO相对于PPO的关键改变是移除critic网络、改用组内奖励归一化估计优势；DAPO保留GRPO的优势估计核心，但系统性地修改了四个关键槽位：

| 直接Baseline | 与DAPO的差异（1行） |
|:---|:---|
| **GRPO** [2] | DAPO在其基础上增加四重机制，解决熵崩溃与训练不稳定，30→50分 |
| **DeepSeek-R1-Zero** [2] | 同Qwen-32B规模下DAPO以50%步数超越其47分，且完全开源细节 |
| **PPO** [21] | DAPO通过Clip-Higher非对称修改其对称裁剪机制，专为长CoT优化 |

后续可拓展方向：（1）**跨模态/跨域验证**：将DAPO四机制迁移至代码生成、科学推理等非数学领域；（2）**多尺度扩展**：验证在7B/70B等更大或更小模型上的有效性，打破仅32B的局限；（3）**与过程奖励模型（PRM）结合**：当前DAPO仅使用结果奖励，引入[16][17][18]的细粒度过程监督可能进一步释放潜力。

标签定位：modality=text | paradigm=reinforcement_learning | scenario=long_chain-of-thought_reasoning | mechanism=decoupled_clipping + dynamic_sampling + reward_shaping | constraint=open-source_reproducibility

## 引用网络

### 后续工作（建立在本文之上）

- [[P__基于单次不确定性估计的高效RL数_UFO-RL]]: Open-source RL system; likely used as comparison baseline for implementation and

