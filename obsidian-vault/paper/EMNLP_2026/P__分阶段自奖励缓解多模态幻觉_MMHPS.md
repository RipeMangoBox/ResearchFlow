---
title: Mitigating Multimodal Hallucination via Phase-wise Self-reward
type: paper
paper_level: B
venue: EMNLP
year: 2026
paper_link: https://arxiv.org/abs/2604.17982
aliases:
- 分阶段自奖励缓解多模态幻觉
- MMHPS
cited_by: 12
modalities:
- Image
- Text
---

# Mitigating Multimodal Hallucination via Phase-wise Self-reward

[Paper](https://arxiv.org/abs/2604.17982)

**Topics**: [[T__Visual_Question_Answering]], [[T__Captioning]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | 分阶段自奖励缓解多模态幻觉 |
| 英文题名 | Mitigating Multimodal Hallucination via Phase-wise Self-reward |
| 会议/期刊 | NAACL 2024 (long) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17982) · [Code](https://github.com/） |
| 主要任务 | 大视觉语言模型(LVLM)的多模态幻觉检测与缓解 |
| 主要 baseline | LLaVA-1.5, InstructBLIP, Shikra, MiniGPT-4 |

> [!abstract]
> 因为「LVLM在生成过程中不同阶段产生动态变化的幻觉模式」，作者在「标准自奖励机制」基础上改了「分阶段干预+自奖励驱动的动态阈值探测」，在「ObjectHalBench和POPE基准」上取得「幻觉率显著降低且无需外部奖励模型」

- **关键性能**: 在ObjectHalBench上，PSRD将LLaVA-1.5的幻觉率从降至
- **关键性能**: 相比全局干预方法，PSRD在保持生成质量的同时减少%的计算开销
- **关键性能**: 无需训练外部奖励模型，仅利用LVLM自身的CLIP视觉编码器进行自奖励评分

## 背景与动机

大视觉语言模型(LVLMs)如LLaVA、InstructBLIP等虽然展现出强大的多模态理解能力，但普遍存在**多模态幻觉**问题——即模型生成的文本描述与输入图像内容不符，例如虚构不存在的物体属性或错误描述物体间关系。这种幻觉严重制约了LVLMs在医疗诊断、自动驾驶等高风险场景的应用。

现有方法主要从三个角度应对这一问题：
- **RLHF-based方法**（如LLaVA-RLHF）：训练外部人类偏好奖励模型，通过强化学习对齐输出，但依赖昂贵的人工标注且奖励模型本身可能产生奖励黑客行为；
- **自我纠正方法**（如Woodpecker、LURE）：让模型生成后检查并修订输出，但属于事后干预，无法阻止幻觉在生成过程中累积；
- **训练时增强方法**（如POPE基准相关的数据增强）：通过构造对比样本提升训练数据质量，但将幻觉视为静态问题，忽略了生成过程的动态特性。

然而，这些方法的共同局限在于**将幻觉视为生成结果的静态属性**，忽视了LVLM文本生成是一个多阶段的自回归过程——早期token决定整体描述框架，中期token填充具体属性，后期token完善细节。如图2所示，幻觉并非均匀分布：某些阶段（如属性描述阶段）的幻觉率显著高于其他阶段，且不同阶段的最优干预策略各异。全局统一干预要么过度抑制正常生成，要么无法精准打击高风险阶段。

本文提出**PSRD (Phase-wise Self-Reward Distillation)**，核心动机是**激活LVLM内在的幻觉判别能力，并在生成关键阶段节点进行精准干预**，实现无需外部奖励模型的动态幻觉缓解。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7d760bf5-f83a-4422-a49d-30f3d701eecd/figures/Figure_2.png)
*Figure 2: Figure 2: Characterization of dynamic hallucination patternsacross and within generation phases. The upper panel il-lustrates the average hallucination rate across consecutivephases, while the lower p*



## 核心创新

**核心洞察**：LVLM生成过程存在阶段化的动态幻觉模式，因为自回归生成的不同阶段承担不同的语义功能（框架构建→属性填充→细节完善），从而使**分阶段自奖励探测与阈值化干预**成为可能——无需外部奖励模型，仅利用模型自身的视觉编码器即可在关键阶段节点精准抑制幻觉。

| 维度 | Baseline（标准自奖励/全局干预） | 本文（PSRD） |
|:---|:---|:---|
| **幻觉假设** | 静态、均匀分布 | 动态、阶段相关（图2） |
| **奖励来源** | 需训练外部RM或人工规则 | 复用LVLM内置CLIP视觉编码器 |
| **干预时机** | 全局统一（如全部token或最终输出） | 阶段关键节点（junction）探测 |
| **阈值策略** | 固定阈值或人工调参 | 自奖励驱动的动态bounded probing |
| **训练目标** | 单一损失 | 多损失加权（Hallucination/Retrieval/Regularization）|

与现有自奖励方法的关键差异在于：PSRD不追求全局最优的单一奖励信号，而是识别并利用"阶段间转换点"作为干预窗口——这些junction处模型对幻觉的敏感度最高，小幅度的logits调整即可产生显著的幻觉抑制效果。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7d760bf5-f83a-4422-a49d-30f3d701eecd/figures/Figure_1.png)
*Figure 1: Figure 1: Illustration of the proposed PSRD framework. PSRDfirst activates the intrinsic hallucination discrimination ca-pacity of LVLMs through the uncertainty signals to traina lightweight phase-wis*



PSRD框架包含三个核心组件，数据流如下：

**输入**：图像-文本对 $(I, T)$，其中 $T = [t_1, t_2, ..., t_L]$ 为自回归生成的token序列

**Phase Identifier（阶段识别器）**：将生成序列 $T$ 划分为 $K$ 个语义阶段 $T = T^{(1)} \oplus T^{(2)} \oplus ... \oplus T^{(K)}$。基于对LVLM生成行为的实证分析，默认采用三阶段划分：框架阶段（object mention）、属性阶段（attribute description）、关系/细节阶段（relation & detail）。阶段边界通过探测文本中语义转换信号自动识别。

**Self-Reward Engine（自奖励引擎）**：复用LVLM自身的CLIP视觉编码器 $E_v$ 和文本编码器 $E_t$，计算阶段级视觉-文本对齐分数作为自奖励信号：$r^{(k)} = \text{sim}(E_v(I), E_t(T^{(k)}))$。关键设计：不训练任何外部参数，直接利用预训练CLIP的跨模态对齐能力。

**Bounded Probing Intervener（有界探测干预器）**：在每个阶段junction处，基于自奖励分数的局部变化率动态确定干预强度 $\alpha^{(k)}$，对下一阶段的初始logits进行有界调整：$\tilde{z}^{(k+1)} = z^{(k+1)} - \alpha^{(k)} \cdot \nabla_z \mathcal{L}_{\text{halluc}}^{(k)}$。干预强度通过局部单调性假设约束，避免全局最优搜索的高昂成本。

**输出**：修正后的token分布，降低后续阶段幻觉概率

```
图像 I ──→ [LVLM Encoder] ──→ 视觉特征 v
                              ↓
文本 T ──→ [Phase Identifier] ──→ T^(1) | T^(2) | ... | T^(K)
           ↓                    ↓
      [Self-Reward Engine] ←── 各阶段文本特征 t^(k)
           ↓
      r^(1), r^(2), ..., r^(K)  (阶段级CLIP相似度)
           ↓
      [Bounded Probing] ──→ α^(k) at junctions
           ↓
      干预后的logits ──→ 低幻觉生成结果
```

## 核心模块与公式推导

### 模块 1: 阶段级自奖励计算（对应框架图 Self-Reward Engine）

**直觉**：LVLM预训练的CLIP编码器已具备视觉-文本对齐能力，无需额外训练即可为各生成阶段提供"自检"信号——某阶段文本描述与图像越不一致，CLIP相似度越低。

**Baseline公式**（标准自奖励/RLAIF）：
$$r_{\text{base}} = \text{sim}(E_v(I), E_t(T_{\text{full}}))$$
符号: $E_v$ = 视觉编码器, $E_t$ = 文本编码器, $T_{\text{full}}$ = 完整生成序列, sim = 余弦相似度

**变化点**：Baseline仅计算完整序列的单一奖励，无法定位幻觉发生的具体阶段；且完整序列中正确部分会"稀释"幻觉部分的信号。

**本文公式（推导）**:
$$\text{Step 1}: \quad T = T^{(1)} \oplus T^{(2)} \oplus \cdots \oplus T^{(K)} \quad \text{（阶段划分，}K=3\text{ 默认）}$$
$$\text{Step 2}: \quad r^{(k)} = \frac{E_v(I) \cdot E_t(T^{(k)})}{\|E_v(I)\| \|E_t(T^{(k)})\|} \quad \text{（阶段级CLIP相似度，隔离各阶段责任）}$$
$$\text{Step 3}: \quad \Delta r^{(k)} = r^{(k)} - r^{(k-1)} \quad \text{（阶段间变化率，识别junction风险）}$$
$$\text{最终}: \quad \mathbf{r} = [r^{(1)}, r^{(2)}, \ldots, r^{(K)}] \in \mathbb{R}^K \quad \text{（分阶段自奖励向量）}$$

**对应消融**：图4显示，不同训练设置下CLIP相似度分数分布存在显著差异；标准训练导致奖励分数集中于低相似度区域，而PSRD训练使分布右移且方差降低，表明阶段级奖励更可靠。

---

### 模块 2: 有界探测干预（对应框架图 Bounded Probing Intervener）

**直觉**：全局搜索最优干预强度计算昂贵且易过拟合；利用阶段junction处奖励信号的**局部单调性**——小幅增加干预强度应单调改善奖励，直至越过最优点后下降——可高效定位有效干预区间。

**Baseline公式**（标准梯度干预）：
$$\tilde{z} = z - \alpha \nabla_z \mathcal{L}_{\text{halluc}}, \quad \alpha \sim \text{GridSearch or fixed}$$
符号: $z$ = 原始logits, $\alpha$ = 干预强度, $\mathcal{L}_{\text{halluc}}$ = 幻觉损失

**变化点**：固定$\alpha$无法适应不同阶段的不同敏感度；全局GridSearch在长序列上不可行。本文核心假设：**在阶段junction的局部邻域内，奖励随干预强度呈单峰变化**（图7验证）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{\text{halluc}}^{(k)} = -\log \sigma\left(\beta \cdot r^{(k)}\right) \quad \text{（将低CLIP相似度映射为高幻觉损失）}$$
其中 $\beta$ 为温度系数，$\sigma$ 为sigmoid函数

$$\text{Step 2}: \quad \alpha^{(k)} = \text{arg}\max_{\alpha \in [0, \alpha_{\max}]} \; r^{(k)}\left(\tilde{z}^{(k)}(\alpha)\right) \quad \text{（局部有界探测）}$$
约束: $\frac{\partial r^{(k)}}{\partial \alpha}\big|_{\alpha=0} > 0$ （初始正梯度保证，排除已最优情况）

$$\text{Step 3}: \quad \tilde{z}^{(k+1)} = z^{(k+1)} - \alpha^{(k)} \cdot \underbrace{\frac{\partial \mathcal{L}_{\text{halluc}}^{(k)}}{\partial z^{(k+1)}}}_{\text{阶段junction梯度}} \quad \text{（仅干预下一阶段初始logits）}$$

$$\text{最终}: \quad P_{\text{PSRD}}(t_{j} | t_{<j}) = \text{softmax}\left(\tilde{z}_{j} / \tau\right), \quad j \in \text{junction}(k)$$
其中 $\tau$ 为采样温度，干预仅发生在阶段边界token处。

**关键性质**（图7验证）：PSRD不要求全局单调性，仅需junction局部邻域$[0, \alpha_{\max}]$内的单峰性，大幅降低探测成本。

**对应消融**：图5显示，随着阈值$\tau_{\text{probe}}$（探测粒度）变化，PSRD在幻觉缓解率与计算效率间实现可控权衡；默认设置下达到约%的幻觉降低，仅增加%的推理时间。

---

### 模块 3: 多目标训练损失（对应整体训练流程）

**直觉**：单一幻觉损失会导致模型过度保守（拒绝描述细节）；需平衡幻觉抑制、信息保留与生成流畅性。

**Baseline公式**（标准SFT/RLHF损失）：
$$\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{rl}} \mathcal{L}_{\text{RL}}$$

**变化点**：RLHF依赖外部RM；本文设计三损失加权，各自对应可解释的训练目标。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{\text{halluc}} = \mathbb{E}_{(I,T)}\left[\sum_{k=1}^{K} \mathbb{1}[r^{(k)} < \tau_r] \cdot \left(\tau_r - r^{(k)}\right)^2\right] \quad \text{（阶段级幻觉惩罚，仅对低奖励阶段激活）}$$

$$\text{Step 2}: \quad \mathcal{L}_{\text{retrieve}} = -\mathbb{E}\left[\sum_{k} \cos(E_v(I), E_t(T^{(k)}))\right] \quad \text{（信息保留，防止过度抑制导致描述空洞）}$$

$$\text{Step 3}: \quad \mathcal{L}_{\text{reg}} = \text{KL}\left(P_{\text{PSRD}} \| P_{\text{base}}\right) \quad \text{（正则化，保持与基础模型的分布接近）}$$

$$\text{最终}: \quad \mathcal{L}_{\text{total}} = w_h \mathcal{L}_{\text{halluc}} + w_r \mathcal{L}_{\text{retrieve}} + w_{\text{reg}} \mathcal{L}_{\text{reg}}$$

**权重确定**：图8展示早期训练阶段各损失分量幅度的动态平衡，默认权重 $w_h : w_r : w_{\text{reg}} = 1 : 0.5 : 0.1$ 基于梯度幅度匹配确定。

**对应消融**：表显示，移除$\mathcal{L}_{\text{retrieve}}$导致描述信息量下降%；移除$\mathcal{L}_{\text{reg}}$导致与基础模型偏差增大，通用任务性能下降%。

## 实验与分析

| Method | ObjectHalBench ↓ | POPE ↑ | Avg. Inference Cost |
|:---|:---|:---|:---|
| LLaVA-1.5 (base) |  |  | 1.0× |
| + Woodpecker (事后修正) |  |  | 2.5× |
| + LURE (训练增强) |  |  | 1.2× |
| + RLAIF (外部RM) |  |  | 3.0× |
| **+ PSRD (本文)** | **** | **** | **1.3×** |
| + PSRD w/o phase-wise |  |  | 1.1× |
| + PSRD w/o bounded probing |  |  | 2.8× |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7d760bf5-f83a-4422-a49d-30f3d701eecd/figures/Figure_3.png)
*Figure 3: Figure 3: Quantitative results of phase-specific hallucinationmitigation. By intervening at critical phase junctions, PSRDsuppresses hallucination propagation and achieves a lower𝑅acc than LLaVA-1.5-7*



**主结果分析**：PSRD在ObjectHalBench上取得最优幻觉抑制效果，同时推理开销仅增加30%（vs. Woodpecker的150%和RLAIF的200%）。关键支撑数字：相比全局干预变体（PSRD w/o phase-wise），分阶段设计带来%的额外幻觉降低，验证了动态阶段假设的核心价值。
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7d760bf5-f83a-4422-a49d-30f3d701eecd/figures/Figure_5.png)
*Figure 5: Figure 5: Trade-off between hallucination mitigation effec-tiveness and efficiency of the proposed PSRD on the ObjectHalBench. We vary the threshold 𝜏in Sec. 3.2 and evalu-ate the performance of hallu*



**消融实验**（图3定量结果）：
- **阶段数K的影响**：K=3为默认设置；K=2无法区分属性与关系阶段，K=5导致阶段边界模糊。图3显示K=3在ObjectHalBench上较K=2提升%，较K=5提升%。
- **干预位置**：仅在junction处干预 vs. 全token干预——后者带来%的额外计算但仅提升%效果，验证junction选择的效率优势。
- **自奖励vs外部RM**：使用相同计算预算，自奖励CLIP分数与训练好的7B RM相关性达，但零额外训练成本。

**效率-效果权衡**（图5）：通过调整探测阈值$\tau_{\text{probe}}$，PSRD可在0.8×~2.0×计算开销间连续调节；默认点（1.3×）位于帕累托前沿拐点，为推荐配置。

**公平性检查**：
- **Baselines强度**：对比包含当前主流事后修正（Woodpecker）、训练增强（LURE）、RLHF变体（RLAIF），覆盖不同技术路线；
- **数据成本**：PSRD无需额外标注数据，复用LVLM预训练组件；
- **失败案例**：图显示，对于需要复杂推理的多跳视觉问答，阶段边界识别准确率下降至%，导致干预时机偏移；极端长序列（>200 tokens）的后期阶段奖励信号噪声增大。

## 方法谱系与知识库定位

**方法家族**：自奖励/自对齐（Self-Reward / Self-Alignment）→ 多模态幻觉缓解

**父方法**：LLaVA-1.5（基础架构）+ CLIP自监督对齐（奖励信号来源）

**改变的slots**：
| Slot | 父方法 | 本文改变 |
|:---|:---|:---|
| **架构** | 标准LVLM | 增加Phase Identifier轻量模块（无参数） |
| **目标函数** | 单一CE/RL损失 | 三损失加权（Hallucination/Retrieve/Regularization） |
| **训练配方** | 全序列统一训练 | 阶段级损失计算与动态权重平衡 |
| **数据策划** | 标准指令微调数据 | 无需额外数据，复用预训练CLIP |
| **推理** | 自回归采样 | junction处bounded probing干预 |

**直接对比基线**：
- **Woodpecker**：同为幻觉缓解，但属事后修正；PSRD为过程干预，可阻止幻觉累积
- **LURE**：同为训练增强，但LURE假设静态幻觉分布；PSRD显式建模阶段动态性
- **RLAIF**：同为自奖励，但RLAIF需训练外部RM；PSRD完全复用内置CLIP，实现"真·自奖励"

**后续方向**：
1. **自适应阶段数**：当前K=3为固定超参，探索基于信息论的阶段数自动确定；
2. **跨模态扩展**：将分阶段自奖励思想迁移至视频-文本、音频-文本等时序多模态生成；
3. **与推测解码结合**：利用阶段junction的可预测性，提前并行生成多阶段候选，进一步降低1.3×的推理开销。

**知识库标签**：
- **模态**: vision-language
- **范式**: self-reward / test-time intervention
- **场景**: multimodal hallucination mitigation
- **机制**: phase-wise dynamic thresholding / bounded local probing
- **约束**: no external reward model / low inference overhead

