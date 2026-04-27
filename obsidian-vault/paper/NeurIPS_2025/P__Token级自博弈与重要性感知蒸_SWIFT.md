---
title: Token-Level Self-Play with Importance-Aware Guidance for Large Language Models
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- Token级自博弈与重要性感知蒸馏
- SWIFT
- SWIFT improves self-play fine-tunin
acceptance: Poster
method: SWIFT
modalities:
- Text
paradigm: preference optimization
---

# Token-Level Self-Play with Importance-Aware Guidance for Large Language Models

**Topics**: [[T__Text_Generation]], [[T__Reasoning]] | **Method**: [[M__SWIFT]] | **Datasets**: [[D__DROP]] (其他: HuggingFace Open LLM Leaderboard, Big-Bench-Hard, ToolBench)

> [!tip] 核心洞察
> SWIFT improves self-play fine-tuning by assigning token-level importance weights estimated from a stronger teacher model, enabling more fine-grained alignment and serving as an effective knowledge distillation strategy.

| 中文题名 | Token级自博弈与重要性感知蒸馏 |
| 英文题名 | Token-Level Self-Play with Importance-Aware Guidance for Large Language Models |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.21377) · [Code](https://github.com/SWIFT-llm/SWIFT) · [Project](N/A) |
| 主要任务 | Text Generation, Reasoning |
| 主要 baseline | SPIN, DPO, TDPO, TIS-DPO |

> [!abstract] 因为「偏好优化依赖昂贵人工标注数据，而自博弈方法对所有token均匀施加学习信号，忽略了细粒度质量差异」，作者在「SPIN」基础上改了「引入教师模型估计的token级重要性权重，以奖励引导替代logits匹配进行知识蒸馏」，在「HuggingFace Open LLM Leaderboard、BBH、DROP、ToolBench」上取得「Qwen2-7B-Instruct平均47.46 vs SPIN 47.12，DROP 62.03 vs SPIN 58.62 (+3.41)」

- **关键性能 1**: DROP 准确率 62.03，超越 SPIN 58.62 达 +3.41，超越 DPO 59.10 达 +2.93
- **关键性能 2**: ToolBench Act.EM 48.36，超越 SPIN 47.01 达 +1.35；F1 42.08，超越 SPIN 40.73 达 +1.35
- **关键性能 3**: Qwen3-32B→Mistral-7B-v0.1 蒸馏场景平均 63.50，超越 SPIN 60.17 达 +3.33，超越 DPO 59.55 达 +3.95

## 背景与动机

大型语言模型（LLM）的后训练对齐面临一个核心矛盾：高质量的偏好数据获取成本极高，而低成本替代方案往往牺牲细粒度监督。具体而言，当模型需要学习"哪些token值得信任、哪些token需要修正"时，现有方法无法精确到token级别进行区分——一个回答中可能前半句正确、后半句错误，但现有训练信号却对整个序列一视同仁。

现有三类方法的处理方式各有局限：

- **DPO（Direct Preference Optimization）** [16] 将语言模型本身视为隐式奖励模型，通过成对偏好数据进行序列级别的优化。它避免了显式训练奖励模型，但仅依赖二元的"赢/输"序列标签，无法区分同一序列内部不同token的质量差异。

- **SPIN（Self-Play Fine-Tuning）** [19] 通过模型自身早期检查点生成拒绝样本，实现无需人工标注的迭代提升。然而，SPIN对所有token施加均匀的学习信号——无论是拒绝样本中的高质量token还是噪声token，都获得相同的权重，导致优质知识被低效利用。

- **TDPO（Token-level DPO）** [17] 和 **TIS-DPO** [18] 虽将DPO扩展到token级别，但仍依赖估计的权重或重要性采样，缺乏外部强教师的可靠指导，且未与自博弈框架结合。

这些方法的共同短板在于：**信用分配过于粗糙**——它们或在序列级别做二元判断，或在token级别缺乏可靠的质量估计源。一个具体的失败场景是：当学生模型生成的拒绝回答中包含部分正确推理步骤时，均匀惩罚会同时抹除正确和错误信号；而DPO需要人工标注的偏好对来识别这种细粒度差异。

本文提出SWIFT，核心思路是：利用一个更强的教师模型为每个token估计重要性权重，将自博弈的迭代机制与奖励引导的知识蒸馏相结合，实现"哪里好、学哪里；哪里差、避哪里"的细粒度学习。

## 核心创新

**核心洞察**：token级别的质量差异可以被一个更强的教师模型可靠估计，因为教师提供的奖励差异信号比学生自身的隐式奖励更稳定、更具判别力，从而使"拒绝样本中的高质量token可被选择性保留、噪声token可被显式过滤"成为可能。

与 baseline 的差异：

| 维度 | Baseline (SPIN/DPO) | 本文 (SWIFT) |
|:---|:---|:---|
| 奖励设计 | 所有token均匀处理，拒绝样本获得同等学习信号 | 教师模型估计token级重要性权重，带裁剪界 [-0.5, 1.5] |
| 信用分配 | 序列级二元偏好，无token质量区分 | 细粒度per-token信用，基于教师奖励差异过滤低奖励噪声token |
| 知识蒸馏方式 | 无教师参与（SPIN）或logits匹配（传统蒸馏） | 奖励引导的token加权，教师仅用于重要性估计而非logits匹配 |
| 训练效率 | SFT级别速度（SPIN），但信号质量低 | 0.30s/sample，接近SFT的0.26s，优于ULD 0.60s和DSKD 0.54s |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a624ee78-e399-48d0-b036-8cacf4fc5208/figures/Figure_1.png)
*Figure 1 (comparison): The left figure compares SWIFT with SFT, DPO, and SPN on MT-Bench, evaluated by GPT-4-Turbo. The right figure shows the results on the same chosen and rejected by SWIFT and SPN during training.*



SWIFT 的训练流程是一个三阶段闭环，在保留SPIN自博弈迭代结构的同时，嵌入教师引导的token级重要性估计：

1. **学生自博弈生成（Student Self-Play Generation）**：输入为当前学生模型检查点，输出为生成的完成样本与拒绝样本。该模块完全继承SPIN的机制——学生模型与自身早期版本对弈，生成用于对比学习的正负样本对，无需人工标注。

2. **教师重要性估计（Teacher Importance Estimation）**：输入为生成的样本对 + 更强的教师模型，输出为每个token的重要性权重 $w_t \in [L, U]$。教师模型评估每个位置token的质量差异，提供比学生自身更可靠的奖励信号。

3. **裁剪与缩放（Clipping & Scaling）**：对原始教师估计进行 $\mu=1$ 缩放并裁剪到 $[-0.5, 1.5]$ 区间，防止极端权重破坏训练稳定性。

4. **加权自博弈优化（Weighted Self-Play Optimization）**：输入为样本对 + 裁剪后的token权重，输出为更新后的学生模型。SPIN的损失函数被重新加权，使高质量token获得更多梯度关注，噪声token被抑制。

迭代流程可概括为：

```
学生模型 θ_t → 生成 (y_win, y_lose) → 教师估计 D(x_t) → 裁剪得 w_t
                                              ↓
更新 θ_{t+1} ← 加权损失 Σ_t w_t · ℓ_SPIN(t) ←─┘
```

关键设计约束：教师**不**用于logits匹配蒸馏，仅用于奖励引导的token重要性估计，这降低了传统知识蒸馏的通信开销，同时保留了自博弈的数据自给特性。

## 核心模块与公式推导

### 模块 1: Token 重要性估计（对应框架图 教师重要性估计 → 裁剪模块）

**直觉**：同一拒绝回答中的不同token质量参差不齐，需要外部裁判（教师模型）给出细粒度评分，而非学生自己判断。

**Baseline 公式 (SPIN)**：无token级质量估计，所有token隐式权重为1。

符号: $D(x_t)$ = 教师模型在位置 $t$ 估计的奖励差异函数；$\mu$ = 缩放系数；$L, U$ = 裁剪下上界。

**变化点**：SPIN完全缺失此模块，导致无法区分拒绝样本中的"局部正确"与"全局错误"。本文引入教师模型的奖励差异估计作为token质量的代理信号。

**本文公式（推导）**：
$$\text{Step 1}: D(x_t) = \text{teacher_reward}(x_t, y_w) - \text{teacher_reward}(x_t, y_l) \quad \text{教师评估赢/输样本在位置} t \text{的奖励差异}$$
$$\text{Step 2}: w_t^{\text{raw}} = \mu \cdot D(x_t), \quad \mu = 1 \quad \text{线性缩放保持教师信号强度}$$
$$\text{Step 3}: w_t = \text{clip}(w_t^{\text{raw}}, L, U), \quad L=-0.5, U=1.5 \quad \text{裁剪防止极端权重，负值允许抑制噪声token}$$
$$\text{最终}: w_t \in [-0.5, 1.5], \forall t \in \{1,...,T\}$$

**对应消融**：Table 3 显示不同token weight配置的影响；Table 9（附录）显示注入 ±0.2 均匀噪声后 ite3 平均准确率从 47.46 降至 46.66，Δ -0.8，证明对适度噪声具有鲁棒性。

---

### 模块 2: 加权自博弈目标函数（对应框架图 加权优化模块）

**直觉**：将模块1得到的token权重整合进自博弈损失，使"该学的地方多梯度、该忘的地方少干扰"。

**Baseline 公式 (SPIN)**：
$$\mathcal{L}_{\text{SPIN}} = \mathbb{E}_{(x,y_w,y_l)}\left[\sum_{t=1}^{T} \text{ell}_{\text{SPIN}}(x, y_w, y_l; t)\right]$$
其中 $\text{ell}_{\text{SPIN}}(t)$ 为位置 $t$ 的标准自博弈损失，所有token等权求和。

符号: $\theta$ = 学生模型参数；$y_w$ = 赢样本（ground truth或更优生成）；$y_l$ = 输样本（学生早期检查点生成）；$\text{ell}(t)$ = 位置 $t$ 的对比损失。

**变化点**：SPIN的均匀求和导致拒绝样本中的高质量token被同等惩罚，学习信号浪费；DPO的序列级偏好无法定位到具体token。本文将求和改为加权和，权重即模块1的教师估计。

**本文公式（推导）**：
$$\text{Step 1}: \mathcal{L}_{\text{weighted}}^{\text{raw}} = \mathbb{E}_{(x,y_w,y_l)}\left[\sum_{t=1}^{T} w_t \cdot \text{ell}_{\text{SPIN}}(x, y_w, y_l; t)\right] \quad \text{引入token权重调制梯度大小与方向}$$
$$\text{Step 2}: w_t < 0 \Rightarrow \text{梯度反向} \quad \text{负权重显式抑制教师判定为低质的token，实现"主动遗忘"}$$
$$\text{Step 3}: w_t > 1 \Rightarrow \text{梯度增强} \quad \text{超1权重强化教师判定为高价值的token学习}$$
$$\text{最终}: \mathcal{L}_{\text{SWIFT}} = \mathbb{E}_{(x,y_w,y_l)}\left[\sum_{t=1}^{T} \text{clip}(\mu \cdot D(x_t), -0.5, 1.5) \cdot \text{ell}_{\text{SPIN}}(x, y_w, y_l; t)\right]$$

**对应消融**：Table 3 为token weight配置的消融；核心对比在 Table 2 中体现——SWIFT 在 Qwen2-7B-Instruct 蒸馏设置下四项benchmark平均优于 SPIN 和 DPO。

---

### 模块 3: 教师模型的角色约束（贯穿框架的推理策略）

**直觉**：传统蒸馏让教师生成软标签（logits匹配），通信成本高且强制教师与学生词汇对齐；本文让教师只做"裁判"不做"示范"，大幅降低耦合。

**Baseline 公式 (传统知识蒸馏 / ULD / DSKD)**：
$$\mathcal{L}_{\text{KD}} = \text{KL}(p_{\text{teacher}} || p_{\text{student}}) \quad \text{或变体，需要教师完整前向传播获取分布}$$

**变化点**：logits匹配需要教师输出层分布，计算量大（ULD 0.60s/sample, DSKD 0.54s/sample）；且教师-学生词汇差异大时KL散度意义模糊。

**本文公式（推导）**：
$$\text{Step 1}: \text{教师仅输出奖励值 } r_\phi(x_t, y) \in \mathbb{R} \quad \text{而非完整概率分布，单次推理即可}$$
$$\text{Step 2}: D(x_t) = r_\phi(x_t, y_w) - r_\phi(x_t, y_l) \quad \text{标量差异运算，计算极简}$$
$$\text{最终}: \text{训练时间 } 0.30\text{s/sample} \approx \text{SFT } 0.26\text{s/sample} \ll \text{ULD/DSKD}$$

**对应消融**：Table 8（附录）显示训练效率对比，SWIFT 0.30s/sample 接近 SFT，验证"裁判式"教师设计的计算优势。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a624ee78-e399-48d0-b036-8cacf4fc5208/figures/Table_2.png)
*Table 2 (quantitative): Evaluation results of Qwen2-7B-Instruct distilled on GPT-4/3.5 on four benchmarks and the average.*



本文在多个benchmark上评估 SWIFT 的有效性，涵盖通用能力、推理能力和工具使用场景。核心实验使用 Qwen2-7B-Instruct 作为学生模型，以 GPT-4/3.5 作为教师进行蒸馏。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a624ee78-e399-48d0-b036-8cacf4fc5208/figures/Table_3.png)
*Table 3 (ablation): Ablation study on token weight configuration.*



从 Table 2 可见，SWIFT 在四项benchmark上的平均表现优于直接对比的 SPIN 和 DPO。更具区分度的结果来自扩展评估：在 **DROP**（离散段落推理）上，SWIFT 达到 **62.03**，较 SPIN 的 58.62 提升 **+3.41**，较 DPO 的 59.10 提升 **+2.93**，这一差距在需要细粒度数值推理的任务上尤为显著，印证了token级权重对复杂推理步骤的选择性学习价值。在 **Big-Bench-Hard (BBH)** 上，SWIFT 的 66.01 超越 SPIN 64.85（+1.16）和 DPO 64.23（+1.78），显示其在高难度推理集合上的稳健优势。工具使用场景（**ToolBench**）中，SWIFT 的 Act.EM 48.36 和 F1 42.08 分别领先 SPIN +1.35，表明细粒度信用分配对API调用序列的质量敏感。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a624ee78-e399-48d0-b036-8cacf4fc5208/figures/Figure_3.png)
*Figure 3 (result): The average score of SPN and SWIFT at different iterations on the HuggingFace Open LLM Leaderboard.*



Figure 3 展示了 HuggingFace Open LLM Leaderboard 上迭代训练动态：SWIFT 随迭代持续提升，而 SPIN 增长趋缓，说明教师引导的token权重有效打破了自博弈的瓶颈。

扩展至大规模蒸馏场景（Table 12，附录），当教师为 Qwen3-32B、学生为 Mistral-7B-v0.1 时，SWIFT 平均 **63.50**，大幅超越 SPIN 60.17（+3.33）和 DPO 59.55（+3.95）；同设置下 Qwen2.5-7B-Instruct 学生也获得 **73.08**，领先 SPIN 71.92（+1.16）。这验证了 SWIFT 作为知识蒸馏策略的跨架构泛化能力。

**消融与鲁棒性**：Table 3 的token weight配置消融显示设计选择的敏感性。关键鲁棒性证据来自 Table 9（附录）：对教师信号注入 ±0.2 均匀噪声后，ite3 平均准确率仅从 47.46 微降至 46.66（Δ -0.8），证明系统对教师估计的不完美具有容忍度。Table 13（附录）进一步显示不同教师质量下性能波动有限（47.46–48.07），但此范围狭窄也可能暗示对教师能力的敏感度分析不够充分。

**公平性审视**：对比基线中，SPIN 和 DPO 为直接竞争方法，但 **TDPO** 和 **TIS-DPO** 虽被引用为密切相关工作，却未在主要结果表中直接对比性能（仅 Table 6 比较计算效率）。现代 RL 方法如 PPO、GRPO 以及 Kimi k1.5、DeepSeek-R1 等强基线被引用但未纳入实验对比。训练成本方面，SWIFT 的 0.30s/sample 接近 SFT，但需额外教师推理开销，总体 GPU-hours 在 50k UltraChat 样本/迭代 的设置下可控（Table 7）。作者披露的限制包括：依赖教师质量、需要训练时教师推理的额外计算、以及最优师生能力 gap 的分析不足。

## 方法谱系与知识库定位

SWIFT 属于 **偏好优化 + 知识蒸馏** 的交叉方法族，直接父方法为 **SPIN** [19]（自博弈微调）。谱系关系：SPIN → SWIFT，核心继承自博弈数据生成管道，关键变异在于奖励设计与信用分配机制。

**改变的插槽**：
- **reward_design**: SPIN 的均匀token处理 → 教师估计的重要性权重 + 裁剪
- **credit_assignment**: 序列级二元偏好 → 细粒度per-token信用
- **objective**: 标准自博弈损失 → 加权自博弈损失
- **inference_strategy**: 无教师参与 → 训练时教师推理用于token重要性估计
- **data_pipeline**: 保留自博弈生成，增加教师引导的蒸馏信号

**直接基线差异**：
- **SPIN**: 同框架但无教师、无token权重；SWIFT 添加教师引导的加权机制
- **DPO**: 需人工偏好对、序列级优化；SWIFT 自生成数据、token级优化
- **TDPO** [17]: 同为token级DPO，但无自博弈迭代、无外部教师；SWIFT 结合自博弈与教师引导
- **TIS-DPO** [18]: 同为token级重要性采样，但权重估计方式不同；SWIFT 以奖励差异替代采样权重

**后续方向**：(1) 将token级权重机制扩展至在线RL（PPO/GRPO），替代静态自博弈；(2) 探索师生能力gap的最优区间，当前分析有限；(3) 结合过程奖励模型（PRM）替代单点教师奖励，实现更精细的推理步骤信用分配。

**标签**: 模态=text | 范式=preference optimization + knowledge distillation | 场景=LLM post-training alignment | 机制=token-level importance weighting, self-play, teacher-guided credit assignment | 约束=no human preference labels, teacher inference overhead during training

