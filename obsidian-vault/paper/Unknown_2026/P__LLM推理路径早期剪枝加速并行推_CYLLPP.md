---
title: Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.16029
aliases:
- LLM推理路径早期剪枝加速并行推理
- CYLLPP
modalities:
- Text
paradigm: Reinforcement Learning
---

# Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning

[Paper](https://arxiv.org/abs/2604.16029)

**Topics**: [[T__Reasoning]], [[T__Compression]] (其他: Pruning)

| 属性 | 内容 |
|------|------|
| 中文题名 | LLM推理路径早期剪枝加速并行推理 |
| 英文题名 | Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16029) · [Code](https://github.com/xxx ⭐
| 主要任务 | 大型语言模型并行推理中的路径早期剪枝，降低计算开销同时保持推理准确率 |
| 主要 baseline | Best-of-N (BoN)、Self-Consistency (SC)、Speculative Decoding、Lookahead Decoding、EAGLE、Medusa、Tree-of-Thoughts (ToT) |

> [!abstract] 因为「并行推理中大量候选路径早期即出现错误且无法恢复，导致计算浪费」，作者在「Best-of-N 等并行采样方法」基础上改了「引入 STOP 模块学习早期路径潜力评分并动态剪枝」，在「GSM8K、MATH、GPQA、MMLU-Pro 等数学与 STEM benchmark」上取得「用 50% 计算量达到 BoN 全路径推理的准确率」。

- **关键性能 1**: 在 GSM8K 上，STOP 仅用 50% 的 forward passes 达到 Best-of-N (N=16) 的 95.2% 准确率（完整 BoN 为 95.8%）
- **关键性能 2**: 在 MATH 上，STOP 以 40% 计算成本达到 BoN 的 92% 准确率水平
- **关键性能 3**: 在 GPQA 钻石级问题上，STOP 相比 Self-Consistency 减少 60% 推理路径数，准确率提升 3.2%

## 背景与动机

大型语言模型（LLM）的推理能力可通过并行生成多条候选路径（如 Best-of-N 采样、Self-Consistency 投票、Tree-of-Thoughts 分支搜索）得到显著提升。然而，这种"大力出奇迹"的策略代价高昂：多数候选路径在推理早期就已偏离正确方向，却仍需消耗完整的生成预算。例如，在数学问题求解中，一条路径可能在第 3 个 token 就选择了错误的运算符号，但模型仍会继续生成至最大长度，浪费大量计算。

现有方法从不同角度应对这一效率问题：
- **Best-of-N (BoN)**：并行采样 N 条完整路径，最终用奖励模型或多数投票选择最优答案。该方法简单有效，但所有路径均需完整解码，无法避免早期错误路径的计算浪费。
- **Speculative Decoding / Medusa / EAGLE**：通过草稿模型或多头预测加速单条路径的 token 生成，但聚焦于单序列速度优化，未解决多路径并行中的冗余问题。
- **Tree-of-Thoughts (ToT)**：显式维护推理树并通过启发式评估剪枝，但评估函数通常为任务特定的硬编码规则，缺乏通用学习能力，且剪枝决策仍偏晚。

这些方法的共同瓶颈在于：**缺乏对路径"早期潜力"的通用学习机制**。BoN 不做任何中间判断；ToT 的评估过于粗糙且依赖人工设计；投机解码系列完全不处理多路径选择。核心观察是：早期错误往往导致不可逆的失败（如图 1 所示），若在生成初期即可识别并终止低潜力路径，将大幅节省计算。然而，如何定义"早期潜力"、如何在极短前缀上做出可靠预测、如何平衡探索与利用，仍是开放挑战。

本文提出 STOP（Score The Optimal Prefix）框架，首次将路径早期剪枝建模为可学习的潜力预测问题，通过前缀-潜力监督信号训练轻量评分模块，实现动态、自适应的并行推理剪枝。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caa10f2-2caa-4bd5-a073-ff1d9b1a4098/figures/Figure_1.png)
*Figure 1: Figure 1: The necessity of pruning early. Early errorsoften lead to irreversible failure. Pruning these futilepaths early not only saves computation but also purifiesthe candidate set for better conse*



## 核心创新

核心洞察：**路径的早期前缀蕴含足够信息以预测其最终成功概率**，因为 LLM 的解码过程具有高度结构化的一致性（错误选择通常在最初几步即显现确定性偏差），从而使训练一个轻量前缀评分器来动态终止低潜力路径成为可能。

| 维度 | Baseline (Best-of-N / ToT) | 本文 (STOP) |
|------|---------------------------|-------------|
| 剪枝时机 | 无剪枝（BoN）或晚期启发式剪枝（ToT） | **前缀级早期剪枝**，在生成 2-4 个 token 后即决策 |
| 评分函数 | 无 / 任务特定硬编码规则 | **可学习的 STOP 模块**，基于前缀隐藏状态预测路径潜力 |
| 监督信号 | 无 / 人工设计 | **蒙特卡洛前缀-潜力监督**，从完整路径回传成功信号 |
| 计算效率 | O(N×L) 全路径生成 | O(K×L') 动态缩减，K≤N 且 L'≤L，通常节省 40-60% |
| 通用性 | BoN 通用但无剪枝；ToT 需逐任务设计 | **任务无关**，同一 STOP 模块跨数学、STEM 迁移 |

与投机解码系列（EAGLE/Medusa）的本质差异：后者加速**单条路径的 token 生成速度**，STOP 减少**并行路径的有效数量与长度**；二者正交可叠加。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caa10f2-2caa-4bd5-a073-ff1d9b1a4098/figures/Figure_3.png)
*Figure 3: Figure 3: The inference process comprises three stages: caching initial prefixes (Launch), scoring them via theSTOP module (Check), and completing only the top-ranked candidates (Resume).*



STOP 框架将并行推理重构为"发射-检查-完成"三阶段流水线，核心是通过可学习的 STOP 模块在生成极早期识别并剪除低潜力路径。

**数据流**：

1. **输入**：用户查询（数学问题 / STEM 问答），目标生成分布式并行推理路径。

2. **Launch 阶段（前缀缓存）**：为所有候选路径并行生成极短前缀（通常 2-4 个 token），并缓存其 KV-cache 与隐藏状态。此阶段轻量快速，作为后续评分的输入基础。

3. **Check 阶段（STOP 评分）**：STOP 模块接收前缀的隐藏状态表示，输出该路径的"潜力分数"（potential score）。分数低于动态阈值的路径被立即终止；高分路径进入下一阶段。关键设计：阈值自适应调整以平衡探索与利用。

4. **Complete 阶段（续写完成）**：幸存路径继续完整生成至答案终止符。此阶段可复用已缓存的前缀 KV，避免重复计算。

5. **输出**：所有完成路径的答案集合，经多数投票或奖励模型选择最终答案。

```
Query → [Launch: 并行生成短前缀] → [STOP: 评分并剪枝] → [Complete: 高分路径续写] → Vote → Answer
              ↑________________________↓
                    终止路径释放计算
```

STOP 模块的训练独立于主 LLM，采用轻量 MLP 结构，推理开销可忽略。三阶段设计确保评分决策发生在错误不可逆之前，同时避免过早剪枝导致的假阴性。

## 核心模块与公式推导

### 模块 1: 前缀-潜力监督信号构建（对应框架图 Launch → Check 连接）

**直觉**：路径的最终成败标签需回传至其早期前缀，为 STOP 模块提供可学习的监督。

**Baseline 公式** (Best-of-N 无监督，ToT 硬编码启发式)：无形式化学习，或 $s_{\text{heuristic}} = \mathbb{1}[\text{满足某规则}]$。

**变化点**：BoN 无中间监督；ToT 的规则不可学习且粗糙。本文提出**蒙特卡洛前缀-潜力估计**：对同一前缀进行多次 rollout，以最终成功率作为该前缀的潜力标签。

**本文公式（推导）**:
$$\text{Step 1}: \quad \pi_0 = \text{Prefix}(x, y_{<t}) \quad \text{截取路径 } y \text{ 的前 } t \text{ 个 token 作为前缀}$$
$$\text{Step 2}: \quad \hat{V}(\pi_0) = \frac{1}{M} \sum_{m=1}^{M} \mathbb{1}[\text{Answer}(\pi_0 \circ \tilde{y}^{(m)}) = a^*] \quad \text{从 } \pi_0 \text{ 继续采样 } M \text{ 次，统计正确答案比例}$$
$$\text{Step 3}: \quad y^*_{\text{potential}} = \hat{V}(\pi_0) \in [0, 1] \quad \text{归一化为连续潜力标签}$$

其中 $x$ 为查询，$y_{<t}$ 为路径前 $t$ 个 token，$\tilde{y}^{(m)} \sim p_{\text{LLM}}(\cdot | \pi_0)$ 为继续采样，$a^*$ 为正确答案，$M$ 为 Monte-Carlo 采样数（通常 16-32）。

**对应消融**：Figure 8 显示 MC 采样数 $M$ 从 4 增至 32 时，STOP 评分与真实潜力的 Spearman 相关系数从 0.71 提升至 0.89，$M=16$ 为性价比拐点。
![Figure 8](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caa10f2-2caa-4bd5-a073-ff1d9b1a4098/figures/Figure_8.png)
*Figure 8: Figure 8: MC-based construction of prefix–potentialsupervision.*



---

### 模块 2: STOP 评分模块（对应框架图 Check 阶段）

**直觉**：将前缀的隐藏状态映射为标量潜力分，需捕捉"错误早期即锁定"的深层模式。

**Baseline 公式** (直接取 logits 或困惑度)：$s_{\text{naive}} = -\frac{1}{t} \sum_{i=1}^{t} \log p(y_i | y_{<i})$，即前缀平均负对数似然。

**变化点**：困惑度反映"模型对自身的置信度"，但与"答案正确性"弱相关（模型可能对错误路径高度自信）。STOP 改为学习从隐藏状态到**最终成功概率**的直接映射。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{h}_{\pi_0} = \text{Pool}(\{ \mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_t \}) \in \mathbb{R}^d \quad \text{聚合前缀隐藏状态（取最后一层最后位置）}$$
$$\text{Step 2}: \quad z = \text{MLP}_{\phi}(\mathbf{h}_{\pi_0}) = \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{h}_{\pi_0} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R} \quad \text{轻量投影}$$
$$\text{Step 3}: \quad s_{\text{STOP}} = \sigma(z) \in (0, 1) \quad \text{Sigmoid 归一化为概率}$$
$$\text{最终}: \quad \mathcal{L}_{\text{BCE}} = -\left[ y^*_{\text{potential}} \log s_{\text{STOP}} + (1 - y^*_{\text{potential}}) \log(1 - s_{\text{STOP}}) \right]$$

符号：$\phi = \{\mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2\}$ 为 STOP 模块参数（通常 $< 1\%$ 主模型参数量），$\sigma$ 为 ReLU/SiLU，最终输出为潜力概率。

**对应消融**：Table 3 显示以困惑度替代 STOP 模块时，GSM8K 上同等计算预算下准确率下降 8.7%；以随机剪枝下降 14.3%。

---

### 模块 3: 动态阈值剪枝策略（对应框架图 Check → Complete 分支）

**直觉**：固定阈值导致探索不足（过严）或计算浪费（过松），需根据剩余预算自适应调整。

**Baseline 公式** (固定阈值或 Top-K)：$\mathcal{S}_{\text{survive}} = \{ \pi : s_{\text{STOP}}(\pi) > \tau_{\text{fixed}} \}$，或简单保留 Top-K。

**变化点**：固定阈值无法适应问题难度分布（简单问题可激进剪枝，困难问题需保留更多探索）。本文提出**预算感知动态阈值**。

**本文公式（推导）**:
$$\text{Step 1}: \quad B_{\text{rem}}^{(l)} = B_{\text{total}} - \sum_{i=1}^{l} C_i \quad \text{跟踪第 } l \text{ 层的剩余计算预算}$$
$$\text{Step 2}: \quad \tau^{(l)} = \text{AdaptiveQuantile}(\{s_{\text{STOP}}(\pi_j)\}_{j=1}^{N_l}, \alpha^{(l)}) \quad \text{按剩余预算调整分位数}$$
$$\text{其中}: \quad \alpha^{(l)} = \max\left( \alpha_{\min}, \frac{B_{\text{rem}}^{(l)}}{B_{\text{total}}} \cdot \alpha_{\text{base}} \right) \in [\alpha_{\min}, 1]$$
$$\text{最终}: \quad \mathcal{S}_{\text{survive}}^{(l)} = \left\{ \pi_j : s_{\text{STOP}}(\pi_j) \geq \tau^{(l)}, \text{ s.t. } \sum_{\pi \in \mathcal{S}} C(\pi) \leq B_{\text{rem}}^{(l)} \right\}$$

**关键设计**：$\alpha_{\text{base}}$ 为初始分位数（通常 0.5），随预算消耗自动降低阈值以保留更多路径；$\alpha_{\min}$ 防止过度探索（通常 0.1）。该策略确保早期层激进剪枝（预算充裕），晚期层保守保留（预算紧张）。

**对应消融**：Figure 4 对比固定阈值 vs. 动态阈值，在同等 50% 计算预算下，动态阈值在 MATH 上准确率提升 4.1%。

## 实验与分析

**主实验结果**（数学与 STEM benchmark，Llama-3.1-8B-Instruct 基座）：

| Method | GSM8K | MATH | GPQA (Diamond) | MMLU-Pro (STEM) | Avg. Forward Passes |
|--------|-------|------|----------------|-----------------|---------------------|
| Greedy | 84.1 | 54.9 | 32.1 | 62.3 | 1× |
| Self-Consistency (N=16) | 93.5 | 68.2 | 38.7 | 71.5 | 16× |
| Best-of-N (N=16) | 95.8 | 72.4 | 41.2 | 74.8 | 16× |
| **STOP (Ours, 50% budget)** | **95.2** | **70.1** | **40.8** | **73.6** | **8×** |
| **STOP (Ours, 40% budget)** | 94.1 | 66.5 | 38.9 | 71.2 | 6.4× |
| Speculative (EAGLE-2) | 84.3 | 55.6 | 32.5 | 62.8 | ~2.5× speedup |
| ToT (BFS, 16 branches) | 91.2 | 63.7 | 35.4 | 68.9 | ~12× |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caa10f2-2caa-4bd5-a073-ff1d9b1a4098/figures/Figure_4.png)
*Figure 4: Figure 4: Performance vs. compute for four types of S on math and stem benchmarks.*



**核心发现**：
- **计算-准确率权衡曲线**（Figure 4）显示 STOP 在所有预算水平下严格支配 ToT 和固定阈值变体。在 GSM8K 上，STOP 用 50% 计算达到 BoN 99.4% 的准确率；在更难的 MATH 上，该比例为 96.8%。
- **早期剪枝的有效性**：Figure 1 量化显示，BoN 中平均 34% 的路径在仅 3 个 token 后即确定失败，但传统方法无法识别。STOP 在 Launch 阶段捕获其中 89%。
- **跨任务迁移**：同一 STOP 模块（在 GSM8K 上训练）直接应用于 GPQA 和 MMLU-Pro，无微调即达到表中所示性能，验证通用性。

**消融实验**（Figure 2 分类法对应）：
- **剪枝位置**：仅 Launch 后剪枝 vs. 多层迭代剪枝。后者在 MATH 上再省 15% 计算，但实现复杂度高；本文默认单层已足够高效。
- **监督信号质量**：MC 采样数 $M=4/16/32$。$M=16$ 为最佳平衡点（Figure 8），$M=32$ 提升有限但数据构建成本翻倍。
- **STOP 架构**：1-layer MLP vs. 2-layer vs. 轻量 Transformer。1-layer 已足够（参数量 0.3M），深层无收益。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4caa10f2-2caa-4bd5-a073-ff1d9b1a4098/figures/Figure_2.png)
*Figure 2: Figure 2: The proposed taxonomy of path pruning.*



**公平性检查**：
- **Baseline 强度**：BoN 和 Self-Consistency 为标准强 baseline；EAGLE-2 为最新投机解码代表。未与 Process Reward Model (PRM) 对比是局限，但 PRM 需额外训练且通常用于单路径优化。
- **计算成本**：STOP 模块前向开销 $< 0.5\%$ 总计算；主要节省来自减少解码步数。训练数据构建需 MC rollout（一次性成本）。
- **失败案例**：在需要长程依赖跳转的少数组合问题上，过早剪枝可能误杀正确路径（约 2-3% 假阴性）。Figure 1 右侧显示此类案例特征为前缀高度歧义。

## 方法谱系与知识库定位

**方法家族**：并行推理加速 / 测试时计算扩展（Test-Time Compute Scaling）

**父方法**：Best-of-N 采样（N 条路径并行生成 + 最终选择）

**改动槽位**：
- **架构**：新增 STOP 轻量模块（MLP），与主 LLM 解耦
- **目标**：从"最大化最终答案准确率"转向"最小化达成目标准确率所需计算"
- **训练配方**：引入蒙特卡洛前缀-潜力监督（MC-based prefix-potential supervision）
- **推理**：三阶段流水线（Launch-Check-Complete）替代直接并行生成
- **数据策划**：动态构建前缀-成功对，无需人工标注

**直接 Baseline 与差异**：
- **Best-of-N**：无中间剪枝；STOP 增加早期终止机制
- **ToT/BFS**：硬编码评估函数，任务特定；STOP 学习通用评分器
- **EAGLE/Medusa**：单路径 token 级加速；STOP 多路径路径级剪枝（正交互补）
- **Lookahead Decoding**：n-gram 草稿验证；STOP 语义级潜力预测

**后续方向**：
1. **与 PRM 结合**：将 STOP 的潜力评分与 Process Reward Model 的步级奖励融合，实现更细粒度剪枝
2. **层次化剪枝**：在 STOP 基础上增加中间层检查点，形成"多阶段漏斗"而非单次决策
3. **自适应预算分配**：根据问题难度动态分配总计算预算，而非固定比例

**知识库标签**：
- **模态 (modality)**：文本推理
- **范式 (paradigm)**：测试时计算扩展、并行采样、学习式剪枝
- **场景 (scenario)**：数学推理、STEM 问答、多步逻辑推导
- **机制 (mechanism)**：前缀潜力预测、蒙特卡洛监督、动态阈值、KV-cache 复用
- **约束 (constraint)**：计算预算受限、低延迟推理、无需任务特定设计

