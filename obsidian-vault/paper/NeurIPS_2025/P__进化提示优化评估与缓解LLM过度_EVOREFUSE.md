---
title: 'EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 进化提示优化评估与缓解LLM过度拒绝
- EVOREFUSE
acceptance: Poster
cited_by: 4
code_url: https://github.com/FishT0ucher/EVOREFUSE
method: EVOREFUSE
modalities:
- Text
paradigm: evolutionary optimization
---

# EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions

[Code](https://github.com/FishT0ucher/EVOREFUSE)

**Topics**: [[T__Text_Generation]], [[T__Benchmark_-_Evaluation]] | **Method**: [[M__EVOREFUSE]] | **Datasets**: EVOREFUSE-TEST vs baselines, EVOREFUSE-TEST with system prompt, Fine-tuning with EVOREFUSE-ALIGN, DPO with EVOREFUSE-ALIGN, Mutation strategy effectiveness

> [!tip] 核心洞察
> EVOREFUSE, an evolutionary prompt optimization algorithm, generates diverse pseudo-malicious instructions that consistently elicit confident refusals across LLMs, enabling both evaluation and mitigation of over-refusals through novel benchmark and alignment datasets.

| 中文题名 | 进化提示优化评估与缓解LLM过度拒绝 |
| 英文题名 | EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.23473) · [Code](https://github.com/FishT0ucher/EVOREFUSE) · [Project](待补充) |
| 主要任务 | Over-refusal Detection and Mitigation, Prompt Optimization, Benchmark / Evaluation |
| 主要 baseline | XSTEST, SGTEST, OR-BENCH, PHTEST/PH-GEN, HITEST, PromptAgent-rewritten |

> [!abstract]
> 因为「LLM 对语义无害的伪恶意指令(pseudo-malicious instructions)存在过度拒绝(over-refusal)现象，现有基准缺乏多样性和有效性」，作者在「AutoPrompt 自动提示生成」基础上改了「进化算法框架，引入9种情感/语义变异策略、重组与模拟退火选择，并构建 ELBO 拒绝概率目标」，在「9个 LLM 的 EvoRej@T(T) 基准」上取得「相比次优基准 SGTEST 平均拒绝触发率提升 140.41%」

- **关键性能 1**: EVOREFUSE-TEST 在 LLaMA-3.1-8B-Instruct 上 PRR 达 0.80，相比 SGTEST (0.17) 提升 370.59%，相比 XSTEST (0.13) 提升 515.38%
- **关键性能 2**: DPO 训练 with EVOREFUSE-ALIGN 降低过度拒绝 40.04%，安全性仅下降 3.87%
- **关键性能 3**: Scenario 变异策略成功率 20%，为9种策略中最高，Disgust/Prejudiced 仅 5%

## 背景与动机

大型语言模型在安全对齐过程中常表现出「过度拒绝」(over-refusal)：用户输入语义上完全无害的查询，仅因包含某些敏感关键词（如"bomb""attack"），模型便断然拒绝回答。例如，用户询问"How does a bomb calorimeter work?"（弹式量热器的工作原理），这是一个标准物理实验设备问题，但模型可能因"bomb"一词触发安全拦截。这种过度保守的行为严重损害了模型的有用性(helpfulness)。

现有方法主要从三个方向应对此问题：
- **XSTEST** [4]：构建测试套件识别夸张安全行为，但依赖人工设计模板，覆盖场景有限；
- **OR-BENCH** [7]：专门化过度拒绝基准，然而静态数据集难以持续发现新型伪恶意模式；
- **PH-GEN** [8]：自动生成伪有害提示评估错误拒绝，但采用简单重写或梯度搜索，探索路径狭窄、多样性不足。

这些方法的共同瓶颈在于**缺乏系统性的多样化探索机制**：要么依赖人工枚举（XSTEST/OR-BENCH），要么沿单一梯度方向优化（PH-GEN），无法全面覆盖触发过度拒绝的语义空间。此外，现有工作多聚焦于「评估」而忽视「缓解」——即使发现了过度拒绝问题，也缺乏高质量的对齐数据来微调模型。

本文提出 EVOREFUSE，首次将进化算法引入提示优化领域，通过多策略变异、重组与模拟退火选择，系统生成多样化伪恶意指令，并同步产出评估基准(EVOREFUSE-TEST)与对齐数据(EVOREFUSE-ALIGN)，实现从「发现」到「修复」的闭环。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/37cb1de9-761b-498a-b410-9ecf298f1356/figures/Figure_1.png)
*Figure 1 (qualitative): Top-5 tokens with highest information flow across Transformer layers for the word 'bomb' in jailbreak and normal contexts (Left) and word clouds for 8 behaviors (Right).*



## 核心创新

**核心洞察**：LLM 的过度拒绝源于对敏感关键词的过度关注而忽略上下文，因为现有搜索方法沿狭窄路径探索指令空间，从而使「情感/语义多样化的群体进化搜索」能够系统性地发现被忽视的伪恶意模式。

| 维度 | Baseline (AutoPrompt/PH-GEN) | 本文 (EVOREFUSE) |
|:---|:---|:---|
| 探索策略 | 梯度-based 搜索或简单重写，单一路径 | 9种变异策略(anger/controversial/despair/disgust/harmful/scenario/violent/prejudiced/other) + 重组，群体并行探索 |
| 优化目标 | 单一拒绝概率或无显式目标 | ELBO (evidence lower bound) 拒绝概率下界，兼顾多样性与收敛性 |
| 选择机制 | 贪婪选择或随机保留 | 模拟退火(τ₀=0.1, β=0.005)，温度调度平衡探索与利用 |
| 数据产出 | 静态基准或一次性生成 | 迭代进化产出 TEST(582条) 与 ALIGN(3000条)，支持评估+对齐训练 |
| 安全验证 | 人工检查或无 | GPT-4O 自动验证语义无害性，确保伪恶意指令不越界 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/37cb1de9-761b-498a-b410-9ecf298f1356/figures/Figure_2.png)
*Figure 2 (ablation): Validation of (1+1)-ES using XSTest as eval. Left: Robust over-refusal mitigation across various safety contexts with increasing iteration. Middle: Single-objective optimization. Right: Fitness trace across 30 independent runs with random seeds. Solid lines indicate mean values, while shaded regions represent standard deviation.*



EVOREFUSE 采用经典进化算法骨架，针对「伪恶意指令生成」任务定制各模块。数据流如下：

**输入**: 初始无害种子指令集合（如日常问答、科学问题）

→ **[1] 多策略变异引擎(Mutation Engine)**: 对当前种群中每个个体，随机应用9种情感/语义变异策略之一（如 Scenario 将问题嵌入虚构危险场景，Anger 注入愤怒情绪），输出多样化候选指令变体。

→ **[2] 适应度评估(Fitness Evaluation)**: 将候选指令输入目标 LLM，采样 K=10 次响应，计算触发拒绝的比例作为适应度分数。该估计作为真实拒绝概率的 ELBO 下界。

→ **[3] 重组(Recombination)**: 从种群中选取适应度最高的 L=4 个个体，进行 N=2 次交叉重组，产生融合多优质片段的后代指令。

→ **[4] 模拟退火选择(Simulated Annealing Selection)**: 以当前温度 τₜ = 0.1·exp(−0.005t) 计算接受概率，决定是否用后代替换父代，早期容忍较差解以维持多样性，后期趋于收敛。

→ **[5] 安全验证(Safety Verification)**: 通过 GPT-4O 审核进化后的指令，过滤真正有害的实例，仅保留「语义无害但触发拒绝」的伪恶意指令。

→ **[6] 数据集构建(Dataset Assembly)**: 验证后的指令搭配模型响应，组装为 EVOREFUSE-TEST(582条，纯评估) 或 EVOREFUSE-ALIGN(3000条，含响应用于 SFT/DPO)。

```
种子指令 ──→ [变异: 9策略] ──→ 候选池
                              ↓
                    [评估: K=10响应] → 适应度排序
                              ↓
                    [选择: Top-L=4] → [重组: N=2]
                              ↓
                    [模拟退火: τ₀=0.1, β=0.005]
                              ↓
                         迭代或收敛
                              ↓
                    [安全验证: GPT-4O]
                              ↓
              EVOREFUSE-TEST(582) / EVOREFUSE-ALIGN(3000)
```

## 核心模块与公式推导

### 模块 1: 适应度评估与 ELBO 目标（对应框架图 [2] Fitness Evaluation）

**直觉**: 单次查询目标 LLM 判断拒绝与否方差过大，需多次采样获得稳定估计，并将经验频率提升为概率下界以指导优化。

**Baseline 公式** (PH-GEN 等单点评估方法): 
$$\hat{p}_{\text{refuse}} = \mathbb{1}[\text{refuse}(M(x))]$$
符号: $M$ = 目标语言模型, $x$ = 候选指令, $\mathbb{1}[\cdot]$ = 指示函数（单次响应拒绝则为1，否则为0）。

**变化点**: 单次评估噪声大，且贪婪优化易导致过拟合特定模型响应；需要统计稳定的概率估计作为进化选择依据。

**本文公式（推导）**:
$$\text{Step 1}: \hat{p}_{\text{refuse}}^{(K)} = \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[\text{refuse}(M(x)_k)] \quad \text{K=10次采样降低方差，蒙特卡洛估计}$$
$$\text{Step 2}: P(\text{refuse}|x) \geq \text{ELBO}(x; M) = \hat{p}_{\text{refuse}}^{(K)} \quad \text{将经验频率解释为真实拒绝概率的证据下界}$$
$$\text{最终}: \text{Fitness}(x) = \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[\text{refuse}(M(x)_k)]$$

**对应消融**: Table 7 显示不同变异策略在该适应度框架下的成功率差异显著，Scenario 策略达 20%，验证 ELBO 目标能有效区分策略质量。

---

### 模块 2: 模拟退火选择（对应框架图 [4] Simulated Annealing Selection）

**直觉**: 纯贪婪选择会快速收敛到局部最优，丧失种群多样性；需引入温度机制，早期探索、后期利用，维持对指令空间的广泛覆盖。

**Baseline 公式** (标准进化策略的贪婪选择):
$$x_{\text{next}} = \text{arg}\max_{x \in \{\text{parents}\} \cup \{\text{offspring}\}} \text{Fitness}(x)$$

**变化点**: 贪婪选择在伪恶意指令空间中过早收敛，无法发现需要组合多种语义模式才能触发的隐蔽过度拒绝案例；需概率性接受劣解以跨越适应度峡谷。

**本文公式（推导）**:
$$\text{Step 1}: \Delta E = \text{Fitness}(x_{\text{new}}) - \text{Fitness}(x_{\text{old}}) \quad \text{计算能量差（适应度变化）}$$
$$\text{Step 2}: P(\text{accept}|x_{\text{new}}) = \exp\left(-\frac{\max(0, -\Delta E)}{\tau_t}\right) = \begin{cases} 1 & \text{if } \Delta E > 0 \\ \exp\left(\frac{\Delta E}{\tau_t}\right) & \text{if } \Delta E \leq 0 \end{cases}$$
$$\text{Step 3}: \tau_t = \tau_0 \cdot \exp(-\beta t) = 0.1 \cdot \exp(-0.005t) \quad \text{指数冷却，初始温度} \tau_0=0.1, \text{冷却系数} \beta=0.005$$
$$\text{最终}: x_{\text{next}} \sim \text{Categorical}\left(P(\text{accept}|x_{\text{new}})\right)$$

符号: $\tau_t$ = 第 $t$ 代温度, $\tau_0$ = 初始温度, $\beta$ = 冷却系数, $t$ = 迭代轮次。当 $\Delta E > 0$（新解更优）时必定接受；当 $\Delta E \leq 0$ 时以与温度相关的概率接受劣解，早期高温允许大幅跳跃，后期低温趋于稳定。

**对应消融**: Figure 2 验证 (1+1)-ES 在 XSTest 上的鲁棒性，显示随迭代增加过度拒绝缓解效果持续提升，证明温度调度有效避免早熟收敛。

---

### 模块 3: 重组操作与多策略变异（对应框架图 [1]+[3] Mutation + Recombination）

**直觉**: 单一变异策略只能探索局部语义邻域，组合多个优质个体的片段可发现更复杂的伪恶意模式；同时情感/场景维度的多样化变异是触发不同过度拒绝机制的关键。

**Baseline 公式** (AutoPrompt 的梯度-based 提示优化):
$$x' = x + \epsilon \cdot \nabla_x \mathcal{L}_{\text{task}}(M(x))$$
沿损失梯度方向微调提示嵌入，探索路径受梯度方向约束。

**变化点**: 梯度搜索沿窄路径进行，且需可微访问模型内部；本文改为离散文本空间上的进化操作，无需梯度，通过语义重组实现大范围跳跃。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{M} = \{\text{Anger}, \text{Controversial}, \text{Despair}, \text{Disgust}, \text{Harmful}, \text{Other}, \text{Prejudiced}, \text{Scenario}, \text{Violent}\} \quad \text{9种变异策略池}$$
$$\text{Step 2}: x' = m(x), \quad m \sim \text{Uniform}(\mathcal{M}) \quad \text{随机策略变异}$$
$$\text{Step 3}: \{x_{(1)}, x_{(2)}, x_{(3)}, x_{(4)}\} = \text{Top-4}(\text{Population}, \text{Fitness}) \quad \text{选择L=4个优质父代}$$
$$\text{Step 4}: x_{\text{child}}^{(i)} = \text{Crossover}(x_{(a)}, x_{(b)}), \quad a,b \sim \{1,2,3,4\}, i \in \{1,2\} \quad \text{N=2次重组}$$
$$\text{最终}: \text{Population}_{t+1} = \text{SA-Select}\left(\text{Population}_t \cup \{x_{\text{child}}^{(1)}, x_{\text{child}}^{(2)}\}\right)$$

**对应消融**: Table 7 显示各变异策略成功率：Scenario 0.20 > Violent 0.15 > Anger 0.14 > Other 0.12 > Despair 0.08 > Controversial 0.07 > Harmful 0.06 > Disgust 0.05 = Prejudiced 0.05。去掉 Scenario 策略后整体成功率下降最显著，验证多样化策略设计的必要性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/37cb1de9-761b-498a-b410-9ecf298f1356/figures/Table_1.png)
*Table 1 (result): Evaluations of red-teaming LLMs on EvoRej@T(T) benchmarks using PRR*



本文在两类任务上展开评估：一是**伪恶意指令的拒绝触发能力评估**，覆盖9个主流 LLM（含 GPT-4o、Claude-3.5-Sonnet、LLaMA-3.1-8B/70B-Instruct、Qwen2.5-72B-Instruct 等）；二是**过度拒绝缓解训练**，验证 EVOREFUSE-ALIGN 的对齐效果。

在拒绝触发评估中，EVOREFUSE-TEST 表现突出。以 LLaMA-3.1-8B-Instruct 为例（Table 8），在系统提示条件下 PRR（Positive Refusal Rate）达 0.80，CRR（Conditional Refusal Rate）达 0.74。相比之下，次优基准 SGTEST 仅 0.17/0.13，XSTEST 为 0.13/0.10，HITEST 仅 0.08/0.06。这意味着 EVOREFUSE-TEST 的拒绝触发率是 SGTEST 的 4.7 倍、XSTEST 的 6.2 倍。平均来看，EVOREFUSE-TEST 相比次优基准 SGTEST 提升 140.41%，相比更广泛的基准集合提升 85.34%。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/37cb1de9-761b-498a-b410-9ecf298f1356/figures/Table_3.png)
*Table 3 (comparison): Evaluations of prompt-based and alignment-based over-refusal mitigation methods on representative harmful behaviors. Best results are highlighted in bold. 'x' denotes toxic response. Lower PRR indicates better mitigation. Refuse rates are reported as False Refusal Rate (FRR) / Attack Success Rate (ASR) / Benign Refusal Rate (BRR).*



在过度拒绝缓解方面（Table 9），基于 EVOREFUSE-ALIGN 的监督微调(SFT)相比最佳微调基线降低过度拒绝 14.31%，相比最佳提示方法降低 14.76%。更值得注意的是 DPO（Direct Preference Optimization）训练：在系统提示条件下，DPO with EVOREFUSE-ALIGN 实现过度拒绝降低 40.04%，而安全性（ADVBENCH/HARMBENCH/JAILBREAKV 上的攻击成功率）仅下降 3.87%。这一 trade-off 显著优于现有对齐方法。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/37cb1de9-761b-498a-b410-9ecf298f1356/figures/Table_4.png)
*Table 4 (qualitative): Visualization of gradient scores for target tokens within representative pseudo-malicious instructions. Red indicates higher gradient scores, highlighting words that contribute most to over-refusal.*



消融实验进一步验证关键设计。Figure 2 展示 (1+1)-ES 在 XSTest 上的迭代优化过程，可见随迭代轮次增加，各类安全上下文下的过度拒绝缓解效果持续提升，证明模拟退火选择的收敛稳定性。Table 7 的变异策略消融显示，Scenario 策略贡献最大（20% 成功率），而 Disgust/Prejudiced 策略最弱（5%），说明**场景化嵌入**是触发过度拒绝的最有效语义模式。此外，将 GPT-4O 替换为开源模型 DarkIdol2 作为变异器后，PRR 从 72% 降至 46%（Table 10），表明方法对强变异器的依赖——这也是作者明确披露的限制之一。

公平性检查：本文比较的 baselines（XSTEST、SGTEST、OR-BENCH、PHTEST）均为该领域代表性基准，但缺少与近期强对抗方法（如 GCG、AutoDAN、PAIR、TAP）的比较，这些方法在越狱攻击中表现更强，可能也能发现过度拒绝模式。此外，实验重度依赖 GPT-4O 进行变异与安全验证，在完全开源链路中的适用性受限。作者亦承认系统提示条件会影响绝对指标，但相对排序保持稳定。

## 方法谱系与知识库定位

**方法家族**: 提示优化(Prompt Optimization) → 自动提示生成(AutoPrompt) → 进化提示优化(EVOREFUSE)

**父方法**: AutoPrompt [19] —— 通过自动生成的提示从语言模型中引出知识。EVOREFUSE 继承其「自动搜索有效提示」的核心思想，但将搜索空间从连续嵌入梯度优化扩展至离散文本空间的群体进化，引入多样性约束与重组机制。

**改动槽位**:
- **exploration_strategy**: 梯度/手动搜索 → 9策略变异 + 重组 + 模拟退火
- **objective**: 单一任务损失 → ELBO 拒绝概率下界 + 多样性感知
- **data_pipeline**: 静态数据集/单次生成 → 迭代进化(K=10, L=4, N=2) + 安全验证
- **training_recipe**: 标准 SFT → EVOREFUSE-ALIGN 进化数据 + DPO 对齐

**直接 baselines 及差异**:
- **PH-GEN [8]**: 同为伪有害提示生成，但用简单重写/梯度搜索；EVOREFUSE 以进化算法替代，多样性显著提升
- **XSTEST [4]**: 人工设计测试套件；EVOREFUSE 自动生成且拒绝触发率提升 6 倍以上
- **PromptAgent**: 提示重写用于微调；EVOREFUSE 直接优化拒绝概率并产出对齐数据

**后续方向**:
1. **自适应变异策略学习**: 当前9种策略为人工设计，未来可用元学习或 LLM 自身生成新策略
2. **跨模型迁移优化**: 降低对 GPT-4O 的依赖，开发开源模型可承载的完整链路
3. **动态安全边界估计**: 将 EVOREFUSE 与实时安全分类器结合，实现拒绝阈值的自适应校准

**标签**: 文本模态 / 进化优化范式 / 安全评估场景 / 群体搜索机制 / 安全-有用性权衡约束

