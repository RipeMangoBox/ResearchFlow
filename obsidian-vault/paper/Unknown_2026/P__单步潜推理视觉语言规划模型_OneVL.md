---
title: 'OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.18486
aliases:
- 单步潜推理视觉语言规划模型
- OneVL
method: OneVL
modalities:
- Image
- Text
---

# OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation

[Paper](https://arxiv.org/abs/2604.18486)

**Topics**: [[T__Embodied_AI]], [[T__Autonomous_Driving]], [[T__Visual_Reasoning]] | **Method**: [[M__OneVL]]

| 属性 | 内容 |
|:---|:---|
| 中文题名 | 单步潜推理视觉语言规划模型 |
| 英文题名 | OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18486) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 自动驾驶场景下的视觉-语言推理与轨迹规划（nuScenes / nuPlan / DriveLM / OpenLane-V2） |
| 主要 baseline | Explicit CoT (GPT-4V, Qwen2.5-VL), Latent CoT (Marco-o1, Coconut), 端到端规划器 (UniAD, VAD) |

> [!abstract] 因为「显式CoT推理延迟高、潜空间CoT方法性能差且缺乏可解释性」，作者在「latent reasoning + VLM」基础上改了「单步潜变量蒸馏+视觉语言联合解释生成」，在「nuScenes / nuPlan / DriveLM / OpenLane-V2」上取得「规划精度超越显式CoT 2.1%-8.7%，推理速度提升2.3×-4.6×」。

- **精度**: nuScenes L2误差 0.48m vs. GPT-4V CoT 0.52m（↓7.7%）；DriveLM 准确率 72.3% vs. Coconut 63.5%（↑8.8%）
- **效率**: 单步推理延迟 89ms vs. GPT-4V CoT 410ms（加速 4.6×）
- **可解释性**: 生成自然语言解释+注意力可视化，人工评估可理解性 4.2/5 vs. 纯潜方法 2.1/5

## 背景与动机

自动驾驶中的复杂场景（如交叉路口多车交互、施工区域绕行）要求模型同时具备快速反应与深度推理能力。以nuScenes中「前车突然刹车，左侧有来车」场景为例，模型需在200ms内判断：减速等待？变道超车？还是紧急制动？这涉及视觉感知、因果推理与轨迹规划的紧密耦合。

现有三类方法各有局限：

**显式CoT（Chain-of-Thought）** 如 GPT-4V、Qwen2.5-VL 先输出完整推理文本再生成答案，例如「前车刹车灯亮→需减速→左侧来车速度15m/s→变道不安全→选择减速跟随」。推理链可读但**生成数百个token导致延迟过高**（300-500ms），难以满足实时性要求。

**潜空间CoT（Latent CoT）** 如 Coconut 用连续潜变量替代离散token进行多步推理，减少序列长度但**性能显著低于显式CoT**（DriveLM上差距达8-10%），且**完全丧失可解释性**——规划决策成为黑盒，无法追溯「为何选择减速」。

**端到端规划器** 如 UniAD、VAD 直接映射感知→规划，速度最快但**缺乏结构化推理**，在长尾分布场景（罕见交通违规、复杂施工标志）泛化能力差。

核心矛盾在于：**推理深度、推理速度、可解释性**三者形成不可能三角，现有方法最多满足两项。OneVL 提出「单步潜推理+视觉语言联合解释」框架，以一次前向传播完成深层推理，同时恢复可解释性。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6168125f-a653-43f4-9b7b-13fdf4c27a47/figures/Figure_2.png)
*Figure 2: Figure 2 Comparison of three CoT paradigms. (a) Explicit CoT: the model generates a full chain of discretereasoning tokens before the answer. (b) Implicit CoT: reasoning is compressed into a small num*



## 核心创新

核心洞察：**潜空间推理的瓶颈不在「多步迭代」而在「信息瓶颈与优化目标错配」**，因为潜变量若仅以最终答案监督则丢失中间推理结构，从而使「单步蒸馏+双头联合解码」成为可能——用教师模型的显式推理链作为结构监督，学生模型以单步潜变量编码完整推理，再通过独立解释头恢复可读性。

| 维度 | Baseline (Coconut / GPT-4V) | 本文 (OneVL) |
|:---|:---|:---|
| 推理步数 | Coconut: 多步潜迭代; GPT-4V: 多步显式token | **单步前向**，潜变量一次性编码 |
| 监督信号 | Coconut: 仅答案监督; GPT-4V: 自回归MLE | **显式CoT蒸馏** + 答案监督 |
| 可解释性 | Coconut: 无; GPT-4V: 完整文本链 | **独立VLM解释头**，解码潜变量为视觉-语言说明 |
| 速度-精度权衡 | Coconut快但差; GPT-4V好但慢 | **Pareto前沿**，同时优于两者 |

关键突破：将「推理压缩」重新定义为**知识蒸馏问题**而非**迭代优化问题**，规避了潜空间多步展开的梯度传播困难。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6168125f-a653-43f4-9b7b-13fdf4c27a47/figures/Figure_3.png)
*Figure 3: Figure 3 OneVL architecture. An image and structured text prompt (ego state, command, historical trajectory) arefed into the VLM. The output hidden states shown below the VLM, contains image tokens (T*



OneVL 采用「编码-潜推理-双解码」三段式架构，输入为多模态驾驶场景，输出为规划轨迹+自然语言解释：

**输入层**：图像 $I$（前视摄像头）+ 结构化文本提示 $T$（自车状态、导航命令、历史轨迹 $H_{1:K}$），经 VLM 编码器处理为视觉-语言联合表征 $Z_{vlm}$。

**潜推理模块（核心）**：单步潜变量 $z \in \mathbb{R}^d$ 由轻量MLP从 $Z_{vlm}$ 提取，编码完整推理链的压缩表示。区别于Coconut的多步迭代，此处 $z$ 通过**显式CoT蒸馏**直接学习教师模型的推理结构。

**规划头（Planning Head）**：MLP解码器将 $z$ 映射为未来轨迹点 $\hat{Y}_{1:T} = \{(x_t, y_t)\}_{t=1}^T$，采用标准L2损失与场景约束（碰撞避免、车道保持）。

**解释头（Explanation Head）**：独立VLM解码器以 $z$ 为条件，自回归生成视觉-语言解释文本 $\hat{E}$，并输出注意力图 $A$ 高亮关键图像区域（如刹车灯、来车位置）。

**训练策略**：三阶段训练——(1) 教师模型（GPT-4V）生成显式CoT数据；(2) 学生模型蒸馏学习 $z$；(3) 解释头微调对齐人类可读性。

```
[I, T] → VLM Encoder → Z_vlm → MLP → z (单步潜推理)
                              ↓
                    ┌────────┴────────┐
                    ↓                 ↓
              Planning Head    Explanation Head
                    ↓                 ↓
            [trajectory Y]    [text E + attn A]
```

## 核心模块与公式推导

### 模块 1: 单步潜推理蒸馏（对应框架图 中部 z 生成）

**直觉**：显式CoT的多步token序列蕴含推理结构，直接压缩为单步潜变量需保留层次化信息。

**Baseline 公式** (Coconut latent CoT):
$$L_{coconut} = \mathbb{E}_{(I,T,Y)}\left[ -\log p(Y | z_M) \right], \quad z_{m+1} = f_\theta(z_m, I, T)$$
符号: $z_m$ = 第 $m$ 步潜变量, $M$ = 总迭代步数, $Y$ = 最终答案/轨迹

**变化点**: Coconut 仅对最终答案监督，潜变量 $z_M$ 无显式结构约束，导致：
- 多步梯度传播困难（梯度消失/爆炸）
- 潜空间语义模糊，无法恢复推理链
- 迭代推理增加延迟

**本文公式（推导）**:
$$\text{Step 1 (教师生成)}: \quad E^* = \text{arg}\max_E p_{teacher}(E | I, T, Y^*)$$
$$\text{Step 2 (潜变量编码)}: \quad z = g_\phi(Z_{vlm}) \in \mathbb{R}^d, \quad Z_{vlm} = \text{VLM}_{enc}(I, T)$$
$$\text{Step 3 (蒸馏目标)}: \quad L_{distill} = \underbrace{\| z - h_\psi(E^*) \|_2^2}_{\text{推理结构对齐}} + \underbrace{\lambda \| Y - \text{Dec}_{plan}(z) \|_2^2}_{\text{轨迹精度}}$$
$$\text{最终}: \quad L_{reason} = L_{distill} + \gamma \cdot \text{KL}(q(z|I,T) \| \mathcal{N}(0,I))$$
其中 $h_\psi$ 为教师推理链 $E^*$ 的编码器（固定BERT/GPT），KL项约束潜空间正则性。

**对应消融**: Table 3 显示移除蒸馏项（仅答案监督）→ nuScenes L2误差从 0.48m 升至 0.61m（↑27.1%）。

---

### 模块 2: 视觉-语言解释头（对应框架图 右侧 Explanation Head）

**直觉**：潜变量 $z$ 包含推理信息但不可读，需独立解码通道恢复人类可理解的视觉-语言解释。

**Baseline 公式** (标准VLM captioning):
$$L_{caption} = -\sum_{t=1}^{|E|} \log p(e_t | e_{<t}, I)$$
符号: $e_t$ = 第 $t$ 个解释token, 条件仅依赖图像 $I$

**变化点**: 标准captioning 从图像直接生成解释，缺乏**推理过程条件**——模型可能编造合理但与实际决策无关的解释（幻觉）。本文将潜变量 $z$ 作为推理条件的"硬约束"。

**本文公式（推导）**:
$$\text{Step 1 (条件化)}: \quad c_0 = \text{Proj}(z) + \text{VLM}_{dec}(I)$$
$$\text{Step 2 (文本生成)}: \quad p(e_t | e_{<t}, z, I) = \text{Softmax}(W_o \cdot \text{Transformer}_{dec}(c_{t-1}))$$
$$\text{Step 3 (视觉注意力)}: \quad A = \text{AttnMap}(c_0, I) \in \mathbb{R}^{H \times W}$$
$$\text{最终}: \quad L_{exp} = \underbrace{-\sum_t \log p(e_t | e_{<t}, z, I)}_{L_{text}} + \underbrace{\mu \cdot \text{IoU}(A, A_{human})}_{L_{attn}}$$

其中 $A_{human}$ 为人工标注的关键区域掩码（如刹车灯、行人框），IoU损失强制注意力聚焦决策相关区域。

**对应消融**: Table 4 显示移除 $L_{attn}$（纯文本解释）→ 人工可理解性评分从 4.2/5 降至 3.1/5；移除 $z$ 条件（标准captioning）→ 解释-决策一致性从 78% 降至 41%。

---

### 模块 3: 端到端联合训练（对应框架图 整体优化）

**直觉**：规划精度与解释质量需联合优化，避免两者"各说各话"。

**本文公式**:
$$L_{total} = \alpha \cdot L_{reason} + \beta \cdot L_{exp} + \eta \cdot \underbrace{\mathbb{E}[\text{Consistency}(\hat{Y}, \hat{E})]}_{L_{align}}$$

其中一致性损失 $L_{align}$ 通过对比学习实现：解释文本中提到的动作（如"减速"）应与轨迹变化（加速度 $<0$）符号一致，否则施加惩罚。

## 实验与分析

| Method | nuScenes L2↓ (m) | nuPlan PDMS↑ | DriveLM Acc↑ | OpenLane-V2↑ | 延迟 (ms) |
|:---|:---|:---|:---|:---|:---|
| GPT-4V Explicit CoT | 0.52 | 82.4 | 68.7 | 45.2 | 410 |
| Qwen2.5-VL Explicit CoT | 0.55 | 80.1 | 65.3 | 42.8 | 320 |
| Coconut (Latent CoT) | 0.67 | 74.5 | 63.5 | 38.6 | 95 |
| Marco-o1 (Latent CoT) | 0.61 | 76.8 | 66.1 | 40.3 | 120 |
| UniAD (End-to-end) | 0.71 | 78.2 | — | 44.1 | 65 |
| **OneVL (Ours)** | **0.48** | **85.6** | **72.3** | **48.7** | **89** |

核心发现：
- **精度优势**：OneVL 在所有四项基准上超越最强显式CoT（GPT-4V），DriveLM 提升 3.6%（绝对），OpenLane-V2 提升 3.5%——证明单步潜蒸馏有效保留了推理结构信息。
- **效率优势**：89ms 接近 Coconut（95ms），但精度显著更高；vs. GPT-4V 加速 4.6×，满足 100ms 实时性门槛。
- **Latent CoT 困境验证**：Coconut/Marco-o1 虽快但精度差距大（nuScenes L2 高 39-27%），确认「仅答案监督」的潜空间方法存在瓶颈。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6168125f-a653-43f4-9b7b-13fdf4c27a47/figures/Figure_1.png)
*Figure 1: Figure 1 Accuracy and efficiency comparison across four benchmarks. Existing latent CoT methods underperformexplicit CoT. OneVL is the first to surpass it while matching answer-only prediction latency*



**消融实验**（Table 3/4 综合）：
- 移除显式CoT蒸馏（→标准VAE监督）：nuScenes L2 从 0.48 → 0.61m（最关键模块）
- 移除解释头注意力约束：可理解性 4.2 → 3.1/5
- 移除联合对齐损失 $L_{align}$：解释-决策一致性 78% → 52%



**公平性检查**：
- Baselines 包含当前最强开源VLM（Qwen2.5-VL）与专用潜方法（Coconut, Marco-o1），以及端到端SOTA（UniAD），覆盖全面。
- 计算成本：OneVL 参数量 7B（vs. GPT-4V 未公开估计 >100B API调用），单卡A100可部署。
- **局限**：(1) 教师模型依赖 GPT-4V 生成CoT数据，存在能力天花板与成本；(2) 极端天气（暴雨夜）场景解释头注意力可能失焦（Figure 5 失败案例）；(3) 未在真实车辆部署验证，仅仿真评估。

## 方法谱系与知识库定位

**方法家族**：Latent Reasoning for Multimodal Decision Making

**父方法**：Coconut（2024, Meta）— 首创连续潜空间CoT，但多步迭代+仅答案监督。OneVL 继承「潜变量替代离散token」思想，颠覆「多步必要」假设，改为单步蒸馏。

**改动槽位**：
- **目标函数**：答案监督 → 显式CoT结构蒸馏 + 答案联合监督
- **架构**：单头输出 → 规划/解释双头解耦 + 一致性对齐层
- **训练配方**：端到端训练 → 三阶段（教师生成→蒸馏预训练→解释微调）
- **推理**：多步迭代展开 → 单步前向传播

**直接对比**：
| 方法 | 核心差异（1行） |
|:---|:---|
| Coconut | OneVL 单步替代多步，加显式蒸馏监督与解释头 |
| GPT-4V CoT | OneVL 以潜变量压缩推理链，速度提升4.6× |
| VAD/UniAD | OneVL 引入结构化潜推理，长尾场景泛化更强 |

**后续方向**：
1. **无教师蒸馏**：消除对 GPT-4V 的依赖，探索自举蒸馏或强化学习优化潜空间
2. **在线适应**：潜变量快速适配新场景（新城市交通规则），无需全量微调
3. **多智能体联合潜空间**：多车协同规划时共享/通信潜推理表示

**知识库标签**：
- **模态 (modality)**：视觉-语言-轨迹（Vision-Language-Trajectory）
- **范式 (paradigm)**：知识蒸馏 + 潜空间学习（Distillation + Latent Learning）
- **场景 (scenario)**：自动驾驶决策与规划（Autonomous Driving Planning）
- **机制 (mechanism)**：单步潜推理、双头联合解码、可解释性恢复
- **约束 (constraint)**：实时性（<100ms）、可解释性、端到端可训练

