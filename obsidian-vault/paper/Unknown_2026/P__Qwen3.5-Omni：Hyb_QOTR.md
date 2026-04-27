---
title: Qwen3.5-Omni Technical Report
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.15804
aliases:
- Qwen3.5-Omni：Hybrid-Attention MoE全模态大模型
- QOTR
- 核心变化是将 Thinker 和 Talker 从 dense att
modalities:
- Text
---

# Qwen3.5-Omni Technical Report

[Paper](https://arxiv.org/abs/2604.15804)

**Topics**: [[T__Speech_Processing]], [[T__Video_Understanding]]

> [!tip] 核心洞察
> 核心变化是将 Thinker 和 Talker 从 dense attention 升级为 Hybrid-Attention MoE，以稀疏激活换取长序列推理效率，从而解锁 256k 上下文建模能力。ARIA 则以最小侵入性解决了流式语音合成中文本/语音 token 速率不匹配这一具体工程瓶颈。两者有效的原因在于：MoE 的稀疏性与 Hybrid Attention 的局部-全局切换天然适配超长多模态序列的计算需求；ARIA 的交错对齐策略直接消除了编码效率差异导致的合成抖动，而无需重新设计整个语音解码器。

| 中文题名 | Qwen3.5-Omni：Hybrid-Attention MoE全模态大模型 |
| 英文题名 | Qwen3.5-Omni Technical Report |
| 会议/期刊 | arXiv预印本 (2026) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15804) · [Code] · [Project] |
| 主要任务 | 全模态理解（文本/图像/音频/视频）、流式语音合成（TTS）、语音识别（ASR）、音视频对话、语音克隆、代码生成 |
| 主要 baseline | Qwen3-Omni、Gemini-3.1 Pro、Qwen3.5-Plus-Nothink |

> [!abstract]
> 因为「dense attention架构在处理超长多模态序列时推理效率低下，且流式语音合成中文本与语音token编码效率不匹配导致韵律失真」，作者在「Qwen3-Omni的Thinker-Talker架构」基础上改了「将双模块升级为Hybrid-Attention MoE，并引入ARIA动态对齐机制与多码本codec」，在「MMAU、MMAR、VoiceBench、URO-Bench-pro」上超越Gemini-3.1 Pro，在215个音频及音视频子任务上取得SOTA。

- **MMAU/MMAR/VoiceBench/URO-Bench-pro**：Qwen3.5-Omni-Plus 超越 Gemini-3.1 Pro（具体数值待补充）
- **纯文本能力保留**：与 Qwen3.5-Plus-Nothink 在 MMLU-Pro、GPQA、LiveCodeBench v6、IFEval 上表现相当
- **长上下文扩展**：256k token，支持10小时音频输入与400秒720P视频（1 FPS采样）

## 背景与动机

当前全模态大语言模型（Omni LLM）正从实验室走向实际部署，但在成为真正可用的智能体过程中面临系统性瓶颈。以实时音视频对话场景为例：用户上传一段10小时的会议录音并要求模型实时总结、同时以自然语音流式回复——现有模型要么因序列过长而推理崩溃，要么语音输出出现明显的"卡顿-加速"节奏紊乱，仿佛机器在"喘不过气"地说话。

现有方法如何应对？**Qwen3-Omni** 作为直接前代，采用 Thinker-Talker 双模块架构：Thinker负责多模态理解与文本生成，Talker负责语音解码，两者通过特定接口协同。该设计初步实现了端到端全模态能力，但采用 dense attention，每 token 计算量随序列长度线性增长，导致长上下文推理成本过高。**Gemini 系列** 在音视频理解上表现强劲，但其架构细节未完全公开，且流式语音交互的实时性优化路径与 Qwen 系列不同。**GPT-4o** 同样具备全模态能力，但语音模式采用独立 pipeline 而非完全端到端，延迟与一致性控制存在 trade-off。

这些方法的共同短板在于三方面：**其一**，dense attention 的二次复杂度使 256k 级别长上下文成为计算噩梦，10小时音频或400秒视频的 token 序列远超高效推理的舒适区；**其二**，文本 tokenizer（如 BPE）与语音 tokenizer（如 SoundStream）的编码效率存在数量级差异——一个文本词可能对应数个 token，而一秒语音可能对应数十个声学 token，这种密度不匹配在流式生成时造成文本"等待"语音或语音"追赶"文本的节律失衡；**其三**，全模态联合训练往往以牺牲纯文本推理能力为代价，形成"多模态增益、单模态受损"的尴尬权衡。

Qwen3.5-Omni 的核心动机正是以最小架构侵入性解决上述瓶颈：用稀疏化架构解锁长上下文，用轻量对齐插件修复流式语音节奏，用精细数据配方维持能力平衡。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d60253c4-1efe-4479-a703-a99db98ba48e/figures/Figure_1.png)
*Figure 1: Figure 1: Qwen3.5-Omni is a unified end-to-end model capable of processing multiple modalities, suchas text, audio, image and video, and generating real-time text or speech response. Based on thesefea*



## 核心创新

**核心洞察**：将 Thinker 与 Talker 同时从 dense attention 升级为 Hybrid-Attention MoE，以稀疏激活降低每 token 计算量，同时以局部-全局动态注意力机制维持长距离依赖建模精度，从而使 256k token 超长多模态序列的高效端到端推理成为可能；ARIA 则以交错插入策略在 Talker 解码阶段动态平衡文本与语音 token 的生成速率，无需重构整个语音解码器即可消除流式合成的节律抖动。

| 维度 | Baseline (Qwen3-Omni) | 本文 (Qwen3.5-Omni) |
|:---|:---|:---|
| **核心架构** | Dense Attention (Thinker + Talker) | Hybrid-Attention MoE (Thinker + Talker) |
| **长上下文能力** | 受限于 dense 计算，实际支持较短 | 256k token，10h音频/400s视频 |
| **语音流式生成** | 直接串行解码，文本-语音速率不匹配导致韵律失真 | ARIA 动态交错对齐 + 多码本 codec 单帧即时合成 |
| **训练数据规模** | 较少多语言语音数据 | 4000万小时监督数据，ASR 113种语言，TTS 36种语言 |
| **纯文本能力保留** | 未明确验证 | 与 Qwen3.5-Plus-Nothink 相当（置信度0.80） |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d60253c4-1efe-4479-a703-a99db98ba48e/figures/Figure_2.png)
*Figure 2: Figure 2: The overview of Qwen3.5-Omni. Qwen3.5-Omni adopts the Thinker-Talker architecture.Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens byreceives*



Qwen3.5-Omni 延续并升级了 Qwen3-Omni 的 **Thinker-Talker** 双模块架构，整体数据流如下：

**输入层**：接收任意组合的多模态输入——文本 token、图像 patch、音频波形（通过 AuT 编码器）、视频帧序列。所有模态统一编码为兼容的 token 序列。

**Thinker（思考者）**：核心认知模块，负责多模态理解、推理与文本生成。输入为多模态 token 序列，输出为文本 token 流（隐藏状态同时传递给 Talker）。**关键升级**：从 dense attention 替换为 **Hybrid-Attention MoE**，在全局注意力与局部滑动窗口注意力间动态切换，MoE 层稀疏激活以降低计算量。

**Talker（说话者）**：语音生成模块，接收 Thinker 输出的文本隐藏状态，实时合成语音波形。包含三个关键子组件：
- **ARIA（Adaptive Rate Interleave Alignment）**：动态监测文本 token 与语音 token 的生成速率差异，通过交错插入策略调整解码节奏；
- **多码本 codec**：采用 multi-codebook 表示，实现单帧即时语音合成，降低首包延迟；
- **语音解码器**：将 codec 表示转换为最终波形输出。

**输出层**：同时输出文本流与语音流，支持实时打断（turn-taking intent recognition）与语义级交互控制。

```
[多模态输入] ──→ [统一编码] ──→ [Thinker: Hybrid-Attention MoE] ──┬──→ [文本输出]
                                      ↑                           │
                                      └──────── [隐藏状态传递] ────┼──→ [Talker]
                                                                   │     ├── [ARIA: 速率对齐]
                                                                   │     ├── [多码本codec: 单帧合成]
                                                                   │     └── [语音解码器]
                                                                   └──→ [语音输出]
```


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d60253c4-1efe-4479-a703-a99db98ba48e/figures/Figure_3.png)
*Figure 3: Figure 3: The overview of AuT. Consuming 40 million hours of supervised data especially more multi-lingual data, AuT encoder in Qwen3.5-Omni obtain stronger general purpose audio representation in6.25*



## 核心模块与公式推导

### 模块 1: Hybrid-Attention MoE（Thinker/Talker 核心，对应框架图 Thinker/Talker 主体）

**直觉**：超长多模态序列中，并非所有位置都需要全局注意力；MoE 的稀疏激活可将计算聚焦于"专家"子网络，Hybrid Attention 则在全局与局部模式间自适应切换，两者结合实现线性-次线性复杂度的长上下文建模。

**Baseline 公式** (Dense Attention, Qwen3-Omni):
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

符号: $Q,K,V$ = 查询/键/值矩阵; $d_k$ = 键维度; 输出维度 = $d_{model}$. Dense 模式下每层所有参数参与计算，序列长度 $L$ 时注意力复杂度为 $O(L^2 \cdot d)$。

**变化点**: Qwen3-Omni 的 dense attention 在 $L=256k$ 时 $L^2$ 项导致内存与计算爆炸；且全局注意力对局部声学/视觉模式存在过度建模。本文引入双重稀疏化：

**本文公式（推导）**:
$$\text{Step 1: 路由选择}: \quad g(x) = \text{TopK}(\text{softmax}(W_g \cdot x), K_{exp})$$
$$\text{仅激活 } K_{exp} \text{ 个专家，计算量降为 } O(K_{exp}/N_{total}) \text{ 倍}$$

$$\text{Step 2: Hybrid Attention 切换}: \quad \text{Attn}_{hybrid} = \alpha(x) \cdot \text{Attn}_{global} + (1-\alpha(x)) \cdot \text{Attn}_{sliding}$$
$$\text{其中 } \alpha(x) = \sigma(W_\alpha \cdot x), \text{ 对长距离依赖位置趋近1，局部模式趋近0}$$

$$\text{最终}: \quad \text{MoE-Layer}(x) = \sum_{i \in \text{TopK}} g(x)_i \cdot E_i(\text{Attn}_{hybrid}(x)) + x$$

符号: $W_g$ = 门控网络; $K_{exp}$ = 激活专家数（通常2）；$N_{total}$ = 总专家数；$\alpha(x)$ = 动态混合系数；$E_i$ = 第 $i$ 个专家网络；残差连接 $+x$ 保证梯度稳定。

**对应消融**: 论文未提供 MoE 相对 dense 基线的直接效率量化数据（如推理速度up或内存节省比例），该部分。

---

### 模块 2: ARIA — Adaptive Rate Interleave Alignment（Talker 解码插件，对应框架图 Talker 内 ARIA 模块）

**直觉**：文本 tokenizer 的压缩率（~0.1-0.3 token/字符）与语音 codec 的压缩率（~50-100 token/秒）存在数量级错配，直接串行解码会导致语音段过长"拖慢"文本节奏，或文本过快"抽空"语音缓冲；ARIA 通过动态缓冲与交错插入实现速率均衡。

**Baseline 形式** (Qwen3-Omni 直接串行解码):
$$\text{Decode}_{base}: \quad t_1, t_2, ..., t_n \rightarrow s_1, s_2, ..., s_m \quad \text{（固定映射，无速率调控）}$$

符号: $t_i$ = 第 $i$ 个文本 token; $s_j$ = 第 $j$ 个语音 token/frame; 映射关系由训练分布隐式决定。

**变化点**: 流式场景下 $t_i$ 与 $s_j$ 的生成速率 $r_t = dT/dt$、$r_s = dS/dt$ 动态变化且 $r_s \gg r_t$，固定映射导致韵律断裂。ARIA 引入显式速率监测与缓冲调度：

**本文公式（推导）**:
$$\text{Step 1: 速率估计}: \quad \hat{r}_t^{(t)} = \frac{\Delta N_t}{\Delta \tau}, \quad \hat{r}_s^{(t)} = \frac{\Delta N_s}{\Delta \tau}$$
$$\text{基于最近窗口的 token 产出速率滑动平均}$$

$$\text{Step 2: 缓冲差计算}: \quad B_t = \int_0^t (\hat{r}_s(\tau) - \kappa \cdot \hat{r}_t(\tau)) d\tau$$
$$\text{其中 } \kappa = \mathbb{E}[|s|/|t|] \text{ 为语料级平均语音-文本长度比，作为目标对齐系数}$$

$$\text{Step 3: 交错决策}: \quad \text{Insert}_j = \mathbb{1}[B_{t_j} > \theta_{upper}] \cdot \text{Pad}_s + \mathbb{1}[B_{t_j} < \theta_{lower}] \cdot \text{Skip}_s$$
$$\text{当缓冲过剩时插入填充帧，不足时跳过/压缩非关键语音段}$$

$$\text{最终输出}: \quad \{s_{out}\} = \text{Interleave}(\{t_i\}, \{s_j\}, \{B_t, \theta_{upper}, \theta_{lower}\})$$

**对应消融**: 论文未提供 ARIA 的独立消融实验，其贡献无法从整体性能提升中分离。

---

### 模块 3: 多码本 Codec 与 AuT 编码器（语音侧，对应框架图 AuT 模块）

**直觉**：单码本语音表示需要多步迭代才能生成一帧波形，延迟高；多码本并行表示可在单步内同时预测多个子带，实现"单帧即时合成"。

**Baseline 形式** (传统单码本或逐帧自回归):
$$\hat{c}_t = \text{arg}\max_c P(c | c_{<t}, \text{context}) \quad \text{（逐帧自回归，T步延迟）}$$

**变化点**: 多码本将单帧分解为 $M$ 个并行子带码本，每步同时预测 $M$ 个离散 token：

**本文公式**:
$$\text{Frame}_t = (c_t^{(1)}, c_t^{(2)}, ..., c_t^{(M)}) \sim P(c^{(1:M)} | \text{context})$$
$$\text{重构}: \quad \hat{x}_t = \text{CodecDecoder}(\text{VQ-Lookup}(c_t^{(1:M)}))$$

AuT 编码器侧： consuming 4000万小时监督数据（尤其增加多语言数据），编码器输出与 Thinker 对齐的语义表示。

**对应消融**: 多码本 codec 的独立延迟降低数据、AuT 多语言扩展的边际增益。

## 实验与分析

主实验结果覆盖 215 个音频及音视频子任务，以下为关键基准对比：

| Method | MMAU | MMAR | VoiceBench | URO-Bench-pro | 纯文本 (MMLU-Pro/GPQA/LiveCodeBench/IFEval) |
|:---|:---|:---|:---|:---|:---|
| Gemini-3.1 Pro | 
| Qwen3.5-Omni-Plus | **超越** | **超越** | **超越** | **超越** | 与 Qwen3.5-Plus-Nothink **相当** |
| Qwen3-Omni | 



**核心发现分析**：

1. **音视频理解优势**：Qwen3.5-Omni-Plus 在 MMAU（音频理解）、MMAR（音视频推理）、VoiceBench（语音对话）、URO-Bench-pro（语音鲁棒性）上均超越 Gemini-3.1 Pro。这一结果主要支撑了「MoE 架构升级+数据扩展带来全模态能力跃升」的核心声明。但需注意，主要竞争基线仅为 Gemini-3.1 Pro，**未见与 GPT-4o、Gemini-2.0 Flash 等同期模型的系统性对比**。

2. **纯文本能力保留**：与 Qwen3.5-Plus-Nothink（同规模纯文本模型，无思维链）在 MMLU-Pro、GPQA、LiveCodeBench v6、IFEval 上表现相当，支持「全模态训练不损害文本能力」的声明。但此处基线为 **Nothink 版本**（无思维链），若与完整推理版本对比结论可能不同；且该结论置信度标注为 0.80，非绝对成立。

3. **215 个 SOTA 的数字膨胀风险**：该数字包含大量语言特定 ASR/S2TT 子任务（113 种语言 ASR），细分粒度过高可能夸大整体领先幅度。

**消融与公平性检查**：

- **ARIA 与多码本 codec**：均**缺乏消融实验**，无法从整体提升中分离其独立贡献（Table 。
- **MoE 效率增益**：论文声称稀疏激活降低每 token 计算量，但**未提供相对 dense 基线的直接量化数据**（如推理吞吐量、内存占用、端到端延迟）。
- **自评基准偏向**：OmniGAIA、OmniCloze、SongFormBench 等基准由作者团队关联方发布，存在基准选择偏向风险。
- **Audio-Visual Vibe Coding**：「涌现」声明缺乏严格定义和对照实验，置信度仅 0.45，定性存疑。
- **计算成本**：模型规模与训练成本未完全披露，MoE 的专家并行可能引入额外通信开销，实际部署效率待验证。

## 方法谱系与知识库定位

**方法家族**：端到端全模态大语言模型（Omni LLM）→ 基于双模块解耦（理解-生成分离）的架构路线。

**父方法**：**Qwen3-Omni**（Thinker-Talker 架构的首次提出者）。Qwen3.5-Omni 继承其双模块拓扑，但将核心计算单元从 dense attention 替换为 Hybrid-Attention MoE，属于「同架构家族内的结构性升级」。

**关键 slot 变更**：
| Slot | 变更内容 |
|:---|:---|
| **架构** | Dense Attention → Hybrid-Attention MoE（唯一结构性变更） |
| **目标函数** | 未明确变更，沿用联合训练目标 |
| **训练配方** | 数据扩展至 4000万小时，ASR 113种语言，TTS 36种语言 |
| **数据策划** | 多语言语音数据显著增加 |
| **推理** | 新增 ARIA 插件、多码本 codec 单帧合成；上下文扩展至 256k |

**直接基线与差异**：
- **Qwen3-Omni**：父代，dense attention，无 ARIA，无多码本 codec，短上下文
- **Gemini-3.1 Pro**：主要竞争基线，架构未公开，Qwen3.5-Omni 在语音专项基准上超越但对比维度有限
- **GPT-4o**：语音模式非完全端到端，Qwen3.5-Omni 强调流式实时性与长上下文

**后续方向**：
1. **可验证的模块贡献**：为 ARIA、多码本 codec、MoE 各自设计独立消融，量化分离各组件边际增益；
2. **更广泛的基线对比**：纳入 GPT-4o、Gemini-2.0 Flash、Claude 等同期模型的 head-to-head 评测；
3. **推理效率实证**：提供 MoE 相对 dense 在 256k 上下文下的端到端延迟、吞吐量、内存占用数据，验证稀疏化的实际部署价值；
4. **涌现能力严格验证**：对 Audio-Visual Vibe Coding 等「涌现」声明设计对照实验，明确定义评估标准。

**知识库标签**：
- **模态**：text / image / audio / video / speech（全模态）
- **范式**：end-to-end unified model, dual-module (Thinker-Talker)
- **场景**：real-time streaming conversation, long-context multimodal understanding, zero-shot voice cloning
- **机制**：Mixture-of-Experts (MoE), hybrid local-global attention, adaptive rate alignment, multi-codebook neural codec
- **约束**：长上下文效率（256k）、流式低延迟、多语言覆盖（113种ASR）、能力权衡（全模态vs纯文本）

