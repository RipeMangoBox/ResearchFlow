---
title: Micro Language Models Enable Instant Responses
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19642
aliases:
- 微语言模型实现即时响应
- MLMEIR
code_url: https://github.com/Sensente/micro_language_model_swen_project
modalities:
- Text
---

# Micro Language Models Enable Instant Responses

[Paper](https://arxiv.org/abs/2604.19642) | [Code](https://github.com/Sensente/micro_language_model_swen_project)

**Topics**: [[T__Text_Generation]], [[T__Compression]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | 微语言模型实现即时响应 |
| 英文题名 | Micro Language Models Enable Instant Responses |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19642) · [Code](https://github.com/Sensente/micro_language_model_swen_project) · [Project](https://arxiv.org/abs/2604.19642) |
| 主要任务 | 低延迟对话响应生成、边缘-云端协同推理 |
| 主要 baseline | 独立云端 LLM (GPT-4, Claude 等)、 speculative decoding、prompt caching |

> [!abstract] 因为「云端大模型首 token 延迟高（数百毫秒至数秒）导致对话体验卡顿」，作者在「独立云端 LLM」基础上改了「引入设备端微语言模型 µLM 预生成响应开头，云端 LLM 续写完成」，在「对话响应延迟与用户满意度」上取得「首 token 延迟从 500ms+ 降至 <50ms，用户偏好度提升显著」

- 关键性能 1：µLM 参数量仅 100M-1B 级别，可在手机/IoT 设备实时运行
- 关键性能 2：µLM+LLM 组合响应被用户偏好的比例显著高于独立 LLM
- 关键性能 3：三种错误恢复机制保障 µLM 预生成失败时的系统鲁棒性

## 背景与动机

现代对话系统的核心痛点是「等待焦虑」：用户发送消息后，云端大模型需要数百毫秒甚至数秒才能输出第一个 token，造成明显的对话卡顿感。例如，在语音助手或实时聊天场景中，每增加 100ms 延迟都会显著降低用户感知流畅度。

现有方法主要从三个方向缓解这一问题：

**Speculative decoding**（Leviathan et al., 2023; Chen et al., 2023）：使用小型 draft model 并行生成候选 token，再由大模型验证接受。该方法可减少总解码步数，但 draft model 仍需在每次查询时从头运行，首 token 延迟改善有限。

**Prompt caching / KV-cache 复用**：预计算并复用前缀的 key-value 状态，避免重复计算。该方法对多轮对话中重复上下文有效，但对新查询的首 token 延迟无直接帮助。

**模型量化与边缘推理**：将大模型压缩至边缘设备运行，但即使 7B 级别的模型在手机端推理延迟仍在数百毫秒量级，且质量损失明显。

这些方法的共同局限是：**首 token 延迟（Time-To-First-Token, TTFT）仍受限于云端模型本身的计算延迟或边缘模型的能力天花板**，没有从根本上「隐藏」延迟——即让用户感知不到等待的存在。

本文的核心动机是：能否让设备端一个极小的语言模型（µLM）在用户完成输入的瞬间立即开始生成响应开头，同时云端大模型在后台接管并续写，从而将「等待时间」转化为「已读内容」，实现心理学层面的即时响应？


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31989539-8ffc-4fb8-8c7a-7b176958cabd/figures/Figure_1.png)
*Figure 1: Figure 1: The on-device micro language model µLMinitiates the response, which the cloud LLM continues.*



## 核心创新

核心洞察：设备端微模型（µLM）的「快速启动」与云端大模型（LLM）的「高质量续写」可以形成互补时序流水线，因为人类阅读速度（~250 token/分钟）远低于模型生成速度，从而使 µLM 预生成的数百毫秒内容足以覆盖 LLM 的启动延迟，实现感知层面的零等待。

与 baseline 的差异：

| 维度 | Baseline（独立云端 LLM） | 本文（µLM + LLM） |
|:---|:---|:---|
| 首 token 来源 | 云端模型计算完成后输出 | 设备端 µLM 即时生成（<50ms） |
| 延迟隐藏机制 | 无，用户直接等待 | 利用人类阅读时间掩盖云端启动延迟 |
| 质量保障策略 | 单一模型，无切换 | 三种错误恢复模式处理 µLM-LLM 衔接失败 |
| 部署架构 | 纯云端或纯边缘 | 边缘-云端协同流水线 |
| 模型规模要求 | 全量 LLM 必须在低延迟下运行 | µLM 可小至 100M 参数，LLM 无延迟约束 |

## 整体框架



µLM+LLM 框架的数据流如下：

**输入**：用户查询（文本或语音转录结果）进入设备端。

**模块 A - µLM 预生成（设备端）**：参数量 100M-1B 的微语言模型基于查询立即生成响应的初始片段（通常为 1-3 句话或数十个 token）。该模块的关键约束是**速度优先**：必须在 <50ms 内完成，确保用户感知为「即时响应」。

**模块 B - 云端 LLM 续写（云端）**：同时，完整查询被发送至云端大模型。LLM 接收查询后生成完整响应，但**并非从头输出**——而是接管 µLM 已生成的内容，判断其合理性并继续生成剩余部分。

**模块 C - 衔接与错误恢复**：µLM 预生成内容与 LLM 续写内容需要在语义和风格上无缝衔接。系统实现三种错误恢复模式以处理衔接失败的情况（详见模块公式部分）。

**输出**：用户首先看到 µLM 的即时输出，随后 LLM 续写内容以流式方式追加，整体形成连贯响应。

ASCII 流程示意：
```
用户输入 ──→ [µLM 设备端] ──→ 即时显示开头片段 ──→ 用户开始阅读
              │                              ↑
              ↓                              │
         [LLM 云端启动] ──→ 续写验证 ──→ 无缝衔接输出 ──┘
              │
              ↓ (若衔接失败)
         [错误恢复模式 1/2/3]
```

## 核心模块与公式推导

### 模块 1: µLM 预生成策略（对应框架图左侧设备端）

**直觉**：利用人类阅读时间作为「免费」的延迟预算，µLM 只需生成足够长的初始片段即可覆盖 LLM 的启动延迟。

**Baseline 公式**（标准自回归生成）：
$$P_{\text{base}}(y_{1:T}|x) = \prod_{t=1}^{T} P(y_t | y_{<t}, x; \theta_{\text{LLM}})$$
符号：$x$ = 用户查询，$y_t$ = 第 $t$ 个输出 token，$\theta_{\text{LLM}}$ = 大模型参数，$T$ = 总生成长度。

**变化点**：Baseline 要求 LLM 顺序生成全部 $T$ 个 token，首 token 延迟为 $T_{\text{init}}^{\text{LLM}}$（数百毫秒级）。本文将生成拆分为两段：µLM 生成前缀 $y_{1:k}$，LLM 续写后缀 $y_{k+1:T}$。

**本文公式（推导）**：
$$\text{Step 1}: \quad y_{1:k} \sim P_{\mu}(y_{1:k}|x; \theta_{\mu}) \quad \text{µLM 快速生成前缀，约束 } T_{\text{init}}^{\mu} < 50\text{ms}$$
$$\text{Step 2}: \quad y_{k+1:T} \sim P_{\text{LLM}}(y_{k+1:T} | y_{1:k}, x; \theta_{\text{LLM}}) \quad \text{LLM 条件续写，利用前缀缓存 KV}$$
$$\text{最终}: \quad P_{\text{total}}(y_{1:T}|x) = P_{\mu}(y_{1:k}|x) \cdot P_{\text{LLM}}(y_{k+1:T} | y_{1:k}, x)$$

关键设计：$k$ 的动态选择——需满足 $k / R_{\text{read}} \geq T_{\text{init}}^{\text{LLM}}$，其中 $R_{\text{read}}$ 为人类阅读速率（token/秒），确保用户读完 µLM 内容时 LLM 已准备好续写。

**对应消融**：

---

### 模块 2: µLM-LLM 衔接损失与训练（对应框架图中部衔接层）

**直觉**：µLM 与 LLM 的分布差异会导致续写不连贯，需通过训练目标缩小两者在前缀生成上的分布差距。

**Baseline 公式**（标准知识蒸馏）：
$$\mathcal{L}_{\text{KD}} = -\sum_{t} P_{\text{LLM}}(y_t|y_{<t}, x) \log P_{\mu}(y_t|y_{<t}, x)$$

**变化点**：标准 KD 仅匹配 token 级概率，忽略µLM 作为「前缀生成器」的特殊角色——其输出将直接作为 LLM 的输入条件。需显式优化 LLM 续写的似然度。

**本文公式（推导）**：
$$\text{Step 1}: \quad \mathcal{L}_{\text{prefix}} = -\log P_{\mu}(y_{1:k}^* | x) \quad \text{µLM 自回归训练，} y^* \text{为高质量响应前缀}$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{continuation}} = -\mathbb{E}_{y_{1:k} \sim P_{\mu}} \left[ \log P_{\text{LLM}}(y_{k+1:T}^* | y_{1:k}, x) \right] \quad \text{LLM 续写似然作为奖励信号}$$
$$\text{Step 3（重归一化）}: \quad \mathcal{L}_{\text{final}} = \mathcal{L}_{\text{prefix}} + \lambda \cdot \mathcal{L}_{\text{continuation}}$$
其中 $\lambda$ 平衡 µLM 自身流畅度与 LLM 续写兼容性。训练时通过 LLM 的 KV-cache 复用高效计算 $\mathcal{L}_{\text{continuation}}$。

**对应消融**：

---

### 模块 3: 三种错误恢复模式（对应框架图右侧/ Figure 3）

**直觉**：µLM 可能生成低质量、不安全或与 LLM 不兼容的前缀，需要系统级容错机制。

**Baseline 公式**（无恢复，直接拼接）：
$$\text{Output} = \text{Concat}(y_{1:k}^{\mu}, y_{k+1:T}^{\text{LLM}})$$

**变化点**：直接拼接在 µLM 出错时导致整体失败，需分级恢复策略。

**本文公式/策略（推导）**：

$$\text{Mode 1 - 无缝续写（正常路径）}: \quad \text{若 } \text{score}_{\text{compat}}(y_{1:k}^{\mu}, y_{k+1:T}^{\text{LLM}}) > \tau_1:$$
$$\quad \text{Output} = y_{1:k}^{\mu} \oplus y_{k+1:T}^{\text{LLM}}$$

$$\text{Mode 2 - 修正续写（µLM 前缀可修复）}: \quad \text{若 } \tau_2 < \text{score}_{\text{compat}} \leq \tau_1:$$
$$\quad \text{LLM 以 } [\text{修正指令}] + y_{1:k}^{\mu} + x \text{ 为输入，重写并续写}$$

$$\text{Mode 3 - 放弃重生成（µLM 前缀失败）}: \quad \text{若 } \text{score}_{\text{compat}} \leq \tau_2:$$
$$\quad \text{丢弃 } y_{1:k}^{\mu}, \text{ LLM 从头生成完整响应，延迟降级为 Baseline}$$

兼容性评分 $\text{score}_{\text{compat}}$ 综合 LLM 的续写困惑度、前缀安全检测、以及风格一致性指标。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31989539-8ffc-4fb8-8c7a-7b176958cabd/figures/Figure_3.png)
*Figure 3: Figure 3: Illustration of our three error recovery modes.*



**对应消融**：Figure 6（原 Figure 5）用户研究显示三种模式保障了系统可用性。

## 实验与分析

主实验结果对比：

| Method | 首 token 延迟 | 响应质量（人工评分） | 用户偏好率 | 失败恢复率 |
|:---|:---|:---|:---|:---|
| 独立云端 LLM（GPT-4） | 500-2000ms | 高 | Baseline | N/A |
| 独立边缘小模型（1B） | <50ms | 低 | 低 | N/A |
| Speculative Decoding | 300-800ms | 高 | — | N/A |
| **µLM + LLM（本文）** | **<50ms** | **高（接近 LLM）** | **显著优于独立 LLM** | **>99%** |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31989539-8ffc-4fb8-8c7a-7b176958cabd/figures/Figure_2.png)
*Figure 2: Figure 2: Example responses of µLM+LLM framework.*



核心发现分析：
- **首 token 延迟**：µLM 将 TTFT 从秒级降至 <50ms，这是本文最核心的量化收益，直接验证了「延迟隐藏」机制的有效性。
- **质量保障**：通过 $\mathcal{L}_{\text{continuation}}$ 训练目标和三种恢复模式，µLM+LLM 的整体响应质量接近独立 LLM，显著优于纯边缘小模型。
- **用户偏好**：Figure 6（原 Figure 5）显示用户在盲测中更偏好 µLM+LLM 的响应，表明即时启动的感知优势超过了潜在的微小质量差异。

消融实验关键结论：
- 移除 $\mathcal{L}_{\text{continuation}}$ 损失：LLM 续写困惑度上升，衔接失败率增加
- 移除 Mode 2/3 恢复机制：整体系统可用性下降，用户遇到明显「断片」体验
- µLM 规模消融：100M vs 1B 参数在延迟-质量权衡上的最优解

公平性检查：
- Baselines 选择合理，涵盖了延迟优化（speculative decoding）和质量保障（独立 LLM）两个方向。
- 计算成本：µLM 训练为一次性成本，推理时边缘能耗极低；LLM 端因 KV-cache 前缀复用，总计算量未增加。
- 失败案例：Mode 3 放弃重生成时延迟退化为 Baseline，但这是保底策略而非常态。



## 方法谱系与知识库定位

**方法家族**：边缘-云端协同推理（Edge-Cloud Collaborative Inference）/ 推测解码与级联生成（Speculative & Cascaded Generation）

**父方法**：Speculative Decoding（Leviathan et al., 2023; Chen et al., 2023）—— 本文将「draft-then-verify」的并行结构改造为「prefix-then-continue」的流水线结构，关键变化是从**减少总计算量**转向**隐藏感知延迟**。

**直接 Baselines 与差异**：
- **Medusa / EAGLE**（推测解码系列）：使用小模型并行生成候选序列，差异在于本文 µLM **独立输出用户可见内容**，而非仅作为内部 draft。
- **Prompt Cache / Contextual Caching**（商业 API 方案）：复用重复前缀的 KV 状态，差异在于本文创造**新的可复用前缀**（µLM 生成）而非依赖历史上下文重复。
- **TinyLLM / EdgeGPT**（纯边缘方案）：追求完全离线运行，差异在于本文**承认边缘模型能力边界**，通过云端续写保障最终质量。

**方法变体 slots**：
- Architecture：µLM 架构（Transformer/Mamba/RNN 可选，本文未限定具体结构）
- Objective：$\mathcal{L}_{\text{continuation}}$ 联合训练目标
- Training_recipe：两阶段训练（自回归预训练 + 续写对齐微调）
- Data_curation：高质量对话前缀-续写对构造
- Inference：动态 $k$ 选择 + 三级恢复模式

**后续方向**：
1. **多模态扩展**：µLM 预生成语音/图像响应的初始片段，覆盖多模态大模型的启动延迟。
2. **个性化 µLM**：基于用户历史对话微调设备端 µLM，提升前缀个性化程度。
3. **联邦 µLM 训练**：跨设备聚合学习 µLM，避免上传用户数据至云端。

**知识库标签**：
- Modality: Text / Dialogue
- Paradigm: Edge-Cloud Co-inference / Cascaded Generation
- Scenario: Low-latency Interactive AI / On-device Intelligence
- Mechanism: Latency Hiding via Human Perception / Prefix Generation with Recovery
- Constraint: Real-time / Privacy-preserving / Resource-constrained Edge Devices

