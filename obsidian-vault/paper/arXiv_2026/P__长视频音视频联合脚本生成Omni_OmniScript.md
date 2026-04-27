---
title: 'OmniScript: Towards Audio-Visual Script Generation for Long-Form Cinematic Video'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.11102
aliases:
- 长视频音视频联合脚本生成OmniScript
- OmniScript
- 现有视频理解模型将叙事内容压缩为粗粒度段落摘要
method: OmniScript
modalities:
- Image
---

# OmniScript: Towards Audio-Visual Script Generation for Long-Form Cinematic Video

[Paper](https://arxiv.org/abs/2604.11102)

**Topics**: [[T__Video_Understanding]], [[T__Text_Generation]] | **Method**: [[M__OmniScript]]

> [!tip] 核心洞察
> 现有视频理解模型将叙事内容压缩为粗粒度段落摘要，导致时间锚定丢失和多模态线索混杂。OmniScript的核心洞察是：将输出空间从自由文本重新定义为严格解耦的原子字段元组（时间戳×角色×对话×动作×表情×音频），迫使模型在生成时显式分离每个叙事维度。这种结构化输出约束配合音视频联合输入和CoT中间推理链，使模型能够在长视频上维持细粒度时间对齐的叙事理解，而非退化为全局摘要。有效性的关键在于任务形式化本身——正确定义了「什么是好的脚本」，从而使训练信号和评估指标都能精确指向目标能力。

| 中文题名 | 长视频音视频联合脚本生成OmniScript |
| 英文题名 | OmniScript: Towards Audio-Visual Script Generation for Long-Form Cinematic Video |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.11102) · [Code] · [Project] |
| 主要任务 | Video-to-Script (V2S)：将长视频（电影/电视剧/短剧）转换为时间对齐的结构化脚本（Scene→Event→Field层级） |
| 主要 baseline | Qwen3VL-235B-A22B, Qwen3VL-8B, Gemini-3-pro, TimeChat-Captioner, Qwen3-Omni-30/3B, MiniCPM-O-4.5 |

> [!abstract] 因为「现有MLLM将长视频压缩为粗粒度摘要导致时间锚定丢失、多模态线索混杂、长上下文性能崩溃」，作者在「Qwen3VL-8B + Whisper音频编码器」基础上改了「严格解耦的原子字段输出格式 + 记忆增强渐进式标注 + 三阶段训练（含CoT推理链与RL时间分段奖励）」，在「自建V2S基准（横版电影+竖版短剧）」上取得「事件级Overall 37.0 / tIoU@0.1 69.0，超越Qwen3VL-235B-A22B达+4.0 Overall和+7.0 tIoU@0.1」

- **事件级Overall 37.0** vs. Qwen3VL-235B-A22B 33.0（+4.0），tIoU@0.1 69.0 vs. 62.0（+7.0）
- **场景级Overall 52.4**，tIoU@0.1 74.6，时间定位接近Gemini-3-pro（75.3）
- **原生长视频处理能力**：TimeChat-Captioner在分段协议下从7.7跃升至32.3，OmniScript无需分段即达37.0

## 背景与动机

将一部90分钟的电影转化为可读的、时间精确到秒的完整脚本——包含谁、何时、说了什么、做了什么、表情如何、背景音是什么——这一任务对影视制作、无障碍辅助和内容检索具有核心价值，但现有技术远未解决。以《教父》中经典的餐厅暗杀场景为例：观众需要知道迈克尔何时从座位站起（动作）、何时从厕所取出藏枪（时间戳）、背景音乐何时从爵士乐骤停（音频线索）、他面部从紧张到决绝的微表情变化（表情）、以及那句低语的意大利语对话内容（对话）——这些多模态叙事线索必须在时间轴上精确对齐，而非混为一谈。

现有方法如何处理这一挑战？**TimeChat-Captioner** 引入带时间戳的场景级描述，但将角色动作、意图和对话混杂在粗粒度段落中，无法实现原子级解耦。**Qwen3VL系列** 作为强大的视觉语言模型，在短视频理解上表现优异，但直接处理5分钟完整视频时事件级Overall仅3.9-7.7，暴露出严重的长上下文瓶颈。**Gemini-3-pro** 作为闭源商业模型，场景级性能较强（Overall 57.0），但缺乏公开的训练细节和可复现性，且同样未针对脚本生成的严格字段解耦进行优化。

这些方法的共同缺陷在于：**输出空间设计错误**。它们将视频理解定义为自由文本生成或粗粒度摘要任务，导致三个连锁问题：(1) 时间锚定在压缩过程中丢失；(2) 视觉、音频、语义线索在统一文本中相互干扰；(3) 评估缺乏精确对应，无法区分"对话准确但时间错位"与"时间正确但角色混淆"等不同错误模式。更深层的数据瓶颈在于：MovieNet、Movie101等现有数据集要么缺乏时间锚定，要么将多模态线索混杂在段落摘要中，2分钟片段的细粒度标注即需约4000 tokens，人工标注成本极高。

OmniScript的核心动机正是从**任务形式化**本身入手：重新定义输出空间为严格解耦的原子字段元组，使训练信号、模型架构和评估指标三者精确对齐。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9b77a20-2ea6-4df5-aef3-06294ca7e210/figures/Figure_1.png)
*Figure 1: Figure 1 Overview of our Video-to-Script (V2S) framework. Given a long-form cinematic video, our pipeline performstemporally grounded scene-event parsing and generates a structured script with multimo*



## 核心创新

核心洞察：将输出空间从自由文本重新定义为严格解耦的原子字段元组（时间戳×角色×对话×动作×表情×音频），因为结构化输出约束迫使模型在生成时显式分离每个叙事维度，从而使长视频上的细粒度时间对齐叙事理解成为可能——而非退化为全局摘要。

| 维度 | Baseline（TimeChat-Captioner / Qwen3VL） | 本文（OmniScript） |
|:---|:---|:---|
| **输出格式** | 自由文本段落或粗粒度时间戳描述 | 严格层级：Scene → Event → Field，Event为六元组 $(\tau, c, d, a, x, u)$ |
| **模态输入** | 纯视觉（Qwen3VL）或视觉+文本（TimeChat） | 音视频联合：Qwen3VL-8B视觉 + Whisper large-v3音频编码 |
| **训练数据构建** | 现有数据集（MovieNet/Movie101，无时间锚定或粗粒度） | 记忆增强渐进式标注：角色档案管理器动态注入历史，Gemini-based标注器生成细粒度脚本 |
| **推理机制** | 直接生成 | CoT风格中间推理链（情节演化+角色关系推理）→ 结构化字段解码 |
| **长视频处理** | 分段或直接崩溃（5分钟Overall 3.9-7.7） | 原生长视频训练 + RL时间分段奖励优化 |

这种"任务形式化即方法"的设计哲学贯穿数据、模型、训练、评估全流程，而非单一模块改进。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9b77a20-2ea6-4df5-aef3-06294ca7e210/figures/Figure_2.png)
*Figure 2: Figure 2 Overview of the memory-augmented progressive annotation pipeline. The character profile manager injectshistorical profiles to guide plot reasoning and dynamically updates character memory. Th*



OmniScript框架包含三大核心子系统，数据流如下：

**输入端**：长视频（电影/电视剧/短剧，横版或竖版）→ 视频帧序列 + 音频波形，分别进入视觉编码器与音频编码器。

**模块A：记忆增强渐进式标注系统（数据构建）**。角色档案管理器（Character Profile Manager）维护每个角色的历史属性、关系网络和情感状态；在处理新场景时，将历史档案注入提示，辅助情节推理并动态更新角色记忆；Gemini-based标注器据此生成细粒度V2S标注。输出：约240万双语视频预训练数据 + 约4.5万精选SFT数据（2.1万电影/电视 + 2.4万短剧）。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9b77a20-2ea6-4df5-aef3-06294ca7e210/figures/Figure_3.png)
*Figure 3: Figure 3 Overview of the proposed architecture. Instruction, video, and audio are encoded into multimodal tokensand fused in the LLM via AV-DeepStack across multiple layers. The model first performs m*



**模块B：Omni-Modal融合编码器（多模态对齐）**。视频帧经Qwen3VL视觉编码器提取特征，音频经Whisper large-v3编码器提取特征，分别通过模态投影器对齐至LLM的文本嵌入空间，拼接为多模态token序列输入Qwen3VL-8B骨干网络。

**模块C：结构化生成解码器（输出）**。LLM先执行CoT风格推理（情节演化分析、角色关系推理），再解码为严格格式的层级结构：Scene层级（场景分割）→ Event层级（事件六元组）→ Field层级（六个独立字段）。

**三阶段训练流程**：(1) 模态对齐阶段——冻结LLM，训练投影器，随机帧遮蔽增强鲁棒性；(2) 预训练阶段——全量微调，多任务目标（ASR±时间戳、密集字幕、摘要、时间定位）；(3) SFT+RL阶段——Schema监督 + CoT推理链 + 时间分段奖励RL。

```
长视频输入
  ├──→ 视觉帧 → Qwen3VL视觉编码器 → 投影器 ──┐
  │                                            ├──→ 多模态token序列 → Qwen3VL-8B LLM
  └──→ 音频波形 → Whisper large-v3 → 投影器 ──┘
                                                    ↓
                                              CoT推理链（情节+角色）
                                                    ↓
                                              结构化解码
                                                    ↓
                                          Scene → Event(τ,c,d,a,x,u) → 脚本输出
```

## 核心模块与公式推导

### 模块 1: 任务形式化与输出空间设计（对应框架图 输出端）

**直觉**：现有模型的自由文本输出导致评估模糊、训练信号混杂；将输出约束为严格字段元组，可使每个维度的预测误差独立可追踪。

**Baseline 形式**（TimeChat-Captioner 风格密集字幕）：
$$S_{base} = \{(t_i, c_i)\}_{i=1}^{N}$$
其中 $t_i$ 为时间戳，$c_i$ 为自由文本描述，所有叙事维度压缩于单一文本字符串。

**变化点**：自由文本无法区分"对话内容正确但说话人错误"与"说话人正确但时间错位"等错误模式；且长视频中 $c_i$ 的token长度随复杂度爆炸。

**本文形式化**：
$$\text{Scene}_i = \{e_{i,j}\}_{j=1}^{M_i}$$
$$e_{i,j} = (\tau_{i,j},\; c_{i,j},\; d_{i,j},\; a_{i,j},\; x_{i,j},\; u_{i,j})$$
符号：$\tau$=时间戳区间，$c$=角色，$d$=对话，$a$=动作，$x$=表情，$u$=音频线索。每个字段独立解码，损失函数可加权分解。

**对应消融**：结构化输出使评估可精确到字段级F1，Table 1/2显示各字段独立优化空间。

---

### 模块 2: 记忆增强渐进式标注（对应框架图 模块A/数据构建）

**直觉**：长视频角色关系随时间演化，静态标注导致前后矛盾；动态角色档案可维持跨场景一致性。

**Baseline 形式**（标准LLM-based视频标注）：
$$\hat{S} = \text{LLM}(V, P_{static})$$
其中 $V$ 为视频输入，$P_{static}$ 为固定提示模板，无历史记忆机制。

**变化点**：静态提示无法处理角色更名、关系反转、身份揭示等长程叙事结构；且对同一角色在不同场景的属性描述易冲突。

**本文公式（推导）**：
$$\text{Step 1}: \mathcal{M}_0 = \text{InitProfiles}(\text{cast list}) \quad \text{从演职员表初始化角色档案}$$
$$\text{Step 2}: \mathcal{M}_t = \text{Update}(\mathcal{M}_{t-1},\; \text{Extract}(V_t, \mathcal{M}_{t-1})) \quad \text{逐场景提取并更新记忆}$$
$$\text{Step 3}: \hat{S}_t = \text{Gemini}(V_t, P_{dynamic} \oplus \mathcal{M}_t) \quad \text{动态提示注入历史档案生成标注}$$
**最终**：标注质量受 $|\mathcal{M}_t - \mathcal{M}_{t-1}|$ 约束，确保跨场景角色一致性。

**对应消融**：角色档案管理器的独立贡献未在摘录中显式消融。

---

### 模块 3: 时间分段奖励RL与长视频优化（对应框架图 训练阶段3）

**直觉**：自回归生成详细脚本的token消耗极高（2分钟约4000 tokens），标准RL奖励在长视频上稀疏且时间定位误差累积；将奖励按时间区间分段可提供更密集的优化信号。

**Baseline 形式**（标准RLHF/RLAIF）：
$$R_{base}(S, \hat{S}) = \text{sim}_{\text{overall}}(S, \hat{S})$$
单一全局奖励，对长视频中局部时间错位不敏感。

**变化点**：全局奖励无法区分"第3分钟对话错位"与"第15分钟动作遗漏"；长视频的全局相似度度量易被高频出现的正确字段主导。

**本文公式（推导）**：
$$\text{Step 1}: S = \text{bigcup}_{k=1}^{K} S^{(k)}, \quad \hat{S} = \text{bigcup}_{k=1}^{K} \hat{S}^{(k)} \quad \text{将真值与预测按时间分为}K\text{段}$$
$$\text{Step 2}: R_{temp}^{(k)} = \sum_{f \in \mathcal{F}} w_f \cdot \text{F1}(S_f^{(k)}, \hat{S}_f^{(k)}) + \lambda \cdot \mathbb{1}[\text{tIoU}^{(k)} \geq 0.1] \quad \text{每段字段F1加时间定位奖励}$$
$$\text{Step 3}: R_{final} = \frac{1}{K}\sum_{k=1}^{K} R_{temp}^{(k)} - \beta \cdot \text{KL}[\pi_{\theta} \| \pi_{ref}] \quad \text{分段奖励均值+KL约束}$$
**最终**：$\mathcal{L}_{RL} = -\mathbb{E}_{S \sim \pi_{\theta}}[R_{final}(S)]$，其中 $\mathcal{F}=\{\tau, c, d, a, x, u\}$ 为六个字段集合。

**对应消融**：RL阶段（时间分段奖励）的独立贡献未在摘录中呈现消融实验。

## 实验与分析

**主结果（事件级，Table 1）**：

| Method | Overall | tIoU@0.1 | 备注 |
|:---|:---|:---|:---|
| MiniCPM-O-4.5 | 3.9 | — | 直接长视频，崩溃 |
| TimeChat-Captioner (直接) | 7.7 | — | 直接长视频，崩溃 |
| TimeChat-Captioner (†分段) | 32.3 | — | 分段协议显著提升，暴露长上下文瓶颈 |
| Qwen3VL-8B | 
| Qwen3VL-235B-A22B | 33.0 | 62.0 | 大模型纯视觉 |
| **OmniScript-8B** | **37.0** | **69.0** | 本文，+4.0 Overall, +7.0 tIoU vs. 235B |

**主结果（场景级，Table 2）**：

| Method | Overall | tIoU@0.1 | 备注 |
|:---|:---|:---|:---|
| Qwen3-Omni-30/3B (MoE) | 21.0 | — | 简单加音频不保证提升，架构差异显著 |
| Qwen3VL-8B | 40.9 | 
| **OmniScript-8B** | **52.4** | **74.6** | 本文 |
| Gemini-3-pro | 57.0 | 75.3 | 闭源商业模型，Overall高4.6但tIoU仅高0.7 |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9b77a20-2ea6-4df5-aef3-06294ca7e210/figures/Figure_5.png)
*Figure 5: Figure 5 Performance comparison across video durations on multiple metric dimensions.*



**关键分析**：
- **核心声明支持**：OmniScript-8B以1/30参数超越Qwen3VL-235B-A22B，验证结构化输出+音视频联合+CoT的有效性，非单纯规模优势。
- **时间定位优势**：tIoU@0.1 69.0 vs. 62.0（+7.0）是最大亮点，说明原子字段解耦直接优化了时间敏感性；场景级74.6接近Gemini-3-pro的75.3，时间定位能力确实"相当"。
- **Overall差距语境化**：场景级Overall 52.4 vs. Gemini-3-pro 57.0，差距4.6分，摘要"相当"存在选择性夸大，但开源8B模型逼近闭源商业模型仍具价值。

**消融与边界条件**（
![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9b77a20-2ea6-4df5-aef3-06294ca7e210/figures/Figure_4.png)
*Figure 4: Figure 4 Comparison of two strategies for extending OmniScript to long videos. Left (Strategy 1): Direct contextextension, trained with long-video annotations (including memory-refine labels) and cros*

）：
- **分段协议敏感性**：TimeChat-Captioner从7.7→32.3（+24.6）证明长上下文瓶颈对输入协议高度敏感；OmniScript无需分段即达37.0，说明原生长视频训练是真实能力而非协议红利。
- **CoT变体观察**：thinking变体往往不如非thinking版本，但缺乏具体数值支撑，可能暗示过度推理导致格式偏离或推理链与结构化输出目标冲突。
- **音频模态非充分条件**：Qwen3-Omni-30/3B（含音频MoE）仅21.0 vs. Qwen3VL-8B 40.9，说明简单模态扩展不足，需配合结构化任务设计。


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9b77a20-2ea6-4df5-aef3-06294ca7e210/figures/Figure_6.png)
*Figure 6: Figure 6 Additional long-video evaluation results. (a) Event-level, scene-level, and overall F1 scores across durations.(b) Event-level fine-grained F1 scores on character, action, and dialogue. (c) E*



**公平性检查**：
- **基线强度**：缺少GPT-4o等OpenAI视频模型对比，最强闭源基线为Gemini-3-pro；开源基线覆盖较全但最新版本可能已更新。
- **计算/数据成本**：4.5万SFT视频规模适中，但依赖Gemini标注引入专有模型依赖，复现成本与质量上限受约束。
- **失败案例**：未在摘录中呈现；竖版短剧（2.4万）与横版电影（2.1万）的格式差异是否导致域间性能差距待验证。字幕遮蔽分析数值缺失。

## 方法谱系与知识库定位

**方法家族**：长视频多模态理解 → 结构化密集预测 → 音视频联合脚本生成

**父方法**：Qwen3VL-8B（视觉语言模型骨干）+ Whisper large-v3（音频编码器）。OmniScript继承Qwen3VL的视觉理解能力，扩展音频模态，但通过**任务形式化重构**而非简单模态堆叠实现突破。

**改动槽位**：
| 槽位 | 父方法/基线 | OmniScript改动 |
|:---|:---|:---|
| 架构 | Qwen3VL-8B纯视觉 | +Whisper音频编码器，AV-Space融合 |
| 目标函数 | 自由文本生成/摘要 | 严格字段元组解码 + 字段级F1分解 |
| 训练配方 | 标准SFT | 三阶段（对齐→预训练→SFT+CoT+RL时间分段奖励） |
| 数据策划 | 现有粗粒度数据集 | 记忆增强渐进式标注，角色档案动态管理 |
| 推理方式 | 直接生成 | CoT中间推理链（情节演化+角色关系）→结构化解码 |

**直接基线差异**：
- **TimeChat-Captioner**：同带时间戳，但输出为粗粒度自由文本；OmniScript严格字段解耦+原生长视频训练。
- **Qwen3VL-235B-A22B**：同视觉骨干家族，但纯视觉+无结构化约束；OmniScript以1/30参数+音频超越。
- **Gemini-3-pro**：同商业级性能区间，但OmniScript开源可复现，且时间定位tIoU差距仅0.7。
- **Qwen3-Omni**：同音频扩展方向，但MoE架构+无结构化任务设计导致性能崩溃，反证任务形式化的关键性。

**后续方向**：
1. **端到端影视制作**：将V2S输出接入自动分镜、配音替换、剪辑决策系统，形成完整AIGC工作流。
2. **实时/流式脚本生成**：当前针对离线长视频，扩展至直播、体育赛事等实时场景需重新设计记忆与延迟权衡。
3. **跨语言/跨文化迁移**：现有240万双语数据以中英为主，非英语电影（如宝莱坞、尼日利亚瑙莱坞）的标注与评估适配。

**知识库标签**：模态[音视频联合] / 范式[结构化生成/CoT推理/RLHF] / 场景[长视频理解/影视内容分析] / 机制[任务形式化/原子字段解耦/时间分段奖励] / 约束[数据稀缺/长上下文/评估困难]

