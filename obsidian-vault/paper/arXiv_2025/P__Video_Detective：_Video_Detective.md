---
title: 'Video Detective: Seek Critical Clues Recurrently to Answer Question from Long Videos'
type: paper
paper_level: B
venue: arXiv
year: 2025
paper_link: https://arxiv.org/abs/2512.17229
aliases:
- Video Detective：递归搜索关键线索的长视频问答
- Video Detective
method: Video Detective
modalities:
- Image
---

# Video Detective: Seek Critical Clues Recurrently to Answer Question from Long Videos

[Paper](https://arxiv.org/abs/2512.17229)

**Topics**: [[T__Visual_Question_Answering]], [[T__Video_Understanding]] | **Method**: [[M__Video_Detective]] | **Datasets**: VideoMME, MLVU, NextQA, VideoVista

| 中文题名 | Video Detective：递归搜索关键线索的长视频问答 |
| 英文题名 | Video Detective: Seek Critical Clues Recurrently to Answer Question from Long Videos |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2512.17229) · [Code](https://arxiv.org/abs/2512.17229) · [Project](https://arxiv.org/abs/2512.17229) |
| 主要任务 | 长视频问答 (Long Video Question Answering) |
| 主要 baseline | LongVILA, Video-XL, LongVA, LongLLava |

> [!abstract] 因为「长视频视觉token数量爆炸导致现有MLLM无法有效处理小时级视频」，作者在「Qwen2.5-VL-7B」基础上改了「递归子段处理+问题感知记忆token压缩机制」，在「VideoMME (w/ sub.) Long」上取得「63.5 accuracy，相比LongVILA提升+6.1」

- **VideoMME (w/ sub.) - Long**: 63.5 accuracy，超越 LongVILA (+6.1)、Video-XL (+8.6)
- **VideoMME (w/o sub.) - Long**: 56.0 accuracy，超越 LongVILA (+3.0)、Video-XL (+6.8)
- **VideoVista**: 74.3 accuracy，超越 Video-XL (+3.7)、LongVA (+6.9)

## 背景与动机

长视频问答的核心困境在于：一部两小时的电影包含超过20万帧视觉信息，而现有7B参数的多模态大语言模型(MLLM)的上下文窗口无法容纳如此庞大的视觉token序列。例如，要回答"楚门在什么时刻发现了世界的真相"，模型必须跨越多个电影片段追踪关键线索，而非简单浏览全部帧。
![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5fe9b2f0-9e54-4d1b-bdf7-3905bb431cfa/figures/Figure_6.png)
*Figure 6: Figure 6: Two data examples in our GLVC dataset comes from the movie “The Truman Show” and“Harry Potter and the Deathly Hallows”.A.3LLM USAGE*



现有方法主要采用两种策略应对此问题。**LongVILA** 通过改进的序列并行技术扩展上下文长度，但仍需一次性编码全部视觉token，计算成本随视频长度线性增长。**Video-XL** 引入全局-局部token压缩机制，然而其压缩过程与具体问题解耦，可能丢失与问题相关的关键细节。**LongVA** 和 **LongLLava** 则通过帧采样或视频摘要减少输入量，但均匀采样容易遗漏稀疏出现的关键事件。

这些方法的共同短板在于：**压缩或采样过程缺乏问题引导**。当问题聚焦于特定人物、物体或事件时，模型仍需处理大量无关视觉信息，导致关键线索被淹没。此外，现有方法多为单遍处理，无法像人类观影那样"边看边记"——随着剧情推进不断更新对关键信息的记忆。

本文提出 Video Detective，核心思想是模拟人类侦探的推理方式：将长视频分段递归处理，每段提取与问题相关的压缩记忆，并在后续段中复用历史记忆，实现问题感知的渐进式线索积累。

## 核心创新

核心洞察：**问题感知记忆token可以在递归处理中充当"信息瓶颈"**，因为记忆token通过交叉注意力主动查询与问题相关的视觉特征，从而使长视频被压缩为紧凑且问题相关的表示成为可能。

| 维度 | Baseline (Qwen2.5-VL-7B) | 本文 |
|:---|:---|:---|
| 推理策略 | 单遍全序列编码 | 递归子段处理，记忆token跨段传递 |
| 视觉-问题交互 | 无显式交互，视觉token直接输入LLM | 门控融合g([F_v ∘ F_Q])，提取即与问题交互 |
| 历史信息利用 | 无，每帧独立编码 | 历史键值F^past_{k,v}与当前特征拼接复用 |
| 信息压缩 | 帧采样或均匀池化 | 可学习记忆token，注意力驱动的问题相关压缩 |
| 训练方式 | 端到端视频QA训练 | 两阶段：视频caption预热→长视频QA微调 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5fe9b2f0-9e54-4d1b-bdf7-3905bb431cfa/figures/Figure_2.png)
*Figure 2: Figure 2: (a) The architecture of our VideoDetective model. The video segment is divided intomultiple sub-segments, which are processed by visual encoder to get multi-modal embeddings.Then these embed*



Video Detective 的整体数据流遵循"分段-感知-记忆-累积-生成"的递归范式：

1. **视频分段 (Video Segmentation)**：输入完整视频V和问题Q，按固定长度（32帧）分割为子段 V_1, V_2, ..., V_S，用`<split>` token分隔。

2. **视觉-问题特征提取 (Visual-Question Feature Extraction)**：对每个子段V_t，提取视觉特征F^t_v，与问题特征F_Q通过门控函数g拼接融合，生成问题感知的视觉表示。

3. **记忆注意力 (Memory Attention)**：可学习的记忆token M_t通过嵌入矩阵E^m和投影矩阵W^m_{q,k,v}转换为查询Q^m、键K^m、值V^m，交叉注意力聚焦于当前段中与问题相关的区域。

4. **历史感知键值构建 (History-Aware Key-Value Construction)**：将过去记忆的键值F^past_{k,v}、当前段键值K,V、以及记忆键值K^m,V^m三层拼接，形成完整的注意力上下文。

5. **递归记忆更新 (Recurrent Memory Update)**：当前段输出更新记忆token M_{t+1}，作为下一段的输入，实现信息累积。

6. **答案生成 (Answer Generation)**：所有段处理完毕后，最终记忆序列M_1,...,M_S与问题Q输入LLM，自回归生成答案。

```
V,Q → [Seg_1: V_1+Q+M_1] → Memory Update → M_2
            ↓                    ↓
        <split>              <split>
            ↓                    ↓
      [Seg_2: V_2+Q+M_2] → Memory Update → M_3 → ... → M_S,Q → Answer
```

## 核心模块与公式推导

### 模块 1: 问题感知视觉投影（对应框架图 步骤2）

**直觉**: 让视觉特征在提取之初就"知道"要找什么，避免无关信息进入后续压缩。

**Baseline 公式** (标准视觉编码): $$Q, K, V = W_{q,k,v} \cdot F^t_v$$
符号: $F^t_v$ = 第t段视觉特征, $W_{q,k,v}$ = 投影矩阵

**变化点**: Baseline 仅依赖视觉特征，无法区分问题相关与无关内容。本文引入问题特征F_Q，通过拼接和门控融合实现条件化。

**本文公式（推导）**:
$$\text{Step 1}: \quad F_{\text{fused}} = g([F^t_v \circ F_{\mathcal{Q}}]) \quad \text{将视觉与问题特征拼接后输入门控函数}$$
$$\text{Step 2}: \quad Q, K, V = W_{q,k,v} \cdot F_{\text{fused}} \quad \text{投影为标准注意力三元组}$$
**最终**: $$Q, K, V = W_{q,k,v} \cdot g([F^t_v \circ F_{\mathcal{Q}}])$$

**对应消融**: 去掉问题感知压缩后，MLVU Test accuracy 从 45.8 降至 41.5 (Δ -4.3)。

---

### 模块 2: 记忆token投影与注意力扩展（对应框架图 步骤3-4）

**直觉**: 用少量可学习token作为"信息瓶颈"，通过注意力机制主动"询问"当前段：哪些问题相关内容值得记住？

**Baseline 公式** (标准自注意力，无记忆): 
$$\text{query} = Q, \quad \text{key} = K, \quad \text{value} = V$$

**变化点**: Baseline 缺乏跨段信息传递机制。本文增加并行的记忆分支，并将历史、当前、记忆三层信息融合。

**本文公式（推导）**:
$$\text{Step 1}: \quad Q^m, K^m, V^m = W^m_{q,k,v} \cdot E^m(M_t) \quad \text{记忆token嵌入并投影到注意力空间}$$
符号: $M_t$ = 第t段记忆token, $E^m$ = 可学习嵌入(初始化自bos token), $W^m_{q,k,v}$ = 记忆专用投影矩阵

$$\text{Step 2}: \quad \text{query} = g([Q \circ Q^m]) \quad \text{当前查询与记忆查询融合，兼顾当下与历史目标}$$

$$\text{Step 3}: \quad \text{key} = g([F^{\text{past}}_k \circ K \circ K^m]) \quad \text{历史键+当前键+记忆键三重拼接}$$

$$\text{Step 4}: \quad \text{value} = g([F^{\text{past}}_v \circ V \circ V^m]) \quad \text{对称的值向量融合}$$

**最终注意力计算**:
$$\text{Attention}(\text{query}, \text{key}, \text{value}) = \text{softmax}\left(\frac{\text{query} \cdot \text{key}^T}{\sqrt{d_k}}\right) \cdot \text{value}$$

**对应消融**: 去掉历史上下文复用后，NextQA accuracy 从 79.3 降至 75.2 (Δ -4.1)。

---

### 模块 3: 分段表示与训练目标（对应框架图 整体结构）

**直觉**: 显式的分段格式让模型学会"一段一段看"，而训练目标强制记忆token承载足够信息。

**Baseline 公式** (标准自回归损失):
$$\mathcal{L} = \sum_{j=1}^{l} -\log p(x_j | V, \mathcal{Q}, x_0, \ldots, x_{j-1})$$

**变化点**: 推理时原始视觉tokenV不可见，只能依赖记忆tokenM。训练目标需反映这一约束。

**本文公式（推导）**:
$$\text{Step 1}: \quad \underbrace{\{V_1, \mathcal{Q}, M_1\}}_{Seg_1}, \texttt{<split>}, \underbrace{\{V_2, \mathcal{Q}, M_2\}}_{Seg_2}, \texttt{<split>}, \cdots, \underbrace{\{V_s, \mathcal{Q}, M_s\}}_{Seg_S}, \texttt{<split>}, \mathcal{Q}$$
$$\text{Step 2}: \quad \mathcal{L} = \sum_{j=1}^{l} -\log p(x_j | V_1, M_1, \mathcal{Q}, \cdots, V_S, M_S, \mathcal{Q}, x_0, \ldots, x_{j-1}) \quad \text{训练时保留视觉token引导学习}$$

**推理时分布** (关键差异):
$$p(x_j | M_1, \ldots, M_S, x_0, \ldots, x_{j-1})$$
即推理仅依赖记忆token，实现高效长视频理解。

## 实验与分析



本文在5个长视频问答基准上评估 Video Detective，涵盖不同视频类型和难度。核心结果来自 Table 1。在 **VideoMME (w/ sub.) - Long** 上，Video Detective 达到 63.5 accuracy，较最强开源baseline LongVILA (57.4) 提升 +6.1，较 Video-XL (54.9) 提升 +8.6。这一设置包含字幕信息，更接近实际观影场景。在 **VideoMME (w/o sub.) - Long** 上，56.0 accuracy 同样领先 LongVILA (+3.0) 和 Video-XL (+6.8)，证明方法不依赖字幕先验。**VideoVista** 上 74.3 accuracy 超越 Video-XL (+3.7) 和 LongVA (+6.9)，显示跨数据集泛化能力。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5fe9b2f0-9e54-4d1b-bdf7-3905bb431cfa/figures/Figure_5.png)
*Figure 5: Figure 5: The impact of compression ratio α.*



然而，**MLVU Test** 上 45.8 仅微幅领先 Video-XL (45.5) +0.3，提升空间有限。**NextQA** 上 79.3 甚至低于 LongVILA (80.7) -1.4，表明在较短视频或因果推理场景下优势不明显。**Cinepile** 67.1 标记*号，提示训练集污染，无法与GPT-4o/Gemini-1.5-Pro做零样本公平比较。

消融实验验证了核心组件的必要性：去掉问题感知压缩导致 MLVU Test 45.8 → 41.5 (Δ -4.3)；去掉历史上下文复用导致 NextQA 79.3 → 75.2 (Δ -4.1)。两者均为显著性能损失，支持"问题引导"和"递归累积"的核心主张。

公平性方面存在若干问题：Table 1 多处 baseline 值缺失（LongVILA在MLVU和VideoVista无数据），阻碍全面对比；未与更新的长视频模型如 VILA-1.5 或 mPLUG-Owl3 比较；仅7B规模，缺乏缩放分析；Cinepile结果因训练集包含而不可靠。此外，Figure 5 显示压缩比α的影响，但具体数值未在文本中详述。

## 方法谱系与知识库定位

Video Detective 属于**长视频理解MLLM**方法族，直接继承自 **Qwen2.5-VL-7B**（LLM参数初始化），在四个关键slot上进行扩展：

| 改动slot | 具体变化 |
|:---|:---|
| inference_strategy | 单遍全序列 → 递归子段+记忆传递 |
| architecture | 添加可学习记忆token嵌入E^m、记忆投影W^m_{q,k,v}、门控融合g([·∘·]) |
| training_recipe | 端到端训练 → 两阶段（8K视频caption预热 + 长视频QA微调） |
| data_pipeline | 标准帧采样 → 32帧分段+<split>分隔符+每段前置问题 |

**直接baseline差异**：
- **LongVILA**: 序列并行扩展上下文，无显式压缩机制；本文用记忆token实现问题相关压缩
- **Video-XL**: 全局-局部token压缩，压缩与问题解耦；本文压缩过程由问题引导
- **LongVA/LongLLava**: 帧采样/摘要减少输入；本文保留全部帧但递归压缩，避免采样遗漏

**后续方向**：(1) 记忆token数量的自适应学习，替代固定压缩比；(2) 与更强闭源模型（GPT-4o, Gemini）的公平对比协议；(3) 扩展到视频-文本检索、长视频字幕生成等下游任务。

**标签**: 模态=video+text | 范式=recurrent_memory_compression | 场景=long_video_QA | 机制=cross_attention+gated_fusion | 约束=7B_parameter, fixed_segment_length

