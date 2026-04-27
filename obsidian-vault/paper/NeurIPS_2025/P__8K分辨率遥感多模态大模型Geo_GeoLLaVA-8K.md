---
title: 'GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 8K分辨率遥感多模态大模型GeoLLaVA
- GeoLLaVA-8K
- By introducing high-resolution RS d
acceptance: Spotlight
cited_by: 23
method: GeoLLaVA-8K
modalities:
- Image
- Text
paradigm: supervised
---

# GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution

**Topics**: [[T__Visual_Question_Answering]], [[T__Visual_Reasoning]] | **Method**: [[M__GeoLLaVA-8K]] | **Datasets**: XLRS-Bench

> [!tip] 核心洞察
> By introducing high-resolution RS datasets and two token compression strategies—Background Token Pruning and Anchored Token Selection—GeoLLaVA-8K becomes the first RS-focused MLLM capable of handling 8K×8K resolution imagery while improving performance over baselines.

| 中文题名 | 8K分辨率遥感多模态大模型GeoLLaVA |
| 英文题名 | GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution |
| 会议/期刊 | NeurIPS 2025 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.21375) · [Code](N/A) · [Project](N/A) |
| 主要任务 | Visual Question Answering, Visual Reasoning (Remote Sensing Image Understanding) |
| 主要 baseline | GeoChat, LLaVA-Next, LLaVA-OneVision, Qwen2.5-VL, InternVL2.5/3, GPT-4o, Gemini 2.0 Flash |

> [!abstract] 因为「超高分辨率遥感图像（8K×8K）导致视觉令牌爆炸、现有训练数据分辨率不足4K」，作者在「LLaVA」基础上改了「两阶段视觉令牌压缩（Background Token Pruning + Anchored Token Selection）+ 新建SuperRS-VQA/HighRS-VQA数据集」，在「XLRS-Bench」上取得「Scene Classification 59.0、Planning 66.0、Spatiotemporal Reasoning 50.0等多项SOTA，平均精度51.5（压缩比24:1）」

- **Scene Classification**: 59.0，超越最强baseline Gemini 2.0 Flash的43.3，领先GeoChat 22.8达+36.2
- **Planning**: 66.0，超越所有对比模型，较LLaVA-Next 30.0提升+36.0
- **Spatiotemporal Reasoning**: 50.0，超越Qwen2.5-VL-72B的49.3，成为该任务最优7B模型

## 背景与动机

遥感图像的分辨率正在快速提升，商业卫星已能采集8K×8K像素级别的地表影像。然而，当研究者尝试将多模态大语言模型（MLLM）应用于这类超高分辨率（UHR）遥感图像时，面临双重困境：一是**训练数据稀缺**——现有遥感视觉问答数据集（如VRSBench）分辨率普遍不超过4K，且多为合成生成；二是**令牌爆炸**——8K图像经视觉编码器（如ViT）后产生的视觉令牌序列长度可达普通图像的数十倍，直接超出GPU显存上限，导致无法训练甚至无法推理。

现有方法如何应对？**LLaVA** 及其后续版本（LLaVA-Next、LLaVA-OneVision）采用标准视觉编码器+全序列投影架构，所有视觉令牌无差别输入LLM，在8K分辨率下直接OOM。**Qwen2-VL** 提出"任意分辨率"处理机制，通过动态分辨率切分缓解问题，但未针对遥感图像的地理语义特性优化，且缺乏8K级遥感训练数据。**GeoChat** 作为专为遥感设计的MLLM，虽具备领域知识，但受限于低分辨率训练数据（≤4K）和全令牌处理策略，在XLRS-Bench的UHR任务上表现疲软（Scene Classification仅22.8）。

这些方法的共同短板在于：**将遥感图像视为普通自然图像处理，忽视了遥感场景的核心语义结构——大面积同质背景（海洋、森林、沙漠）与稀疏分布的关键地物目标**。全令牌处理不仅浪费计算于冗余背景，还稀释了真正需要关注的目标信息。本文据此提出核心假设：通过识别并压缩背景令牌、保留对象中心令牌，可在显存约束下实现8K遥感图像的高效理解。为此，作者构建了首个平均分辨率达8,376×8,376的遥感VQA数据集，并设计了面向遥感语义的两阶段令牌压缩机制。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ebd76192-3969-4c49-97ee-0a0ebc6f04e1/figures/fig_001.png)
*Figure: We introduce SuperRS-VQA and HighRS-VQA, the highest-resolution VQA datasets*



## 核心创新

核心洞察：**遥感图像的地理语义具有显著的前景-背景不对称性**，因为大面积同质背景区域（如海洋、森林）的视觉令牌语义高度冗余，而地物目标（如建筑、舰船、道路）的令牌携带关键判别信息，从而使「先剪枝背景、再锚定对象」的两阶段压缩策略成为可能——在24:1压缩比下不仅避免OOM，还能提升任务精度。

| 维度 | Baseline (LLaVA/Qwen2-VL) | 本文 (GeoLLaVA-8K) |
|:---|:---|:---|
| 数据分辨率 | ≤4K，合成数据为主 | 8K+真实遥感，SuperRS-VQA (8,376×8,376) + HighRS-VQA (2,000×1,912) |
| 视觉令牌处理 | 全序列输入，无差别对待 | 两阶段压缩：Background Token Pruning → Anchored Token Selection |
| 对象感知 | 隐式学习，无显式引导 | Warmup Model显式检测对象区域，指导令牌保留 |
| 训练策略 | 标准SFT | SFT阶段即应用压缩，模型自适应压缩后的令牌分布 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ebd76192-3969-4c49-97ee-0a0ebc6f04e1/figures/fig_002.png)
*Figure: Our datasets have the perception and*



GeoLLaVA-8K的整体流程遵循LLaVA式编码器-投影器-LLM架构，但在视觉令牌进入LLM前插入关键的两阶段压缩模块：

1. **Visual Encoder**（输入：8K×8K遥感图像；输出：密集视觉令牌序列 $T_{encoder} \in \mathbb{R}^{N \times d}$）：采用标准ViT类编码器提取图像特征，N极大（8K图像可达数万令牌）。

2. **Background Token Pruning**（输入：$T_{encoder}$；输出：$T_{compressed} \in \mathbb{R}^{N' \times d}, N' < N$）：基于语义冗余分析，识别并移除对应大面积同质背景（海洋、森林等）的令牌，首阶段大幅削减令牌数量。

3. **Warmup Model**（输入：图像/中间特征；输出：检测对数几率 $\text{logits}_{warmup}$）：轻量级辅助模型，预训练用于识别图像中的对象区域，为第二阶段提供空间指导信号。

4. **Anchored Token Selection**（输入：$T_{compressed}$ + $\text{logits}_{warmup}$；输出：$T_{final} \in \mathbb{R}^{N'' \times d}, N'' \ll N$）：基于warmup模型的检测结果，从压缩后序列中保留与地物目标相关的对象中心令牌，确保遥感理解所需的关键语义不丢失。

5. **Projector/Adapter**（输入：$T_{final}$；输出：LLM兼容的视觉嵌入）：将压缩后的视觉令牌投影到LLM的输入空间。

6. **LLM Backbone**（输入：视觉嵌入 + 文本指令；输出：生成回答）：7B参数语言模型，执行VQA或视觉推理任务。

整体训练分为两步：先用筛选后的高分辨率数据训练warmup模型，再在SFT阶段将两阶段压缩集成到完整pipeline中联合优化。压缩比经网格搜索确定为24:1，在单节点8 GPU上稳定训练。

```
8K RS Image → [Visual Encoder] → Dense Tokens (N)
                                    ↓
                         [Background Token Pruning]
                                    ↓
                         Reduced Tokens (N') + [Warmup Model] → logits
                                    ↓
                         [Anchored Token Selection]
                                    ↓
                         Compressed Tokens (N'' = N/24)
                                    ↓
                         [Projector] → [LLM] + Text Query → Answer
```

## 核心模块与公式推导

### 模块 1: Background Token Pruning（对应框架图 Stage 1）

**直觉**: 遥感图像中大面积同质背景区域的视觉令牌语义高度重复，移除它们不会损失判别信息，反而能减少噪声干扰。

**Baseline 公式** (LLaVA): $$T_{input} = T_{encoder} \in \mathbb{R}^{N \times d}$$
符号: $T_{encoder}$ = 视觉编码器输出的完整令牌序列, $N$ = 令牌总数（8K图像下极大）, $d$ = 令牌维度。

**变化点**: Baseline将全部$N$个令牌输入LLM，导致8K分辨率下显存溢出。本文假设背景令牌语义可聚类为少数同质簇，通过语义冗余分析识别并剔除这些簇对应的令牌。

**本文公式（推导）**:
$$\text{Step 1}: \quad S_i = \text{SemanticClass}(T_{encoder}^{(i)}) \in \{bg, obj\} \quad \text{（通过语义分析将每个令牌分类为背景或对象）}$$
$$\text{Step 2}: \quad T_{compressed} = \{T_{encoder}^{(i)} \text{mid} S_i = obj\} \cup \{r \cdot T_{encoder}^{(j)} \text{mid} S_j = bg, \text{randomly sampled}\} \quad \text{（保留全部对象令牌，背景令牌随机采样保留比例} r\text{）}$$
$$\text{最终}: \quad T_{compressed} = \text{BackgroundTokenPruning}(T_{encoder}) \in \mathbb{R}^{N' \times d}, \quad N' \approx N/2 \sim N/3$$

**对应消融**: Figure 5显示将背景令牌减半（halving）甚至提升了性能；Table 12中完全移除背景剪枝的变体精度显著下降。

---

### 模块 2: Anchored Token Selection（对应框架图 Stage 2）

**直觉**: 背景剪枝后仍有过剩令牌，需进一步筛选出与遥感任务真正相关的对象中心令牌；利用轻量warmup模型的检测能力作为"锚点"指导选择。

**Baseline 公式**: 无显式基线，Qwen2-VL等采用动态分辨率切分，但未引入对象感知的选择机制。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{logits}_{warmup} = f_{warmup}(x) \in \mathbb{R}^{H \times W \times C} \quad \text{（warmup模型输出空间检测对数几率，} C \text{为对象类别数）}$$
$$\text{Step 2}: \quad A_{i} = \max_{c} \sigma(\text{logits}_{warmup}^{(i,c)}) \quad \text{（计算每个空间位置的对象存在置信度）}$$
$$\text{Step 3}: \quad T_{final} = \text{TopK}(T_{compressed}, A, k = N/r^*) \quad \text{（按置信度选取Top-K令牌，} r^*=24\text{）}$$
$$\text{最终}: \quad T_{final} = \text{AnchoredTokenSelection}(T_{compressed}, \text{logits}_{warmup}) \in \mathbb{R}^{N'' \times d}$$
符号: $f_{warmup}$ = 预热检测模型, $\sigma$ = sigmoid函数, $r^*$ = 最优压缩比, $N'' = N/24$。

**对应消融**: Table 12显示压缩比24:1时平均精度51.5，优于32:1的50.3（-1.2），而16:1因OOM无法训练；Figure 6展示对象令牌消融，验证对象中心假设。

---

### 模块 3: 带压缩令牌的监督微调损失（对应框架图 Training）

**直觉**: 压缩需在训练阶段即引入，使LLM学会从稀疏对象令牌中重建完整语义，而非仅在推理时被动适应。

**Baseline 公式** (标准LLaVA SFT): $$\mathcal{L}_{SFT}^{base} = -\sum_{t} \log P(y_t \text{mid} y_{<t}, T_{encoder}(x), x_{text})$$
符号: $y_t$ = 第$t$个目标token, $y_{<t}$ = 历史生成, $T_{encoder}(x)$ = 完整视觉令牌, $x_{text}$ = 文本指令。

**变化点**: 将条件中的$T_{encoder}(x)$替换为两阶段压缩后的$T_{final}(x)$，使模型在训练时就适应压缩令牌的分布特性。

**本文公式（推导）**:
$$\text{Step 1}: \quad T_{final}(x) = \text{ATS}(\text{BTP}(T_{encoder}(x)), f_{warmup}(x)) \quad \text{（两阶段压缩作为可微/近似可微操作）}$$
$$\text{Step 2}: \quad \mathcal{L}_{SFT} = -\sum_{t} \log P(y_t \text{mid} y_{<t}, T_{final}(x), x_{text}) \quad \text{（标准自回归损失，条件为压缩后令牌）}$$
$$\text{约束}: \quad \text{Memory}(T_{final}(x)) \leq \text{Budget}_{GPU} \Rightarrow r^* = 24 \text{（经网格搜索验证）}$$
$$\text{最终}: \quad \mathcal{L}_{SFT} = -\sum_{t} \log P(y_t \text{mid} y_{<t}, T_{final}(x; r^*=24), x_{text})$$

**对应消融**: Table 13显示仅使用VRSBench-train（低分辨率数据）时平均精度暴跌至34.8（-16.7），证明高分辨率数据+压缩训练的必要性；移除HighRS-VQA降至49.4（-2.1）。

## 实验与分析



本文在 **XLRS-Bench** 上评估GeoLLaVA-8K，该benchmark专为评估MLLM对超大超高分辨率遥感图像的理解能力设计，包含Scene Classification (SC)、Complex Reasoning (CR)、Planning (PL)、Spatiotemporal Reasoning (STR)、Object Counting (OC)、Remote Sensing Object Classification (RC)、Object Level Urban Change (OLUC)、Object Spatial Relationship (OSR)八项子任务。Table 11展示了与12个强baseline的详细对比。

**主实验结果**: GeoLLaVA-8K（7B）在 **Scene Classification** 达到 **59.0**，超越最强闭源模型Gemini 2.0 Flash的43.3达+15.7，较遥感专用baseline GeoChat的22.8提升+36.2；**Planning** 任务以 **66.0** 大幅领先所有对比模型，较LLaVA-Next的30.0提升+36.0；**Spatiotemporal Reasoning** 取得 **50.0**，超越Qwen2.5-VL-72B的49.3，成为该任务最优7B模型。在Object Properties维度，GeoLLaVA-8K（46.1）超越Qwen2.5-VL-7B（44.0）。唯一未登顶的子任务是Object Spatial Relationship（35.0），低于InternVL2.5-8B的50.0和Qwen2.5-VL-7B的46.7，显示空间关系推理仍有提升空间。



**消融实验**（Table 12 & 13）揭示关键设计选择：压缩比方面，**24:1最优**（平均精度51.5），32:1降至50.3（-1.2），16:1在8-16 GPU上均OOM无法训练；数据集方面，**仅使用VRSBench-train**（低分辨率合成数据）平均精度暴跌至**34.8**（-16.7），Object Counting从26.7降至13.3，RC从48.0降至11.0，证明8K真实数据的决定性作用；移除HighRS-VQA降至49.4（-2.1），验证双数据集互补性。Figure 5直观展示背景令牌减半甚至提升性能，支持"背景冗余"假设。

**公平性检验**: 对比存在若干局限。优势方面，XLRS-Bench是领域内最新UHR benchmark，对比覆盖通用MLLM（LLaVA系列、Qwen2.5-VL、InternVL）、闭源API（GPT-4o、Claude、Gemini）及遥感专用模型（GeoChat），较为全面。不足方面：多个文献中出现的遥感MLLM（LHRS-Bot、RSGPT、EarthGPT、SkySenseGPT、H2RSVLM）未进入Table 11对比，可能因评测框架差异；GeoChat使用原生框架而其他模型统一用LMMs-Eval，存在潜在框架偏差；本文仅展示7B模型，而部分baseline提供72B版本，规模不对等；压缩比16:1的OOM表明方法向更高分辨率（如16K）扩展仍需进一步优化。此外，代码未公开， reproducibility受限。

## 方法谱系与知识库定位

**方法家族**: LLaVA-style Vision-Language Models（视觉指令调谐家族）

**父方法**: **LLaVA** [6] —— GeoLLaVA-8K明确继承其「视觉编码器 + 投影器 + LLM」三段式架构，以及视觉指令调谐的训练范式。同时受 **Qwen2-VL** [13] 的"任意分辨率"处理启发，但将通用动态切分升级为遥感语义感知的对象锚定压缩。

**改动槽位**:
- **data_pipeline**: 替换为自建的SuperRS-VQA + HighRS-VQA（8K级真实遥感数据）
- **inference_strategy**: 替换为两阶段压缩（BTP + ATS），替代全令牌输入
- **training_recipe**: 修改为SFT阶段即集成压缩，联合优化
- **architecture**: 轻量修改，增加warmup模型作为辅助检测分支

**直接基线与差异**:
| 基线 | 差异 |
|:---|:---|
| GeoChat [9] | 同为遥感MLLM，但GeoLLaVA-8K通过8K数据+令牌压缩实现分辨率飞跃 |
| LLaVA-Next [6衍生] | 通用MLLM改进版，无遥感特化、无UHR处理能力 |
| Qwen2.5-VL [13衍生] | 具备任意分辨率基础能力，但缺乏遥感语义感知的对象锚定机制 |
| InternVL2.5/3 [8衍生] | 强通用视觉理解，空间关系任务更强，但UHR遥感数据与任务适配不足 |

**后续方向**: (1) 将压缩比优化扩展至16K/32K卫星影像，解决16:1 OOM瓶颈；(2) 融合时序多光谱信息，从单帧静态理解迈向遥感视频/变化检测；(3) 开源代码与数据，填补当前reproducibility缺口。

**标签**: 模态=image+text | 范式=supervised fine-tuning | 场景=remote sensing / Earth observation | 机制=visual token compression / object-centric selection | 约束=GPU memory budget / ultra-high resolution

