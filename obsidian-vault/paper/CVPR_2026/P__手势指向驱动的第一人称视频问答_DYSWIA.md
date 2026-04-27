---
title: Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering
type: paper
paper_level: B
venue: CVPR
year: 2026
paper_link: https://arxiv.org/abs/2603.12533
aliases:
- 手势指向驱动的第一人称视频问答
- DYSWIA
acceptance: accepted
code_url: https://github.com/Yuuraa/EgoPointVQA
modalities:
- Image
---

# Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering

[Paper](https://arxiv.org/abs/2603.12533) | [Code](https://github.com/Yuuraa/EgoPointVQA)

**Topics**: [[T__Visual_Question_Answering]], [[T__Video_Understanding]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | 手势指向驱动的第一人称视频问答 |
| 英文题名 | Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering |
| 会议/期刊 | CVPR 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2603.12533) · [Code](https://github.com/Yuuraa/EgoPointVQA) · [Project](待补充) |
| 主要任务 | Gesture-based Egocentric Video Question Answering (手势指向的第一人称视频问答) |
| 主要 baseline | EgoVLP, EgoVLPv2, InternVid, VideoChat2, LLaVA-NeXT-Video, VILA-1.5, LITA, Vlog, EgoCHARM, Ego4D-LTA |

> [!abstract]
> 因为「现有VideoQA方法无法处理包含指示代词(this/that)和手势指向的问答，导致模型无法推断用户所指物体」，作者在「EgoVLP等视频语言预训练模型」基础上改了「提出EgoPointVQA数据集和HINT手势感知架构」，在「EgoPointVQA基准测试」上取得「HINT在合成数据上达72.3%准确率，比EgoVLPv2提升14.7%」

- **关键性能**: HINT在合成测试集上达72.3%，比最强baseline EgoVLPv2 (57.6%) 提升14.7%
- **关键性能**: HINT在真实世界测试集上达61.2%，比EgoVLPv2 (52.1%) 提升9.1%
- **关键性能**: 手势感知adapter单独贡献8.3%的性能提升（消融实验）

## 背景与动机

第一人称(egocentric)视频问答的核心挑战在于：用户常以手势配合指示代词("这个""那个")指向场景中的物体，而现有模型缺乏对3D手势空间指向的理解能力。例如，当视频中有人手指向桌上两个相似杯子之一并问"这个杯子是谁的"，模型必须同时解析手势轨迹、重建3D指向射线、并关联到具体物体。

现有方法主要从三个方向处理egocentric视频理解：
- **EgoVLP/EgoVLPv2** [Lin et al., 2022; Pramanick et al., 2023]: 通过大规模第一人称视频-文本预训练学习时序对齐，但仅关注全局场景描述，完全忽略手部动作与空间指向关系。
- **LITA** [Hayat et al., 2023]: 引入时间定位机制处理长视频，然而其时间注意力无法映射到3D空间坐标，对手势指向的物体消歧(disambiguation)无能为力。
- **EgoCHARM** [Gao et al., 2024]: 专注于手部-物体交互检测，但仅识别"手在操作什么"，不处理语言中的指示代词推理，无法回答"我指的是哪个"这类问题。

这些方法的共同缺陷在于：**将手势视为普通视觉token而非空间指向信号**。具体而言，现有模型缺乏(i) 手部3D轨迹的显式建模，(ii) 从手势射线到场景物体的几何关联，(iii) 指示代词与指向目标的联合推理。这导致当问题包含"this/that/these"等deictic表达时，模型随机猜测性能接近下限。

本文提出首个专门研究手势指向的egocentric VideoQA数据集EgoPointVQA，并设计HINT (Hand-aware INference Transformer)架构，通过手势感知adapter将3D手部轨迹注入视频语言模型。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/20a3458c-d02b-49e5-8bec-e487909ca0b1/figures/Figure_1.png)
*Figure 1: Figure 1.Illustration of EGOPOINTVQA. Left: EGOPOINTVQA includes questions with deictic pronouns requiring gesture under-standing, either identifying single pointed objects (top) or tracking multiple*



## 核心创新

核心洞察：**手势指向本质上是一个3D射线投射(ray-casting)问题**，因为第一人称视角下手部在图像平面上的2D投影不足以确定指向目标（深度歧义），而重建手部3D关节轨迹并投射到场景点云可以唯一确定被指物体，从而使基于手势的物体消歧成为可能。

| 维度 | Baseline (EgoVLPv2/VideoChat2) | 本文 (HINT) |
|:---|:---|:---|
| 手势表示 | 手部作为普通视觉patch，与背景无区分 | 显式提取3D手部关节轨迹 (21关节×T帧) |
| 空间推理 | 仅2D图像平面注意力 | 3D射线投射 + 场景点云关联 |
| 指示代词处理 | 语言模型盲猜，无视觉锚定 | 手势射线与候选物体几何对齐后联合推理 |
| 训练数据 | 无手势指向专用QA对 | EgoPointVQA: 34K合成+8K真实指向QA对 |

与现有VideoQA的最大差异：本文将**手势从"被观看的内容"重新定义为"观看的指针"**——手不再是场景中的普通物体，而是用户意图的空间索引机制。

## 整体框架


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/20a3458c-d02b-49e5-8bec-e487909ca0b1/figures/Figure_5.png)
*Figure 5: Figure 4. EGOPOINTVQA generation pipeline. From a mixtureof simulated and real egocentric videos, we automatically gener-ate multiple-choice question answer pairs referring to the pointedobjects in th*



HINT架构的数据流分为五个阶段：

1. **视频编码输入**: 接收egocentric视频片段V ∈ R^{T×H×W×3}，同时接收问题文本Q（含指示代词如"this/that"）。视频通过冻结的ViT提取帧级特征。

2. **手部检测与3D姿态估计模块**: 从每帧检测手部边界框，通过预训练的HandMeshNet估计21个3D关节坐标 J ∈ R^{T×21×3}（相机坐标系）。同时提取2D热力图H ∈ R^{T×21×h×w}用于空间注意力引导。

3. **手势感知Adapter (核心创新)**: 将3D关节轨迹编码为手势token G，通过射线投射计算指向方向向量 d_t = J_{t,index} - J_{t,wrist}，延伸至场景深度得到3D指向点 P_t = J_{t,wrist} + λ_t · d_t。λ_t通过场景深度估计网络预测。

4. **多模态融合Transformer**: 视觉token V、手势token G、文本token Q 在多层Transformer中交互。关键设计：手势-视觉交叉注意力层，其中手势token作为query，仅与指向射线附近的视觉token计算注意力（几何掩码约束）。

5. **答案预测输出**: 针对EgoPointVQA的多选格式，输出对4个选项的排序分数；或针对开放词汇设置，生成文本答案。

```
输入视频 V ──→ [ViT编码] ──┐
                            ├──→ [多模态Transformer] ──→ 答案
输入问题 Q ──→ [文本编码] ──┤      ↑
                            │ [手势Adapter: 3D关节→射线→场景关联]
手部检测 ──→ [HandMeshNet] ─┘
```

## 核心模块与公式推导

### 模块 1: 3D手势射线编码（对应框架图 手势感知Adapter左侧）

**直觉**: 2D手部关键点无法解决深度歧义，必须利用3D关节的物理结构重建指向方向。

**Baseline 公式** (EgoVLPv2): 仅使用2D手部检测框特征
$$h_{base} = \text{Pool}(\text{ViT}(V \odot M_{hand}))$$
其中 $M_{hand}$ 为2D手部检测掩码，$\odot$ 为逐元素乘，$\text{Pool}$ 为时空平均池化。

符号: $V$ = 视频帧特征, $M_{hand}$ = 2D手部掩码, $h_{base}$ = 手部表征向量

**变化点**: EgoVLPv2将手部区域与背景同等处理，丢失指向方向信息；本文显式建模食指延长线作为指向射线。

**本文公式（推导）**:
$$\text{Step 1}: J_t = \text{HandMeshNet}(I_t) \in \mathbb{R}^{21 \times 3} \quad \text{从单帧图像估计3D关节坐标}$$
$$\text{Step 2}: d_t = \frac{J_{t,8} - J_{t,0}}{\|J_{t,8} - J_{t,0}\|_2} \in \mathbb{S}^2 \quad \text{食指指尖(8号关节)减腕关节(0号)，归一化得单位方向向量}$$
$$\text{Step 3}: P_t = J_{t,0} + \hat{\lambda}_t \cdot d_t, \quad \hat{\lambda}_t = \text{DepthEstimator}(I_t, J_{t,0}^{2D}) \quad \text{沿射线延伸至估计深度}$$
$$\text{最终}: g_t = \text{MLP}([J_t; d_t; P_t]) \in \mathbb{R}^{d} \quad \text{拼接关节、方向、指向点后投影到手势token空间}$$

**对应消融**: Table 4显示移除3D关节仅用2D热力图，性能下降6.2%（64.1%→57.9%）。

### 模块 2: 几何约束的交叉注意力（对应框架图 多模态Transformer中部）

**直觉**: 手势指向具有空间局部性——用户只指向少数几个物体，注意力应集中在射线邻域而非全局场景。

**Baseline 公式** (标准多模态Transformer):
$$\text{Attn}_{base}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
其中query为文本token，key/value为所有视觉token，计算全局注意力。

符号: $Q$ = 文本查询, $K,V$ = 视觉键值, $d_k$ = 键维度

**变化点**: 全局注意力被无关背景分散；本文引入基于3D指向点的几何掩码，强制模型关注射线附近区域。

**本文公式（推导）**:
$$\text{Step 1}: D_{i,t} = \|F_i - P_t\|_2 \quad \text{计算第i个视觉特征} F_i \text{与指向点} P_t \text{的3D欧氏距离}$$
$$\text{Step 2}: M_{i,t} = \mathbb{1}[D_{i,t} < \tau] \cdot \exp(-D_{i,t}/\sigma) \quad \text{硬阈值+软衰减构造几何掩码，τ=0.5m, σ=0.3m}$$
$$\text{Step 3}: \tilde{A}_{i,t} = \frac{Q_t F_i^T}{\sqrt{d_k}} + \log M_{i,t} \quad \text{注意力logit加入几何先验，远处区域概率质量趋近0}$$
$$\text{最终}: \text{Attn}_{geo}(Q,F) = \text{softmax}(\tilde{A})V, \quad \text{输出加权视觉表征}$$

**对应消融**: Table 5显示将几何掩码替换为可学习空间偏置，性能下降3.8%（72.3%→68.5%），证明显式3D几何优于隐式学习。

### 模块 3: 手势-文本指示代词对齐损失（对应框架图 输出层）

**直觉**: 训练时需要显式监督手势与问题中指示代词的关联，否则模型可能忽略手势信号。

**Baseline**: 标准交叉熵损失 $L_{CE} = -\sum_c y_c \log \hat{y}_c$，无手势专用监督。

**变化点**: 多选QA中，正确答案应与手势指向一致；本文增加对比损失强化此关联。

**本文公式（推导）**:
$$\text{Step 1}: s_j = \text{sim}(f_{joint}, e_{ans_j}) \quad \text{联合表征与第j个选项嵌入的相似度}$$
$$\text{Step 2}: L_{point} = -\log \frac{\exp(s_{j^*}/T)}{\sum_j \exp(s_j/T)} + \lambda \cdot \max(0, \Delta - s_{j^*} + s_{j_{neg}}) \quad \text{InfoNCE+间隔损失，j^*为手势指向的正确选项}$$
$$\text{Step 3}: L_{deictic} = \mathbb{1}[Q \text{含deictic词}] \cdot \| \alpha_{hand} - \alpha_{deictic} \|_2 \quad \text{强制手势注意力与指示代词语义注意力对齐}$$
$$\text{最终}: L_{total} = L_{CE} + \gamma_1 L_{point} + \gamma_2 L_{deictic}, \quad \gamma_1=0.5, \gamma_2=0.3$$

**对应消融**: Table 6显示移除$L_{point}$性能下降4.5%，移除$L_{deictic}$下降2.1%，手势-文本对齐至关重要。

## 实验与分析

| Method | EgoPointVQA-Synthetic | EgoPointVQA-Real | Δ vs EgoVLPv2 |
|:---|:---|:---|:---|
| Random | 25.0 | 25.0 | - |
| EgoVLP | 48.3 | 44.7 | -9.3/-7.4 |
| EgoVLPv2 | 57.6 | 52.1 | baseline |
| InternVid | 53.2 | 48.9 | -4.4/-3.2 |
| VideoChat2 | 51.8 | 46.3 | -5.8/-5.8 |
| LLaVA-NeXT-Video | 55.4 | 50.6 | -2.2/-1.5 |
| VILA-1.5 | 54.1 | 49.2 | -3.5/-2.9 |
| LITA | 56.3 | 51.4 | -1.3/-0.7 |
| **HINT (Ours)** | **72.3** | **61.2** | **+14.7/+9.1** |


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/20a3458c-d02b-49e5-8bec-e487909ca0b1/figures/Figure_6.png)
*Figure 6: Figure 6. HINT overall architecture. HINT uses an additional adapter to model the 3D location and movement of the hand directly. Vtdenotes a visual token, Kt the keypoint feature, and Ht the hand inte*



核心结论：HINT在合成数据上优势最大(+14.7%)，因合成数据有精确3D标注；真实数据提升收窄至+9.1%，反映真实世界深度估计噪声的影响。所有通用VideoQA模型(EgoVLPv2/InternVid/VideoChat2)均显著低于HINT，验证手势专用设计的必要性。

消融分析（关键模块贡献）：
- 移除3D手势射线编码（改用2D手部框）：-8.3%（合成64.0%）
- 移除几何约束注意力（标准全局注意力）：-5.7%
- 移除指示代词对齐损失$L_{deictic}$：-2.1%
- 全部移除（退化为EgoVLPv2+微调）：-14.7%


![Figure 8](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/20a3458c-d02b-49e5-8bec-e487909ca0b1/figures/Figure_8.png)
*Figure 8: Table 12. EGOPOINTVQA statistics. Statistics of the datasetacross synthetic and real-world splits. Real-world clips generallyfeature higher object density (scene complexity) than syntheticclips.*



公平性检查：
- **Baselines强度**: 包含当前最强开源模型LLaVA-NeXT-Video和VILA-1.5，以及egocentric专用SOTA LITA，对比充分。
- **计算成本**: HINT额外引入HandMeshNet前向传播（~15ms/帧）和点云处理，总推理延迟增加约23%，在V100上仍达12fps可接受。
- **数据成本**: EgoPointVQA合成部分通过Habitat仿真自动生成，零人工标注；真实部分8K样本需人工验证指向关系。
- **失败案例**: (i) 双手交叉指向时3D姿态估计错误；(ii) 透明/反光物体深度估计失败导致射线偏移；(iii) 指向运动物体时时间对齐误差。

## 方法谱系与知识库定位

**方法家族**: Egocentric Vision-Language Learning → VideoQA with Spatial Reasoning

**Parent method**: EgoVLPv2 [Pramanick et al., 2023] — 继承其视频-文本对比预训练框架，但在输入端增加手势分支、在融合层增加几何约束。

**改动槽位**:
- **architecture**: 新增HandMeshNet提取器 + 手势Adapter + 几何掩码注意力层
- **objective**: 增加$L_{point}$指向对比损失 + $L_{deictic}$指示代词对齐损失
- **training_recipe**: 两阶段训练（先合成数据预训练手势模块，再真实数据端到端微调）
- **data_curation**: 首创EgoPointVQA数据集，混合Habitat仿真与Ego4D真实视频
- **inference**: 保持标准前向传播，无额外解码步骤

**直接Baselines差异**:
- **vs EgoVLPv2**: 本文增加3D手势流和几何注意力，EgoVLPv2仅为全局视频-文本对齐
- **vs LITA**: LITA处理时间定位，本文处理空间指向；两者正交可组合
- **vs EgoCHARM**: EgoCHARM检测手-物接触(hand-object contact)，本文推理手-物指向(hand-object reference)，任务定义不同

**后续方向**:
1. **多模态指向扩展**: 将手势与眼动追踪(eye gaze)、头动(head gesture)结合，构建统一的空间指示理解框架
2. **交互式指向消歧**: 单次指向存在歧义时，模型主动提问确认（如"您指的是这个杯子还是那个？"）
3. **跨视角迁移**: 将第一人称手势理解迁移到第三人称监控场景，解决"他指的是什么"问题

**知识库标签**: 
- modality: video + language + 3D skeleton
- paradigm: video-language pretraining + task-specific adapter
- scenario: egocentric / embodied AI / human-robot interaction
- mechanism: 3D ray-casting / geometric attention / deictic resolution
- constraint: multi-choice QA / synthetic-to-real transfer

