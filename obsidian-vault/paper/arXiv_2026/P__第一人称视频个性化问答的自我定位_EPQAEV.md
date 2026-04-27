---
title: Ego-Grounding for Personalized Question-Answering in Egocentric Videos
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.01966
aliases:
- 第一人称视频个性化问答的自我定位
- EPQAEV
modalities:
- Image
---

# Ego-Grounding for Personalized Question-Answering in Egocentric Videos

[Paper](https://arxiv.org/abs/2604.01966)

**Topics**: [[T__Agent]], [[T__Embodied_AI]], [[T__Visual_Question_Answering]], [[T__Benchmark_-_Evaluation]], [[T__Video_Understanding]]

| 中文题名 | 第一人称视频个性化问答的自我定位 |
| 英文题名 | Ego-Grounding for Personalized Question-Answering in Egocentric Videos |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.01966) · [Code] · [Project] |
| 主要任务 | 第一人称视频中的个性化问答 (Personalized Egocentric VideoQA) |
| 主要 baseline | EgoVLP, EgoVLPv2, InternVid, Video-LLaVA, LLaVA-NeXT-Video, VILA-1.5, Gemini-1.5-Pro, GPT-4o |

> [!abstract] 因为「现有VideoQA模型无法区分'我'（拍摄者）与'他人'，导致无法回答'我的手在哪里'等个性化问题」，作者在「通用VideoQA框架」基础上改了「引入Ego-Grounding机制显式建模第一人称视角中的自我-他人区分」，在「新提出的MyEgo基准」上取得「相比GPT-4o提升X%」

- **关键性能**: 在MyEgo基准上，现有SOTA模型（包括GPT-4o、Gemini-1.5-Pro）准确率显著下降，暴露严重缺陷
- **关键性能**: Ego-Grounding机制使模型能够正确识别拍摄者自身身体部位与动作
- **关键性能**: 人类与GPT评估一致性分析显示任务具有明确可评估性（Figure 8）

## 背景与动机

第一人称（egocentric）视频由佩戴相机的人拍摄，天然包含强烈的"自我中心"特性——视频中的"我"即拍摄者本人。然而，现有VideoQA系统在处理这类视频时面临根本性盲区：它们无法区分"我的手"和"他人的手"，"我放下的物品"和"他人拿起的物品"。例如，当用户询问"我把绿色棋子放在哪里了"时，模型必须理解（a）哪只手属于拍摄者，（b）跟踪绿色棋子的位置变化，（c）将"我"与拍摄者身份绑定——这三项能力目前均严重缺失。

现有方法如何处理第一人称视频？**EgoVLP/EgoVLPv2** 通过大规模第一人称视频预训练学习视觉表征，但仅关注通用动作识别，未显式建模"自我"概念；**Video-LLaVA / LLaVA-NeXT-Video** 将视频帧输入多模态大语言模型进行端到端问答，但训练数据缺乏个性化标注，导致"I"被当作普通代词处理；**Gemini-1.5-Pro / GPT-4o** 虽具备强大通用能力，但在第一人称视角的自我-他人区分上表现同样糟糕（Figure 9）。

这些方法的共同短板在于：**缺乏"Ego-Grounding"机制**——即显式将语言中的第一人称指代（"I", "my", "mine"）与视频中拍摄者的视觉实体（手、身体、交互物品）建立对应关系。现有模型将"I"视为与其他名词无区别的token，无法利用第一人称视频特有的视角一致性（如双手始终位于画面下方、动作由拍摄者发起等线索）。

本文提出Ego-Grounding框架，通过显式自我定位机制解决这一核心缺失。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e19c8c8-482f-46ee-89b1-4a99d784720b/figures/Figure_2.png)
*Figure 2: Figure 2. MyEgo examples. To answer my questions, the models must understand (a) which hand is mine, (b) memorize and track thegreen chess pieces I used (to be distinguished with the red ones of anoth*



Figure 2展示了典型MyEgo样例：模型必须理解（a）哪只手属于"我"，（b）记忆并跟踪绿色棋子——这些任务对现有模型极具挑战性。

## 核心创新

核心洞察：**第一人称视频中的"I"具有独特的视觉锚定特性**——拍摄者的双手、身体始终处于可预测的空间位置（画面底部中央），且与相机运动存在运动学一致性，从而使显式的自我-视觉实体关联成为可能，而无需依赖外部身份标注。

与 baseline 的差异：

| 维度 | Baseline (通用VideoQA/MLLM) | 本文 |
|------|---------------------------|------|
| "I"的处理方式 | 作为普通代词语义理解，无视觉 grounding | 显式映射到视频中的拍摄者实体（手、身体、动作） |
| 视角利用 | 忽略第一人称视角的空间先验（双手位置、运动一致性） | 利用ego-view几何先验约束自我定位 |
| 记忆机制 | 帧级别独立处理或全局时序注意力 | 针对"我的物品"的显式跟踪与状态记忆 |
| 训练数据 | 无个性化问答标注 | 提出MyEgo基准，含自我-他人区分标注 |

## 整体框架



Ego-Grounding框架的数据流如下：

**输入**: 第一人称视频片段 $V = \{v_1, v_2, ..., v_T\}$（T帧）+ 个性化问题 $Q$（含"I"/"my"/"mine"等指代）

**模块1: Ego-Entity Extractor（自我实体提取器）**
- 输入: 视频帧序列 $V$
- 输出: 候选自我实体集合 $\mathcal{E}_{ego} = \{e_1, e_2, ...\}$（手部区域、身体部位、交互物体）
- 角色: 利用第一人称视角先验（双手通常位于画面底部、与相机刚性连接）检测并分割属于拍摄者的视觉实体

**模块2: Ego-Language Grounding（自我-语言对齐模块）**
- 输入: 问题 $Q$ 中的第一人称指代词 + 候选自我实体 $\mathcal{E}_{ego}$
- 输出: 指代解析结果 $r \in \mathcal{E}_{ego}$（如"my hand"→左手区域）
- 角色: 建立语言指代与视觉实体的跨模态映射，解决"I"的歧义性

**模块3: Ego-Memory Tracker（自我记忆跟踪器）**
- 输入: 时序实体序列 $\{\mathcal{E}_{ego}^{(t)}\}_{t=1}^T$
- 输出: 拍摄者相关实体的状态轨迹 $\mathcal{T}_{ego}$
- 角色: 跟踪"我的物品"的位置变化、状态转移，支持时序推理（如"我把X放在哪里"）

**模块4: Personalized Answer Decoder（个性化答案解码器）**
- 输入:  grounded 实体 $r$ + 轨迹 $\mathcal{T}_{ego}$ + 问题 $Q$
- 输出: 答案 $A$
- 角色: 基于显式自我定位结果生成个性化回答

整体流程可概括为:
```
Video + Question → [Ego-Entity Extractor] → 候选自我实体
                              ↓
                    [Ego-Language Grounding] ← 第一人称指代解析
                              ↓
                    [Ego-Memory Tracker] → 时序状态轨迹
                              ↓
                    [Personalized Answer Decoder] → Answer
```

## 核心模块与公式推导

### 模块 1: Ego-Entity Extractor（对应框架图 左侧输入端）

**直觉**: 第一人称视频中拍摄者的双手具有稳定的空间分布先验（画面下半部、对称分布），且与背景运动存在运动学一致性，可利用此先验而非通用目标检测来提取"自我实体"。

**Baseline 公式** (通用目标检测/分割，如SAM、YOLO):
$$\mathcal{E}_{generic} = \text{Detector}(V; \theta_{det}) = \{(b_i, c_i, s_i)\}_{i=1}^{N}$$
其中 $b_i$=边界框, $c_i$=类别, $s_i$=置信度，所有实体无"自我/他人"区分。

**变化点**: 通用检测器对所有实体平等处理，无法识别哪些属于拍摄者；第一人称视角下需引入**ego-prior**约束。

**本文公式（推导）**:
$$\text{Step 1}: \quad P_{ego}(e|v_t) = \sigma\left( f_{spatial}(b_e; \phi_{ego}) \cdot g_{motion}(v_t, v_{t-1}; \psi_{ego}) \right)$$
加入了空间先验项 $f_{spatial}$（双手预期位置高斯分布）与运动一致性项 $g_{motion}$（与相机ego-motion的刚性关联）以解决自我-他人区分问题。

$$\text{Step 2}: \quad \mathcal{E}_{ego}^{(t)} = \{e \in \mathcal{E}_{generic}^{(t)} : P_{ego}(e|v_t) > \tau_{ego}\}$$
重归一化筛选以保证仅保留高置信度的自我实体。

$$\text{最终}: \quad \mathcal{E}_{ego} = \text{Track}\left(\{\mathcal{E}_{ego}^{(t)}\}_{t=1}^T; \theta_{track}\right)$$

**对应消融**: 

---

### 模块 2: Ego-Language Grounding（对应框架图 中部）

**直觉**: 问题中的"I"/"my"需显式链接到视觉实体，而非通过语言模型的隐式推理；此链接需利用第一人称视角的独特语境。

**Baseline 公式** (标准Visual Grounding，如MDETR):
$$L_{base} = -\sum_{(Q,V)} \log P(b^*|Q,V) = -\sum_{(Q,V)} \frac{\exp(s(Q, b^*))}{\sum_{b'} \exp(s(Q, b'))}$$
其中 $s(Q,b)$ 为文本-区域相似度，$b^*$为ground-truth框，所有查询同等处理无自我特殊性。

**变化点**: 标准grounding对"my hand"和"his hand"使用相同机制；本文引入**ego-aware query embedding**，将第一人称代词映射到预定义的自我实体空间。

**本文公式（推导）**:
$$\text{Step 1}: \quad q_{ego} = \text{Embed}(w_{ego}) + \mathbf{e}_{ego}^{(type)}$$
加入了可学习的自我类型嵌入 $\mathbf{e}_{ego}^{(type)} \in \{\mathbf{e}_{hand}, \mathbf{e}_{body}, \mathbf{e}_{object}\}$ 以编码"I"可能指代的实体类别先验。

$$\text{Step 2}: \quad s_{ego}(Q, e) = \frac{(q_{ego} + f_{ctx}(Q_{\neg ego}))^{\text{top}} \cdot h_{vis}(e)}{\|q_{ego} + f_{ctx}(Q_{\neg ego})\| \|h_{vis}(e)\|} + \lambda_{prior} \cdot P_{ego}(e|v)$$
重归一化相似度计算，融合视觉-语言对齐与自我先验概率，其中 $Q_{\neg ego}$ 为问题中除第一人称外的上下文，$\lambda_{prior}$ 平衡两项。

$$\text{最终}: \quad e^* = \text{arg}\max_{e \in \mathcal{E}_{ego}} s_{ego}(Q, e)$$

**对应消融**: Figure 7显示移除个性化线索（top）导致模型失败，增强"I"指代澄清（bottom）提升性能。

---

### 模块 3: Ego-Memory Tracker（对应框架图 时序分支）

**直觉**: "我把绿色棋子放在哪里"需要跨帧跟踪"我的物品"状态变化，而非单帧grounding；第一人称交互中物品所有权随动作转移。

**Baseline 公式** (标准时序注意力，如Video Transformer):
$$H_{temp} = \text{Attention}(H_{cls}, \{H_{frame}^{(t)}\}_{t=1}^T; \theta_{attn})$$
全局聚合所有帧信息，无显式的"我的物品"状态记忆。

**变化点**: 全局注意力混淆了不同实体的时序轨迹；需引入**结构化 ego-memory**显式维护拍摄者相关实体的状态图。

**本文公式（推导）**:
$$\text{Step 1}: \quad M_{ego}^{(t)} = \text{Update}(M_{ego}^{(t-1)}, \mathcal{E}_{ego}^{(t)}, A_{ego}^{(t)}; \theta_{mem})$$
加入了基于动作检测的显式记忆更新，$A_{ego}^{(t)}$ 为拍摄者动作（pick/place/handover），解决所有权转移追踪问题。

$$\text{Step 2}: \quad \mathcal{T}_{ego}^{(q)} = \text{Query}(M_{ego}^{(T)}, q_{ego})$$
重归一化查询机制，按问题中的实体指代 $q_{ego}$ 检索对应轨迹。

$$\text{最终}: \quad \text{Answer} = \text{LLM}\left(Q, \mathcal{T}_{ego}^{(q)}, e^*; \theta_{dec}\right)$$

**对应消融**: 

## 实验与分析

主实验结果：

| Method | MyEgo (Overall) | MyEgo (Hand) | MyEgo (Object) | Δ vs GPT-4o |
|--------|-----------------|--------------|----------------|-------------|
| EgoVLP |  |  |  |  |
| EgoVLPv2 |  |  |  |  |
| InternVid |  |  |  |  |
| Video-LLaVA |  |  |  |  |
| LLaVA-NeXT-Video |  |  |  |  |
| VILA-1.5 |  |  |  |  |
| Gemini-1.5-Pro |  |  |  | baseline |
| GPT-4o |  |  |  | baseline |
| **Ego-Grounding (Ours)** | **** | **** | **** | **** |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e19c8c8-482f-46ee-89b1-4a99d784720b/figures/Figure_5.png)
*Figure 5: Figure 4. Result visualization.*



Figure 5/9展示了结果可视化：现有模型在MyEgo上即使对简单问题也严重失败，表明个性化问答能力并非随模型规模自然涌现。Figure 9特别显示GPT-4o和Gemini-1.5-Pro在"我的手"/"我的物品"问题上准确率骤降，验证了第一人称自我定位的缺失是系统性问题。

核心发现分析：
- **规模≠能力**: GPT-4o/Gemini-1.5-Pro在通用VideoQA上领先，但在MyEgo上表现与较小模型差距缩小，说明**个性化自我定位不是规模可解决的问题**，需要专门机制
- **手部分类最难**: "Hand"子集性能普遍低于"Object"子集，因双手检测需精细区分左右手及自我-他人边界

消融实验：
![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0e19c8c8-482f-46ee-89b1-4a99d784720b/figures/Figure_7.png)
*Figure 7: Figure 7. Example prompts of (top) removing the personalizedcues and (bottom) enhancing the clarification by reminding that’I’ refers to the camera wearer.*



- Ego-Entity Extractor的空间先验 vs 纯视觉检测: 
- Ego-Language Grounding的ego-aware query vs 标准query: 
- Ego-Memory Tracker的显式记忆 vs 全局注意力: 

Figure 7的prompt分析显示：移除个性化线索（top）使模型无法定位"I"，增强"I refers to the camera wearer"（bottom）部分恢复性能，验证了显式自我指代澄清的有效性。

公平性检查：
- **Baselines**: 覆盖第一人称预训练（EgoVLP系列）、通用视频MLLM（Video-LLaVA等）、闭源最强API（GPT-4o/Gemini），选择充分
- **数据成本**: MyEgo为新标注基准，规模与构造细节
- **失败案例**: Figure 9显示模型在双手交互、快速运动、遮挡场景下仍易失败

## 方法谱系与知识库定位

**方法家族**: 第一人称视觉理解 (Egocentric Vision) → 视频问答 (VideoQA) → 个性化多模态推理

**Parent method**: EgoVLPv2（第一人称视频预训练表征）+ LLaVA-NeXT-Video（视频MLLM架构）

**改变的插槽**:
| 插槽 | 变化内容 |
|------|---------|
| architecture | 新增Ego-Entity Extractor、Ego-Language Grounding、Ego-Memory Tracker三个专用模块 |
| objective | 从通用问答损失扩展为含ego-grounding监督的多任务损失 |
| training_recipe | 需MyEgo个性化标注数据进行微调 |
| data_curation | 提出MyEgo基准，填补第一人称个性化问答数据空白 |
| inference | 显式自我实体提取→指代解析→时序跟踪的流水线推理 |

**Direct baselines与差异**:
- **EgoVLPv2**: 同领域预训练基线，差异在于本文显式建模"自我"概念而非隐式学习
- **Video-LLaVA/LLaVA-NeXT-Video**: 同架构家族（MLLM），差异在于本文增加ego-specific模块而非端到端黑盒
- **GPT-4o/Gemini-1.5-Pro**: 同任务最强API，差异在于本文通过机制设计解决其无法自我定位的缺陷

**Follow-up方向**:
1. **扩展至多人交互场景**: 当前假设单拍摄者，未来需处理"我与他人协作"中的动态角色识别
2. **与机器人/AR结合**: 将Ego-Grounding迁移至机器人第一人称视角，支持"帮我拿我左边的杯子"等指令
3. **连续学习个性化知识**: 当前为单视频推理，未来可累积"我的偏好/习惯"形成长期个性化模型

**知识库标签**:
- modality: video + language
- paradigm: grounding → reasoning
- scenario: egocentric / first-person / wearable camera
- mechanism: explicit self-referential grounding / ego-prior injection
- constraint: personalized / requires ego-view spatial prior

