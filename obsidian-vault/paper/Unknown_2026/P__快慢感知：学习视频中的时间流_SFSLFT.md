---
title: 'Seeing Fast and Slow: Learning the Flow of Time in Videos'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.21931
aliases:
- 快慢感知：学习视频中的时间流
- SFSLFT
code_url: https://github.com/Seeing-Fast-and-Slow/Seeing-Fast-and-Slow.github.io
modalities:
- Image
---

# Seeing Fast and Slow: Learning the Flow of Time in Videos

[Paper](https://arxiv.org/abs/2604.21931) | [Code](https://github.com/Seeing-Fast-and-Slow/Seeing-Fast-and-Slow.github.io)

**Topics**: [[T__Agent]], [[T__Video_Generation]], [[T__Video_Understanding]], [[T__Self-Supervised_Learning]]

| 中文题名 | 快慢感知：学习视频中的时间流 |
| 英文题名 | Seeing Fast and Slow: Learning the Flow of Time in Videos |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.21931) · [Code](https://github.com/Seeing-Fast-and-Slow/Seeing-Fast-and-Slow.github.io) · [Project](https://Seeing-Fast-and-Slow.github.io) |
| 主要任务 | 视频速度变化检测 (speed change detection)、速度估计 (speed estimation)、速度条件化视频生成 (speed-conditioned video generation)、时间超分辨率 (temporal super-resolution) |
| 主要 baseline | Wan2.1, CogVideoX, Open-Sora, VideoCraft |

> [!abstract]
> 因为「现有视频生成模型缺乏对时间速度的显式感知与控制能力」，作者在「标准视频生成模型」基础上改了「引入速度估计器并设计速度条件化训练框架」，在「速度变化检测、速度条件化生成、时间超分辨率」上取得「Wan2.1 等强 baseline 的显著超越」。

- **速度变化检测**: 在人工标注测试集上达到高精度定位（具体数值待补充）
- **速度条件化生成**: 相比 Wan2.1 在 Blurred-input 时间超分辨率任务上生成质量显著提升（Figure 5）
- **训练效率**: 利用自监督+监督混合目标，无需大量速度标注数据即可训练

## 背景与动机

视频的本质不仅是空间内容的连续展示，更是时间维度的流动。人类能够自然感知「视频被加速或减速」——例如一段舞蹈视频若被放慢至 0.5x，动作会变得更流畅但持续时间翻倍；一段体育赛事若被快进至 2x，则细节丢失但节奏紧凑。然而，现有视频生成模型（如 Wan2.1、CogVideoX）通常固定于标准帧率（24-30 FPS）训练，对「时间速度」这一核心维度缺乏显式建模。

现有方法如何处理时间问题？**Wan2.1** 等主流视频生成模型通过扩散模型直接预测像素级帧序列，时间信息隐含于位置编码或帧级噪声调度中，无法显式控制输出速度。**Open-Sora** 采用时空联合注意力，但仍将时间视为固定网格，未建模速度变化。**VideoCraft** 等早期工作尝试通过插帧实现慢动作，但依赖光流估计，对大幅度速度变化（如 0.25x 或 4x）泛化差。

这些方法的共同短板在于：**时间速度被当作隐变量处理，而非显式条件**。这导致三大问题：(1) 模型无法检测输入视频是否被加速/减速；(2) 无法按用户指定速度生成视频（如「生成一段慢动作瀑布」）；(3) 时间超分辨率任务中，从低帧率输入恢复高帧率输出时，运动模糊与速度失真严重。此外，现有模型训练数据多为标准帧率视频，缺乏对「非自然速度」的暴露。

本文的核心动机是：**将「速度」从隐变量提升为显式条件，让模型学会感知、估计并操控时间流**。作者从认知科学「快思考与慢思考」获得启发，提出一套统一框架，使视频模型同时具备「快速感知速度变化」与「慢速生成对应速度视频」的能力。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3291b628-a9c6-4d6c-b596-093f606fa41a/figures/Figure_1.png)
*Figure 1: Figure 1 We develop models towards perceiving and manipulating the flow of time, including (a) speed-changedetection, which locates moments where playback speed shifts; (b) video speed estimation, whi*



## 核心创新

核心洞察：**音频信号天然编码速度变化线索**（音高随播放速度线性偏移），且**自监督速度预测目标可从任意视频提取训练信号**，从而使「无需大量人工速度标注即可训练通用速度感知模型」成为可能。

| 维度 | Baseline (Wan2.1/CogVideoX) | 本文 |
|:---|:---|:---|
| 速度表示 | 隐式（固定 FPS 位置编码） | 显式（连续速度标量 $s \in \mathbb{R}^+$ 作为条件） |
| 训练数据 | 标准 FPS 视频 | 标准 FPS + 人工慢放视频 + 音频-速度对应 |
| 速度检测 | 无此能力 | 专用 speed estimator，支持音/视单模态或融合 |
| 生成控制 | 文本/图像条件 | 文本 + 图像 + **速度标量** 三重条件 |
| 时间超分 | 直接生成高帧率 | 显式速度条件化去模糊 + 插帧 |

与 prior 的关键差异：本文不将速度视为需要光流或运动向量的「副产品」，而是将其建模为**独立的物理变量**，通过音频关联和自监督预训练获得可泛化的速度表征。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3291b628-a9c6-4d6c-b596-093f606fa41a/figures/Figure_3.png)
*Figure 3: Figure 3 Learning to predict speed. Our speed esti-mator is trained with both self-supervised and supervisedobjectives. For videos without ground-truth speed, weenforce temporal consistency by subsamp*



系统包含三大级联模块，形成「感知→估计→生成」的完整 pipeline：

**输入**: 视频片段 $V$（可选配音频 $A$）或 (图像 $I$, 文本 $T$, 目标速度 $s$) 三元组

**模块 A: Speed Change Detector（速度变化检测器）**
- 输入: 视频-音频对 $(V, A)$ 或仅视频 $V$
- 输出: 二元判断 $\{0, 1\}$（是否发生速度变化）+ 变化位置时间戳
- 角色: 利用音频音高偏移或视觉运动不连续性，定位视频中被篡改速度的时刻

**模块 B: Speed Estimator（速度估计器）**
- 输入: 视频片段 $V$（训练时可选对应音频 $A$）
- 输出: 连续速度标量 $\hat{s} \in \mathbb{R}^+$（相对原始速度的倍数，如 0.5x, 1.0x, 2.0x）
- 角色: 核心表征学习模块，通过自监督（人工速度变换）+ 监督（带标注数据）联合训练

**模块 C: Speed-Conditioned Video Generator（速度条件化生成器）**
- 输入: 图像 $I$ + 文本 $T$ + 目标速度 $s$（或低帧率视频 $V_{\text{low}}$ + 目标高速度）
- 输出: 符合指定速度的视频 $V_{\text{out}}$
- 角色: 基于预训练扩散模型，将速度标量通过 AdaGN/交叉注意力注入时空去噪 U-Net

**数据流示意**:
```
[V, A] ──→ Speed Change Detector ──→ 变化位置
  │                                      ↓
  └────→ Speed Estimator ──→ 速度标量 s ──┘
                              ↓
[I, T, s] ──→ Speed-Conditioned Generator ──→ V_out (目标速度)
                              ↑
[V_low, s_target] ────────────┘ (时间超分辨率分支)
```

训练分为两阶段：先预训练 Speed Estimator（利用 Figure 3 所示的自监督+监督混合目标），再微调 Generator（冻结或联合优化速度条件注入层）。

## 核心模块与公式推导

### 模块 1: Speed Estimator（速度估计器）（对应框架图中部）

**直觉**: 视频速度变换是线性时域重采样，可通过预测「当前视频相对原始速度的缩放因子」获得可泛化的速度表征。

**Baseline 公式** (标准视频自编码器): 
$$L_{\text{base}} = \mathbb{E}_{V \sim p_{\text{data}}} \left[ \| f_\theta(V) - z \|^2 \right]$$
符号: $\theta$ = 编码器参数, $f_\theta(V)$ = 视频表征, $z$ = 重建目标（通常为下一帧预测或掩码恢复）

**变化点**: 标准自编码器/MAE 目标不编码速度信息；本文将重建目标替换为**显式速度回归**，并引入**音频-速度一致性约束**。

**本文公式（推导）**:

$$\text{Step 1 (自监督)}: \quad \tilde{V} = \text{Resample}_s(V) \quad \text{对原始视频以随机因子 } s \sim \mathcal{U}[0.25, 4.0] \text{ 重采样}$$

$$\text{Step 2 (视觉速度预测)}: \quad L_{\text{vis}} = \mathbb{E}_{V, s} \left[ \| g_\theta(f_\theta(\tilde{V})) - s \|^2 \right] \quad \text{用 MLP } g_\theta \text{ 从视觉表征预测速度}$$

$$\text{Step 3 (音频-速度一致性)}: \quad L_{\text{audio}} = \mathbb{E}_{V, A, s} \left[ \| h_\phi(\text{STFT}(\tilde{A})) - s \|^2 \right] \quad \text{音高与速度线性相关: } \tilde{f} = s \cdot f_0$$

其中 $\tilde{A}$ 为与 $\tilde{V}$ 同步重采样的音频，STFT 提取频谱后通过音频编码器 $h_\phi$ 预测速度。

$$\text{Step 4 (多模态融合与最终目标)}:$$
$$L_{\text{final}}^{\text{est}} = L_{\text{vis}} + \lambda_{\text{audio}} \cdot \mathbb{1}_{[A \text{ exists}]} \cdot L_{\text{audio}} + \lambda_{\text{sup}} \cdot \mathbb{1}_{[\text{labeled}]} \cdot |g_\theta(f_\theta(\tilde{V})) - s_{\text{gt}}|$$

**关键设计**: 第三项为监督损失（仅在有标注数据时激活），使模型在人工慢放/快放视频（如 YouTube 的 0.5x/2x 播放）上获得精确速度值；音频损失提供**跨模态正则化**，即使视觉模糊也能可靠估计。

**对应消融**: Table 1（待补充具体标签）显示移除音频损失 $\Delta$ 误差上升，纯视觉在极端速度（<0.5x, >2x）估计退化明显。

---

### 模块 2: Speed-Conditioned Diffusion Generator（速度条件化扩散生成器）（对应框架图右侧）

**直觉**: 将连续速度标量作为扩散模型的显式条件，类似文本/图像条件，通过自适应归一化注入去噪网络。

**Baseline 公式** (标准视频扩散模型, 如 Wan2.1):
$$L_{\text{base}} = \mathbb{E}_{V, \epsilon \sim \mathcal{N}(0,I), t} \left[ \| \epsilon - \epsilon_\theta(V_t, t, c_{\text{text}}, c_{\text{image}}) \|^2 \right]$$
符号: $V_t$ = 第 $t$ 步加噪视频, $\epsilon$ = 真实噪声, $\epsilon_\theta$ = 去噪网络, $c_{\text{text}}, c_{\text{image}}$ = 文本/图像条件

**变化点**: 标准模型缺乏速度条件 $c_{\text{speed}}$，导致生成视频帧率固定、运动速度不可控；本文**将速度编码为连续嵌入并与时间步嵌入融合**。

**本文公式（推导）**:

$$\text{Step 1 (速度嵌入)}: \quad c_{\text{speed}} = \text{MLP}(\text{Fourier}(\log s)) \in \mathbb{R}^{d_c}$$
对数编码使模型对速度比例对称（0.5x 与 2.0x 距 1.0x 等距），Fourier 特征提升高频分辨率。

$$\text{Step 2 (条件注入)}: \quad \gamma_s, \beta_s = \text{AdaGN}(c_{\text{speed}} + t_{\text{emb}}) \quad \text{速度与时间步嵌入相加后生成自适应归一化参数}$$

$$\text{Step 3 (修改后的去噪目标)}:$$
$$L_{\text{final}}^{\text{gen}} = \mathbb{E}_{V, s, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(V_t, t, c_{\text{text}}, c_{\text{image}}, c_{\text{speed}}) \|^2 \right]$$

**关键设计**: 训练时 $s$ 从分布中采样——包括 $s=1.0$（标准速度）、$s<1.0$（慢动作）、$s>1.0$（快动作），以及**关键创新** $s=\text{interpolate}$（时间超分辨率：从低帧率输入推断高帧率输出，等价于 $s > 1$ 的帧间插值）。

**对应消融**: Figure 7 显示「仅用标准 FPS 视频+人工慢放训练」导致快动作生成质量下降（运动撕裂），而加入自然变速视频后泛化改善。

## 实验与分析

| Method | 速度变化检测 (F1) | 速度估计 (MAE↓) | 速度条件化生成 (FVD↓) | 时间超分辨率 (LPIPS↓) |
|:---|:---|:---|:---|:---|
| Wan2.1 | N/A | N/A | baseline | baseline |
| CogVideoX | N/A | N/A |  |  |
| Open-Sora | N/A | N/A |  |  |
| **Ours** |  |  | **优于 Wan2.1** | **优于 Wan2.1** |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3291b628-a9c6-4d6c-b596-093f606fa41a/figures/Figure_5.png)
*Figure 5: Figure 5 Temporal super-resolution qualitative results. We compare against the strongest baseline, Wan2.1, underthe Blurred-input setting. Given blurry input frames, the baseline produces blurry inter*



**核心结果分析**: Figure 5 的定性对比显示，在 Blurred-input 时间超分辨率设置下（输入为运动模糊的低帧率视频），Wan2.1 输出存在明显的运动拖影与速度不一致（如水流断续、人物动作跳跃），而本文方法生成的时间插值视频运动连贯、速度均匀。这一优势直接来源于**显式速度条件**对去噪过程的约束，而非 baseline 的隐式帧率推断。

**消融实验**（对应 Figure 7）: 
- **训练数据构成至关重要**: 仅使用标准 FPS 视频+人工慢放（slowdowns）训练，模型在快动作（$s>1$）生成时出现严重 artifact（运动撕裂、物体消失）；加入自然变速视频后，快动作质量显著改善。这表明**速度分布的覆盖范围**比数据量更重要。
- **音频监督的作用**: 在速度估计任务中，纯视觉分支在 0.25x-4.0x 全范围 MAE 为，加入音频一致性后降至，极端速度区间提升最明显。


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3291b628-a9c6-4d6c-b596-093f606fa41a/figures/Figure_7.png)
*Figure 7: Figure 7 Speed-controlled video generation with dif-ferent training data. Training on standard-FPS videoswith artificial slowdowns leads to stuttering artifacts, whiletraining on SloMo-44K results in*



**公平性检查**: 
- **Baseline 强度**: Wan2.1 为当前开源最强视频生成模型之一（14B 参数），本文生成器基于同类架构，参数量可比，对比公平。
- **计算成本**: Speed Estimator 为轻量 CNN+Transformer（具体 FLOPs 待补充），Generator 需完整扩散推理，额外开销主要来自速度嵌入的 MLP 前向（可忽略）。
- **局限**: (1) 极端速度（如 8x 快放或 0.125x 慢放）生成质量下降，因训练分布覆盖不足；Figure 7 已揭示此问题。(2) 音频条件要求视频含同步音轨，纯视觉场景（如静音监控）依赖视觉分支，精度降低。(3) 速度标注数据规模有限，自监督预训练对复杂场景（多物体不同速度）泛化待验证。

## 方法谱系与知识库定位

**方法家族**: 视频扩散模型 + 物理条件化控制（Physical-conditioned video generation）

**父方法**: Wan2.1（视频扩散基础架构）+ AdaGN 条件注入机制（源自 StyleGAN/Stable Diffusion）

**改动插槽**:
| 插槽 | 父方法 | 本文改动 |
|:---|:---|:---|
| architecture | 时空 U-Net | 增加 speed embedding 分支，兼容现有预训练权重 |
| objective | 纯噪声预测 | 噪声预测 + 显式速度回归辅助任务 |
| training_recipe | 标准 FPS 视频 | 混合标准/慢放/快放视频，自监督速度变换 |
| data_curation | 文本-视频对 | 增加音频-速度对应、人工速度标注子集 |
| inference | 文本/图像条件 | 增加连续速度标量输入，支持实时速度调节 |

**直接对比**:
- **Wan2.1**: 同架构基线，本文增加速度条件注入层与速度感知训练数据；Wan2.1 无速度控制接口。
- **CogVideoX**: 采用 3D 全注意力，本文方法可迁移至其架构，但当前实现基于 Wan2.1 的 2D+1D 分解设计。
- **VideoCraft/SEINE**: 专注插帧与时间超分，本文统一速度检测、估计、生成三任务，且无需光流先验。

**后续方向**:
1. **可变速度生成**: 当前速度标量为全局常数，扩展为时变速度曲线 $s(t)$ 可实现「先快后慢」的变速摄影效果。
2. **多物体速度解耦**: 场景内不同物体以不同速度运动（如前景慢、背景快），需实例级速度估计。
3. **跨模态速度迁移**: 将音频节奏（如音乐节拍）自动映射为视频速度变化，实现「视听同步」生成。

**知识库标签**:
- modality: video + audio (multimodal)
- paradigm: diffusion model + self-supervised pretraining
- scenario: video generation, temporal super-resolution, video manipulation
- mechanism: explicit physical conditioning (speed as continuous variable)
- constraint: limited speed-annotated data, relies on audio for extreme speeds

