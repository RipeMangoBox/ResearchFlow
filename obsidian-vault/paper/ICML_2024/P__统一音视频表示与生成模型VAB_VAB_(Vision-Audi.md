---
title: 'From Vision to Audio and Beyond: A Unified Model for Audio-Visual Representation and Generation'
type: paper
paper_level: C
venue: ICML
year: 2024
paper_link: null
aliases:
- 统一音视频表示与生成模型VAB
- VAB (Vision-Audi
- VAB (Vision-Audio-Beyond)
acceptance: Poster
cited_by: 22
method: VAB (Vision-Audio-Beyond)
---

# From Vision to Audio and Beyond: A Unified Model for Audio-Visual Representation and Generation

**Topics**: [[T__Video_Generation]], [[T__Retrieval]], [[T__Representation_Learning]] | **Method**: [[M__VAB]] | **Datasets**: VGGSound V+A, VGGSound Audio, VGGSound V→A

| 中文题名 | 统一音视频表示与生成模型VAB |
| 英文题名 | From Vision to Audio and Beyond: A Unified Model for Audio-Visual Representation and Generation |
| 会议/期刊 | ICML 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2409.19132) · [Code](https://github.com/） ⭐
| 主要任务 | 视频到音频生成、音视频分类、跨模态检索、线性探测 |
| 主要 baseline | CAV-MAE、MAE、Vampnet |

> [!abstract] 因为「现有音视频模型多为任务专用架构，无法同时支持判别式表示学习与生成式音频合成」，作者在「CAV-MAE 掩码自编码框架」基础上改了「引入模态特定专家层（前12层）+ 对比微调两阶段训练 + 双音频分词器支持」，在「VGGSound 分类/生成/检索」上取得「V+A 分类 65.2、FAD 4.18、V→A 检索 R@1 28.2」

- **VGGSound V+A 分类**: VAB-Encodec + 对比微调达到 65.2，无对比微调仅 56.2，相对提升 16.0%
- **视频到音频生成**: eva-CLIP 视觉编码器实现 FAD 4.18，相比 MAE 编码器的 4.93 提升 15.2%
- **跨模态检索**: V→A R@1 达 28.2，相比 MAE 编码器的 18.4 提升 53.3%

## 背景与动机

当前音视频理解领域存在一个根本性割裂：判别式模型（如 CAV-MAE）擅长提取紧凑的跨模态表示用于分类和检索，生成式模型（如 Vampnet）则专注于从视觉条件合成高质量音频，但两者架构分离、目标冲突，导致研究者必须为每个任务训练独立模型。例如，一个视频平台若需同时实现「视频内容分类」和「自动生成匹配音效」，就要维护两套参数和训练流程。

现有方法如何处理这一问题？**CAV-MAE** 采用共享 Transformer 层处理视觉和音频模态，通过掩码自编码和对比学习联合训练，但其统一架构牺牲了模态特异性——视觉和音频在早期层就强制交互，导致低级特征混淆。**MAE** 作为纯视觉预训练方法，虽能提取强视觉特征，但缺乏音频建模能力，需外挂音频编码器。**Vampnet** 专注于音频生成，采用 DAC 分词器的 Coarse2Fine 迭代解码，但完全依赖音频上下文，无法利用视觉信息引导生成。

这些方法的共同短板在于：**架构层面缺乏真正的模态特异性与跨模态融合的分层设计，训练层面缺乏表示学习与生成质量的平衡机制**。CAV-MAE 的共享层导致视觉和音频在特征提取阶段互相干扰；固定高掩码比（如 MAE 的 0.75）虽利于表示学习，却严重损害生成任务所需的细粒度重建能力；此外，现有工作多绑定单一音频分词器（Encodec 或 DAC），限制了框架的灵活性与扩展性。

本文提出 VAB（Vision-Audio-Beyond），通过**模态特定专家层**实现早期模态专属处理、**自适应掩码采样**平衡表示与生成、**对比微调**增强判别性能，首次在单一框架内统一支持音视频分类、检索与视频到音频生成三大任务。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ed7001c9-3337-48d9-86a2-0d3899219313/figures/Figure_1.png)
*Figure 1: Figure 1. VAB is a unified audio-visual model capable of support-ing various audio-visual tasks within a single framework.*



## 核心创新

核心洞察：**模态特定专家层（Modal-specific Experts）** 能够在不增加总参数量冗余的前提下，为视觉和音频分别保留独立的早期表征空间，因为不同模态的低级特征统计特性差异显著（视觉为局部时空相关性、音频为频域谐波结构），从而使统一的 24 层 Transformer 既能学到模态内强先验、又能在高层实现高效跨模态对齐与生成成为可能。

| 维度 | Baseline (CAV-MAE) | 本文 (VAB) |
|:---|:---|:---|
| 架构设计 | 共享 Transformer 层，视觉/音频无区分 | 前 12 层为模态特定专家（253M），后 12 层跨模态融合 |
| 掩码策略 | 固定掩码比 0.75 | 自适应采样 $m \sim \mathcal{N}(0.55, 0.25)$ |
| 训练阶段 | 单阶段掩码自编码预训练 | 两阶段：掩码自编码 + 对比微调 |
| 音频分词器 | 单一 Encodec | 双支持：Encodec（单阶段）/ DAC（Coarse2Fine） |
| 推理策略 | 标准解码 | Classifier-free guidance（scale=5）+ 温度调参 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ed7001c9-3337-48d9-86a2-0d3899219313/figures/Figure_2.png)
*Figure 2: Figure 2. Pre-processing (left) and masked audio token predictionpre-training (right) of VAB framework.*



VAB 的数据流遵循「编码 → 模态专属处理 → 跨模态融合 → 任务输出」的四阶段范式：

1. **视频编码器（eva-CLIP）**：输入为 1fps 采样的视频帧，输出 1024 维 CLIP 视觉嵌入。采用 BLIP 同款 eva-CLIP 编码器，替代 CAV-MAE 的 MAE 编码器，提供更强的跨模态对齐先验。

2. **音频分词器（Encodec / DAC）**：输入 16kHz 原始音频，输出离散音频 token。Encodec 为单阶段 4 层码本；DAC 为 12 层码本，需遵循 Vampnet 的 Coarse2Fine 两阶段训练。

3. **模态特定专家层（12 层，253M 参数）**：接收视觉嵌入和音频 token 分别进行模态内自注意力计算，输出模态专属表示。这是与 CAV-MAE 共享层的本质区别。

4. **跨模态融合层（12 层）**：将专家层输出的视觉和音频表示拼接后进行联合自注意力，输出统一音视频表示。

5. **任务头**：
   - **分类头**：取 [CLS] 标记做音视频联合分类
   - **检索头**：用融合层输出计算跨模态相似度
   - **生成头**：以视觉条件 + 掩码音频 token 为输入，迭代预测被掩码的音频 token

```
视频帧 (1fps) ──→ eva-CLIP ──→ 视觉嵌入 ──┐
                                           ├──→ [模态特定专家 12 层] ──→ [跨模态融合 12 层] ──→ 任务头
原始音频 (16kHz) ──→ Encodec/DAC ──→ 音频 token ──┘
```

## 核心模块与公式推导

### 模块 1: 掩码自编码损失（对应框架图预训练阶段）

**直觉**: 联合重建被掩码的视觉和音频内容，迫使模型学习模态内和模态间的有意义表征。

**Baseline 公式 (CAV-MAE)**: 
$$\mathcal{L}_{MAE}^{CAV} = \mathbb{E}_{(v,a) \sim \mathcal{D}} \left[\mathcal{L}_{recon}(a, \hat{a}) + \mathcal{L}_{recon}(v, \hat{v})\right]$$

符号: $(v,a)$ = 视频-音频样本对, $\hat{v}, \hat{a}$ = 重建输出, $\mathcal{L}_{recon}$ = 均方误差或交叉熵损失

**变化点**: CAV-MAE 采用固定掩码比且共享所有层；VAB 引入模态特定专家层，使重建前的特征提取更具模态针对性。

**本文公式（推导）**:
$$\text{Step 1}: h_v^{(1:12)} = \text{Expert}_{\text{visual}}(v_{\text{masked}}), \quad h_a^{(1:12)} = \text{Expert}_{\text{audio}}(a_{\text{masked}})$$
$$\text{Step 2}: h^{(13:24)} = \text{Fusion}([h_v^{(12)}; h_a^{(12)}])$$
$$\text{最终}: \mathcal{L}_{MAE}^{VAB} = \mathbb{E}_{(v,a), m \sim \mathcal{N}(0.55,0.25)} \left[\mathcal{L}_{recon}(a, \hat{a}(h^{(24)})) + \mathcal{L}_{recon}(v, \hat{v}(h^{(24)}))\right]$$

**对应消融**: Table 12 显示掩码比 0.55（均值）在分类（56.2）和生成（FAD 5.30）间取得最佳平衡；0.75 时分类略升但生成显著恶化。

---

### 模块 2: 自适应掩码采样（对应框架图预训练右半部分）

**直觉**: 固定高掩码比利于表示学习但破坏生成所需细节，固定低掩码比则相反；通过随机采样实现任务间的隐式数据增强。

**Baseline 公式 (MAE-style)**:
$$m = 0.75 \quad \text{(fixed)}$$

**变化点**: MAE 的 0.75 固定掩码比针对视觉表示优化，直接迁移至音频-视觉联合生成会导致音频重建质量严重不足；VAB 通过正态分布采样引入动态难度调节。

**本文公式（推导）**:
$$\text{Step 1}: m \sim \mathcal{N}(\mu=0.55, \sigma=0.25), \quad m \in [0.35, 0.75] \text{ (clipped)}$$
$$\text{Step 2}: \text{mask}_v = \text{RandomSample}(\text{positions}(v), m_v), \quad \text{mask}_a = \text{RandomSample}(\text{positions}(a), m_a)$$
$$\text{最终}: \mathcal{L}_{MAE}^{VAB} = \mathbb{E}_{m} \left[\mathcal{L}_{recon}\right] \quad \text{其中期望覆盖不同掩码难度}$$

**对应消融**: Table 12 显示 $m=0.35$ 时分类与生成均次优；$m=0.75$ 时 VGGSound V+A 分类达 58.1 但 FAD 恶化至 6.55，而 $m=0.55$ 取得 56.2 / 5.30 的最佳权衡。

---

### 模块 3: 对比微调损失（对应框架图下游任务优化）

**直觉**: 掩码自编码优化的是重建似然，对判别式任务（分类、检索）的表征区分度不足；通过显式对比学习拉近匹配对、推开非匹配对。

**Baseline 公式**: 无（CAV-MAE 将对比损失与重建损失联合训练，非分阶段）

**变化点**: VAB 采用**分阶段策略**——先纯掩码自编码预训练，再冻结部分层进行对比微调，避免生成与判别目标的直接冲突。

**本文公式（推导）**:
$$\text{Step 1}: z_v = \text{Pool}(h_v^{(24)}), \quad z_a = \text{Pool}(h_a^{(24)}) \quad \text{提取全局表征}$$
$$\text{Step 2}: sim(z_v, z_a) = \frac{z_v \cdot z_a}{\|z_v\| \|z_a\|} \quad \text{余弦相似度}$$
$$\text{最终}: \mathcal{L}_{contrastive} = -\log \frac{\exp(sim(z_v, z_a)/\tau)}{\sum_{a' \in \mathcal{N}} \exp(sim(z_v, z_{a'})/\tau)}$$

符号: $z_v, z_a$ = 视觉/音频全局表征, $\tau$ = 温度参数, $\mathcal{N}$ = 负样本集合（批次内非匹配对）

**对应消融**: Table 11 显示 VAB-Encodec 无对比微调时 VGGSound V+A 仅 56.2，加入对比微调后跃升至 65.2（+9.0，+16.0%）；VAB-DAC 从 60.4 提升至 63.9（+3.5）。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ed7001c9-3337-48d9-86a2-0d3899219313/figures/Table_2.png)
*Table 2: Table 2. Mean Opinion Score (MOS) for video-to-audio generation.Values in bold indicate the best value.*



本文在 VGGSound 测试集上评估视频到音频生成，同时覆盖 AudioSet、MSR-VTT 的跨模态检索，以及 ESC-50、SPC-1 的音频分类。核心结果如下：VAB-Encodec 在 VGGSound 音视频联合分类上达到 65.2（+对比微调），相比无对比微调的 56.2 提升显著；在生成任务上，eva-CLIP 视觉编码器实现 FAD 4.18，优于 MAE 编码器的 4.93。V→A 检索 R@1 达 28.2，较 MAE 编码器的 18.4 提升 53.3%。值得注意的是，VAB-Encodec  consistently 优于 VAB-DAC，尽管 DAC 的理论码本容量更大，暗示 12 层 DAC 的 Coarse2Fine 训练存在稳定性挑战。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ed7001c9-3337-48d9-86a2-0d3899219313/figures/Table_3.png)
*Table 3: Table 3. Cross-modal retrieval results on AudioSet, VGGSound, and MSR-VTT. Values in bold highlight the best performance.*



与先前音频视觉模型的分类对比（Table 4）显示，VAB 在 VGGSound、AS-2M、AS-20K 的音视频（V+A）、纯视频（V）、纯音频（A）设置下均具竞争力，但**并非每项任务的绝对 SOTA**——论文的核心主张是统一能力而非单项最优。跨模态检索（Table 3）在 AudioSet、VGGSound、MSR-VTT 上展示了双向检索性能。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ed7001c9-3337-48d9-86a2-0d3899219313/figures/Table_5.png)
*Table 5: Table 5. Comparison with ESC-50 and SPC-1 audio only classifi-cation accuracy.*



消融实验揭示关键组件的贡献度：
- **对比微调**: 最关键组件，VAB-Encodec V+A 分类 56.2 → 65.2（+9.0），但音频单独任务受益有限甚至轻微下降，说明对比损失对跨模态对齐的特异性。
- **视觉编码器选择**: eva-CLIP 替换为 MAE 导致全面退化——V+A 分类 -6.5（65.2 → 58.7）、FAD 从 4.18 恶化至 4.93、V→A 检索 R@1 从 28.2 跌至 18.4。
- **掩码比例**: 偏离 0.55 均值均造成性能漂移，0.75 均值使 FAD 从 5.30 恶化至 6.55（+1.25），验证自适应采样的必要性。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ed7001c9-3337-48d9-86a2-0d3899219313/figures/Figure_4.png)
*Figure 4: Table 1. Quantitative evaluation for video-to-audio generation onthe VGGSound test set. Values in bold indicate the best value.*



公平性审视：本文 baselines 选择合理（CAV-MAE、MAE 为直接可比方法），但缺少与专用音频生成模型（AudioLM、MusicLM）和扩散模型的对比，也未与 GPT-4o、Gemini 等新兴统一多模态模型比较。模型总参数量 403M（专家层 253M），DAC 推理需 24-36 次迭代解码，计算成本高于 Encodec。作者披露 VAB-DAC 表现不及预期，但未深入分析训练不稳定的根因。

## 方法谱系与知识库定位

VAB 属于**统一多模态基础模型**家族，直接继承自 **CAV-MAE**（Contrastive Audio-Visual Masked Autoencoder）。谱系关系为：CAV-MAE 提供掩码自编码 + 对比学习的核心范式，VAB 在此基础上进行四 slot 改造——**架构**（共享层 → 模态特定专家 + 融合层）、**训练配方**（单阶段固定掩码 → 两阶段自适应掩码 + 对比微调）、**数据管道**（单一 Encodec → 双分词器 Encodec/DAC）、**推理策略**（标准解码 → classifier-free guidance）。

直接 baselines 差异：
- **vs CAV-MAE**: VAB 引入模态特定专家层和分阶段训练，CAV-MAE 全程共享参数
- **vs MAE**: VAB 扩展至多模态联合建模，MAE 纯视觉且仅支持表示学习
- **vs Vampnet**: VAB 整合视觉条件实现视频到音频生成，Vampnet 纯音频生成

后续方向：(1) 解决 DAC 12 层码本的训练不稳定问题，释放其理论容量优势；(2) 扩展至更多模态（文本、深度）实现真正的「Beyond」；(3) 探索专家层与融合层的动态路由机制，进一步压缩计算。

**标签**: 模态[视觉+音频] / 范式[掩码自编码+对比学习] / 场景[统一表示与生成] / 机制[模态特定专家+自适应掩码] / 约束[单模型多任务，非单项 SOTA]

