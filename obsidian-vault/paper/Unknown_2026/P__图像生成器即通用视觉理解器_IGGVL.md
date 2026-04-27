---
title: Image Generators are Generalist Vision Learners
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.20329
aliases:
- 图像生成器即通用视觉理解器
- IGGVL
- 核心直觉是：图像生成预训练与LLM文本预训练在本质上同构——大规模生成
modalities:
- Image
---

# Image Generators are Generalist Vision Learners

[Paper](https://arxiv.org/abs/2604.20329)

**Topics**: [[T__Agent]], [[T__Image_Generation]], [[T__Semantic_Segmentation]], [[T__Depth_Estimation]]

> [!tip] 核心洞察
> 核心直觉是：图像生成预训练与LLM文本预训练在本质上同构——大规模生成训练迫使模型内化视觉世界的完整数据分布，从而隐式习得强大的视觉表征。将视觉任务输出统一编码为RGB图像，使得感知问题在形式上等价于条件图像生成问题，无需架构改动即可复用生成模型的全部能力。生成模型天然建模完整输出分布，因此能优雅处理视觉任务中的一对多歧义，而无需判别式模型所需的定制损失设计。有效性的关键在于基础生成模型的表征质量足够强，使得少量任务数据的指令微调即可激活潜在的理解能力。

| 中文题名 | 图像生成器即通用视觉理解器 |
| 英文题名 | Image Generators are Generalist Vision Learners |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20329) · [Code](https://github.com/...) · [Project](待补充) |
| 主要任务 | 语义分割、实例分割、指代分割、表面法线估计、度量深度估计 |
| 主要 baseline | SAM3, Lotus-2, Marigold, StableNormal, DSINE, DINO-X |

> [!abstract] 因为「生成式视觉预训练的隐式理解能力从未被系统性地转化为可量化的视觉感知性能」，作者在「Nano Banana Pro 图像生成模型」基础上改了「混合数据指令微调 + RGB统一输出空间编码」，在「NYUv2法线估计、RefCOCOg指代分割、Cityscapes语义分割」上取得「超越专用SOTA模型的性能」。

- **NYUv2 表面法线估计**: mean error 15.549，低于 Lotus-2 的 16.558（越低越好）
- **RefCOCOg 指代分割**: cIoU 0.738，超越 SAM3 Agent 的 0.734
- **Cityscapes 语义分割**: mIoU 0.699，超越 SAM3 的 0.652

## 背景与动机

视觉理解领域存在一个根本性悖论：扩散模型等图像生成器在合成高保真图像时，显然已内化了物体结构、空间关系和场景几何等深层视觉知识，但这些"隐式理解"从未被成功提取为可度量的感知能力。例如，Stable Diffusion 能生成符合物理规律的光影效果，却无法直接输出一张可用于定量评估的深度图。

现有工作沿两条路径尝试破解这一矛盾。**零样本观察类工作**（Wiedemer et al., 2025; Zuo et al., 2025）发现生成模型在特定提示下会自发产生类似分割或深度图的输出，但输出格式不受约束——颜色映射任意变化，无法反解码为精确数值标注。**任务适配类工作**（Lotus、Marigold、StableNormal）则走向另一极端：为每个任务添加专用解码头或进行全量微调，虽在单一任务达到SOTA，却牺牲了模型的通用性，每个任务需要独立部署的专用模型。此外，判别式范式在处理视觉任务固有的**一对多歧义**时（如一个物体存在多个合理分割边界），需要定制复杂架构——典型如 SAM 系列的多掩码输出加单掩码选择机制，显著增加了系统复杂度。

本文的核心动机由此明确：能否通过**最小侵入式的训练配方**，在不修改模型架构、不牺牲基础生成能力的前提下，将单一图像生成模型转化为跨任务的通用视觉理解器？这一思路直接借鉴 NLP 领域的成功经验——GPT 等语言模型通过预训练获得通用语言能力，再通过指令微调激活特定任务性能。作者假设：图像生成预训练与文本预训练在本质同构，大规模生成训练迫使模型内化完整视觉数据分布，少量任务数据的指令微调即可激活潜在理解能力。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/584f4788-2fb1-4341-9f26-a0882187ae74/figures/Figure_1.png)
*Figure 1: Figure 1 | We demonstrate the hidden visual understanding capabilities of image generators byinstruction-tuning Nano Banana Pro. The instruction-tuned model, Vision Banana, can producevisualizations i*



## 核心创新

**核心洞察**：图像生成预训练与LLM预训练在本质同构，因为大规模生成训练迫使模型内化视觉世界的完整数据分布，从而将感知任务统一重构为条件图像生成问题，使单一模型无需架构改动即可处理多任务成为可能。

| 维度 | Baseline (Lotus/Marigold/StableNormal) | 本文 (Vision Banana) |
|:---|:---|:---|
| 架构修改 | 添加任务专用解码头或全量微调 | **零架构改动**，纯训练配方 |
| 任务泛化 | 每任务独立专用模型 | **单一模型**覆盖分割/深度/法线 |
| 输出格式 | 任务特定张量（深度图/法线图等） | **统一RGB图像编码**，指令控制可视化方案 |
| 歧义处理 | 需定制损失（如SAM多掩码机制） | 生成模型天然建模完整分布，**隐式处理一对多** |
| 基础能力保留 | 牺牲生成能力 | 混合原始数据防止灾难性遗忘（声明待验证） |

与先前工作的本质差异：Vision Banana 不是"为理解任务设计更好的模型"，而是"用更好的训练配方激活生成模型已有的理解能力"。

## 整体框架



Vision Banana 的整体流程可概括为"**一个骨干、一种编码、一套指令**"的三位一体设计：

**输入阶段**：接收自然语言指令（如"segment the cat in red, the dog in blue"）与条件图像（待分析的输入图）。指令同时承担两项功能：指定任务类型（分割/深度/法线）与定义输出可视化方案（颜色映射规则）。

**骨干网络**：Nano Banana Pro（NBP），一个领先的图像生成模型，直接继承其在大规模图像生成训练中积累的视觉表征。**不做任何架构修改**，不添加输出头，不改变层数或通道维度。

**输出编码**：所有视觉任务输出被强制编码为RGB图像。具体而言——语义分割用类别→RGB的固定映射；实例分割用实例ID→RGB的哈希映射；深度图用标量距离→RGB的**双射曲线编码**（Figure 5 展示该双射的可视化）；法线图用方向向量→RGB的标准球面映射。这一设计将感知问题形式上等价于条件图像生成问题。

**解码阶段**：按照预定义的可视化方案的逆映射，将生成的RGB图像反解码回任务标注格式（如深度数值、类别标签），以计算定量评估指标。

**训练数据**：NBP原始生成训练数据与少量视觉任务标注数据的混合，混合比例防止灾难性遗忘。

```
[输入图像 + 自然语言指令] → Nano Banana Pro (生成模型) → [RGB输出图像] → [任务特定解码器] → [定量标注结果]
         ↑___________________________________________________________↓
                              预定义可视化方案的双向映射
```

## 核心模块与公式推导

### 模块 1: 混合数据指令微调（对应框架图：训练阶段）

**直觉**：生成模型的预训练分布与下游任务分布存在域间隙，纯任务数据微调会导致灾难性遗忘，需用原始生成数据"锚定"基础能力。

**Baseline 公式**（标准任务微调）：$$\mathcal{L}_{\text{task}} = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{task}}} \left[ \| f_\theta(x) - y \|^2 \right]$$
符号：$x$ = 输入（图像+指令），$y$ = 任务标注，$f_\theta$ = 生成模型，$\mathcal{D}_{\text{task}}$ = 任务数据集。

**变化点**：标准微调仅优化任务数据，导致生成模型遗忘预训练分布；本文假设生成能力本身蕴含理解能力，遗忘将损害泛化。

**本文公式（推导）**：
$$\text{Step 1}: \mathcal{L}_{\text{mixed}} = \lambda \cdot \mathcal{L}_{\text{gen}} + (1-\lambda) \cdot \mathcal{L}_{\text{task}} \quad \text{加入原始生成损失以保留分布先验}$$
$$\text{Step 2}: \mathcal{L}_{\text{gen}} = \mathbb{E}_{z \sim \mathcal{D}_{\text{gen}}} \left[ \| f_\theta(z_{\text{noisy}, t}) - z_0 \|^2 \right] \quad \text{标准扩散去噪损失，在原始数据上计算}$$
$$\text{最终}: \mathcal{L}_{\text{final}} = \lambda \mathbb{E}_{\mathcal{D}_{\text{gen}}}\left[\|f_\theta(z_t)-z_0\|^2\right] + (1-\lambda)\mathbb{E}_{\mathcal{D}_{\text{task}}}\left[\|f_\theta(x)-y_{\text{RGB}}\|^2\right]$$

其中 $y_{\text{RGB}} = \text{encode}(y; \text{prompt})$ 为指令依赖的可视化编码。混合权重 $\lambda$ 控制生成能力与任务性能的权衡。（消融

---

### 模块 2: 标量-RGB双射编码（对应框架图：输出编码/解码模块）

**直觉**：深度估计等任务需要输出连续标量值，但生成模型原生输出RGB图像；需构造可逆映射保证信息无损，同时使RGB空间中的"距离"对应标量空间的"距离"以利于生成模型学习。

**Baseline 公式**（朴素线性映射）：$$c_{\text{linear}}(d) = \frac{d - d_{\min}}{d_{\max} - d_{\min}} \cdot [1,1,1] \in [0,1]^3$$
仅使用灰度，三个通道冗余且未利用RGB全空间；生成模型对彩色纹理的学习先验未被激活。

**变化点**：线性映射将一维信息压缩到一维子空间，且视觉单调乏味；本文通过**空间填充曲线**将标量距离 $d \geq 0$ 映射到三维RGB立方体 $[0,1]^3$，实现双射且保持局部连续性。

**本文公式（推导）**：
$$\text{Step 1}: \gamma: [0, +\infty) \to [0,1]^3, \quad d \mapsto \gamma(d) \quad \text{构造Peano-Hilbert型空间填充曲线}$$
$$\text{Step 2}: \text{约束}: \|\gamma(d_1) - \gamma(d_2)\|_2 \approx g(|d_1 - d_2|) \quad \text{RGB距离近似标量距离的单调函数}$$
$$\text{Step 3}: \text{逆映射}: \hat{d} = \gamma^{-1}(c_{\text{RGB}}) \quad \text{解码时精确恢复标量值}$$
$$\text{最终}: c_{\text{RGB}} = \gamma(d), \quad \hat{d} = \gamma^{-1}(f_\theta(x, \text{"depth metric"}))$$

Figure 5 可视化该双射：标量距离沿曲线在RGB立方体中蜿蜒，相邻深度值对应相邻颜色，远距离对应显著色差。此设计使生成模型可利用其强大的颜色纹理生成先验来"绘制"深度图。（消融

---

### 模块 3: 指令驱动的动态可视化（对应框架图：输入编码模块）

**直觉**：同一任务需支持多种提示风格（如"segment the cat" vs "red mask for feline"），模型需从自然语言解析可视化方案而非硬编码。

**Baseline 公式**（固定模板）：$$y_{\text{RGB}} = T_{\text{fixed}}(y), \quad \forall \text{ prompts}$$
模板 $T_{\text{fixed}}$ 与指令无关，无法适应多样化人机交互。

**变化点**：将可视化方案本身作为指令的一部分，模型学习**指令→可视化方案→RGB输出**的联合映射，实现零样本适应新颜色约定。

**本文公式**：
$$\text{Step 1}: (x_{\text{img}}, x_{\text{text}}) \to \text{CLIP/LLM编码} \to h_{\text{cond}}$$
$$\text{Step 2}: f_\theta(x_{\text{img}}, h_{\text{cond}}) \to c_{\text{RGB}}, \quad \text{其中 } c_{\text{RGB}}^{(i,j)} = \text{color}(\text{class}(i,j); x_{\text{text}})$$
$$\text{最终}: \hat{y} = \text{decode}(f_\theta(x); x_{\text{text}}) \quad \text{解码依赖指令中声明的颜色映射}$$

Figure 2、Figure 3 展示该能力：语义分割中"blue for sky, green for grass"与"azure for heavens, emerald for lawn"等不同表述均可正确解析。（定量消融待补充）

## 实验与分析

主实验结果汇总如下（零样本迁移设定，模型未见任何评估基准训练集）：

| Method | NYUv2法线 mean↓ | DIODE-indoor法线 mean↓ | RefCOCOg cIoU↑ | Cityscapes mIoU↑ | SA-1B pmF1↑ |
|:---|:---|:---|:---|:---|:---|
| DSINE | — | **16.4** | — | — | — |
| Lotus-2 | 16.558 | — | — | — | — |
| SAM3 / SAM3 Agent | — | — | 0.734 | 0.652 | — |
| DINO-X | — | — | — | — | **0.552** |
| **Vision Banana (本文)** | **15.549** | 17.778 | **0.738** | **0.699** | 0.540 |


![Figure 8](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/584f4788-2fb1-4341-9f26-a0882187ae74/figures/Figure_8.png)
*Figure 8: Table 4 | Surface normal estimation results. Vision Banana achieves the lowest mean and medianangle errors on the indoor datasets on average, and is on par with previous SOTA on outdoor scenes.*



**核心发现分析**：
- **法线估计**（Figure 6, Figure 7）：NYUv2室内场景mean error 15.549，相对Lotus-2降低6.1%，支持"生成预训练隐含强几何先验"的核心主张；但DIODE-indoor（17.778）不及DSINE（16.4），室外VKitti场景进一步落后，说明生成范式对**复杂光照和室外分布偏移**敏感。
- **指代分割**：RefCOCOg cIoU 0.738 vs SAM3 Agent 0.734，优势微弱（+0.4%）；ReasonSeg gIoU 0.793 vs 0.770（+3.0%）稍显著，表明自然语言推理能力受益于生成模型的开放词汇理解。
- **语义分割**（Figure 2）：Cityscapes mIoU 0.699显著超越SAM3（+7.2%），但基线选择存疑——未纳入Mask2Former等判别式SOTA，对比不公平。
- **实例分割**（Figure 3）：SA-1B/Gold上pmF1 0.540低于DINO-X的0.552（-2.2%），**反例明确**：生成范式在需要精确边界定位的任务上尚未全面超越专用模型。



**关键缺失与公平性**：
- "不牺牲生成能力"声明**完全缺乏定量支撑**，论文未报告FID/CLIP Score等生成质量指标的前后对比。
- 部分基线结果来自第三方HuggingFace demo而非统一复现，引入不可控变量。
- **计算成本**：作者明确承认NBP的推理成本显著高于轻量级专用模型，实际部署障碍显著。
- 失败案例：室外深度估计、精确边界实例分割为明显短板。

## 方法谱系与知识库定位

**方法家族**：生成式视觉预训练 → 任务适配微调

**父方法**：Nano Banana Pro（NBP）图像生成模型。本文未修改其架构，仅施加训练配方。

**改动槽位**：
- **训练_recipe**：混合数据指令微调（核心创新）
- **data_curation**：原始生成数据 + 任务标注数据混合
- **inference**：增加RGB↔任务标注的双向编解码
- 架构 / 目标函数：未改动

**直接基线与差异**：
- **Lotus / Marigold / StableNormal**：任务专用全量微调，牺牲通用性；本文零架构改动、多任务统一
- **Wiedemer et al. / Zuo et al. (零样本观察)**：无训练，输出格式不受控；本文指令微调确保可解码性
- **SAM 系列**：判别式架构，需定制多掩码机制处理歧义；本文利用生成分布隐式处理

**后续方向**：
1. 验证"不牺牲生成能力"声明：需补充生成质量定量评估与混合权重 $\lambda$ 的敏感性分析
2. 降低计算成本：探索模型蒸馏或更高效生成骨干的迁移
3. 扩展至视频/3D理解：检验RGB统一编码在时序一致性和三维结构上的可扩展性

**知识库标签**：
- **modality**: 图像（2D视觉）
- **paradigm**: 生成式预训练 + 指令微调
- **scenario**: 通用视觉理解（分割、深度、法线多任务）
- **mechanism**: RGB统一输出空间编码、混合数据防遗忘
- **constraint**: 零样本迁移、零架构改动、高计算成本

