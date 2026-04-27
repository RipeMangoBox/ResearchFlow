---
title: Explainable Disentangled Representation Learning for Generalizable Authorship Attribution in the Era of Generative AI
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.21300
aliases:
- 可解释解耦表征的作者归因新框架
- EDRLGA
- 现有方法用单一编码器隐式学习风格
modalities:
- Text
paradigm: Reinforcement Learning
---

# Explainable Disentangled Representation Learning for Generalizable Authorship Attribution in the Era of Generative AI

[Paper](https://arxiv.org/abs/2604.21300)

**Topics**: [[T__Classification]], [[T__Few-Shot_Learning]], [[T__Interpretability]]

> [!tip] 核心洞察
> 现有方法用单一编码器隐式学习风格，导致风格与内容在同一嵌入空间中纠缠。EAVAE的核心洞察是：解耦不能靠损失函数「劝说」模型分离，而要靠架构设计「强制」分离——用两个独立编码器分别负责风格和内容，从信息流路径上切断两者的混合可能。可解释判别器则通过对抗信号进一步强化这种分离，同时借助LLM生成的自然语言解释提供可解释性。这一思路将「解耦」从训练目标层面提升到架构设计层面，是其相对于前序工作的核心结构性差异。

| 中文题名 | 可解释解耦表征的作者归因新框架 |
| 英文题名 | Explainable Disentangled Representation Learning for Generalizable Authorship Attribution in the Era of Generative AI |
| 会议/期刊 | 2026 (arXiv预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.21300) · Code · Project |
| 主要任务 | 作者归因（Authorship Attribution）、AI生成文本检测（AI Text Detection） |
| 主要 baseline | LUAR、Contra-X、Man and Nguyen (2024) |

> [!abstract] 因为「内容混淆问题导致模型将作者身份与话题虚假关联」，作者在「单编码器对比学习框架」基础上改了「双编码器VAE架构+可解释判别器」，在「Amazon Reviews / PAN21 / HRS / M4」上取得「SOTA性能（具体数值待补充）」

- 关键性能：Amazon Reviews、PAN21、HRS 数据集上 MRR 与 R@8 达到 SOTA（具体数值待补充）
- 关键性能：M4 数据集少样本 AI 文本检测 pAUC@k 表现优异，假阳率 < 1%（具体数值待补充）
- 关键性能：2740万文档、130万作者的大规模预训练数据 + 132k 文档对微调数据

## 背景与动机

作者归因任务面临的核心困境是「内容混淆问题」（content confounding problem）：模型容易把「谁写的」和「写了什么」混为一谈。想象一个侦探小说爱好者——现有方法可能仅因为某作者常写侦探题材，就将一篇侦探小说错误地归给他，而非真正识别其独特的句式节奏、词汇偏好等风格指纹。这种虚假相关在跨域场景下暴露无遗：当测试文档的话题与训练分布偏移时，模型性能急剧崩塌。

现有方法如何应对？Altakrori et al. (2021) 和 Sawatphol et al. (2022) 采用单一编码器配合对比学习目标，试图隐式地将风格与内容分离，但两者仍纠缠于同一嵌入空间；Man and Nguyen (2024) 延续了这一范式，仅将编码器规模扩大，未触及架构根本。这些方法的共同瓶颈有三：其一，「隐式解耦」本质上是损失函数对模型的「劝说」，无法从信息流路径上阻断风格-内容混合；其二，依赖 SLM（小型语言模型）编码器，表征容量不足以捕捉复杂风格模式；其三，完全缺乏可解释性——在法证调查、学术诚信等高风险场景中，「黑箱」决策不可接受。

更紧迫的是，LLM 生成文本的能力正快速逼近人类风格，传统基于风格的检测方法面临双重挤压：既要少样本有效，又要对抗越来越「像人」的 AI 文本。本文正是回应这一复合挑战——通过架构层面的强制解耦，同时实现泛化性与可解释性。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/15ac2cdd-1a5c-4f83-8bf3-522ccc033415/figures/Figure_1.png)
*Figure 1: Figure 1: An example of content-style entanglement.*



## 核心创新

核心洞察：解耦不能靠损失函数「劝说」模型分离，而要靠架构设计「强制」分离——用两个独立编码器分别负责风格和内容，从信息流路径上切断混合可能，从而使可解释的风格证据提取与跨域泛化成为可能。

| 维度 | Baseline（Man and Nguyen, 2024 等） | 本文（EAVAE） |
|:---|:---|:---|
| 编码器架构 | 单一编码器，风格/内容共享嵌入空间 | 双编码器 Es/Ec，架构强制分离 |
| 解耦机制 | 隐式：仅靠对比损失约束 | 显式：信息流路径隔离 + 对抗判别 |
| 可解释性 | 无 | QwQ-32B 生成自然语言解释，监督判别器 |
| 训练目标 | 纯监督对比学习 | 对比预训练 → VAE 重构 + 对抗判别微调 |
| 负样本策略 | 随机采样 | BM25 硬负挖掘 + K-means 话题结构挖掘 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/15ac2cdd-1a5c-4f83-8bf3-522ccc033415/figures/Figure_2.png)
*Figure 2: Figure 2: The architecture of Explainable AuthorshipVariational Autoencoder (EAVAE). EAVAE employsseparate encoders for style and content, with an explain-able discriminator that distinguishes whether*



EAVAE 采用两阶段训练框架，数据流如下：

**输入**：原始文档 d（如用户评论、文章片段）

**第一阶段：大规模监督对比预训练**
- **LLM 骨干编码器**：接收文档 d，输出上下文表征；引入双向注意力增强上下文建模
- **BM25 硬负样本挖掘模块**：从 2740万文档池中检索词汇相似但风格不同的文档，构建负样本对
- **监督对比损失**：拉近同作者文档、推远不同作者文档，学习强基础表征

**第二阶段：可解释 VAE 微调**
- **风格编码器 Es**：将 d 映射到风格潜空间，输出高斯参数 (μs, σs)，重参数化采样得 zs
- **内容编码器 Ec**：将 d 映射到内容潜空间，输出高斯参数 (μc, σc)，重参数化采样得 zc
- **共享重构器 Grec(zs, zc)**：从两个潜变量重建原始文档，提供重构监督
- **风格判别器 Ds**：判断风格表示对是否来自同一作者，输出二元判断 + 自然语言解释
- **内容判别器 Dc**：判断内容表示对是否来自同一内容源，输出二元判断 + 自然语言解释
- **QwQ-32B 解释生成器**（仅数据构建阶段）：为 132k 文档对预生成解释，作为判别器训练监督

**输出**：风格表征 zs（用于作者归因）、内容表征 zc（用于话题分析）、自然语言解释（用于决策可信性）

```
原始文档 d → [Es] → zs ~ N(μs,σs) ─┐
           → [Ec] → zc ~ N(μc,σc) ─┼→ [Grec] → 重建 d'
                                    │
           文档对 (d₁,d₂) → [Es×Es] → [Ds] → {同作者? + 解释}
                        → [Ec×Ec] → [Dc] → {同内容? + 解释}
```

## 核心模块与公式推导

### 模块 1: 监督对比预训练（对应框架图 第一阶段）

**直觉**：在解耦之前，必须先学到「什么是作者风格」的强基础表征；BM25 硬负样本迫使模型超越表面词汇，捕捉深层风格模式。

**Baseline 公式** (SimCLR/标准监督对比学习):
$$\mathcal{L}_{\text{supcon}} = -\sum_{i \in I} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

符号: $\mathbf{z}_i$ = 文档 $i$ 的表征, $P(i)$ = 同作者正样本集, $A(i)$ = 所有样本集, $\tau$ = 温度系数

**变化点**：标准对比学习随机采样负样本，模型易依赖话题关键词作弊；本文引入 BM25 检索词汇相似但作者不同的文档作为硬负样本，强制模型区分「相似内容下的不同风格」。

**本文公式**：
$$\text{Step 1: 硬负挖掘} \quad \mathcal{N}_{\text{hard}}(i) = \text{TopK}_{\text{BM25}}\left(q=d_i, \text{filter: } a_j \neq a_i, k=K\right)$$
$$\text{Step 2: 扩展负样本集} \quad A'(i) = P(i) \cup \mathcal{N}_{\text{hard}}(i) \cup \mathcal{N}_{\text{rand}}(i)$$
$$\text{最终: } \mathcal{L}_{\text{pretrain}} = -\sum_{i \in I} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{n \in A'(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_n / \tau)}$$

**对应消融**：

---

### 模块 2: 双编码器 VAE 解耦（对应框架图 第二阶段左侧）

**直觉**：既然风格和内容在单一空间必然纠缠，不如从源头用独立网络分别编码，让架构本身成为解耦的保证。

**Baseline 公式** (标准 VAE):
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

符号: $q_\phi$ = 编码器, $p_\theta$ = 解码器, $\mathbf{z}$ = 单一潜变量

**变化点**：标准 VAE 用单一潜变量编码全部信息；本文强制分解 $\mathbf{z} = [\mathbf{z}_s; \mathbf{z}_c]$，由独立编码器分别生成，KL 散度项相应分解，确保各潜空间只负责对应因子。

**本文公式推导**：
$$\text{Step 1: 独立编码} \quad q_{\phi_s}(\mathbf{z}_s|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_s, \boldsymbol{\sigma}_s^2), \quad q_{\phi_c}(\mathbf{z}_c|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\sigma}_c^2)$$
$$\text{Step 2: 重参数化采样} \quad \mathbf{z}_s = \boldsymbol{\mu}_s + \boldsymbol{\sigma}_s \odot \boldsymbol{\epsilon}_s, \quad \mathbf{z}_c = \boldsymbol{\mu}_c + \boldsymbol{\sigma}_c \odot \boldsymbol{\epsilon}_c, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
$$\text{Step 3: 联合重构} \quad \hat{\mathbf{x}} = G_{\text{rec}}([\mathbf{z}_s; \mathbf{z}_c])$$
$$\text{最终: } \mathcal{L}_{\text{VAE}}^{\text{ours}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{重构损失}} - \underbrace{\beta_s D_{\text{KL}}(q_{\phi_s} \| p(\mathbf{z}_s))}_{\text{风格正则}} - \underbrace{\beta_c D_{\text{KL}}(q_{\phi_c} \| p(\mathbf{z}_c))}_{\text{内容正则}}$$

**对应消融**：

---

### 模块 3: 可解释对抗判别器（对应框架图 第二阶段右侧，核心创新）

**直觉**：仅靠重构损失无法保证潜空间的语义纯净性；引入对抗判别器，让「专家」分别检验风格/内容表示的判别可行性，同时用 LLM 生成的解释将判别依据显式化。

**Baseline 公式** (标准对抗解耦，如 β-VAE / FactorVAE 的判别器):
$$\mathcal{L}_{\text{adv}} = \mathbb{E}[\log D(\mathbf{z})] + \mathbb{E}[\log(1-D(\tilde{\mathbf{z}}))]$$

符号: $D$ = 二元判别器, $\mathbf{z}$ = 真实样本潜变量, $\tilde{\mathbf{z}}$ = 打乱/合成样本

**变化点**：标准判别器只输出二元标签，无法解释「为什么」；本文判别器额外输出自然语言解释 $e$，且解释由 QwQ-32B 在数据构建阶段预生成、作为监督信号训练，实现「可解释的对抗」。

**本文公式推导**：
$$\text{Step 1: 判别器输出扩展} \quad D_s(\mathbf{z}_s^{(i)}, \mathbf{z}_s^{(j)}) \rightarrow (y_s \in \{0,1\}, \mathbf{e}_s \in \mathcal{V}^*)$$
其中 $y_s=1$ 表示同作者，$\mathbf{e}_s$ 为解释文本序列

$$\text{Step 2: 解释监督信号} \quad \mathbf{e}_s^* = \text{QwQ-32B}\left(\text{prompt: } "\text{判断 } d_i, d_j \text{ 是否同作者并解释风格证据}"\right)$$
$$\text{Step 3: 联合损失} \quad \mathcal{L}_s^{\text{dis}} = \underbrace{-\left[y_s \log \hat{y}_s + (1-y_s)\log(1-\hat{y}_s)\right]}_{\text{判别交叉熵}} + \lambda \underbrace{\text{CE}(\mathbf{e}_s^*, \hat{\mathbf{e}}_s)}_{\text{解释生成损失}}$$

$$\text{最终: } \mathcal{L}_{\text{dis}} = \mathcal{L}_s^{\text{dis}} + \mathcal{L}_c^{\text{dis}}$$
内容判别器 $\mathcal{L}_c^{\text{dis}}$ 形式对称，判断「同话题/内容源」并生成内容层面解释。

**对应消融**：消融实验确认可解释判别器为关键组件（具体 Δ% 待补充）

## 实验与分析

| Method | Amazon Reviews MRR | PAN21 MRR | HRS R@8 | M4 pAUC@k |
|:---|:---|:---|:---|:---|
| LUAR |  |  |  |  |
| Contra-X |  |  |  |  |
| Man and Nguyen (2024) |  |  |  |  |
| **EAVAE (本文)** | **SOTA** | **SOTA** | **SOTA** | **优异（FPR<1%）** |



**核心发现分析**：论文声称在三个作者归因数据集上达到 SOTA，但提供的文本片段中缺乏具体数值，主要结果表格未完整呈现。M4 数据集上的 AI 文本检测声称「假阳率 < 1%」，这一指标在检测场景中尤为关键——低假阳意味着较少冤枉人类作者，但具体 pAUC@k 数值待补充。

**消融实验**：架构解耦（双编码器 vs. 单编码器）和可解释判别器被确认为关键组件。推断移除双编码器设计会导致风格-内容重新纠缠，跨域性能显著下降；移除判别器则解耦纯度不足。具体 Δ% 数值待补充。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/15ac2cdd-1a5c-4f83-8bf3-522ccc033415/figures/Figure_3.png)
*Figure 3: Figure 3: The architecture of unified generator withhybrid prompting mechanism.*



**基线公平性检查**：论文明确对比了 LUAR、Contra-X、Man and Nguyen (2024)，但未明确是否包含 Kandula et al. (2025) 的集成风格嵌入方法——后者作为潜在强基线存在遗漏风险。计算成本方面，2740万文档预训练 + QwQ-32B 解释生成意味着较高资源消耗；推理时需前向传播两个编码器，延迟约为单编码器基线的 2 倍。

**失败案例与局限**：作者主动披露三项局限——(1) 解释质量绑定 QwQ-32B 能力，可能与人类直觉错位；(2) 框架针对二元归因设计，多作者/协作场景需扩展；(3) LLM 持续模仿人类风格，长期鲁棒性存疑。



## 方法谱系与知识库定位

**方法家族**：解耦表征学习（Disentangled Representation Learning）→ 作者归因应用

**父方法**：β-VAE / FactorVAE（潜变量解耦框架）+ 监督对比学习（Khosla et al., 2020）

**改变的插槽**：
- **架构**：单编码器 → 双编码器（风格 Es / 内容 Ec）
- **目标**：纯对比 → 对比预训练 + VAE 重构 + 对抗判别三阶段
- **训练配方**：BM25 硬负挖掘 + K-means 话题结构挖掘构建 132k 对
- **数据策划**：2740万文档、130万作者大规模预训练 + QwQ-32B 生成 132k 解释
- **推理**：双潜变量 [zs; zc] 联合用于归因，zs 单独用于风格分析

**直接基线对比**：
- **LUAR**：单编码器跨域检索，无显式解耦 → 本文架构强制分离
- **Contra-X**：对比学习隐式分离，无 VAE 概率框架 → 本文引入生成式建模增强表征鲁棒性
- **Man and Nguyen (2024)**：同团队前作，扩大 SLM 规模但未改架构 → 本文 LLM 骨干 + 双编码器结构性变革

**后续方向**：(1) 将可解释判别器扩展至多作者混合场景；(2) 探索更轻量的解释生成（蒸馏 QwQ-32B 至小模型）；(3) 动态适应 LLM 风格进化，引入持续学习机制。

**标签**：文本模态 / 解耦表征范式 / 作者归因与 AI 检测场景 / 对抗机制 + VAE 概率推断 / 可解释性约束

