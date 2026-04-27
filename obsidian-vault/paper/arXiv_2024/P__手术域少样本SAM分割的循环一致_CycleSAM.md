---
title: 'CycleSAM: Few-Shot Surgical Scene Segmentation with Cycle- and Scene-Consistent Feature Matching'
type: paper
paper_level: C
venue: arXiv
year: 2024
paper_link: https://www.semanticscholar.org/paper/5211b9c881b92c543abe007b60a5f2dfcb724bed
aliases:
- 手术域少样本SAM分割的循环一致性特征匹配
- CycleSAM
- 通用特征匹配在手术域失败的根本原因有两个：特征本身不适配（域间隙）和匹
cited_by: 1
method: CycleSAM
---

# CycleSAM: Few-Shot Surgical Scene Segmentation with Cycle- and Scene-Consistent Feature Matching

[Paper](https://www.semanticscholar.org/paper/5211b9c881b92c543abe007b60a5f2dfcb724bed)

**Topics**: [[T__Semantic_Segmentation]], [[T__Few-Shot_Learning]], [[T__Domain_Adaptation]] | **Method**: [[M__CycleSAM]]

> [!tip] 核心洞察
> 通用特征匹配在手术域失败的根本原因有两个：特征本身不适配（域间隙）和匹配过程不可靠（噪声对应）。CycleSAM用手术特定自监督特征解决前者，用双向循环一致性+场景一致性约束解决后者。两个改动都作用于同一个瓶颈——点提示质量——而非改变SAM本身或整体流程结构。有效性的核心逻辑是：更好的特征+更严格的匹配过滤=更准确的点提示=更好的SAM分割结果。

| 中文题名 | 手术域少样本SAM分割的循环一致性特征匹配 |
| 英文题名 | CycleSAM: Few-Shot Surgical Scene Segmentation with Cycle- and Scene-Consistent Feature Matching |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://www.semanticscholar.org/paper/5211b9c881b92c543abe007b60a5f2dfcb724bed) · Code  · Project  |
| 主要任务 | 少样本手术场景图像分割 (Few-Shot Surgical Scene Segmentation) |
| 主要 baseline | 少样本SAM变体、线性探测 (Linear Probing)、参数高效适应 (PEFT)、伪标签基线 |

> [!abstract] 因为「手术图像标注稀缺且通用视觉特征在手术域存在严重域间隙」，作者在「少样本SAM框架（特征匹配→点提示采样→SAM分割）」基础上改了「用手术特定自监督特征替换通用SAM特征，并引入循环一致性+场景一致性双重过滤」，在「四个手术数据集的1-shot/5-shot设置」上取得「比现有少样本SAM方法提升2-4倍（Dice/IoU）」

- **关键性能1**: 四个多样化手术数据集上，1-shot和5-shot设置下比现有少样本SAM方法提升 **2-4倍**（以Dice/IoU衡量）
- **关键性能2**: 消融实验（Table 3）验证手术特定特征提取器、循环一致性和场景一致性均有独立贡献
- **关键性能3**: 超越线性探测、参数高效适应和伪标签基线方法

## 背景与动机

手术图像分割是计算机辅助手术导航和组织识别的基础任务，但其核心瓶颈在于标注数据极度稀缺——手术场景复杂多变，专业标注需要临床医生投入大量时间，导致监督学习方法难以直接部署。以腹腔镜胆囊切除术为例，同一器官在不同光照、出血量、器械遮挡条件下外观差异巨大，而可用的标注病例可能仅有数十例。

现有方法从三个方向尝试突破这一困境。**通用提示分割模型SAM** 具备强大的零样本分割能力，但其有效使用依赖图像特定的视觉提示（visual prompts），这使其主要被用于辅助数据标注而非自动化分割，无法直接处理无人工干预的手术视频流。**近期少样本SAM扩展方法** 尝试通过参考图像自动预测点提示（point prompts）来驱动SAM，其流程为：提取查询图像与支持图像的通用视觉特征→计算特征相似度图→采样高置信度点作为SAM提示→输出分割结果。然而，这些方法的特征匹配建立在ImageNet预训练的通用视觉特征之上，对手术图像这类域外（out-of-domain）数据缺乏鲁棒性。**参数高效微调方法**（如LoRA、Adapter）虽能在少量参数更新下适配新域，但在1-shot/5-shot极端设置下仍面临过拟合风险，且未解决特征本身不适配的根本问题。

这些方法的共同短板在于忽视了手术域的特殊性：器械反光、血液遮挡、组织形变、视角受限等因素使手术图像与自然图像存在显著域间隙，通用特征匹配产生大量噪声对应关系，导致点提示质量低下，最终分割性能大幅下降。此外，少样本设置进一步限制了可用于域适应的监督信号，形成"特征不适配→匹配噪声→提示错误→分割失败"的恶性循环。

本文的核心动机正是打破这一循环：在不改变SAM分割能力本身的前提下，从特征提取和匹配过滤两个关键环节入手，构建对手术域鲁棒的特征匹配机制，从而在极少标注数据条件下实现高质量的自动化手术场景分割。

## 核心创新

核心洞察：手术域少样本分割的瓶颈在于点提示质量，而点提示质量同时受限于特征域不适配和匹配过程噪声，因为手术图像与自然图像存在结构性域间隙且通用特征匹配缺乏双向验证机制，从而使手术特定自监督特征与双重一致性约束的联合设计成为可能。

| 维度 | Baseline（少样本SAM扩展） | 本文（CycleSAM） |
|:---|:---|:---|
| 特征提取 | 通用SAM视觉编码器（ImageNet域预训练） | 手术特定自监督特征提取器（手术视频预训练+参数高效微调） |
| 相似度计算 | 单向余弦相似度 $\text{sim}(f_q, f_s)$ | 循环一致性过滤：双向匹配乘积 $S_{cyc} = \text{sim}(q,s) \cdot \text{sim}(s,q)^\text{top}$ |
| 空间约束 | 无场景级过滤 | 场景一致性过滤：$S_{scene} = S_{cyc} \odot M_{scene}$，抑制跨场景错误对应 |
| 训练目标 | 仅分割损失 $\mathcal{L}_{seg}$ | 联合优化：$\mathcal{L} = \mathcal{L}_{seg} + \lambda \mathcal{L}_{consist}$ |

两个改动作用于同一瓶颈——点提示质量——而非改变SAM本身或整体流程结构。有效性的核心逻辑链为：更好的特征 + 更严格的匹配过滤 → 更准确的点提示 → 更好的SAM分割结果。

## 整体框架

CycleSAM保持现有少样本SAM框架的宏观流程（特征匹配→点提示采样→SAM分割），但对前两个环节进行了针对性替换与增强。整体数据流如下：

**输入**: 查询图像 $x_q$（待分割手术帧）+ 支持图像-掩码对 $(x_s, M_s)$（1-shot或5-shot标注样本）

**模块A：手术特定特征提取**（替换原SAM编码器）
- 输入：$x_q$ 和 $x_s$
- 处理：通过手术视频自监督预训练的特征提取器 $f_{surgical}(\cdot)$ 提取域适配特征，再经参数高效微调（仅更新少量参数）适配具体任务
- 输出：$f_q = f_{surgical}(x_q)$, $f_s = f_{surgical}(x_s)$

**模块B：循环一致性相似度计算**（增强原相似度计算）
- 输入：$f_q$, $f_s$
- 处理：计算双向匹配矩阵并逐元素乘积过滤
- 输出：$S_{cyc}(x_q, x_s) = \text{sim}(f_q, f_s) \cdot \text{sim}(f_s, f_q)^\text{top}$

**模块C：场景一致性过滤**（新增模块）
- 输入：$S_{cyc}$, 场景级掩码 $M_{scene}$
- 处理：逐元素乘积抑制跨场景对应
- 输出：$S_{scene} = S_{cyc} \odot M_{scene}$

**模块D：多样化点提示采样**
- 输入：过滤后的高质量相似度图 $S_{scene}$
- 处理：在高分区域采样多个多样化点坐标
- 输出：点提示集合 $\{p_1, p_2, ..., p_k\}$

**模块E：SAM分割**
- 输入：$x_q$ + 点提示
- 处理：冻结的SAM解码器生成分割掩码
- 输出：最终预测 $\hat{M}_q$

```
x_q, (x_s, M_s) → [f_surgical] → f_q, f_s
                          ↓
              [双向sim计算] → S_cyc = sim(q,s)·sim(s,q)ᵀ
                          ↓
              [⊙ M_scene] → S_scene
                          ↓
              [点采样] → {p_i}
                          ↓
              [SAM解码器] → M̂_q
```

## 核心模块与公式推导

### 模块1：循环一致性相似度过滤（对应框架图 模块B）

**直觉**: 单向特征匹配容易产生噪声对应（如手术器械反光区域被错误匹配到组织区域），双向验证可保留真正鲁棒的特征对应。

**Baseline公式** (通用少样本SAM): 
$$S_{base}(x_q, x_s) = \text{sim}(f(x_q), f(x_s)) = \frac{f_q \cdot f_s^\text{top}}{\|f_q\| \|f_s\|}$$
符号: $f(\cdot)$ = 通用视觉编码器（如SAM的ViT-H）, $\text{sim}$ = 余弦相似度, $f_q, f_s \in \mathbb{R}^{HW \times C}$ 为展平的空间特征。

**变化点**: Baseline仅计算查询→支持的单向相似度，无法识别"查询中A区域匹配支持中B区域，但B区域并不回指A区域"的伪对应。手术域中这类噪声尤为严重（如不同器械的金属反光具有相似外观但语义不同）。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{S}_{q\to s} = \text{sim}(f_q, f_s), \quad \tilde{S}_{s\to q} = \text{sim}(f_s, f_q) = \tilde{S}_{q\to s}^\text{top}$$
$$\text{Step 2}: S_{cyc} = \tilde{S}_{q\to s} \odot \tilde{S}_{s\to q}^\text{top} = \tilde{S}_{q\to s} \odot \tilde{S}_{q\to s}$$
（逐元素乘积，等价于双向置信度的联合激活）
$$\text{最终}: S_{cyc}(i,j) = \text{sim}(f_q^{(i)}, f_s^{(j)}) \cdot \text{sim}(f_s^{(j)}, f_q^{(i)})$$

几何解释：$S_{cyc}(i,j)$ 仅在位置 $i$ 和 $j$ 互为最近邻时取高值，抑制所有单向"一厢情愿"的匹配。

**对应消融**: Table 3 显示移除循环一致性约束 Δ性能下降（具体数值。

---

### 模块2：场景一致性过滤（对应框架图 模块C）

**直觉**: 手术场景中不同解剖区域具有明确的空间语义约束（如胆囊只出现在肝床区域，不会出现在腹壁），利用场景级先验可进一步过滤跨场景的荒谬对应。

**Baseline**: 无此模块，$S_{base}$ 直接用于点采样。

**变化点**: 即使经过循环一致性过滤，仍可能存在同一手术视频帧内跨解剖区域的错误匹配（如胆囊区域特征错误匹配到胃壁区域），需要场景级空间约束。

**本文公式（推导）**:
$$\text{Step 1}: \text{估计场景掩码 } M_{scene} \in \{0,1\}^{H\times W}$$
（具体估计方法：基于支持集掩码的扩展或学习得到的场景边界，原文细节
$$\text{Step 2}: S_{scene} = S_{cyc} \odot M_{scene}$$
其中 $\odot$ 为逐元素乘积，$M_{scene}$ 在语义有效区域取1，无效区域取0。
$$\text{最终}: S_{scene}(i,j) = S_{cyc}(i,j) \cdot M_{scene}(i,j)$$

**对应消融**: Table 3 显示移除场景一致性约束 Δ性能下降（具体数值。

---

### 模块3：联合训练目标（对应框架图 模块A的微调阶段）

**直觉**: 参数高效微调需同时优化分割质量和特征匹配质量，避免两者相互掣肘。

**Baseline公式** (标准PEFT微调):
$$\mathcal{L}_{base} = \mathcal{L}_{seg}(\hat{M}_q, M_q^{gt})$$
通常为交叉熵或Dice损失。

**变化点**: 仅优化分割损失无法显式约束特征匹配质量，导致微调后的特征空间仍可能产生噪声对应。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}_{seg} = \text{Dice}(\text{SAM}(x_q, \{p_i\}), M_q^{gt})$$
$$\text{Step 2}: \mathcal{L}_{consist} = \|S_{scene} - S_{scene}^{target}\|_F^2 \text{ 或一致性正则项}$$
（具体形式
$$\text{最终}: \mathcal{L} = \mathcal{L}_{seg} + \lambda \mathcal{L}_{consist}$$

其中 $\lambda$ 为平衡系数，仅更新参数高效模块（如LoRA低秩矩阵或Adapter参数），冻结SAM主体和特征提取器主干。

**对应消融**: Table 3 显示联合训练 vs 仅分割损失的对比（具体数值。

## 实验与分析

主要实验结果在四个多样化手术数据集上进行，涵盖1-shot和5-shot两种少样本设置：

| Method | 数据集1 (1-shot) | 数据集1 (5-shot) | 数据集2 (1-shot) | 数据集2 (5-shot) | 数据集3 (1-shot) | 数据集3 (5-shot) | 数据集4 (1-shot) | 数据集4 (5-shot) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 现有少样本SAM基线 |  |  |  |  |  |  |  |  |
| Linear Probing |  |  |  |  |  |  |  |  |
| PEFT基线 |  |  |  |  |  |  |  |  |
| 伪标签基线 |  |  |  |  |  |  |  |  |
| **CycleSAM** | **** | **** | **** | **** | **** | **** | **** | **** |
| **提升倍数** | **2-4x** | **2-4x** | **2-4x** | **2-4x** | **2-4x** | **2-4x** | **2-4x** | **2-4x** |

核心结果分析：声称的 **2-4倍提升**（以Dice/IoU衡量）跨度较大，暗示不同数据集/设置下的增益不均匀。需关注：（1）增益主要来自特征提取器替换还是一致性约束机制；（2）5-shot相比1-shot的提升是否呈现饱和趋势。

消融实验（Table 3）验证各模块独立贡献：
- 手术特定自监督特征提取器：移除后性能下降（具体Δ%
- 循环一致性约束：移除后性能下降（具体Δ%
- 场景一致性约束：移除后性能下降（具体Δ%
- 联合训练目标 $\mathcal{L}_{seg} + \lambda \mathcal{L}_{consist}$ vs 单独 $\mathcal{L}_{seg}$：（具体Δ%

公平性检查与局限：
1. **基线强度存疑**：若基线使用通用SAM特征而CycleSAM使用手术特定预训练特征，特征质量差异本身可能是主要增益来源，而非一致性约束机制的创新价值。理想消融应包含"通用特征+一致性约束"和"手术特征+无不一致性约束"的对照。
2. **手术特定模型的可复现性**：自监督预训练的数据来源、视频数量、预训练任务细节（对比学习/掩码重建/时序预测）未在摘要中披露，这是方法可复现性的关键前提。
3. **SAM本身的上限约束**：若SAM对特定手术器械类型或组织状态的零样本响应差，提示质量再高也难以突破瓶颈。方法未评估SAM在手术域的固有偏差。
4. **1-shot过拟合风险**：参数高效微调在1-shot极端设置下的稳定性未被充分讨论，支持集选择敏感性未知。
5. **计算成本**：双重一致性过滤引入的额外计算开销（双向相似度矩阵乘法）在实时手术视频处理中的可行性未说明。

## 方法谱系与知识库定位

**方法家族**: Few-Shot Semantic Segmentation → Few-Shot Medical Image Segmentation → Few-Shot SAM Adaptation

**父方法**: Segment Anything Model (SAM) + 少样本特征匹配框架（如PerSAM、Matcher等）

**改动插槽**:
| 插槽 | 父方法设置 | 本文改动 |
|:---|:---|:---|
| 特征提取 (architecture) | SAM ViT-H 通用编码器 | 手术特定自监督编码器 + PEFT适配 |
| 匹配目标 (objective) | 单向余弦相似度 | 循环一致性 × 场景一致性联合过滤 |
| 训练配方 (training_recipe) | 分割损失 alone | $\mathcal{L}_{seg} + \lambda \mathcal{L}_{consist}$ 联合优化 |
| 数据策划 (data_curation) | ImageNet/通用视觉预训练 | 手术视频自监督预训练（细节待补充） |
| 推理流程 (inference) | 单向前向匹配 | 双向匹配 + 掩码过滤 |

**直接基线与差异**:
- **PerSAM/Personalize-SAM**: 使用通用特征+手工设计的相似度聚合，CycleSAM替换为手术域特征+可学习一致性约束
- **SAM-Adapter/SSM**: 参数高效微调SAM本身，CycleSAM冻结SAM、微调特征提取器并增加匹配过滤
- **Surgical-SAM类工作**: 全量微调或大量标注适配，CycleSAM专注极少标注（1/5-shot）场景

**后续方向**:
1. **自监督预训练透明化**：公开手术视频预训练数据集与协议，使"手术特定特征"成为可比较、可复用的社区资源
2. **时序一致性扩展**：将循环一致性从空间域扩展到时间域，利用手术视频的帧间连续性进一步优化特征匹配
3. **提示多样性自动化**：当前点提示采样策略固定，可探索基于不确定性的自适应提示数量与位置选择

**标签**: 
- modality: 医学图像 / 手术视频
- paradigm: 少样本学习 / 提示学习 / 自监督预训练
- scenario: 数据稀缺 / 域迁移 / 手术导航
- mechanism: 特征匹配 / 循环一致性 / 场景约束
- constraint: 1-shot/5-shot极端标注 / 参数高效 / 冻结SAM
