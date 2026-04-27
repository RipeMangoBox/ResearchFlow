---
title: 'Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 大规模自监督对抗攻击视觉语言模型
- AnyAttack
acceptance: poster
cited_by: 22
method: AnyAttack
---

# Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models

**Topics**: [[T__Adversarial_Robustness]], [[T__Cross-Modal_Matching]], [[T__Captioning]] | **Method**: [[M__AnyAttack]] | **Datasets**: Commercial VLM ASR, MSCOCO Captions, MSCOCO Retrieval, SNLI-VE

| 中文题名 | 大规模自监督对抗攻击视觉语言模型 |
| 英文题名 | Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.05346) · [Code](https://github.com/AnyAttack) · [Project](https://anyattack.github.io) |
| 主要任务 | 视觉语言模型越狱攻击、可迁移对抗攻击、图像文本检索、图像描述生成 |
| 主要 baseline | AttackVLM-ii/it, SASD-WS-Cos/MSE, SU-Cos/MSE, NI-FGSM, TI-FGSM, Variance Tuning, Admix |

> [!abstract]
> 因为「现有对抗攻击依赖固定目标标签且无法规模化扩展到未标注数据」，作者在「自监督对比学习 + 可迁移攻击方法」基础上改了「温度退火单向对比预训练 + 双向对比微调的两阶段训练范式」，在「Google Gemini / OpenAI GPT 商业 VLM API」上取得「ASR 31 / 38，相比最佳 baseline SASD-WS-Cos 提升 520% / 36%」

- **Google Gemini ASR**: 31 vs. 最佳 baseline SASD-WS-Cos 的 5（提升 520%）
- **OpenAI GPT ASR**: 38 vs. 最佳 baseline SASD-WS-Cos 的 28（提升 36%）
- **MSCOCO 图像-文本检索**: AnyAttack-Bi w/ Aux 在 TR@K, IR@K 等指标上全面优于 Scratch-Cos/Bi 及无辅助模型配置

## 背景与动机

视觉语言模型（VLM）如 GPT-4V、Gemini 等已广泛部署于商业场景，但其对齐机制存在被对抗攻击绕过的风险。现有攻击方法面临一个根本矛盾：一方面，有效的越狱攻击需要针对黑盒商业模型具备强可迁移性；另一方面，主流方法严重依赖人工标注的目标标签或特定类别的优化目标，无法利用互联网规模的未标注图像-文本数据进行规模化训练。

具体而言，现有方法可分为三类局限：**AttackVLM** 通过双模态对抗提示进行越狱，但其 image-to-image (ii) 和 image-to-text (it) 变体在商业 API 上 ASR 近乎为 0（Gemini 上均为 0，GPT 上最高仅 2），表明其迁移性极差；**SASD-WS** 引入系统提示的自对抗攻击，虽在 GPT 上达到 ASR 28，但在 Gemini 上仅 5，且仍需任务特定的监督信号；**SU（Self-Universality）** 方法通过余弦相似度或 MSE 损失增强自通用性，但在商业模型上 ASR 最高仅 12，远未达到实用门槛。这些方法的共同瓶颈在于：它们都是**监督式端到端训练**，攻击生成器从零开始拟合固定数据集，既无法利用 LAION-400M 级别的无标注数据预学习通用对抗模式，也缺乏对干净-对抗嵌入空间双向对齐的显式建模。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fcef7764-9f7a-4a38-a5f7-d985a075e81f/figures/Figure_1.png)
*Figure 1 (pipeline): Comparison of existing targeted adversarial attack strategies and the proposed self-supervised attack paradigm*



Figure 1 直观对比了现有有监督攻击范式与本文提出的自监督攻击范式：前者需要成对的 (图像, 目标标签) 进行优化，后者仅需图像-文本对的自监督信号即可预训练。这一转变使得攻击生成器能够在大规模未标注数据上学习可迁移的对抗扰动先验，再通过少量任务数据微调适配特定攻击目标。本文的核心动机正是打破"攻击必须依赖标注"的假设，通过自监督对比学习实现对抗攻击的规模化与强迁移性。

## 核心创新

核心洞察：对抗攻击可以重新定义为**干净嵌入与对抗嵌入的对比对齐问题**，因为自监督对比学习能够在无标签条件下学习强大的视觉表示，从而使利用 LAION-400M 级未标注数据预训练通用攻击生成器成为可能。

| 维度 | Baseline (AttackVLM / SASD-WS / SU) | 本文 (AnyAttack) |
|:---|:---|:---|
| **监督方式** | 有监督：依赖固定目标标签或类别优化 | 自监督：仅利用图像-文本对的无标注结构 |
| **训练范式** | 端到端从零训练 | 两阶段：温度退火预训练 → 双向对比微调 |
| **损失函数** | 余弦相似度 / MSE / 交叉熵（单向） | L_Pre 单向对比损失 + L_Bi 双向对称对比损失 |
| **数据规模** | 固定标注数据集（如 COCO, ImageNet） | LAION-400M 预训练 + 任务数据微调 |
| **迁移增强** | 单一模型梯度 | 可选辅助模型集成（Auxiliary Models） |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fcef7764-9f7a-4a38-a5f7-d985a075e81f/figures/Figure_2.png)
*Figure 2 (architecture): Overview of the proposed AnyAttack, a self-supervised framework consisting of pre-training and fine-tuning stages*



AnyAttack 采用**两阶段自监督训练框架**，整体数据流如下：

1. **输入**: 干净图像 x 及与之不同的参考图像 x_r（x_r ≠ x），通过冻结的 CLIP/ViT Image Encoder 提取视觉嵌入 z = f_s(x) 和 z^adv = f_s(x_r + δ)

2. **自监督预训练阶段（Self-supervised Pre-training）**: 使用大规模未标注图像-文本对（LAION-400M），通过**温度退火单向对比损失 L_Pre** 训练攻击解码器（Attack Decoder）。该阶段不依赖任何任务标签，仅通过最大化同一样本的干净-对抗嵌入相似度、最小化与其他样本的相似度来学习通用对抗扰动先验

3. **微调阶段（Fine-tuning）**: 将预训练解码器在目标任务数据上微调，采用**双向对比损失 L_Bi** 或余弦相似度 L_Cos 进行优化。L_Bi 通过对称地优化 z→z^adv 和 z^adv→z 两个方向，实现更紧致的嵌入空间对齐

4. **辅助模型集成（Auxiliary Model Integration, 可选）**: 在微调阶段引入额外模型进行集成或梯度引导，进一步提升对黑盒及商业 VLM 的迁移攻击成功率

5. **对抗扰动生成（Adversarial Perturbation Generation）**: 对于任意目标图像，微调后的解码器生成对抗扰动 δ，使得 x_r + δ 能够误导目标 VLM 产生与原始图像 x 相关的错误输出

```
Input Image x_r ──→ [Frozen CLIP/ViT Encoder] ──→ z_r
                          ↓
              [Pre-trained Attack Decoder] ←── LAION-400M (L_Pre, τ-annealed)
                          ↓ (Fine-tuning with L_Bi / L_Cos)
              [Fine-tuned Attack Decoder] ←── Optional: Auxiliary Models
                          ↓
              Adversarial Image x_r + δ ──→ Target VLM (Misled Output)
```

## 核心模块与公式推导

### 模块 1: 自监督预训练目标 L_Pre（对应框架图：预训练阶段）

**直觉**: 无需目标标签，仅通过对比干净图像与其对抗版本的嵌入相似度，让解码器学会生成"在特征空间保持语义一致性"的通用扰动。

**Baseline 形式**: 传统有监督攻击采用固定目标优化，如 $\min_\delta \mathcal{L}_{\text{CE}}(f(x+\delta), y_{\text{target}})$ 或余弦对齐 $\max_\delta \cos(f(x), f(x+\delta))$。这些方法需要预定义 $y_{\text{target}}$ 或特定图像对，无法扩展至无标注数据。

**变化点**: 将目标从"匹配固定标签"改为"在批次内识别自身的对抗版本"，并引入**随时间退火的温度系数**以平衡早期探索与后期锐化。

**本文公式（推导）**:
$$\text{Step 1}: \min_\delta \mathcal{L}(f_s(\delta + x_r), f_s(x)), \quad \text{s.t. } x_r \neq x \quad \text{（核心约束：参考图像与原始图像不同，确保通用性）}$$
$$\text{Step 2}: \mathcal{L}_{\text{Pre}} = -\frac{1}{n} \sum_{i=1}^n \log \frac{ \exp \left( \mathbf{z}_i \cdot \mathbf{z}_i^{(adv)} / \tau(t) \right) }{ \sum_{j=1}^n \exp \left( \mathbf{z}_i \cdot \mathbf{z}_j^{(adv)} / \tau(t) \right) } \quad \text{（InfoNCE 形式，将通用目标实例化为对比学习）}$$
$$\text{Step 3}: \tau(t) = \tau_0 \left( \frac{\tau_{\text{final}}}{\tau_0} \right)^{\frac{t}{T}} = \tau_0 \exp(-\lambda t) \quad \text{（温度退火：早期高温平滑分布促进探索，后期低温聚焦难负样本）}$$

符号: $\mathbf{z}_i = f_s(x_i)$ 为干净嵌入, $\mathbf{z}_i^{(adv)} = f_s(x_{r,i} + \delta_i)$ 为对抗嵌入, $\tau(t)$ 为时变温度, $n$ 为批次大小, $T$ 为总训练步数。

**对应消融**: Figure 4 显示，去掉预训练（Scratch-Cos/Bi）在所有 ViT 架构和检索指标上均显著劣于 AnyAttack-Pre，证明预训练是任务适应的关键。

---

### 模块 2: 双向对比微调损失 L_Bi（对应框架图：微调阶段）

**直觉**: 单向对比仅优化 z 识别 z^adv 的能力，但对抗嵌入也应能反向识别其干净来源；对称双向约束可使两个嵌入空间更紧致对齐。

**Baseline 公式** (L_Cos): 
$$\mathcal{L}_{\text{Cos}} = -\frac{1}{n}\sum_{i=1}^n \cos(\mathbf{z}_i, \mathbf{z}_i^{(adv)})$$
仅最大化余弦相似度，缺乏显式负样本对比，且方向不对称。

**变化点**: 将单向 InfoNCE 扩展为**对称双向形式**，同时优化两个方向的识别能力，保持负样本排斥机制。

**本文公式（推导）**:
$$\text{Step 1}: \text{保留 } \mathbf{z}_i \rightarrow \mathbf{z}_i^{(adv)} \text{ 方向的对比损失（同 L_Pre 结构，固定 } \tau \text{）}$$
$$\text{Step 2}: \text{增加 } \mathbf{z}_i^{(adv)} \rightarrow \mathbf{z}_i \text{ 的反向对比损失，实现对称约束}$$
$$\text{最终}: \mathcal{L}_{\text{Bi}} = \frac{1}{2n} \sum_{i=1}^n \left( -\log \frac{ \exp ( \mathbf{z}_i \cdot \mathbf{z}_i^{(adv)} / \tau ) }{ \sum_{j=1}^n \exp ( \mathbf{z}_i \cdot \mathbf{z}_j^{(adv)} / \tau ) } -\log \frac{ \exp ( \mathbf{z}_i^{(adv)} \cdot \mathbf{z}_i / \tau ) }{ \sum_{j=1}^n \exp ( \mathbf{z}_i^{(adv)} \cdot \mathbf{z}_j / \tau ) } \right)$$

符号: 两项分别对应 (干净→对抗) 和 (对抗→干净) 的识别，取平均保证梯度对称性。

**对应消融**: Figure 4 中 AnyAttack-Bi 一致优于 AnyAttack-Cos（同配置下），且 AnyAttack-Bi w/ Aux 达到最高性能，验证双向对齐的有效性。

---

### 模块 3: 辅助模型集成（对应框架图：微调阶段可选模块）

**直觉**: 单一教师模型的梯度信息有限，集成多个辅助模型的特征空间可扩大对抗扰动的模型间覆盖范围，提升黑盒迁移性。

**Baseline**: 标准单模型攻击生成，无额外模型引导。

**变化点**: 在微调阶段引入**辅助模型集合** $\{f_{\text{aux}}^k\}_{k=1}^K$，通过多模型特征对齐或梯度融合增强扰动的模型无关性。

**本文实现**: 具体融合方式未在提取文本中详述，但实验表明该模块"consistently improves performance"。

**对应消融**: Figure 4 中对比 AnyAttack-Bi 与 AnyAttack-Bi w/ Aux，后者在所有检索指标上均有提升，辅助模型贡献显著。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fcef7764-9f7a-4a38-a5f7-d985a075e81f/figures/Table_1.png)
*Table 1 (comparison): The formulation of different attack strategies*



本文在多个基准上评估 AnyAttack 的有效性，涵盖商业 API 迁移、图像描述生成、图像-文本检索及视觉推理任务。

**商业 VLM 越狱攻击**（核心结果）: 
![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fcef7764-9f7a-4a38-a5f7-d985a075e81f/figures/Table_3.png)
*Table 3 (quantitative): Attack performance comparison on the MM-Vet dataset*

 Table 5 显示，AnyAttack 在 Google Gemini 上达到 ASR 31，在 OpenAI GPT 上达到 ASR 38。这一结果相比最佳 baseline SASD-WS-Cos（Gemini: 5, GPT: 28）分别提升 520% 和 36%，且远超 AttackVLM 系列（最高仅 2）和 SU 系列（最高仅 12）。值得注意的是，此前方法在 Gemini 上几乎完全失效（多数 ASR 为 0），AnyAttack 首次实现对该模型的有效越狱攻击。

**MSCOCO 图像-文本检索**（消融验证）: 
![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fcef7764-9f7a-4a38-a5f7-d985a075e81f/figures/Figure_4.png)
*Figure 4 (ablation): Performance comparison between different configurations of AnyAttack on the image-text retrieval task on MSCOCO*

 Figure 4 展示了七种配置在三种 ViT 架构（B/16, L/14, L/14@336）上的完整消融。关键趋势为：Scratch-Cos/Bi（从零训练）<< AnyAttack-Pre（仅预训练）< AnyAttack-Cos/Bi（预训练+微调）<< AnyAttack-Cos/Bi w/ Aux（+辅助模型），且 Bi 损失一致优于 Cos 损失。具体数值未在提取文本中完整呈现，但相对趋势明确表明：去掉预训练导致最差性能，去掉辅助模型造成显著下降，改用 Cos 替代 Bi 也有明显损失。

**MSCOCO 图像描述生成**: Table 4 显示 AnyAttack-Cos w/ Aux 在所有描述指标上优于 baseline 方法（具体数值待补充）。

**SNLI-VE 视觉推理**: Table 3 评估了对抗攻击对视觉蕴含任务的影响（具体数值待补充）。

**效率分析**: 
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fcef7764-9f7a-4a38-a5f7-d985a075e81f/figures/Figure_5.png)
*Figure 5 (comparison): Comparison of memory usage and time consumption between methods*

 Figure 5 比较了不同方法的内存占用与时间消耗，AnyAttack 在规模化预训练后具备合理的推理效率。

**公平性检查**: 作者选择的 baselines（AttackVLM, SASD-WS, SU）均为同期代表性迁移攻击方法，但遗漏了 FigStep [8]、VLAttack [9] 等最新工作。商业 VLM 实验基于 API 响应的主观标注（"highly/partially relevant"），成功标准可能存在一定主观性。此外，Table 5 未报告总查询次数或攻击预算，难以评估实际部署成本。预训练阶段需要 LAION-400M 规模数据及相应计算资源，对小规模研究者构成门槛。

## 方法谱系与知识库定位

**方法家族**: 自监督对比学习（SimCLR/MoCo 谱系）× 可迁移对抗攻击（NI-FGSM / TI-FGSM / Variance Tuning / Admix 谱系）

**父方法**: Self-supervised contrastive learning + Transferable adversarial attacks。AnyAttack 将对比学习的表示学习能力首次系统应用于对抗攻击生成器的预训练，同时继承了可迁移攻击的梯度优化技术。

**直接 baselines 与差异**:
- **AttackVLM** (ii/it): 双模态对抗提示，有监督端到端训练 → AnyAttack 改为自监督两阶段，无需目标标签
- **SASD-WS** (Cos/MSE): 系统提示自对抗，固定损失微调 → AnyAttack 引入温度退火预训练 + 双向对比损失
- **SU** (Cos/MSE): 自通用性增强，单模型优化 → AnyAttack 增加辅助模型集成与大规模预训练
- **NI-FGSM / TI-FGSM / Variance Tuning / Admix**: 提供梯度优化与迁移增强组件 → AnyAttack 将其整合进自监督框架而非独立使用

**改动槽位**: objective（单向对比预训练 + 双向对比微调）/ training_recipe（两阶段：预训练→微调）/ data_curation（LAION-400M 无标注预训练）/ architecture（可选辅助模型集成）

**后续方向**:
1. 防御视角：针对 AnyAttack 的自监督预训练特性，设计检测对抗嵌入分布异常的防御机制
2. 多模态扩展：将温度退火对比预训练扩展至音频-文本、视频-文本等多模态攻击
3. 高效微调：探索参数高效微调（LoRA/Adapter）替代完整解码器微调，降低任务适配成本

**标签**: 模态(视觉-语言) / 范式(自监督对比学习) / 场景(黑盒攻击/商业API越狱) / 机制(温度退火/双向对齐/辅助模型集成) / 约束(无需任务标签/大规模未标注数据预训练)

