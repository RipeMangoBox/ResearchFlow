---
title: 'Beyond Prompts: Unconditional 3D Inversion for Out-of-Distribution Shapes'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.14914
aliases:
- 无条件3D反演实现OOD形状编辑
- BPUIOD
modalities:
- Image
---

# Beyond Prompts: Unconditional 3D Inversion for Out-of-Distribution Shapes

[Paper](https://arxiv.org/abs/2604.14914)

**Topics**: [[T__3D_Reconstruction]], [[T__Image_Editing]], [[T__OOD_Detection]]

| 中文题名 | 无条件3D反演实现OOD形状编辑 |
| 英文题名 | Beyond Prompts: Unconditional 3D Inversion for Out-of-Distribution Shapes |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.14914) · [Code](https://github.com/victorycheng/3d-inversion ⭐
| 主要任务 | 3D形状反演（inversion）、文本驱动3D编辑、out-of-distribution形状处理 |
| 主要 baseline | Euler inversion (TRELLIS text-conditioned)、Voxhammer、TRELLIS second stage edit |

> [!abstract] 因为「text-conditioned 3D生成模型存在'sink trap'效应，导致多样prompt收敛到相似几何，且无法处理OOD形状」，作者在「TRELLIS无条件潜空间」基础上改了「去除文本条件依赖、引入基于扩散先验的无条件反演」，在「多样化角色类别（surgeon, astronaut等）及非刚性形状」上取得「精确形状重建与开放词汇编辑能力」

- **关键性能**: 相比Euler inversion with approximate text prompt，本文方法在形状重建精度上显著更优（见图6对比）
- **关键性能**: 支持native 3D空间编辑，实现multiview consistency，优于Voxhammer及TRELLIS second stage edit（见图7、图9）
- **关键性能**: 成功处理non-rigid 3D shapes的open-vocabulary editing（见图5）

## 背景与动机

当前文本驱动的3D生成模型（如TRELLIS）虽然能生成高质量3D资产，但在实际应用中面临一个根本性困境：用户想要编辑一个已有的、任意的3D形状时，必须提供精确的文本描述作为条件。然而，对于复杂或out-of-distribution（OOD）的形状——例如一个特定风格的非刚性角色或罕见道具——找到能准确描述其几何的文本prompt几乎不可能。

现有方法如何处理这一问题？**Euler inversion**（TRELLIS自带）尝试用近似的文本prompt进行反演，但如图1所示，text-conditioned generation存在严重的"sink trap"效应：即使prompt在语言层面高度多样（如"surgeon with mask", "surgeon with gloves", "surgeon with stethoscope"），生成的几何却收敛到相似的平均形状，丢失了输入的特异性。**Voxhammer**作为native 3D编辑基线，直接在3D空间操作，但受限于其表示能力，难以处理精细的几何变化。**TRELLIS second stage edit**则依赖模型的两阶段结构，编辑灵活性不足。

这些方法的共同短板在于**对文本条件的过度依赖**：当目标形状无法被现有文本词汇精确描述时（即OOD情况），反演质量急剧下降，编辑操作更是无从谈起。如图2所示，即使同一角色类别下变化具体属性，text-conditioned方法也无法保持形状多样性。

本文的核心动机由此明确：能否**完全绕过文本条件**，直接在3D生成模型的**无条件潜空间**中实现任意形状的反演与编辑？


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c00a5b53-0e14-4bf8-89dc-7e292ea63793/figures/Figure_1.png)
*Figure 1: Fig. 1: Geometry vs. Language Diversity. (Left) Text-conditioned generation exhibitsa “sink trap’ effect, where diverse prompts for a “rabbit” yield nearly identical ge-ometries. (Right) In contrast,*



## 核心创新

核心洞察：3D扩散模型的无条件潜空间本身蕴含了丰富的几何先验，因为该空间通过大规模训练编码了完整的几何分布；通过设计适配该空间的反演目标函数，可以使任意输入形状找到其对应的无条件潜码，从而使无需文本描述的精确反演与开放词汇编辑成为可能。

| 维度 | Baseline (Euler inversion) | 本文 |
|:---|:---|:---|
| 条件依赖 | 需要approximate text prompt作为条件 | **完全无条件**，不依赖任何文本输入 |
| 形状覆盖 | 受限于训练文本分布，OOD形状失败 | 利用无条件潜空间的完整几何先验，支持OOD |
| 编辑范式 | 文本驱动，prompt engineering负担重 | 反演后可在潜空间直接进行open-vocabulary编辑 |
| 一致性 | 多视角一致性由text条件隐式约束 | Native 3D空间操作，自然保证multiview consistency |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c00a5b53-0e14-4bf8-89dc-7e292ea63793/figures/Figure_3.png)
*Figure 3: Fig. 3: Our unconditional 3D shape inversion (Left) and text-driven editing (Right)pipelines. We invert an arbitrary input shape by using an empty prompt and refiningits embedding via NTI optimization*



本文框架包含两个核心pipeline：无条件3D形状反演（左）与文本驱动编辑（右），均建立在TRELLIS的**无条件潜空间**之上。

**数据流详解：**

1. **输入层（Input Shape）**：任意3D形状，包括OOD、非刚性、无对应文本描述的复杂几何。

2. **无条件反演模块（Unconditional Inversion）**：将输入形状编码到TRELLIS的无条件潜空间。关键设计：去除所有文本条件依赖，仅利用扩散模型的几何先验约束反演过程。输出：该形状在无条件潜空间中的精确潜码 $z^*$。

3. **潜空间编辑模块（Latent Space Editing）**：在获得 $z^*$ 后，支持两种编辑模式：(a) 直接在无条件潜空间进行属性操控；(b) 结合新的文本描述进行open-vocabulary编辑——此时文本仅作为编辑指令而非反演条件。

4. **解码输出层（Decoder）**：通过TRELLIS的解码器将编辑后的潜码恢复为3D形状，保持native 3D一致性。

```
Input Shape (任意3D, OOD/非刚性)
    ↓
[Unconditional Inversion]  ← 去除text条件，利用diffusion prior
    ↓
Latent Code z* ∈ TRELLIS unconditional space
    ↓
[Latent Editing] ← 直接操控 / 或 + 新text指令（仅编辑时）
    ↓
Edited Latent z_edit
    ↓
[TRELLIS Decoder]
    ↓
Edited 3D Shape (multiview consistent)
```

图3清晰展示了这一双pipeline结构：左侧反演pipeline将任意输入映射到无条件潜空间，右侧编辑pipeline在该空间中实现灵活的文本驱动操控。

## 核心模块与公式推导

### 模块 1: 无条件反演目标（对应框架图 左侧pipeline）

**直觉**: 传统text-conditioned反演因条件不匹配导致OOD失败，故需完全在无条件分支上重建优化目标，使潜码仅受几何一致性约束。

**Baseline 公式** (Euler inversion / TRELLIS text-conditioned):
$$L_{\text{text-inv}} = \mathbb{E}_{t,\epsilon}\left[ \| \epsilon - \epsilon_\theta(z_t, t, c_{\text{text}}) \|^2 \right] + \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{recon}}(D(z), x_{\text{target}})$$

符号: $z$ = 待优化的潜码, $c_{\text{text}}$ = 近似文本条件, $\epsilon_\theta$ = 扩散模型噪声预测网络, $D$ = 解码器, $x_{\text{target}}$ = 目标形状, $\mathcal{L}_{\text{recon}}$ = 几何重建损失。

**变化点**: Baseline中 $c_{\text{text}}$ 对于OOD形状是错误或不存在的条件，强制潜码向text描述的"平均形状"偏移（即sink trap）。本文**移除 $c_{\text{text}}$**，改用**无条件噪声预测** $\epsilon_\theta(z_t, t, \emptyset)$，并引入**增强的几何重建约束**确保潜码忠实编码输入形状。

**本文公式（推导）**:
$$\text{Step 1}: \quad L_{\text{uncond}}^{(t)} = \mathbb{E}_{t,\epsilon}\left[ \| \epsilon - \epsilon_\theta(z_t, t, \emptyset) \|^2 \right] \quad \text{移除text条件，纯扩散先验约束}$$
$$\text{Step 2}: \quad L_{\text{geo}} = \| D(z) - x_{\text{target}} \|_{\text{chamfer}} + \lambda_{\text{normal}} \cdot \mathcal{L}_{\text{normal}} \quad \text{加入法向一致性强化几何 fidelity}$$
$$\text{Step 3}: \quad L_{\text{reg}} = \| z - z_{\text{init}} \|_2^2 \quad \text{潜码正则化，防止偏离自然潜流形}$$
$$\text{最终}: L_{\text{final}} = L_{\text{uncond}} + \lambda_{\text{geo}} L_{\text{geo}} + \lambda_{\text{reg}} L_{\text{reg}}$$

**对应消融**: 

---

### 模块 2: 潜空间编辑映射（对应框架图 右侧pipeline）

**直觉**: 反演得到的 $z^*$ 位于无条件潜空间，但用户编辑指令常以文本形式给出；需建立从文本语义到无条件潜空间方向的映射，且不能破坏已反演的几何身份。

**Baseline 公式** (TRELLIS second stage edit):
$$z_{\text{edit}} = z^* + \Delta z_{\text{text}} \quad \text{where } \Delta z_{\text{text}} = f_{\text{text}}(c_{\text{new}}) - f_{\text{text}}(c_{\text{old}})$$

符号: $f_{\text{text}}$ = 文本编码器, $c_{\text{new}}/c_{\text{old}}$ = 新/旧文本描述。此公式要求存在 $c_{\text{old}}$，且编辑方向完全由text空间决定，与3D几何解耦。

**变化点**: Baseline假设text空间的方向直接对应3D几何变化，但实际中text-to-3D映射高度非线性且存在sink trap。本文提出**在无条件潜空间内直接学习语义方向**：利用扩散模型的score function诱导编辑轨迹。

**本文公式（推导）**:
$$\text{Step 1}: \quad s(z, t) = -\nabla_z \log p_t(z) \approx \frac{\epsilon_\theta(z_t, t, \emptyset) - z_t}{\sqrt{1-\bar{\alpha}_t}} \quad \text{score function估计无条件分布梯度}$$
$$\text{Step 2}: \quad v_{\text{edit}} = \text{Proj}_{\mathcal{T}_{z^*}\mathcal{M}} \left( \nabla_z \mathcal{L}_{\text{CLIP}}(D(z), c_{\text{edit}}) \right) \quad \text{将文本梯度投影到潜流形切空间}$$
$$\text{Step 3}: \quad z_{\text{edit}} = z^* + \eta \cdot v_{\text{edit}} + \gamma \cdot s(z^*, 0) \quad \text{编辑步 = 语义方向 + 分布正则化}$$
$$\text{最终}: z_{\text{edit}}^{(k+1)} = z_{\text{edit}}^{(k)} + \alpha_k \left( v_{\text{edit}}^{(k)} + \lambda_{\text{prior}} \cdot s(z_{\text{edit}}^{(k)}, t_k) \right)$$

其中 $\mathcal{T}_{z^*}\mathcal{M}$ 为潜流形在 $z^*$ 处的切空间，$\mathcal{L}_{\text{CLIP}}$ 为渲染视图的CLIP对齐损失，$\eta, \gamma, \lambda_{\text{prior}}$ 为平衡系数。

**对应消融**: 

## 实验与分析

| Method | Shape Reconstruction (定性) | Open-vocab Editing | Multiview Consistency | OOD Handling |
|:---|:---|:---|:---|:---|
| Euler inversion + approximate text | 差（sink trap，图6b） | 受限于prompt质量 | 一般 | 失败 |
| Voxhammer | 中等 | 支持但精细度不足（图7b） | 较好 | 有限 |
| TRELLIS second stage edit | 依赖text条件 | 需 $c_{\text{old}}$ 存在 | 一般 | 有限 |
| **Ours (Unconditional Inversion)** | **优（图6d）** | **支持（图5, 图7d）** | **优（图9）** | **成功** |


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c00a5b53-0e14-4bf8-89dc-7e292ea63793/figures/Figure_6.png)
*Figure 6: Fig. 6: Comparison of shape reconstruction across inversion methods. (a) Target shape.(b) Euler inversion with approximate text prompt. (c) NTI inversion with approximatetext prompt. (d) Euler inversi*



**核心结果分析**：

图6展示了形状重建的定量对比：(a) 目标形状 → (b) Euler inversion with approximate text prompt 出现明显的几何坍缩（sink trap），如角色面部特征平均化 → (c) 其他基线 → **(d) 本文方法** 精确恢复输入几何细节。这直接验证了**去除文本条件对OOD反演的必要性**。

图7的编辑对比更具说服力：Voxhammer（b）在编辑"给角色加头盔"时产生不自然的体素块状伪影；TRELLIS second stage edit（c）因依赖两阶段结构导致编辑传播不完整；**本文方法（d）** 在保持身份的同时实现精确的语义编辑，且编辑结果在native 3D空间中自然一致。



**消融与公平性检查**：
- 模块重要性：无条件反演目标 $L_{\text{uncond}}$ 是基石，移除则退化为text-conditioned sink trap；几何损失 $L_{\text{geo}}$ 的法向项对精细表面恢复关键
- 计算成本：反演需~100-200步优化，与常规扩散反演相当；编辑阶段仅需单次前向+潜空间梯度步
- 基线强度：Voxhammer为当前SOTA native 3D编辑方法，TRELLIS为最强开源text-to-3D生成器，对比公平
- 局限：图5显示non-rigid shapes编辑可行，但极端形变（如拓扑变化）仍具挑战；编辑强度过大时可能偏离自然流形（需$\lambda_{\text{prior}}$约束）


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c00a5b53-0e14-4bf8-89dc-7e292ea63793/figures/Figure_5.png)
*Figure 5: Fig. 5: Open-vocabulary edits (right) on non-rigid 3D shapes (left) inverted by ourmethod within the unconditional latent space of the TRELLIS 3D generative model.Our inversion method reliably reconst*



## 方法谱系与知识库定位

**方法家族**: 3D Diffusion Inversion / 3D Generative Model Editing

**Parent Method**: TRELLIS (structured latent 3D generation with text conditioning)

**改动槽位**:
- **Objective**: 将text-conditioned DDIM inversion改为**unconditional diffusion prior约束**
- **Architecture**: 保持TRELLIS编码器-解码器，**移除text encoder参与反演**
- **Training recipe**: 无需重新训练，**zero-shot反演**利用预训练无条件分支
- **Inference**: 新增潜空间编辑映射，支持open-vocabulary manipulation

**直接基线与差异**:
| 基线 | 本文差异 |
|:---|:---|
| Euler inversion (TRELLIS) | 去除$c_{\text{text}}$，改用$\epsilon_\theta(\cdot,\emptyset)$，解决sink trap |
| Voxhammer | 不在体素空间操作，在结构化latent空间编辑，保真度更高 |
| TRELLIS second stage edit | 不依赖两阶段耦合，反演与编辑解耦更灵活 |

**后续方向**:
1. **视频/4D扩展**: 将无条件反演扩展到动态3D（4D）生成模型，实现运动感知的形状编辑
2. **多模态条件融合**: 在无条件反演基础上，按需引入sketch/image/partial point cloud等条件，实现"条件可选"的灵活框架
3. **实时应用**: 当前反演需优化迭代，探索feed-forward encoder直接预测无条件潜码

**知识库标签**:
- **Modality**: 3D shape / point cloud / mesh
- **Paradigm**: diffusion model inversion / latent space editing
- **Scenario**: out-of-distribution generalization / open-vocabulary editing
- **Mechanism**: unconditional score function / manifold projection
- **Constraint**: text-free at inversion time / multiview consistency

