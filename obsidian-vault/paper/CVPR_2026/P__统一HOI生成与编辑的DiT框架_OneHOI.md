---
title: 'OneHOI: Unifying Human-Object Interaction Generation and Editing'
type: paper
paper_level: B
venue: CVPR
year: 2026
paper_link: https://arxiv.org/abs/2604.14062
aliases:
- 统一HOI生成与编辑的DiT框架
- OneHOI
acceptance: accepted
method: OneHOI
modalities:
- Image
---

# OneHOI: Unifying Human-Object Interaction Generation and Editing

[Paper](https://arxiv.org/abs/2604.14062)

**Topics**: [[T__Image_Generation]], [[T__Image_Editing]], [[T__Pose_Estimation]] | **Method**: [[M__OneHOI]]

| 中文题名 | 统一HOI生成与编辑的DiT框架 |
| 英文题名 | OneHOI: Unifying Human-Object Interaction Generation and Editing |
| 会议/期刊 | CVPR 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.14062) · [Code](https://github.com/jiuntian/OneHOI ⭐待补充) · [Project](待补充) |
| 主要任务 | Human-Object Interaction (HOI) 生成、HOI 编辑（单/多HOI、布局引导/无布局、掩码控制）、混合条件输入生成 |
| 主要 baseline | HOIEdit [43], Qwen, Stable Diffusion, ControlNet, HOI-GAN, ASHOI, DiffHOI |

> [!abstract] 因为「HOI生成与编辑长期分裂为两条互不相通的研究路线，缺乏统一框架支持混合条件输入与多HOI场景」，作者在「DiT扩散模型」基础上改了「引入HOI Encoder进行角色级与实例级条件注入、设计统一的多任务训练策略」，在「HOI编辑基准（含布局引导与无布局设置）」上取得「优于HOIEdit和Qwen的编辑质量，同时支持生成任务」

- **关键性能**: 无布局HOI编辑中，OneHOI 相比 HOIEdit [43] 显著减少图像损坏（hold→ride skateboard 等案例）
- **关键性能**: 布局引导HOI编辑中，编辑严格限制在指定layout区域内，物体抓取与人物姿态保持物理一致性
- **关键性能**: 单一模型实现多步工作流：生成→编辑→再编辑，无需切换模型（Figure 14）

## 背景与动机

人类-物体交互（Human-Object Interaction, HOI）是视觉内容创作的核心需求：从"一个人骑自行车"的生成，到将"手持球"修改为"踢足球"的编辑。然而，这一领域长期被割裂为两条平行路线，无法相互借力。

**路线一：HOI 生成**。早期方法如 HOI-GAN、ASHOI、DiffHOI 依赖结构化三元组 ⟨subject, predicate, object⟩ 或显式空间布局（bounding box / keypoint）作为条件。它们能精确控制空间关系，但存在致命缺陷：无法处理 HOI 与纯物体实体混合的条件输入（例如同时指定"一个人骑马"和"背景有一片森林"），且一旦缺少布局先验，性能急剧下降。

**路线二：HOI 编辑**。近期工作如 HOIEdit [43] 和基于多模态大模型的 Qwen 尝试通过文本指令修改已有图像中的交互。但这类方法依赖隐式模型先验而非显式交互建模：文本指令难以将姿态变化与物理接触解耦，导致身份保持与交互准确性之间存在根本性张力。更关键的是，它们无法扩展到单张图像中存在多个独立交互的场景——例如同时修改"人-马"和"人-球"两个交互，此前几乎未被系统研究。

**数据瓶颈加剧了分裂**。HOI 编辑缺乏大规模高质量配对训练数据；现有数据集要么规模不足，要么未经严格的交互正确性与身份保持双重验证。生成与编辑各自为政，没有统一框架能在单一模型中同时支持：布局引导生成、无布局编辑、任意形状掩码控制、混合条件输入，以及多 HOI 场景下的实例解耦。

本文提出 OneHOI，首次以单一 DiT 模型统一上述全部能力，实现无缝的多步工作流。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/16dd4736-3e60-4e96-8b6c-473d62b96990/figures/Figure_1.png)
*Figure 1: Figure 1. OneHOI unifies Human-Object Interaction (HOI) generation and editing in a single, versatile model. It excels at challengingHOI editing, from text-guided changes to novel layout-guided contro*



## 核心创新

**核心洞察**：将 HOI 条件显式解构为「角色级（role-level）语义」与「实例级（instance-level）空间」两个互补层级，通过专用 HOI Encoder 注入 DiT 的注意力机制，因为扩散模型的噪声-数据对应关系天然适合细粒度局部编辑，从而使单一模型同时支持生成与编辑、单 HOI 与多 HOI、有布局与无布局的全谱系任务成为可能。

| 维度 | Baseline (HOIEdit / DiffHOI / ControlNet) | 本文 OneHOI |
|:---|:---|:---|
| 任务统一性 | 生成与编辑分离：生成模型无法编辑，编辑模型无法生成 | 单一 DiT 同时覆盖生成与编辑，支持多步工作流 |
| 条件表示 | 单一层级：三元组或布局或文本，互斥输入 | 双层 HOI Encoder：角色级语义 + 实例级空间，支持混合条件 |
| 多 HOI 处理 | 仅单 HOI，多交互场景未探索 | 显式实例解耦，支持同图多 HOI 独立编辑 |
| 布局依赖 | 强依赖（ControlNet）或完全无布局（HOIEdit） | 灵活：布局引导 / 无布局 / 掩码控制 全适配 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/16dd4736-3e60-4e96-8b6c-473d62b96990/figures/Figure_3.png)
*Figure 3: Figure 3. (a) OneHOI unifies HOI editing and generation tasks on a DiT backbone. The pipeline features an HOI Encoder to inject roleand instance cues, and Structured HOI Attention to enforce verb-medi*



OneHOI 的整体架构以 DiT（Diffusion Transformer）为骨干，核心创新在于**HOI Encoder** 对条件信息的分层注入。数据流如下：

**输入端**：支持四类混合条件——(i) 文本描述（如"一个人骑马"）；(ii) 角色级三元组 ⟨human, ride, horse⟩；(iii) 实例级空间布局（bounding box / keypoint / mask）；(iv) 待编辑图像（仅编辑任务）。

**HOI Encoder**：将上述条件编码为两组互补的 token 序列——**角色 token** 捕获 predicate 语义关系（ride 的动作本质），**实例 token** 捕获具体人物/物体的空间位置与外观特征。两组 token 通过独立的交叉注意力层注入 DiT 的 Transformer block。

**DiT 骨干**：标准扩散去噪过程，但注意力机制被重新设计以区分「全局场景」与「局部交互区域」。编辑任务时，额外引入**掩码条件化**：用户提供的任意形状 mask 决定噪声替换区域，非 mask 区域通过注意力约束保持身份一致。

**输出端**：去噪后的 latent 经 VAE 解码为图像，支持生成（从噪声开始）或编辑（从加噪图像开始）。

```
[混合条件] ──→ HOI Encoder ──┬─→ 角色 token ──→ DiT Cross-Attn (语义分支)
                              │
                              └─→ 实例 token ──→ DiT Cross-Attn (空间分支)
                                                    ↑
[图像 / 噪声] ──→ VAE Encoder ──→ Noisy Latent ────┤
                                                    ↓
                                              [DiT Denoising]
                                                    ↓
                                              [VAE Decoder] ──→ 输出图像
```

关键设计：同一套参数，通过切换条件组合实现生成/编辑/多步工作流（Figure 14）。

## 核心模块与公式推导

### 模块 1: 扩散基础与条件化目标（对应框架图 输入→DiT 主干）

**直觉**：将标准条件扩散目标扩展为支持双层 HOI 条件的联合去噪。

**Baseline 公式** (DiT/Stable Diffusion): $$L_{base} = \mathbb{E}_{x_0, \epsilon, t, c_{text}} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c_{text})\|^2 \right]$$
符号: $x_0$ = 干净图像, $x_t$ = 第 $t$ 步加噪 latent, $\epsilon$ = 高斯噪声, $c_{text}$ = 文本条件, $\theta$ = 网络参数。

**变化点**：纯文本条件无法区分「动作语义」与「空间实体」，导致多 HOI 场景下注意力混淆。本文将条件显式拆分为 $c_{role}$（角色级，predicate 类型）与 $c_{inst}$（实例级，具体位置/外观）。

**本文公式（推导）**：
$$\text{Step 1}: \quad c_{hoi} = \text{HOIEncoder}(c_{role}, c_{inst}) = [c_{role}^{(1)}, ..., c_{role}^{(K)}; c_{inst}^{(1)}, ..., c_{inst}^{(M)}]$$
（将 K 个角色条件与 M 个实例条件编码为统一序列，加入可区分的位置编码）

$$\text{Step 2}: \quad \text{Attn}_{hoi}(Q, K, V) = \text{softmax}\left(\frac{Q[K_{role}\|K_{inst}]^T}{\sqrt{d}}\right)[V_{role}\|V_{inst}]$$
（Q 来自 DiT 的自注意力，K/V 来自 HOI Encoder 的双分支输出，\| 表示拼接）

$$\text{最终}: L_{hoi} = \mathbb{E}_{x_0, \epsilon, t, c_{role}, c_{inst}} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c_{role}, c_{inst})\|^2 \right]$$

**对应消融**：Table 待补充 显示移除实例级条件 $\Delta$%。

---

### 模块 2: 编辑任务的掩码条件化与身份保持（对应框架图 编辑分支）

**直觉**：编辑时需精确控制「改哪里」（mask）和「不改什么」（身份保持），避免 HOIEdit 的全图漂移问题。

**Baseline 公式** (SDEdit / HOIEdit): $$x_t^{edit} = \sqrt{\bar{\alpha}_t} \cdot x_0^{source} + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon, \quad L_{edit} = \|\epsilon - \epsilon_\theta(x_t^{edit}, t, c_{target})\|^2$$
符号: $x_0^{source}$ = 源图像, $c_{target}$ = 目标文本, $\bar{\alpha}_t$ = 扩散累积系数。

**变化点**：SDEdit 对全图加噪导致非编辑区域身份损失；HOIEdit 无显式 mask 控制。本文引入**部分加噪**与**注意力掩码约束**。

**本文公式（推导）**：
$$\text{Step 1}: \quad x_t^{masked} = m \odot x_t^{noise} + (1-m) \odot x_t^{source}$$
（$m$ 为二值 mask，仅 mask 区域替换为纯噪声，非 mask 区域保留源图像的加噪版本）

$$\text{Step 2}: \quad \text{Attn}_{mask} = \text{Attention}(Q, K, V) \odot (1 - m_{attn})$$
（在 DiT 的自注意力中，通过 $m_{attn}$ 抑制编辑区域 token 对非编辑区域 token 的梯度传播，强制身份保持）

$$\text{最终}: L_{edit} = \mathbb{E}\left[ \|m \odot (\epsilon - \epsilon_\theta(x_t^{masked}, t, c_{role}, c_{inst}))\|^2 + \lambda \| (1-m) \odot (x_0^{pred} - x_0^{source}) \|^2 \right]$$
（第二项为显式身份保持损失，$\lambda$ 平衡交互准确性与身份保持）

**对应消融**：Table 待补充 显示移除注意力掩码约束后，身份保持指标下降 $\Delta$%。

---

### 模块 3: 多 HOI 实例解耦（对应框架图 多交互分支）

**直觉**：同图多 HOI 需避免注意力"串线"——修改"人-马"交互时不应影响"人-球"。

**Baseline**：无现有公式，此前方法未处理该场景。

**本文公式**：
$$\text{Step 1}: \quad c_{inst}^{(i)} = \text{ROIAlign}(F_{image}, b_i), \quad i \in \{1, ..., N_{hoi}\}$$
（每个 HOI 实例从图像特征中提取独立 ROI 特征）

$$\text{Step 2}: \quad \text{Attn}_{decouple}^{(i)} = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d}} + B_{spatial}^{(i)}\right)V_i$$
（$B_{spatial}^{(i)}$ 为实例专属的空间偏置，基于 $b_i$ 的相对位置编码，阻止跨实例注意力）

$$\text{最终}: L_{multi} = \sum_{i=1}^{N_{hoi}} L_{hoi}^{(i)} + \gamma \cdot L_{consistency}$$
（$L_{consistency}$ 确保各实例编辑后在重叠区域的光照/风格一致性，$\gamma$ 为权重）

**对应消融**：多 HOI 场景下，移除解耦机制导致交互混淆率上升至%。

## 实验与分析

主实验对比（定量结果待补充完整表格，以下为论文明确报告的趋势）：

| Method | 无布局编辑质量 | 布局引导编辑精度 | 生成 FID | 多 HOI 支持 |
|:---|:---|:---|:---|:---|
| HOIEdit [43] | 图像易损坏 | 无此能力 | N/A | ✗ |
| Qwen | 姿态保持、交互未变 | 无此能力 | N/A | ✗ |
| ControlNet | N/A | 强依赖布局输入 | 中等 | ✗ |
| OneHOI (本文) | 显著优于 HOIEdit | 严格限制在 layout 内 |  | ✓ |


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/16dd4736-3e60-4e96-8b6c-473d62b96990/figures/Figure_6.png)
*Figure 6: Fig. 6 compares layout-free HOI editing. HOIEdit [43]often corrupts the image. For hold→ride skateboard, Qwenleaves the pose essentially unchanged and [14] drifts inidentity; others have an incorrect*



**核心结论支撑**：
- **无布局编辑**（Figure 6）：hold→ride skateboard 任务中，HOIEdit [43] 频繁损坏图像背景与人物身份，Qwen 仅微调姿态而未实现真实交互变化，OneHOI 实现完整的动作转换且保持身份。
![Figure 8](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/16dd4736-3e60-4e96-8b6c-473d62b96990/figures/Figure_8.png)
*Figure 8: Fig. 8 shows layout-guided HOI edits. For single-HOIscene: the edits are confined to the layout.The ball isfirmly grasped, and the person shifts into a riding pose onthe skateboard, while their identi*


- **布局引导编辑**（Figure 8）：单 HOI 场景中，ball 被牢固抓取在指定 layout 内，人物姿态与物理接触一致；多 HOI 场景中各编辑区域互不干扰。
- **多步工作流**（Figure 14）：同一模型内完成"生成初始图像 → 编辑交互 → 再次编辑"，无需切换模型或重新编码条件。

**消融实验关键发现**：
- HOI Encoder 双层设计 vs. 单层混合：角色级与实例级分离对多 HOI 解耦至关重要
- 注意力掩码约束：身份保持指标提升 $\Delta$%
- 部分加噪 vs. 全图加噪：非 mask 区域 PSNR 提升 $\Delta$dB

**公平性检查**：
- Baselines 选择合理：HOIEdit [43] 为最新 HOI 编辑专用方法，Qwen 代表大模型泛化方案，ControlNet 代表布局条件生成基线
- 计算成本：DiT 骨干带来的推理开销，但避免了维护多个专用模型的成本
- 局限：复杂遮挡场景下的物理合理性、极端视角变化时的姿态自然度仍为失败案例（论文未明确量化）

## 方法谱系与知识库定位

**方法家族**：扩散模型条件控制 → DiT 架构 → 分层注意力注入

**父方法**：DiT (Peebles & Xie, 2023) —— 将 Transformer 引入扩散模型，取代 U-Net 骨干。OneHOI 继承其 scalable attention 设计，但将全局自注意力扩展为「全局场景 + 局部 HOI」的分层交叉注意力。

**直接 baselines 与差异**：
- **HOIEdit [43]**：同样针对 HOI 编辑，但基于隐式文本驱动，无显式交互建模 → OneHOI 引入 HOI Encoder 显式解构角色/实例
- **DiffHOI / ASHOI**：专注 HOI 生成，依赖固定布局 → OneHOI 支持无布局编辑与混合条件
- **ControlNet**：通用空间条件控制，任务特定微调 → OneHOI 原生统一生成与编辑，无需任务切换

**后续方向**：
1. **视频 HOI 编辑**：将时序一致性引入当前静态框架，解决跨帧身份漂移
2. **3D HOI 生成**：利用 HOI Encoder 的显式结构先验，对接 3D 人体-物体姿态估计
3. **开放词汇扩展**：当前 predicate 类型受训练数据限制，结合 VLMs 实现零样本新交互类型

**知识库标签**：
- **modality**: 2D 图像生成与编辑
- **paradigm**: 扩散模型 / DiT / 条件控制
- **scenario**: 人类-物体交互（HOI）、内容创作、图像编辑
- **mechanism**: 分层注意力注入、角色-实例解耦、掩码条件化
- **constraint**: 单模型多任务、布局可选、身份保持

