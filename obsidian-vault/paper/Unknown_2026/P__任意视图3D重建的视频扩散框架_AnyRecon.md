---
title: 'AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19747
aliases:
- 任意视图3D重建的视频扩散框架
- AnyRecon
- 现有视频扩散重建方法的核心瓶颈在于：条件化视角数量受限（1-2帧）导致
code_url: https://github.com/longxiang-ai/awesome-video-diffusions
method: AnyRecon
modalities:
- Image
---

# AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model

[Paper](https://arxiv.org/abs/2604.19747) | [Code](https://github.com/longxiang-ai/awesome-video-diffusions)

**Topics**: [[T__3D_Reconstruction]], [[T__Image_Generation]], [[T__Depth_Estimation]] | **Method**: [[M__AnyRecon]]

> [!tip] 核心洞察
> 现有视频扩散重建方法的核心瓶颈在于：条件化视角数量受限（1-2帧）导致全局上下文不足，以及生成与重建的解耦导致大场景重建碎片化。AnyRecon的核心洞察是：将扩散模型从独立的新视角合成器改造为重建流水线的内嵌组件——通过持久全局记忆支持任意数量条件视角，通过生成-重建闭环实现逐段迭代优化，通过移除时序压缩和稀疏注意力解决非顺序输入的效率与对齐问题。有效性的根本原因在于：更丰富的全局上下文（任意参考视图）提供了更强的外观约束，而闭环机制使几何误差可以被持续修正而非累积传播。

| 中文题名 | 任意视图3D重建的视频扩散框架 |
| 英文题名 | AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19747) · [Code](https://github.com/longxiang-ai/awesome-video-diffusions) · [Project] |
| 主要任务 | 稀疏视图三维重建、新视角合成、大场景逐段重建 |
| 主要 baseline | Difix3D+、ViewCrafter、Uni3C |

> [!abstract] 因为「现有视频扩散重建方法仅能以1-2帧为条件，导致全局外观保真度不足且生成与重建解耦造成碎片化」，作者在「Wan2.1-I2V-14B」基础上改了「移除时序压缩、引入全局场景记忆与几何感知闭环、替换为稀疏注意力」，在「DL3DV-Evaluation与Tanks and Temples」上取得「优于Difix3D+/ViewCrafter/Uni3C的重建质量，全局记忆模块单点带来LPIPS 0.205→0.151（26%提升）」。

- **全局场景记忆**：LPIPS 从 0.205 降至 0.151，PSNR 从 20.18 提升至 20.95，SSIM 从 0.634 提升至 0.656
- **推理效率**：通过 DMD2 蒸馏将采样步数压缩至 4 步
- **训练成本**：64×A800 GPU，LoRA rank=32，三阶段共 140k 步

## 背景与动机

稀疏视图三维重建旨在将用户日常随手拍摄的几张无序图像转化为可自由探索的三维场景。例如，游客围绕一座古建筑拍摄十余张不同角度的照片，期望系统能补全被遮挡的背面细节并生成连贯的环绕视频。现有方法在这一需求下暴露出根本性缺陷。

非生成式重建方法如 NeRF 和 3D Gaussian Splatting（3DGS）依赖密集多视角输入，在稀疏条件下几何歧义严重——大遮挡区域无法合成合理内容，直接失效。基于扩散模型的方法通过生成新视角缓解了稀疏性，但普遍存在条件化视角数量受限的瓶颈：大多数方法如 ViewCrafter 仅能以 1-2 帧真实图像作为条件，全局外观保真度不足，且几何一致性弱。视频扩散模型 Wan2.1-I2V-14B 虽具备强大的视频生成能力，但其时序因果潜变量压缩机制假设输入帧按时间顺序排列，对非顺序、大基线视角变化适应性差。更重要的是，现有方法将扩散生成视为独立的新视角合成步骤，生成输出从未反馈回三维几何表示；当需要逐段重建大规模场景时，缺乏跨段的持久几何记忆机制，导致重建碎片化、全局一致性无法保证。

这些挑战共同限制了扩散式稀疏重建在真实世界不规则输入、大视角差和长轨迹场景下的实用性。AnyRecon 的核心动机正是打破这一僵局：将扩散模型从独立生成器改造为重建流水线的内嵌组件，通过持久记忆与闭环迭代实现任意数量条件视角下的连贯重建。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/94d684b5-6bde-498e-818a-6339fef7746e/figures/Figure_1.png)
*Figure 1: Fig. 1: AnyRecon demonstrates robust performance across multiple reconstruction set-tings: (Top) Interpolation, filling in gaps between distant captured views; (Middle)Extrapolation, synthesizing nove*



## 核心创新

核心洞察：将视频扩散模型从独立的新视角合成器改造为重建流水线的内嵌组件，因为持久全局记忆支持任意数量条件视角提供更强外观约束，而生成-重建闭环使几何误差可被持续修正而非累积传播，从而使非顺序、大基线、长轨迹场景的高效一致重建成为可能。

| 维度 | Baseline (Wan2.1-I2V-14B / Difix3D+ / ViewCrafter) | 本文 |
|:---|:---|:---|
| 时序表示 | 时序因果压缩，帧间共享潜变量 | **移除时序压缩**，每帧独立帧级表示 |
| 条件视角数 | 固定1-2帧，硬编码限制 | **全局场景记忆**，支持任意数量参考视图 |
| 注意力机制 | 全自注意力，二次复杂度 | **块稀疏注意力**（2×8×8），线性复杂度 |
| 生成-重建关系 | 解耦，扩散为独立步骤 | **几何感知闭环**，生成输出持续更新共享3D记忆 |
| 推理效率 | 原始多步采样 | **DMD2 4步蒸馏**，效率与保真度平衡 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/94d684b5-6bde-498e-818a-6339fef7746e/figures/Figure_2.png)
*Figure 2: Fig. 2: Pipeline of AnyRecon. Given arbitrary sparse input views organized in acapture view bank Icap, we perform geometry-aware retrieval to select spatially infor-mative views for each novel traject*



AnyRecon 的整体框架围绕"捕获视图库 → 几何感知检索 → 条件化扩散生成 → 记忆更新 → 迭代重建"的闭环流水线展开，核心模块如下：

**输入**：任意稀疏输入视图，组织为捕获视图库 $I_{cap}$，无需时序顺序。

**模块一：几何感知检索（Geometry-aware Retrieval）** —— 输入 $I_{cap}$ 与当前几何记忆 $M_{geo}$，输出精选参考视图子集 $I_{sel}$。区别于 FOV 或相似度检索（Figure 5 对比），该模块基于几何位置关系选择真正有助于当前视角重建的参考帧，避免冗余。

**模块二：全局场景记忆（Global Scene Memory）** —— 输入原始捕获帧，输出持久化的全局外观上下文。通过在渲染先验序列前拼接原始捕获帧（prepended capture view cache），突破 1-2 帧条件限制，训练时随机采样 $N \in [2,4]$ 个条件视角模拟灵活稀疏输入。

**模块三：去时序压缩的视频扩散生成器** —— 输入 $I_{sel}$ 与噪声潜变量，输出生成视角帧。基于 Wan2.1-I2V-14B 移除时序压缩模块，每帧保持独立表示；注意力机制替换为块稀疏注意力（block size 2×8×8），每帧仅关注 ±8 相邻帧和 $I_{sel}$，将二次复杂度降为线性。

**模块四：显式3D几何记忆更新（Explicit 3D Geometry Memory Update）** —— 输入新生成帧，输出更新后的 $M_{geo}$。将扩散生成输出持续积分到共享三维表示中，形成跨段持久记忆，支撑下一段检索与生成（Figure 4 展示无记忆更新时的失效模式）。

**输出**：逐段迭代生成的完整三维场景表示与任意新视角渲染。

```
I_cap ──→ [几何感知检索] ──→ I_sel
              ↑                    ↓
         M_geo ←── [3D记忆更新] ←── [视频扩散生成器]
              └────────────────────┘
                    (闭环迭代)
```

## 核心模块与公式推导

### 模块一：全局场景记忆的条件化机制（对应框架图"Global Scene Memory"位置）

**直觉**：原始视频扩散模型仅支持固定数量的条件帧（通常1-2帧），而真实稀疏重建需要灵活利用任意数量的可用参考视图。

**Baseline 公式** (Wan2.1-I2V-14B 标准条件化):
$$L_{base} = \mathbb{E}_{z_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c_{1:2}) \|^2 \right]$$
符号: $z_t$ = 噪声潜变量, $t$ = 时间步, $c_{1:2}$ = 固定2帧条件嵌入, $\epsilon_\theta$ = 去噪网络

**变化点**：硬编码的 $c_{1:2}$ 限制导致全局上下文不足；训练与推理的条件帧数不匹配时性能骤降。

**本文公式（推导）**:
$$\text{Step 1}: c_{mem} = \text{Concat}(I_{cap}^{(1)}, I_{cap}^{(2)}, ..., I_{cap}^{(N)}), \quad N \sim \mathcal{U}[2,4] \text{（训练时随机采样模拟稀疏输入）}$$
$$\text{Step 2}: c_{render} = \text{RenderPrior}(M_{geo}, \text{camera poses}) \quad \text{（当前几何记忆渲染的先验帧）}$$
$$\text{最终}: L_{mem} = \mathbb{E}_{z_0, \epsilon, t, N} \left[ \| \epsilon - \epsilon_\theta(z_t, t, [c_{mem} \,;\, c_{render}]) \|^2 \right]$$
拼接操作 $[\cdot \,;\, \cdot]$ 将捕获视图缓存置于渲染先验序列之前，形成 prepended capture view cache。

**对应消融**：全局场景记忆模块的消融显示，移除该模块导致 PSNR 20.95→20.18（-3.7%），SSIM 0.656→0.634（-3.4%），**LPIPS 0.151→0.205（+35.8% 退化）**，是感知质量最关键的单一组件。

---

### 模块二：块稀疏注意力与复杂度优化（对应框架图"Sparse Attention"位置）

**直觉**：全自注意力的二次复杂度 $O(T^2)$ 无法扩展至长轨迹大场景，且非顺序输入中远距离帧的注意力权重实际贡献低。

**Baseline 公式** (标准 Transformer 全自注意力):
$$\text{Attention}_{base}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad \text{复杂度 } O(T^2 \cdot H \cdot W)$$
符号: $T$ = 帧数, $H, W$ = 空间分辨率, $d_k$ = 键维度

**变化点**：视频扩散重建中，当前帧主要依赖局部时序邻域和几何相关的参考视图，全局全连接是计算浪费；同时需保留对 $I_{sel}$ 的跨帧几何对齐访问。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{N}(i) = \{j : |j - i| \leq 8\} \cup \{j : I_{cap}^{(j)} \in I_{sel}\} \quad \text{（定义局部邻域+几何相关视图的稀疏索引集）}$$
$$\text{Step 2}: A_{sparse}^{(i,j)} = \begin{cases} \frac{Q^{(i)}K^{(j)T}}{\sqrt{d_k}} & j \in \mathcal{N}(i) \\ -\infty & \text{otherwise} \end{cases} \quad \text{（掩码非相关位置为负无穷）}$$
$$\text{最终}: \text{Attention}_{sparse}^{(i)} = \text{softmax}(A_{sparse}^{(i,\cdot)})V^{(\mathcal{N}(i))}, \quad \text{复杂度 } O(T \cdot K \cdot H \cdot W), \, K=16$$
块实现采用 block size 2×8×8（2帧×8×8空间块），硬件友好的结构化稀疏模式。

**对应消融**：Figure 3 包含稀疏注意力的联合消融，但独立定量贡献未在提供文本中分离呈现。

---

### 模块三：几何感知闭环与记忆更新（对应框架图"3D Memory Update + Retrieval"位置）

**直觉**：逐段重建大场景时，新生成帧必须反馈回三维表示以修正累积误差，而非作为独立输出丢弃。

**Baseline 流程** (Difix3D+ / ViewCrafter 等):
$$\text{Segment}_k = G_\theta(I_{cond}^{(k)}) \rightarrow \text{输出，无反馈} \rightarrow \text{Segment}_{k+1} = G_\theta(I_{cond}^{(k+1)})$$
生成器 $G_\theta$ 每段独立调用，几何误差逐段累积，全局一致性断裂。

**变化点**：生成与重建解耦导致"生成-丢弃-再生成"的碎片化模式；需要显式3D记忆作为跨段信息载体。

**本文公式（推导）**:
$$\text{Step 1}: M_{geo}^{(0)} = \text{Initialize}(I_{cap}, \text{COLMAP/SfM}) \quad \text{（初始点云/高斯表示）}$$
$$\text{Step 2}: I_{sel}^{(k)} = \text{GeomRetrieve}(M_{geo}^{(k-1)}, \text{target pose}_k, I_{cap}) \quad \text{（几何感知检索，Figure 5）}$$
$$\text{Step 3}: \hat{I}_k = \text{Diffuse}(I_{sel}^{(k)}, z_t; \theta) \quad \text{（条件化生成第k段）}$$
$$\text{Step 4}: M_{geo}^{(k)} = \text{Update}(M_{geo}^{(k-1)}, \hat{I}_k, \text{camera}_k) \quad \text{（显式3D记忆更新，Figure 4）}$$
$$\text{最终}: \text{迭代至 } k=K \text{ 完成，输出 } M_{geo}^{(K)}$$

**对应消融**：Figure 4 定性展示了无记忆更新时"新生成轨迹段未整合入重建"的失效模式，但缺乏定量指标。

## 实验与分析

主实验在 DL3DV-Evaluation（10场景）和 Tanks and Temples（5场景）两个数据集上评估插值（interpolation）与外推（extrapolation）两种设置。AnyRecon 与 Difix3D+、ViewCrafter、Uni3C 对比结果如下：

| Method | DL3DV-Eval (PSNR/SSIM/LPIPS) | Tanks&Temples (PSNR/SSIM/LPIPS) | 关键差异 |
|:---|:---|:---|:---|
| Uni3C |  |  | 基线方法 |
| ViewCrafter |  |  | 视频扩散，1-2帧条件 |
| Difix3D+ |  |  | 扩散+3DGS结合 |
| **AnyRecon** | **优于上述基线**（具体数值待补充） | **优于上述基线**（具体数值待补充） | 任意视图条件+闭环记忆 |


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/94d684b5-6bde-498e-818a-6339fef7746e/figures/Figure_6.png)
*Figure 6: Fig. 6: Quality Results on DL3DV Dataset [11].*



定量层面，提供文本未完整呈现 Table 1 的具体数值，仅确认 AnyRecon 在两个数据集两种设置下均优于三个基线。消融实验方面，**全局场景记忆**的独立验证最为充分：PSNR 20.18→20.95（+3.8%），SSIM 0.634→0.656（+3.5%），**LPIPS 0.205→0.151（-26.3%）**，感知质量提升显著。Figure 3 对时序压缩（TC）、4步蒸馏和稀疏注意力进行了联合消融，但各组件的独立贡献未分离报告——"full temporal compression follows Wan by keeping only..."的对比显示移除TC对非顺序输入的必要性，而4步蒸馏在效率与质量间的权衡缺乏单独量化。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/94d684b5-6bde-498e-818a-6339fef7746e/figures/Figure_3.png)
*Figure 3: Fig. 3: Ablation on temporal compression (TC), 4-step distillation andsparse attention. Full temporal compression follows Wan by keeping only the firstframe uncompressed while compressing subsequent f*



定性层面，Figure 7 展示了 Tanks and Temples 数据集上 Difix3D+ 在大视角差场景下产生明显伪影，而 AnyRecon 保持结构完整，支持其鲁棒性优势主张。Figure 6 的 DL3DV 结果展示插值质量。跨域泛化方面，DL3DV 训练→Tanks and Temples 零样本测试验证一定迁移能力，但测试场景仅15个，结论需谨慎。

公平性检查：（1）基线选择未包含 ReconFusion、3D-GS Enhancer 等同期相关方法；（2）所有基线均未针对"任意视图输入"专门优化，评估设置可能对 AnyRecon 有利；（3）训练成本极高（64×A800 GPU），复现门槛高；（4）系统对初始点云质量强依赖，极端视角重叠不足时闭环失效——此失败模式未在实验中量化分析。

## 方法谱系与知识库定位

**方法家族**：视频扩散式新视角合成 → 稀疏视图三维重建

**父方法**：Wan2.1-I2V-14B（阿里万相视频生成大模型）。AnyRecon 直接继承其14B参数规模的视频扩散架构，通过结构性修改（移除时序压缩、替换注意力）和流程扩展（全局记忆、几何闭环）实现领域适配。

**改动槽位**：
- **架构**：移除时序压缩模块；全自注意力→块稀疏注意力（2×8×8）
- **目标函数**：标准扩散损失 + 全局记忆条件化；DMD2 蒸馏目标
- **训练配方**：三阶段渐进（全注意力微调100k步→稀疏热身10k步→蒸馏30k步），LoRA rank=32
- **数据策划**：训练时随机采样 N∈[2,4] 条件视角模拟稀疏输入
- **推理**：4步采样，几何感知检索驱动的迭代闭环

**直接基线与差异**：
- **Difix3D+**：同样结合扩散与3D表示，但无持久记忆机制、无任意视图条件能力
- **ViewCrafter**：视频扩散重建，但固定1-2帧条件、生成-重建解耦、无时序压缩移除
- **Uni3C**：通用3D生成基线，非针对稀疏重建优化

**后续方向**：（1）降低对初始点云的强依赖，探索端到端几何初始化；（2）将稀疏注意力与记忆机制迁移至其他视频扩散骨干（如 CogVideo、OpenSora）验证通用性；（3）引入物理约束或显式深度监督，进一步压缩几何误差累积。

**标签**：modality=图像/视频+3D | paradigm=扩散生成+迭代重建 | scenario=稀疏视图/任意视角/大场景 | mechanism=持久记忆/稀疏注意力/生成-重建闭环 | constraint=高计算成本/初始点云依赖

