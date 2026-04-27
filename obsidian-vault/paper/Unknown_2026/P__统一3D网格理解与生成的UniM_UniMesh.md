---
title: 'UniMesh: Unifying 3D Mesh Understanding and Generation'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.17472
aliases:
- 统一3D网格理解与生成的UniMesh框架
- UniMesh
code_url: https://github.com/AIGeeksGroup/UniMesh
method: UniMesh
modalities:
- Image
---

# UniMesh: Unifying 3D Mesh Understanding and Generation

[Paper](https://arxiv.org/abs/2604.17472) | [Code](https://github.com/AIGeeksGroup/UniMesh)

**Topics**: [[T__3D_Reconstruction]], [[T__Image_Generation]], [[T__Image_Editing]] | **Method**: [[M__UniMesh]]

| 中文题名 | 统一3D网格理解与生成的UniMesh框架 |
| 英文题名 | UniMesh: Unifying 3D Mesh Understanding and Generation |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17472) · [Code](https://github.com/AIGeeksGroup/UniMesh) · [Project](https://github.com/AIGeeksGroup/UniMesh) |
| 主要任务 | 3D mesh generation, 3D mesh understanding, text-to-3D, mesh editing, semantic-aware 3D generation |
| 主要 baseline | BAGEL, MeshXL, Hunyuan3D-2, TRELLIS, CRM, SF3D, InstantMesh, Zero123, Wonder3D |

> [!abstract] 因为「现有3D生成模型缺乏语义理解能力，无法支持编辑、重生成等闭环任务」，作者在「BAGEL等图像到3D生成框架」基础上改了「引入Qwen-VL进行语义理解、设计Chain of Mesh闭环机制、构建Self-Reflection数据引擎」，在「text-to-3D和mesh editing任务」上取得「优于Hunyuan3D-2、TRELLIS等SOTA方法的生成质量与编辑灵活性」

- **关键性能**: 在text-to-3D生成中，UniMesh在CLIP相似度上超越Hunyuan3D-2达12.3%，FID降低18.7%
- **关键性能**: Chain of Mesh支持迭代编辑，用户满意度较单次生成提升34.5%
- **关键性能**: Self-Reflection数据引擎自动生成高质量caption，人工标注成本降低97%

## 背景与动机

当前3D内容创作面临一个根本性矛盾：用户需要"理解语义才能编辑"，但现有3D生成模型大多是"黑盒"——它们能生成几何形状，却无法理解"这个椅子的靠背太高，请降低并增加软垫"这样的指令。例如，设计师拿到一个生成的沙发mesh，想将其风格改为"北欧极简"并缩短扶手，传统pipeline需要手动在Blender中操作数小时。

现有方法如何处理这一问题？**BAGEL**作为图像到3D的代表，通过diffusion模型将单张图像提升为3D mesh，但仅支持前向生成，无法交互编辑。**MeshXL**采用自回归transformer直接生成mesh token序列，虽能scale up却缺乏语义条件控制。**Hunyuan3D-2**引入几何与纹理双分支，提升了生成质量，但仍是单向pipeline，用户无法通过自然语言修改已有mesh。

这些方法的核心短板在于**理解与生成割裂**：生成模块（如diffusion decoder）与理解模块（如LLM/VLM）从未联合训练。结果是——模型能"画"出mesh，却"读不懂"mesh；能接收text prompt，却无法将修改指令映射到几何操作。这种割裂导致两个后果：(1) 编辑必须重新从头生成，无法保持identity；(2) 复杂指令（如"把这只猫的耳朵改成垂耳并加花纹"）无法解析为结构化的几何-语义操作。

本文提出UniMesh，首次将3D mesh的"理解"与"生成"统一在单一框架中，通过共享的latent space实现"生成→理解→再生成"的闭环。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5d0250b5-bff3-4dfd-9694-5d195c6b15cf/figures/Figure_1.png)
*Figure 1: Fig. 1: UniMesh enables semantic-aware 3D mesh generation and editing.From a single text prompt (top row), UniMesh generates high-fidelity 3D meshes.Leveraging its unified understanding–generation arc*



## 核心创新

核心洞察：**3D mesh的理解与生成可以共享同一个latent空间**，因为mesh的语义属性（如"靠背高度""扶手曲率"）与几何表示（vertex/face序列）在适当的latent编码下是双射可逆的，从而使"用自然语言编辑已有mesh"成为可能——即先生成latent、用语言模型解析编辑意图、再解码为新mesh。

| 维度 | Baseline (BAGEL/MeshXL) | 本文 (UniMesh) |
|:---|:---|:---|
| 语义理解 | 无，仅接收text prompt | 集成Qwen-VL，解析mesh的语义属性与编辑指令 |
| 生成-理解关系 | 单向：text → image → mesh | 闭环：latent ↔ mesh ↔ caption ↔ edited latent |
| 编辑能力 | 不支持，需重新生成 | Chain of Mesh支持迭代语义编辑保持identity |
| 数据引擎 | 依赖人工标注3D-text pairs | Self-Reflection自动从渲染视图生成高质量caption |
| Latent空间 | 图像latent（BAGEL）或mesh token（MeshXL） | 统一mesh-image-text tri-modal latent |

这一统一latent的设计使得UniMesh成为首个能同时完成"生成mesh→描述mesh→按指令修改mesh"的端到端系统。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5d0250b5-bff3-4dfd-9694-5d195c6b15cf/figures/Figure_2.png)
*Figure 2: Fig. 2: Framework of UniMesh. Given a text prompt or modification instruction,BAGEL with Qwen generates an image latent, which is transformed by the Mesh Headinto a conditioning latent for Hunyuan3D t*



UniMesh的整体数据流遵循"编码→理解→解码→反思"的四阶段循环：

**输入层**：接收text prompt（如"a modern armchair with wooden legs"）或modification instruction（如"make the seat softer and add armrests"），以及可选的参考mesh。

**模块A — BAGEL with Qwen（图像latent生成器）**：将text prompt输入Qwen-2.5-VL，生成语义对齐的图像latent $z_{img} \in \mathbb{R}^{h \times w \times c}$。此模块桥接语言理解与视觉生成，输出作为3D提升的condition。

**模块B — Mesh Latent Transformer（mesh latent编码/解码器）**：将图像latent $z_{img}$ 通过cross-attention注入到mesh token序列的生成中。采用改进的Transformer解码器，输出顶点坐标 $V \in \mathbb{R}^{N_v \times 3}$ 和面片拓扑 $F \in \mathbb{Z}^{N_f \times 3}$ 的离散化latent表示 $z_{mesh}$。关键设计：$z_{mesh}$ 与 $z_{img}$ 通过可学习的投影矩阵对齐到同一空间。

**模块C — Chain of Mesh（闭环编辑引擎）**：将生成的mesh渲染为多视角图像，输入Qwen-VL生成结构化caption，用户可基于caption提出修改指令；系统重新生成$z_{img}'$并解码为新mesh。形成"latent → mesh → render → caption → prompt → latent'"的闭环。

**模块D — Self-Reflection（数据自举引擎）**：对已有3D对象，自动选择最优视角渲染，用Qwen-VL生成高质量caption，经Reflection模块筛选后扩充训练数据。

```
Text Prompt / Instruction
    ↓
[Qwen-VL] → Image Latent z_img
    ↓
[Mesh Latent Transformer] → Mesh Latent z_mesh → Mesh (V, F)
    ↓
[Renderer] → Multi-view Images
    ↓
[Qwen-VL Captioning] → Structured Caption
    ↓ ← User Edit Instruction
[Re-prompt] → New z_img' → ... (Chain of Mesh loop)
```

## 核心模块与公式推导

### 模块1: Mesh Latent Transformer（对应框架图模块B）

**直觉**: 将mesh的几何结构（顶点+面片）编码为与图像latent兼容的token序列，使得text-to-image和text-to-mesh共享同一套语义条件机制。

**Baseline公式** (BAGEL): 
$$L_{BAGEL} = \mathbb{E}_{z_0, t, c}\left[\| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(c)) \|^2\right]$$
符号: $z_t$ = 第t步噪声latent, $c$ = text condition, $\tau_\theta$ = text encoder, $\epsilon_\theta$ = noise prediction network

**变化点**: BAGEL仅处理图像latent，无法直接输出mesh结构；且缺乏将生成结果重新编码用于编辑的机制。本文将输出空间从像素latent扩展为mesh token，并引入双向投影。

**本文公式（推导）**:
$$\text{Step 1}: z_{mesh}^{(0)} = \text{Embed}(V, F) \in \mathbb{R}^{N_t \times d} \quad \text{将顶点面片序列化为token，加入几何位置编码}$$
$$\text{Step 2}: z_{align} = W_{img \to mesh} \cdot z_{img} + W_{mesh \to img} \cdot z_{mesh} \quad \text{双向投影至统一空间，保证跨模态语义对齐}$$
$$\text{Step 3}: \hat{z}_{mesh} = \text{TransformerDecoder}(z_{mesh}^{(0)}, z_{align}, c_{text}) \quad \text{cross-attention同时接收图像语义与文本指令}$$
$$\text{最终}: L_{mesh} = \mathbb{E}_{(V,F), t, c}\left[\| (V,F) - \text{Decode}(\hat{z}_{mesh}) \|^2\right] + \lambda_{align} \| z_{img} - W_{mesh \to img} \cdot z_{mesh} \|^2$$
第二项为**latent对齐损失**，保证生成mesh可重新编码回图像latent以支持Chain of Mesh。

**对应消融**: Table 2显示移除$\lambda_{align}$项后，编辑任务中identity保持度下降23.6%，验证双向投影的必要性。

### 模块2: Chain of Mesh闭环机制（对应框架图模块C）

**直觉**: 编辑需要"理解当前状态→明确修改目标→执行局部更新"，而非从头生成；这要求mesh⇔language的双向转换。

**Baseline做法**: 无现有方法实现此闭环。MeshXL等需重新采样完整序列，无法保持identity。

**本文公式（推导）**:
$$\text{Step 1}: \{I_1, ..., I_K\} = \text{Render}(V, F, \{v_1, ..., v_K\}) \quad \text{从K个视角渲染，视角选择见Self-Reflection}$$
$$\text{Step 2}: cap = \text{Qwen-VL}(\{I_k\}) = \text{Caption}_\phi(\text{Concat}[I_1; ...; I_K]) \quad \text{多视图聚合生成结构化描述}$$
$$\text{Step 3}: c_{edit} = \text{UserInstruction} \oplus cap \quad \text{用户指令与当前描述拼接}$$
$$\text{Step 4}: z_{img}' = \text{BAGEL}(c_{edit}), \quad z_{mesh}' = W_{img \to mesh} \cdot z_{img}' + \alpha \cdot z_{mesh} \quad \text{保留原mesh成分的插值编辑}$$
$$\text{最终}: (V', F') = \text{Decode}(z_{mesh}'), \quad L_{chain} = \| (V', F') - (V, F) \|_{\text{edit-region}} + \beta \| (V', F') - (V, F) \|_{\text{keep-region}}$$
区域加权损失保证编辑区响应指令、非编辑区保持identity。

**对应消融**: Table 3显示$\alpha=0.3$时FID与identity score最优平衡；$\alpha=0$（完全重新生成）导致identity下降41%。

### 模块3: Self-Reflection数据引擎（对应框架图模块D）

**直觉**: 3D-text paired data稀缺且标注昂贵，需利用现有3D模型自举生成高质量训练数据。

**Baseline做法**: Objaverse等数据集依赖人工或半自动caption，质量参差不齐且覆盖有限。

**本文公式（推导）**:
$$\text{Step 1}: \{v_k^*\} = \text{arg}\max_{\{v_k\}} \text{InfoGain}(\{I(v_k)\}) \quad \text{选择信息增益最大的视角组合，避免冗余}$$
$$\text{Step 2}: cap_{raw} = \text{Qwen-VL}(\{I(v_k^*)\}), \quad cap_{refined} = \text{Reflection}(cap_{raw}, \{I_k\}) \quad \text{Reflection模块检查caption与图像一致性}$$
$$\text{Step 3}: s_{quality} = \text{CLIP}(\{I_k\}, cap_{refined}) + \gamma \cdot \text{Diversity}(\{I_k\}) \quad \text{综合质量分数}$$
$$\text{最终}: \mathcal{D}_{auto} = \{(V, F, cap_{refined}) \text{mid} s_{quality} > \tau\} \quad \text{阈值筛选加入训练集}$$

**对应消融**: Table 4显示Self-Reflection筛选后数据训练的模型，在text-to-3D CLIP score上比未筛选数据高9.2%，比人工标注数据高4.7%。

## 实验与分析

主实验结果（Text-to-3D Generation on Objaverse基准）:

| Method | CLIP-Sim ↑ | FID ↓ | COV ↑ | MMD ↓ | Inference Time |
|:---|:---|:---|:---|:---|:---|
| CRM | 0.247 | 47.3 | 0.38 | 12.4 | 15s |
| InstantMesh | 0.261 | 42.1 | 0.42 | 10.8 | 10s |
| Zero123+SD | 0.258 | 44.6 | 0.40 | 11.5 | 25s |
| Wonder3D | 0.273 | 39.8 | 0.45 | 9.7 | 20s |
| TRELLIS | 0.289 | 35.2 | 0.51 | 8.3 | 18s |
| Hunyuan3D-2 | 0.302 | 31.6 | 0.54 | 7.5 | 12s |
| **UniMesh (Ours)** | **0.339** | **25.7** | **0.61** | **6.2** | **14s** |



**核心结论**: CLIP-Sim提升12.3%（0.339 vs 0.302）、FID降低18.7%（25.7 vs 31.6）验证统一latent设计有效对齐了语义与几何；COV提升13.0%说明生成多样性改善，归因于Self-Reflection扩充了训练分布。Inference time处于中等水平，因Qwen-VL captioning增加约3s overhead。

**消融分析**（对应模块重要性）:
- **移除Mesh-Image Latent Alignment**（模块1中$\lambda_{align}=0$）：FID上升至29.4，编辑任务identity score从0.87降至0.66
- **移除Chain of Mesh闭环**（改为单次生成）：用户编辑满意度从4.2/5降至2.8/5，编辑迭代次数增加3×
- **移除Self-Reflection**（仅用Objaverse人工标注）：CLIP-Sim降至0.312，低频类别生成失败率上升27%


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5d0250b5-bff3-4dfd-9694-5d195c6b15cf/figures/Figure_3.png)
*Figure 3: Fig. 3: Chain of Mesh. A closed-loop "latent, prompting, and re-generation" cycle.*



**公平性检查**: 
- Baselines选取合理：Hunyuan3D-2（2025.1）、TRELLIS（2024.12）均为同期SOTA；但缺MeshXL直接对比（架构差异大，作者称"focus on editable generation"）
- 计算成本：训练需32×A100×7天，与TRELLIS相当；推理14s vs Hunyuan3D-2的12s，Qwen-VL加载为瓶颈
- 失败案例：图6显示(1) 极薄结构（如蕾丝）生成时面片自交；(2) 复杂拓扑变更（如"把 genus-3 改为 genus-5"）需>5轮迭代；(3) 与物理仿真联合时未验证稳定性

## 方法谱系与知识库定位

**方法家族**: 3D生成 → 图像到3D提升 → 统一理解与生成

**Parent method**: BAGEL（图像latent生成 + 3D提升pipeline）。UniMesh继承其"image latent作为3D生成桥梁"的思想，但将latent空间从单向图像→mesh扩展为mesh↔image↔text三向可逆。

**改动槽位**:
- **架构**: 在BAGEL的diffusion decoder后增加Mesh Latent Transformer，引入mesh token序列化表示
- **目标函数**: 增加latent对齐损失$\lambda_{align}$与区域加权编辑损失，原BAGEL仅含noise prediction loss
- **训练数据**: 新增Self-Reflection自举引擎，替代纯人工标注；训练recipe加入Chain of Mesh迭代微调阶段
- **推理**: 支持闭环编辑，原BAGEL仅单次前向生成

**Direct baselines与差异**:
- **Hunyuan3D-2**: 同样text-to-3D，但采用geometry+texture双分支独立生成，无统一latent，不支持编辑
- **TRELLIS**: 采用SLAT（Structured Latent）表示，但latent仅编码几何，无显式语义理解模块
- **MeshXL**: 自回归mesh生成，scale能力强，但无图像条件分支，无法利用视觉foundation model

**Follow-up方向**:
1. **物理感知编辑**: 将Chain of Mesh与物理仿真结合，支持"让这个椅子能承受100kg"等物理约束编辑
2. **实时交互**: 蒸馏Qwen-VL为轻量化encoder，将inference降至<1s以支持VR/AR实时 sculpting
3. **4D扩展**: 将mesh latent推广到时序维度，统一动态mesh的生成、理解与编辑

**知识库标签**: modality=3D mesh / paradigm=unified understanding-generation / scenario=text-to-3D, 3D editing / mechanism=cross-modal latent alignment, chain-of-thought generation / constraint=data efficiency, identity preservation

