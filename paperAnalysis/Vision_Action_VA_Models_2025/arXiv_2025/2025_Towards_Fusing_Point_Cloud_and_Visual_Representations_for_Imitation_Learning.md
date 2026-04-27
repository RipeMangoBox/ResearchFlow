---
title: "Towards Fusing Point Cloud and Visual Representations for Imitation Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/imitation-learning
  - task/robot-manipulation
  - diffusion
  - adaptive-layernorm
  - point-cloud-fusion
  - dataset/RoboCasa
  - opensource/no
core_operator: "以点云token为主干，用RGB全局/局部token汇聚出的条件向量通过AdaLN逐层调制扩散策略，在保留3D几何结构的同时注入2D语义与纹理"
primary_logic: |
  语言指令 + 多视角RGB + 点云 → RGB提取全局/局部token并经CLS汇聚为条件向量，点云经FPS+KNN分组编码为3D token，语言经CLIP嵌入 → 以点云+语言为主序列、RGB为AdaLN条件驱动DiT去噪动作块 → 输出未来动作序列
claims:
  - "在RoboCasa 24个操作任务上，FPV-SUGAR的平均成功率达到50.5%，高于简单PC+RGB拼接的40.42%、RGB-only的30.0%、PC-only的28.0%以及DP3的22.75% [evidence: comparison]"
  - "将点云+语言作为主序列、RGB作为AdaLN条件的非对称融合优于简单拼接和“点云作条件”的反向条件化方案 [evidence: ablation]"
  - "引入ResNet局部RGB特征而非只用全局token，能在按钮、抽屉等细粒度操作上带来更高成功率 [evidence: ablation]"
related_work_position:
  extends: "DiT (Peebles & Xie 2022)"
  competes_with: "3D Diffusion Policy (Ze et al. 2024); 3D Diffuser Actor (Ke et al. 2024)"
  complementary_to: "SUGAR (Chen et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Towards_Fusing_Point_Cloud_and_Visual_Representations_for_Imitation_Learning.pdf"
category: Embodied_AI
---

# Towards Fusing Point Cloud and Visual Representations for Imitation Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.12320)
> - **Summary**: 这篇工作把点云保留为动作策略的主干表示，再用包含全局与局部信息的RGB条件通过AdaLN逐层调制扩散Transformer，从而在机器人模仿学习里同时利用3D几何精度和2D语义纹理。
> - **Key Performance**: RoboCasa 24任务平均成功率最高 50.5%（FPV-SUGAR），较简单 PC+RGB 拼接 40.42% 提升 10.08 个点；FPV-MLP 也达到 49.42%，说明主要收益来自融合机制而不只是更强预训练编码器。

> [!info] **Agent Summary**
> - **task_path**: 语言指令 + 多视角RGB + 点云观测 -> 未来动作块预测
> - **bottleneck**: 2D语义与3D几何都重要，但现有“图像特征贴到点云”或直接拼接会丢失全局上下文，或把跨模态对齐负担全部压给策略网络
> - **mechanism_delta**: 用点云+语言token作为DiT主序列，把RGB全局/局部token经CLS汇聚为条件向量，通过AdaLN在每层残差中持续调制动作去噪
> - **evidence_signal**: RoboCasa上“RGB作条件、点云作主干”的AdaLN融合显著优于单模态、简单拼接和反向条件化；局部特征与CLS条件向量消融也支持该因果链
> - **reusable_ops**: [FPS+KNN点云patch化, CLS汇聚后的AdaLN跨模态调制]
> - **failure_modes**: [将点云压成单个条件向量时几何细节丢失导致性能下降, 仅用全局RGB特征时按钮/抽屉等细粒度操作性能不足]
> - **open_questions**: [该非对称融合是否能迁移到真实机器人与更多数据集, 能否做序列级条件化以避免AdaLN单token瓶颈]

## Part I：问题与挑战

这篇论文研究的是**语言条件下的机器人模仿学习**：给定语言指令、RGB图像和点云，预测未来一段动作块。  
表面上看是“多模态输入”，但真正难点不是把模态简单堆在一起，而是：

1. **点云和RGB承担的角色并不对称**
   - 点云强在3D几何、相对位置、可操作空间结构。
   - RGB强在语义、纹理、材质、类别识别和全局场景上下文。

2. **现有融合方式经常在错误的位置丢信息**
   - 把2D视觉特征lift到3D点上，容易保不住原图里的**全局上下文**。
   - 直接把RGB token和点云token拼起来，虽然信息都在，但**对齐责任全部交给Transformer**，在小样本模仿学习里很吃数据。

3. **RoboCasa把这个瓶颈放大了**
   - 厨房场景多样，训练/测试场景分离。
   - 每个具体场景只有很少示范，泛化压力大。
   - 任务既有大尺度几何操作（开关门、转杆），也有小尺度精细操作（按钮、抽屉把手、杯子插入）。

**真正的瓶颈**可以概括为一句话：  
**如何在不破坏点云几何主干的前提下，把RGB的全局语义与局部细节稳定注入策略。**

### 输入/输出接口与边界

- **输入**：语言指令 + 多视角RGB + 点云
- **输出**：未来动作块（action chunk）
- **训练范式**：离线模仿学习，扩散式策略
- **评测边界**：主要在 RoboCasa 仿真环境的 24 个 manipulation 任务上验证，不涉及真实机器人部署

### 为什么是现在

因为两个条件同时成熟了：

- **任务侧**：像 RoboCasa 这样的高多样性通用操作基准，已经足以暴露“单模态表示不够”和“粗糙融合失效”的问题。
- **方法侧**：DiT/AdaLN 这类条件调制机制提供了一个新旋钮——不必在输入层硬拼模态，而是能在**每层残差路径**持续注入条件信息。

---

## Part II：方法与洞察

FPV-Net 的设计哲学很明确：

> **让点云负责“在哪里操作”，让RGB负责“在操作什么、局部长什么样”，并且用条件化而不是硬拼接来融合。**

### 1）图像分支：不只拿全局token，还保留局部token

作者没有沿用很多IL工作里“ResNet全局池化出一个token就结束”的做法，而是：

- 用 **FiLM-ResNet** 提取图像特征，并让语言指令参与视觉编码；
- 从特征图里同时取：
  - **global token**：场景级语义；
  - **local tokens**：局部空间/纹理细节；
- 再把这些token送入一个小Transformer，用 **CLS token** 汇聚成后续 AdaLN 所需的条件向量。

这一步解决的是**图像侧的信息瓶颈**：  
如果只保留全局token，小物体、按钮、把手这类精细操作线索会被平均掉。

### 2）点云分支：把几何结构保留成token序列

点云侧不是做全局max-pool，而是：

- 先用 **FPS** 选 256 个中心点；
- 再对每个中心点用 **KNN** 取 32 个邻居，形成局部patch；
- 每个patch编码成一个点云token；
- 编码器可选：
  - 轻量 MLP
  - 预训练 3D 表征模型 **SUGAR**

关键点是：**点云以序列形式进入策略，而不是被压成单向量。**  
这让后续自注意力仍然能直接访问局部几何关系。

### 3）融合策略：作者真正动的“因果旋钮”

他们比较了三种融合方式：

1. **直接拼接（Concat）**  
   RGB、点云、语言 token 一起送进策略。
2. **点云作条件（Cond. on PC）**  
   以RGB+语言为主，点云压成条件向量，通过AdaLN调制。
3. **RGB作条件（Cond. on RGB）**  
   以点云+语言为主，RGB压成条件向量，通过AdaLN调制。

最终最好的是第3种，也就是论文主张的 FPV-Net 形态：

- **主序列**：点云token + 语言token
- **条件信号**：RGB token 汇聚出的 CLS 向量
- **注入方式**：在扩散Transformer（DiT block）的层归一化/残差路径中，用 AdaLN 对主序列做 scale/shift 调制

这意味着 RGB 不再和点云“平级抢注意力”，而是作为**逐层控制信号**影响动作去噪。

### 核心直觉

**what changed**  
从“把RGB并入3D token流里一起学”改成“保留点云token流为主干，让RGB作为条件在每层调制”。

**which bottleneck changed**  
- 点云不再被压缩丢掉几何细节；
- RGB不再只提供一个粗糙全局池化向量，而是由全局+局部token先交互后再汇聚；
- 跨模态对齐不再完全依赖输入层token混合，而是变成了**层级、持续、低维的条件控制**。

**what capability changed**  
- 更擅长同时需要空间精度和语义辨识的任务；
- 对按钮、抽屉把手等细粒度操作更稳；
- 在多任务、少示范、跨场景泛化下更数据高效。

### 为什么这个设计有效

可以用一个简洁的因果链理解：

**把RGB从“主输入流”变成“逐层调制信号”**  
→ **降低跨模态对齐难度，同时保住点云几何序列**  
→ **扩散策略在去噪每一步都能拿到语义/纹理偏置**  
→ **动作生成更稳，特别是细粒度操作和语义判别更强**

另一个非常关键的实验洞察是：

- **当点云被拿去做AdaLN条件时，效果反而更差。**

这恰好反向证明了作者的核心判断：  
**点云是不能轻易被压成单token的主干模态**，因为它的价值就在于局部几何结构。

### 策略层面的取舍

| 方案 | 主干信息流 | 优势 | 主要代价/风险 |
|---|---|---|---|
| RGB-only | 图像token | 语义和纹理强 | 缺显式3D结构，受视角/遮挡影响 |
| PC-only | 点云token | 几何定位强 | 语义弱，难区分材质/类别/细节纹理 |
| PC+RGB Concat | 两种token并列输入 | 实现简单，信息都在 | 对齐负担大，数据效率差 |
| RGB主干 + PC条件 | 点云被压成条件向量 | 形式上对称 | 丢失几何细节，实验上比拼接还差 |
| **FPV-Net** | **点云+语言主干，RGB作AdaLN条件** | **保住3D主干，同时逐层注入2D语义和局部细节** | **依赖点云与RGB同步可用；AdaLN仍需把RGB压成单条件向量** |

---

## Part III：证据与局限

### 关键实验信号

- **比较信号：多模态确实必要，但“怎么融合”比“是否融合”更关键。**  
  单模态结果都不理想：PC-only 平均 28.0%，RGB-only 30.0%。  
  简单拼接到 40.42%，说明模态互补是存在的。  
  但 FPV-SUGAR 到了 **50.5%**，FPV-MLP 也有 **49.42%**，说明真正的提升来自**融合机制本身**。

- **比较信号：相对经典3D IL基线有明显优势。**  
  FPV-SUGAR 高于 DP3 的 22.75%，也高于 BC 的 28.8%。  
  不过对 3DA 的巨大优势需要保守解读，因为作者自己也指出 3DA 可能对训练时长更敏感；因此最强证据其实是**对自身融合变体的控制实验**，而不是单看 3DA。

- **消融信号：非对称融合是核心。**  
  “RGB作条件、点云作主干”优于简单拼接；  
  “点云作条件、RGB作主干”则更差。  
  这说明不是 AdaLN 本身万能，而是**谁做主干、谁做条件**非常关键。

- **消融信号：局部RGB特征真的重要。**  
  按钮、抽屉类任务对局部特征敏感；只用全局token不够。  
  这支持作者“全局语义 + 局部细节都要保留”的设计判断。

- **消融信号：CLS汇聚优于max pooling。**  
  在生成AdaLN条件向量时，用Transformer + CLS优于直接max-pool，说明条件向量本身也需要跨视角/跨局部token整合。

- **补充信号：强预训练有帮助，但不是唯一来源。**  
  SUGAR版本最好，但 MLP版本已经接近，说明性能提升不只是“大模型预训练”带来的，而是融合设计确实有效。

### 1-2个最关键指标

- **RoboCasa 平均成功率**：50.5%（FPV-SUGAR）
- **相对简单融合提升**：50.5% vs 40.42%（PC+RGB concat），提升 **10.08 个点**

### 局限性

- **Fails when:** 在未见物体上的精细 pick-and-place 与 insertion 仍然很难；即便最佳模型，若看具体任务，像部分 counter-to-stove、coffee setup 这类任务的绝对成功率仍然偏低，说明该方法更像是“显著改善”而不是“解决了精细操作泛化”。

- **Assumes:** 需要可获得的点云与多视角RGB，并假设语言指令可用；依赖扩散Transformer、CLIP语言编码，以及可选的SUGAR预训练3D编码器；AdaLN当前还需要把条件序列汇聚成单个向量，这本身是一个信息瓶颈。

- **Not designed for:** 纯RGB部署、无深度/点云的场景；真实机器人零样本迁移；导航任务；超长时程、多阶段任务分解也不是本文重点。

### 复现与可扩展性的现实约束

- 评测只在 **一个仿真基准 RoboCasa** 上完成，因此外部泛化证据仍有限。
- 论文中**未看到代码或项目页链接**，因此可复现性一般。
- 方法的最佳版本依赖预训练模块（CLIP/SUGAR），实际部署时需要处理这些依赖。

### 可复用组件

这篇论文最值得迁移到别的 embodied policy 里的，不是某个具体 backbone，而是以下几个“操作子”：

1. **FPS + KNN 的点云patch token化**
2. **全局token + 局部token 的多尺度图像表示**
3. **CLS汇聚条件向量**
4. **“主干模态保序列，辅助模态做AdaLN条件”的非对称融合范式**

如果以后把它接到更大的VLA/世界模型/更强3D encoder上，最可能保留下来的也是这几个模块。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Towards_Fusing_Point_Cloud_and_Visual_Representations_for_Imitation_Learning.pdf]]