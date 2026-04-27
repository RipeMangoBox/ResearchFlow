---
title: "WAS-VTON: Warping Architecture Search for Virtual Try-on Network"
venue: ACM MM
year: 2021
tags:
  - Others
  - task/virtual-try-on
  - neural-architecture-search
  - flow-based-warping
  - dynamic-skip-connections
  - dataset/VITON
  - opensource/full
core_operator: 通过单路径NAS分别搜索服饰类别特异的flow warping架构与跨尺度skip融合网络，以提升服装对齐和试穿合成自然度
primary_logic: |
  人物图像/姿态热图/人体解析先验/平铺服装 → PPP模块预测待替换区域的目标解析与服装mask → NAS-Warping在双层搜索空间中为不同服饰类别搜索flow warping结构并完成服装形变 → NAS-Fusion搜索跨尺度skip连接并融合warped clothing与未改动人体区域 → 输出更自然的虚拟试穿图像
claims:
  - "在扩展VITON上，WAS-VTON的最终试穿结果达到SSIM 0.8430和FID 13.83，优于ACGPN的0.8357和23.49 [evidence: comparison]"
  - "去掉NAS-Fusion中的动态skip连接后，最终试穿FID从13.83恶化到16.79，说明跨尺度特征传递对服装-人体无缝融合有效 [evidence: ablation]"
  - "按服饰类别搜索的NAS-Warping优于每层固定1/2/3个warping block的基线，并且不同类别搜得架构不同，支持类别特异warping的必要性 [evidence: ablation]"
related_work_position:
  extends: "ClothFlow (Han et al. 2019)"
  competes_with: "ACGPN (Yang et al. 2020); CP-VTON (Wang et al. 2018)"
  complementary_to: "Graphonomy (Gong et al. 2019); OpenPose (Cao et al. 2019)"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/arXiv_2021/2021_WAS_VTON_Warping_Architecture_Search_for_Virtual_Try_on_Network.pdf
category: Others
---

# WAS-VTON: Warping Architecture Search for Virtual Try-on Network

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2108.00386) · [DOI](https://doi.org/10.1145/3474085.3475490) · [Code](https://github.com/xiezhy6/WAS-VTON)
> - **Summary**: 这篇工作把虚拟试穿中的“服装形变网络是否应当对所有服饰共享”变成一个可搜索问题：为不同服饰类别自动寻找不同 warping 架构，并进一步搜索更适合试穿任务的融合网络，从而提升服装对齐质量与最终合成自然度。
> - **Key Performance**: 扩展 VITON 上最终试穿 SSIM 0.8430 / FID 13.83；用户偏好评测中 63.07% 的结果被认为最真实。

> [!info] **Agent Summary**
> - **task_path**: 人物图像 + 平铺服装 + 姿态/解析先验 -> 虚拟试穿结果图像
> - **bottleneck**: 现有 try-on 方法通常用固定 warping 架构处理所有服饰类别，无法匹配不同类别间显著不同的形变复杂度；同时固定 U-Net skip 连接限制了最终融合质量
> - **mechanism_delta**: 将试穿系统里最关键的 warping 与 fusion 两个子网都改为可搜索结构，其中 warping 按服饰类别独立搜索，fusion 搜索跨尺度 skip 连接
> - **evidence_signal**: 扩展 VITON 上相对 ACGPN 将最终 FID 从 23.49 降到 13.83，且动态 skip 消融把 FID 从 16.79 降到 13.83
> - **reusable_ops**: [category-specific warping search, cross-scale dynamic skip fusion]
> - **failure_modes**: [parsing或pose错误会级联破坏warping与fusion, 未见服饰类别或大视角偏移时类别特异架构可能失配]
> - **open_questions**: [能否摆脱对人体解析与预训练GMM的依赖, 类别特异搜索能否扩展到开放服饰taxonomy与更高分辨率]

## Part I：问题与挑战

这篇论文解决的是**图像级虚拟试穿**里的核心几何问题：给定目标人物和一件平铺服装，如何把服装**准确变形到目标人体**，再与未被替换的人体区域**无缝融合**，生成自然的试穿图。

### 真正的瓶颈是什么？

作者认为，过去方法的问题不只是 TPS 自由度不够，或者 flow 估计不够准；更深一层的瓶颈是：

- **所有服饰类别共用同一个 warping 架构**；
- 但不同服饰的形变复杂度差异很大：
  - 背心/吊带更接近仿射或较简单变形；
  - 长袖、裤子往往包含更强的局部非刚性形变与遮挡；
- 因此，共享架构会把本应分开的形变模式压进一个统一的、局部纠缠的变形子空间里，导致：
  - 领口、袖口、裤腿边界不自然；
  - 遮挡时纹理和轮廓更容易错位；
  - 下装 try-on 更难做稳。

### 为什么现在值得解决？

因为两件事同时成熟了：

1. **flow-based warping** 已经比 TPS 更有表达力，说明“几何自由度”不是唯一短板；
2. **one-shot NAS** 让“为任务搜结构”变得可行，不必手工为每一类服装设计不同网络。

所以这篇论文的切入点很明确：**不是继续堆固定架构，而是把架构本身变成可学习对象。**

### 输入/输出接口与边界条件

- **输入**：
  - 人物图像
  - 平铺服装图
  - 姿态热图
  - 人体解析/身体轮廓先验
- **输出**：
  - 目标人物穿上指定服装后的试穿图像

### 适用边界

该方法默认的是：

- 前视图、图像级 try-on；
- 依赖人体 parsing、pose estimator 和服装 mask；
- 主要在 VITON 及其扩展数据设置上验证；
- 目标是 2D 图像合成，不是 3D 服装物理模拟。

---

## Part II：方法与洞察

整体框架由三个模块组成：

1. **PPP 模块**：先预测要被替换区域的局部人体解析；
2. **NAS-Warping 模块**：为不同服饰类别搜索专门的 warping 网络；
3. **NAS-Fusion 模块**：搜索更适合 try-on 的融合网络结构。

### 方法拆解

#### 1. PPP：先给出“目标衣服应该落在哪种人体语义区域”

PPP（Partial Parsing Prediction）先预测待替换区域的局部 parsing，给后续两个模块提供先验：

- 给 warping 提供**目标服装 mask**
- 给 fusion 提供**衣物/皮肤/保留区域**的边界指导

这里作者复用了 CP-VTON 的 GMM/TPS 作为辅助形状先验，但 PPP 自己的目标不是做最终高保真变形，而是先把“该变哪一块、目标区域大致长什么样”说清楚。

#### 2. NAS-Warping：把“服装对齐网络结构”也纳入搜索

这是论文的核心。

作者为 warping 设计了一个**双层搜索空间**：

- **网络层级搜索**：每个 multi-scale warping cell 里，用 1/2/3 个 warping block，控制变形复杂度；
- **算子层级搜索**：每个 block 里再选卷积操作，如 1×1、3×3、depthwise-separable 等。

直观上，这相当于同时决定：

- 这类服装需要**几步渐进形变**
- 每一步需要多大的**感受野与局部建模方式**

搜索出来的网络不是“一套通吃”，而是**按服饰类别单独搜索**。这正对论文的核心假设：**不同类别需要不同 warping 架构。**

#### 3. NAS-Fusion：不再默认 U-Net 的固定同尺度 skip

作者认为，最终试穿质量不仅取决于衣服 warp 得好不好，还取决于**融合时信息怎么走**。

所以 fusion 模块不再固定成标准 U-Net，而是继续搜索：

- decoder 每层要接 encoder 的哪一层特征；
- 可以是**同尺度**，也可以是**相邻尺度**；
- 同时搜索上下采样层的卷积操作。

这让网络能更灵活地混合：

- 高分辨率几何边界信息
- 低分辨率语义上下文信息

从而更好地把 warped clothing 与未改动的人体区域接起来。

#### 4. 搜索策略：用 single-path one-shot 降低 NAS 成本

作者没有用耦合很强的 differentiable NAS，而是采用更稳的：

- **single-path one-shot** 训练 supernet
- 再用**evolutionary search** 选最终结构
- 评价指标用 SSIM：
  - warping 搜索时，看 warped cloth 与目标衣物区域的一致性
  - fusion 搜索时，看合成图与真实人物图的一致性

其中：

- **warping 搜索按服饰类别分别做**
- **fusion 搜索在全部类别上统一做**，因为融合模块被假设为类别无关

### 核心直觉

这篇工作的关键不是“NAS 用在 try-on 上”这么表层，而是它改变了系统中的两个硬约束。

#### what changed → which bottleneck changed → what capability changed

1. **从共享 warping 架构，改成按类别搜索 warping 架构**
   - 改变的约束：不再强迫所有服饰共用同一种形变深度与卷积模式
   - 改变的瓶颈：释放了类别间不同的形变子空间
   - 带来的能力：长袖、裤装、遮挡区域的对齐更稳，边界更自然

2. **从固定 U-Net skip，改成跨尺度动态 skip 搜索**
   - 改变的约束：不再只能同尺度传信息
   - 改变的瓶颈：改善了几何细节与语义上下文之间的信息路由
   - 带来的能力：服装边缘、领口、人体保留区域融合更干净

3. **先预测局部 parsing，再做 warping/fusion**
   - 改变的约束：后续模块不再盲猜目标衣物区域
   - 改变的瓶颈：减少了目标形状不明确带来的几何歧义
   - 带来的能力：在遮挡和全身 try-on 场景下更稳

#### 为什么这个设计有效？

因为虚拟试穿本质上不是单纯纹理迁移，而是一个**受人体语义与服饰类别共同约束的条件几何变形问题**。  
作者把这个问题拆成三层：

- PPP 先约束**该变什么**
- NAS-Warping 再决定**怎么变**
- NAS-Fusion 最后决定**怎么接回去**

这样，系统的自由度不再都压在一个固定 warping/fusion backbone 上。

### 战略权衡

| 设计选择 | 放松了什么约束 | 带来的收益 | 代价/风险 |
| --- | --- | --- | --- |
| 按服饰类别搜索 warping | 不再共享同一形变架构 | 更贴合类别特异形变 | 需要为不同类别维护不同子网 |
| 双层搜索空间（块数+算子） | 同时搜索形变深度与感受野 | 多尺度对齐更灵活 | 搜索空间更大、训练更久 |
| 动态跨尺度 skip fusion | 不再局限于标准 U-Net 同尺度连接 | 更好融合几何与语义 | 结构更复杂，解释性稍弱 |
| PPP 解析先验 | 不再让 warping/fusion 直接盲对齐 | 遮挡与边界场景更稳 | 强依赖 parsing 与姿态质量 |
| single-path one-shot NAS | 降低 NAS 计算与不稳定性 | 搜索更实用 | 仍需数天级 supernet 训练 |

---

## Part III：证据与局限

### 关键证据

#### 1. 与 SOTA 的主结果比较：能力跳跃主要体现在最终真实感
在扩展 VITON 上，WAS-VTON 的最终试穿结果达到：

- **SSIM 0.8430**
- **FID 13.83**

对比最强基线 ACGPN：

- ACGPN 为 **SSIM 0.8357 / FID 23.49**

这里最有说服力的信号是 **FID 大幅下降**，说明提升不仅是像素对齐，而是整体感知真实性更强。

#### 2. 用户研究支持“看起来更自然”
用户评测里，**63.07%** 的结果被选为最真实。  
这说明它不是只在自动指标上占优，也确实改善了人眼最敏感的区域：服装轮廓、领口、遮挡边界。

#### 3. 消融实验说明“架构搜索”不是装饰
最关键的两个消融信号：

- **NAS-Fusion 动态 skip**：
  - 去掉后 FID 从 **13.83** 变差到 **16.79**
  - 说明跨尺度特征路由确实影响最终融合质量
- **NAS-Warping vs 固定 block 数**：
  - 搜索得到的架构优于固定 1/2/3 block 的手工基线
  - 说明“类别特异 warping”比“统一固定复杂度”更合理

此外，作者还比较了搜索范式：

- **single-path one-shot** 明显优于 DARTS 式搜索
- 说明在这类生成/配准任务里，更稳定的搜索流程更重要

#### 4. 结构分析与论文假设一致
作者展示了不同服饰类别搜到的 warping 结构不同，并观察到：

- 低分辨率层更偏向小卷积核；
- 高分辨率层更偏向较大卷积核与 depthwise-separable conv；

这支持了一个很重要的结论：**try-on 的最优结构具有尺度依赖性与类别依赖性，固定 backbone 不是最优选择。**

### 局限性

- **Fails when**: parsing 或 pose 预测错误、强自遮挡、非前视角、未见服饰类别、极端宽松/透明/复杂层叠服装时，warping 与 fusion 都可能失败；类别特异搜索也可能在开放类别场景中失配。
- **Assumes**: 依赖 VITON 风格的配对训练数据、人体解析、姿态热图、服装 mask，以及预训练的 GMM/解析器；NAS 训练和搜索本身需要数天级 GPU 计算（文中 supernet 训练约 3 天 + 2.5 天，搜索还需数小时到十余小时）。
- **Not designed for**: 3D 物理布料模拟、任意视角试穿、无解析/无姿态的 parser-free setting、多件服装强交互、开放世界服饰 taxonomy 的统一建模。

### 可复用组件

- **类别特异的 warping 搜索范式**：适合任何“不同子类别形变复杂度差异很大”的图像配准任务。
- **跨尺度动态 skip 搜索**：可迁移到需要精细边界与语义一致性的图像合成任务。
- **PPP 先验分解思路**：先预测目标区域语义，再做几何变形与融合，有利于降低后续模块的歧义。

## Local PDF reference

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/arXiv_2021/2021_WAS_VTON_Warping_Architecture_Search_for_Virtual_Try_on_Network.pdf]]