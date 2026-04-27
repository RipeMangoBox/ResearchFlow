---
title: "CIT: Cloth Interactive Transformer for Virtual Try-On"
venue: CVPR
year: 2022
tags:
  - Others
  - task/virtual-try-on
  - transformer
  - cross-attention
  - thin-plate-spline
  - dataset/VITON
  - opensource/full
core_operator: "在几何匹配与合成两阶段分别用跨模态 Transformer 显式建模人物、衣物与掩码的长程相关，并据此指导 TPS 变形与掩码融合。"
primary_logic: |
  无衣着人物表示 + 商品衣物 → 双向跨注意力匹配得到人衣相关图并回归 TPS 形变 → 生成变形衣物/掩码 → 人物表示、变形衣物、变形掩码做三路交互推理 → 引导掩码组合与渲染，输出最终试穿图
claims:
  - "在 VITON 的 paired retry-on 设置中，CIT 相比 CP-VTON+ 将 SSIM 从 0.817 提升到 0.827，并将 LPIPS 从 0.117 降到 0.115 [evidence: comparison]"
  - "仅替换为 CIT matching block 时，JS 从 0.812 降到 0.800，但 FID 从 25.19 大幅降到 14.76，说明更好的最终写实度并不等价于更高的掩码重叠分数 [evidence: ablation]"
  - "在用户研究中，CIT 在照片真实感与服饰细节保真上分别获得 32.1% 和 35.4% 的最高选择率，超过 CP-VTON、CP-VTON+ 和 ACGPN [evidence: comparison]"
related_work_position:
  extends: "CP-VTON+ (Minar et al. 2020)"
  competes_with: "CP-VTON+ (Minar et al. 2020); ACGPN (Yang et al. 2020)"
  complementary_to: "ZFlow (Chopra et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2022/2022_CIT_Cloth_Interactive_Transformer_for_Virtual_Try_On.pdf
category: Others
---

# CIT: Cloth Interactive Transformer for Virtual Try-On

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2104.05519), [Code](https://github.com/Amazingren/CIT)
> - **Summary**: CIT 保留经典两阶段虚拟试衣框架，但把“局部 CNN 对齐 + 粗拼接合成”改成了双向/三向跨模态 Transformer 交互，因此能更好地对齐人和衣的长程对应，生成更自然的衣物变形与更真实的最终试穿图。
> - **Key Performance**: VITON retry-on 上 SSIM 0.827、LPIPS 0.115；unpaired try-on 上 KID 0.761，用户研究真实感偏好最高。

> [!info] **Agent Summary**
> - **task_path**: 无衣着人物表示 + 商品服饰图 → 变形服饰/掩码 → 最终试穿人像
> - **bottleneck**: 传统两阶段 VTON 主要靠 CNN 或直接拼接隐式建模人衣关系，难以捕捉大位移下的纹理-形状对应，导致变形错位和合成伪影
> - **mechanism_delta**: 在 TPS 之前加入双向人衣 cross-attention，在合成之前加入人物/变形衣物/变形掩码三路交互注意力
> - **evidence_signal**: 相比 CP-VTON+，CIT 在 SSIM、LPIPS、KID 和用户偏好上更优，且消融证明交互块比单纯强化掩码监督更关键
> - **reusable_ops**: [双向人衣跨注意力匹配, 三模态推理引导掩码融合]
> - **failure_modes**: [服饰版型差异过大, 自遮挡严重, 姿态或视角差异过大]
> - **open_questions**: [如何扩展到更多服饰品类与视角, 如何同时优化纹理真实感与形状对齐指标]

## Part I：问题与挑战

这篇论文解决的是 **2D image-based virtual try-on**：输入一张人物图和一张商品服饰图，输出“这个人穿上这件衣服”的结果图。

### 1) 真正的问题不只是“生成图像”，而是“建立正确的人衣对应”
在虚拟试衣里，模型实际上要做两件事：

1. **几何匹配**：从商品服饰图中，决定“哪些像素应该被采样、拉伸、搬运到人体的哪些位置”。
2. **外观合成**：决定“哪些区域应保留人物原图，哪些区域应用变形后的衣物覆盖”。

难点在于：衣物是**非刚体**，而人物的姿态、体型、遮挡和衣物纹理又会显著变化。  
所以，真正瓶颈不是单纯的图像生成能力，而是 **人物布局与衣物外观之间的长程、结构化对应关系**。

### 2) 现有两阶段方法的两个核心短板
作者认为此前方法主要有两类不足：

- **人和衣的互相关系建模不够显式**  
  许多方法把人物表示和商品衣物分别编码，再靠后续模块隐式对齐，容易出现 warp 后衣服位置不对、图案拉坏、logo 变形等问题。

- **CNN 不擅长长程依赖**  
  VTON 需要跨较大空间范围建立对应，例如袖口、领口、条纹、logo 的远距离关联。但纯卷积更偏局部，难以稳定建模这种全局对应。

### 3) 输入/输出接口与边界条件
- **输入**：
  - 人物图像 \(I\)
  - 商品服饰图 \(c\)
  - 人物被处理成 cloth-agnostic representation：姿态、身体形状、保留身份区域（脸/头发/下半身）
- **输出**：
  - 第一阶段：变形后的衣物 \(\hat c\) 与其 mask
  - 第二阶段：最终试穿结果图 \(I_o\)

### 4) 为什么现在值得做
这类任务恰好适合 Transformer 的强项：  
**显式全局建模 + 内容相关的跨位置交互**。  
作者的判断是：如果能在两阶段中的关键接口处，把“人-衣”“人-衣-掩码”关系显式建出来，就能比纯 CNN 更自然地处理复杂纹理和大形变。

---

## Part II：方法与洞察

CIT 的设计很克制：**不推翻经典两阶段框架**，而是在最关键的两个位置插入交互式 Transformer。

### 方法主线

#### 阶段一：CIT Matching Block
目标：让商品服饰先被 **更合理地 warp 到人体上**。

做法不是只看衣服本身，而是把：
- 人物特征 \(X_p\)
- 商品衣物特征 \(X_c\)

一起送进一个 **Interactive Transformer I**：

1. 各自先做 self-attention，得到各自的全局上下文。
2. 再做双向 cross-attention：
   - 人物查询衣物
   - 衣物查询人物
3. 得到更强的人衣相关表示后，生成相关图并回归 TPS 参数。
4. 用 TPS 去 warp 原始衣物和对应 mask。

**作用**：把“warp 该怎么做”从局部匹配，变成“由人体姿态/体型约束下的全局人衣对应”。

#### 阶段二：CIT Reasoning Block
目标：让最终合成更自然，而不是把 warp 后衣服粗暴贴上去。

关键改变是：**不把人物、变形衣物、变形掩码直接拼成一个输入**，而是把三者作为三种不同语义角色分别处理：

- 人物表示：提供身份与姿态
- 变形衣物：提供纹理与外观
- 变形掩码：提供覆盖区域与边界

作者用 **Interactive Transformer II** 对这三路输入做：
- 各自 self-attention
- 任意两两之间的 cross-attention

再用得到的结果去引导 mask composition 和渲染图激活，最后得到试穿结果。

### 核心直觉

#### 发生了什么改变
把以往的：

- “局部卷积提特征”
- “后面再隐式对齐”
- “三路输入直接拼接”

改成了：

- **先对每一路做全局建模**
- **再通过 cross-attention 显式交换信息**
- **让相关图直接服务于 warp 和 fusion**

#### 改变了哪个瓶颈
它改变的是 **信息流的组织方式**：

- 以前：人和衣的对应关系是隐式的、局部传播的
- 现在：人和衣、人与掩码、衣与掩码的关系是显式的、全局交互的

也就是把原先的 **局部感受野瓶颈 + 模态混杂瓶颈**，变成了 **可控的跨模态对应建模**。

#### 能力上带来什么变化
- 第一阶段：warp 更像“按人体约束重排衣物纹理”，而不是“机械拉伸”
- 第二阶段：模型更容易判断
  - 哪些区域该被衣物覆盖
  - 哪些区域要保留人物原图
  - 哪些边界需要清晰过渡

所以结果表现为：
- 纹理保真更好
- 条纹/logo 更不容易被拉坏
- 边界更自然
- 合成伪影更少

#### 为什么这在因果上有效
因为衣物本身是非刚体，**只看衣物很难知道该往哪里变**；  
而人体姿态、体型和遮挡恰好提供了 warp 的约束。  
同理，在合成阶段，如果把“纹理信息”和“覆盖区域信息”混成一个输入，模型容易边界糊、局部错贴；分开建模后，角色更清晰，推理也更稳定。

### 战略取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
| --- | --- | --- | --- |
| 阶段一加入双向人衣 cross-attention | 纯 CNN 难以建模长程 warp 对应 | 复杂纹理、logo、条纹的 warp 更自然 | 计算量上升，仍受 TPS 表达能力限制 |
| 阶段二把人物/衣物/mask 分开推理 | 粗拼接导致角色混淆 | 覆盖边界更清晰，纹理与形状职责分离 | 更依赖 mask 与预处理质量 |
| 保留经典两阶段框架 | 直接端到端生成难训练 | 与已有 VTON 管线兼容、训练更稳 | 第一阶段误差会传给第二阶段 |
| 继续使用 TPS warp | 保留可解释的几何变形接口 | 容易落地到现有方法 | 面对极端姿态/视角差异时仍可能不足 |

---

## Part III：证据与局限

### 关键证据

#### 1) 比较实验信号：相对 CP-VTON+，最终图像质量更稳
在 VITON 的 paired retry-on 设置上，CIT 相比最直接基线 CP-VTON+：
- **SSIM**：0.817 → **0.827**
- **LPIPS**：0.117 → **0.115**

这说明 CIT 至少在“结构相似性”和“感知相似性”上都更优，且改动不是只在视觉样例里成立。

#### 2) 非配对场景信号：整体真实感改善
在 unpaired try-on 上，CIT 的：
- **KID = 0.761**，为表中最佳
- FID 也明显优于 CP-VTON+（13.97 vs 25.19）

这支持一个关键结论：  
CIT 不只是把 paired setting 拟合得更像 GT，也在更难的真实 try-on 场景下提升了整体分布质量。

#### 3) 消融信号：真正有效的是“交互建模”，不是“更强的掩码约束”
论文一个很有价值的发现是：

- 只加 **matching block**，JS 下降，但 FID 大幅改善；
- 完整模型 **B3** 的用户偏好明显高于 **B4**；
- 而 **B4** 虽然加入更严格的 mask L1 约束，JS/SSIM/LPIPS 反而更高，但视觉上更差。

这说明在 VTON 里，**只看 mask overlap（JS/IoU）会误判真实质量**。  
也就是：形状对齐更好，不代表纹理更真、结果更自然。

#### 4) 用户研究信号：主观质量确实更强
30 人用户研究中，CIT 在两项问题上都拿到最高偏好：
- 照片真实感：**32.1%**
- 服饰细节保真：**35.4%**

这对 VTON 很重要，因为该任务的核心目标本来就包含“看起来像真的”。

### 1-2 个最值得记住的结果
- **SSIM 0.827 / LPIPS 0.115**：说明 paired retry-on 下整体质量强。
- **KID 0.761 + 用户研究第一**：说明生成分布与主观真实感都更有说服力。

### 局限性

- **Fails when**: 目标衣物与人物当前穿着在版型/覆盖区域上差异过大；人物存在明显自遮挡；人物姿态或视角与商品衣物差异太大时，结果会模糊或错配。
- **Assumes**: 依赖较好的姿态、身体形状和分割式预处理；训练与评估主要基于 VITON 的前视女性上装数据；仍建立在 TPS warp + 两阶段管线之上；实验只在单一公开数据集上完成，因此证据强度应保守看待。
- **Not designed for**: 多视角/3D 一致试衣、极端视角变化、任意服饰类别泛化、生产级高分辨率全身试衣系统。

### 复现与可扩展性备注
- **开源性**：代码和模型已公开，复现门槛相对可接受。
- **依赖项**：需要标准 VTON 预处理链路（姿态、人体解析/掩码等）。
- **训练代价**：两阶段各训练 200K steps，输入分辨率 256×192，属于中等规模研究设定。
- **比较注意事项**：论文自己也提到 ACGPN 的测试集与其它方法不完全一致，因此跨方法数值对比并非处处完全同协议。

### 可复用组件
1. **双向跨模态匹配块**：适合任何“非刚体外观与目标结构对齐”的任务。
2. **三模态推理块**：适合“外观 + 区域 + 主体”三路信息联合决策的融合任务。
3. **评价洞察**：VTON 不应只看 JS/IoU，形状指标需要与纹理/真实感指标联合看。

## Local PDF reference
![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2022/2022_CIT_Cloth_Interactive_Transformer_for_Virtual_Try_On.pdf]]