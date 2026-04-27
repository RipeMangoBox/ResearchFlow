---
title: "ZFlow: Gated Appearance Flow-based Virtual Try-on with 3D Priors"
venue: ICCV
year: 2021
tags:
  - Others
  - task/image-based-virtual-try-on
  - appearance-flow
  - convgru
  - densepose-prior
  - dataset/VITON
  - dataset/DeepFashion
  - opensource/no
core_operator: "用ConvGRU门控聚合多尺度候选外观流，并在分割与融合阶段注入DensePose的UV/部位先验，以同时稳定服装形变与人体遮挡深度关系"
primary_logic: |
  目标人物图像 + 商品服装图像 + DensePose几何先验
  → 预测多尺度候选外观流并经门控聚合得到稳定服装warp，同时生成换装后服装分割
  → 在UV/部位先验约束下融合warp服装与人物非服装纹理，输出试穿图
claims:
  - "在 VITON 的论文报告结果中，ZFlow 取得 0.885 SSIM、25.46 PSNR 和 15.17 FID，超过表中其他方法的对应最好值 [evidence: comparison]"
  - "将原始像素级外观流替换为 Gated Appearance Flow 后，warp 服装的 SSIM/PSNR 从 0.835/20.54 提升到 0.871/23.14，最终 try-on 的 FID 从 23.68 降到 18.89 [evidence: ablation]"
  - "把 GAF 嵌入 DIF 的 3D flow 回归后，在 DeepFashion 人体姿态迁移上 SSIM 从 0.778 提升到 0.791、PSNR 从 18.59 提升到 19.26 [evidence: comparison]"
related_work_position:
  extends: "ClothFlow (Han et al. 2019)"
  competes_with: "ClothFlow (Han et al. 2019); ACGPN (Yang et al. 2020)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/ICCV_2021/2021_ZFlow_Gated_Appearance_Flow_based_Virtual_Try_on_with_3D_Priors.pdf
category: Others
---

# ZFlow: Gated Appearance Flow-based Virtual Try-on with 3D Priors

> [!abstract] **Quick Links & TL;DR**
> - **Summary**: 这篇工作把虚拟试衣里最关键的两个误差源——服装 warp 过度自由导致的纹理破坏，以及缺少 3D 几何导致的遮挡/领口错误——放进一个端到端框架里，用门控多尺度 appearance flow 和 DensePose 先验同时解决。
> - **Key Performance**: VITON 上达到 **SSIM 0.885 / PSNR 25.46 / FID 15.17**；用户研究中相对 CP-VTON、SieveNet、ACGPN 的偏好率分别为 **92% / 85% / 71%**。

> [!info] **Agent Summary**
> - **task_path**: 目标人物 RGB 图像 + 单件商品服装图 -> 试穿后人物图像
> - **bottleneck**: 稠密服装形变自由度太高却缺少几何约束，导致 over-warp、遮挡顺序错误、领口/皮肤生成异常与颜色串扰
> - **mechanism_delta**: 用 ConvGRU 门控聚合多尺度候选 appearance flow，并把 DensePose 的 UV 与 body-part 先验同时注入分割和融合阶段
> - **evidence_signal**: VITON 上 FID 达到 15.17，且 GAF 相对 vanilla flow 的消融显著改善 warp 质量与最终 try-on 质量
> - **reusable_ops**: [multi-scale flow pyramid, ConvGRU flow gating]
> - **failure_modes**: [前视上衣分布外样本, DensePose/人体解析误差级联到融合结果]
> - **open_questions**: [能否泛化到全身/多视角试衣, 能否摆脱外部 DensePose 与 parser 先验做端到端学习]

## Part I：问题与挑战

这篇论文针对的是 **image-based virtual try-on**：输入目标人物图像和商品服装图，生成“这个人穿上这件衣服”的结果图。

真正的难点不只是“把衣服贴上去”，而是同时满足两类一致性：

1. **几何一致性**：衣服要跟人物姿态、身体形状、遮挡关系一致。  
   典型失败现象是手臂前后顺序错、领口前后关系错、皮肤显露区域错误。
2. **纹理一致性**：条纹、文字、图案、印花不能被拉坏，也不能在边界处 bleeding。

论文指出，已有方法的瓶颈分成两边：

- **TPS warping**：自由度太低，能处理的形变有限，大变形下容易对不齐。
- **稠密 appearance flow**：自由度太高，若没有足够约束，就容易 over-warp，导致纹理扭曲。
- **只靠服装分割做 fusion**：能提供2D布局，但不能表达 3D 深度顺序，所以领口、脖子、手臂遮挡仍然容易错。

为什么现在值得解这个问题？因为：
- 电商虚拟试衣需求强；
- DensePose 一类工具已经能提供较稠密的人体表面几何先验，使得 2D 试衣流程可以“借到一点 3D 信息”，而不用回到昂贵的 3D 扫描方案。

**输入/输出与边界条件**

| 项目 | 内容 |
|---|---|
| 输入 | 目标人物图像 `Im`、商品服装图 `Ip`，以及由 DensePose / parser 得到的结构先验 |
| 输出 | 试穿结果图 `Itryon` |
| 主要场景 | 单张 2D 图像虚拟试衣，以上衣为主 |
| 数据边界 | VITON：前视女性上衣，分辨率 256×192 |
| 训练现实 | 缺少“同一人穿不同衣服”的真实 triplet，因此必须依赖 garment-agnostic 人体表示 |

## Part II：方法与洞察

ZFlow 是一个端到端框架，但内部仍然遵循“**先对齐衣服，再做融合**”的主线，只是把这两步都加上了更强的结构约束。

### 方法主线

1. **Garment Warping：Gated Appearance Flow**
   - 用 Skip-U-Net 在多个解码层预测多尺度候选 flow。
   - 这些 flow 不直接平均，而是通过 **ConvGRU 门控聚合** 成最终的 `fagg`。
   - 直觉上，相当于为每个像素从不同尺度的形变候选里“择优”而不是“放任”。

2. **Conditional Segmentation**
   - 先预测换装后的服装分割 `Mexp`。
   - 输入不是简单的人体 agnostic 表示，而是扩展后的 **Dense Garment-Agnostic Representation (DGAR)**：传统 body shape / pose / head 之外，再加 body-part segmentation。
   - 这一步的作用是先把“换装后大概哪些区域属于衣服/皮肤/背景”确定下来。

3. **Segmentation-Assisted Dense Fusion**
   - 把 warped garment、预测分割 `Mexp`、人物非服装区域纹理，以及 **IUV priors**（UV map + body-part segmentation）一起喂给融合网络。
   - 网络输出最终 try-on 图，并额外重建分割与 UV/body-part 先验，等于迫使融合阶段持续尊重几何信息。

4. **End-to-end fine-tuning**
   - 前面各模块先 warm-up，再整体联合优化。
   - 这样上游 warp 不再只为“局部对齐”负责，而是直接服务于最终视觉质量。

### 核心直觉

- **从“直接回归高自由度稠密 flow”变成“多尺度候选 flow + 门控选择”**  
  → 改变的是形变空间的约束方式：不是放任每个像素随意移动，而是在层级候选里做受控选择  
  → 结果是减少 over-warp，保住条纹、文字、印花等细粒度纹理。

- **从“2D 分割条件”变成“2D 分割 + DensePose 3D proxy”**  
  → 改变的是融合阶段的信息瓶颈：网络不必只靠 RGB 猜前后遮挡和人体表面位置  
  → 结果是领口、手臂、脖子、露肤区域的深度关系更稳定。

- **从“局部分阶段最优”变成“最终图像质量驱动的联合训练”**  
  → 改变的是模块间误差传递方式  
  → 结果是 warp 与 fusion 更协调，最终 FID 进一步下降。

更因果地说，这个设计之所以有效，是因为它分别对准了两个错误来源：
- **GAF** 处理的是“形变自由度过大”的问题；
- **Dense geometric priors** 处理的是“几何/遮挡信息缺失”的问题。  
两者分别约束变形与合成，因此最终提升是叠加而不是互相替代。

| 设计选择 | 改变了什么约束/信息 | 带来的能力变化 | 代价/风险 |
|---|---|---|---|
| TPS / vanilla flow → GAF | 将像素级 warp 限制在多尺度候选并门控聚合 | 大形变下更稳，纹理更少扭曲 | 训练更复杂，需要平滑正则 |
| 普通 agnostic prior → DGAR | 增加 body-part 结构信息 | 条件分割更懂人体布局与遮挡 | 依赖外部解析质量 |
| 仅分割条件融合 → IUV priors 融合 | 补充 UV 和部位级 3D proxy | 领口、手臂前后、露肤更合理 | DensePose 错误会传导 |
| 分模块训练 → 端到端微调 | 上下游目标对齐 | 最终图像质量继续提升 | 训练成本更高 |

## Part III：证据与局限

### 关键证据信号

- **比较信号：整体性能跳变明确**  
  在 VITON 上，ZFlow 达到 **SSIM 0.885 / PSNR 25.46 / FID 15.17**，显著优于表中已有方法，尤其相对 ClothFlow 的 FID 从 **23.68** 降到 **15.17**。  
  这说明提升不是只在局部视觉案例上，而是在主 benchmark 上有稳定收益。

- **消融信号：核心增益主要来自 GAF + 几何先验 + 联合训练**
  - 用 **GAF 替代 vanilla dense flow** 后，warp garment 的 SSIM/PSNR 从 **0.835/20.54** 提升到 **0.871/23.14**，说明“先把 warp 做稳”本身就是关键。
  - 在 fusion 里加入 **Ledge + Lrecon(IUV priors)**，再做 **end-to-end fine-tuning**，最终 try-on 从中间版本继续降 FID 到 **15.17**。  
  这说明几何先验不是装饰性输入，而是实际改变了融合阶段的判别依据。

- **人类偏好信号：主观质量也成立**  
  用户研究里，ZFlow 相对 CP-VTON / SieveNet / ACGPN 的偏好率分别是 **92% / 85% / 71%**。  
  这对“领口、遮挡、皮肤、颜色 bleeding”这类自动指标不完全覆盖的视觉问题尤其重要。

- **跨任务信号：GAF 不是只对 try-on 有效**  
  把 GAF 换进 DIF 的 pose transfer flow regression 后，DeepFashion 上 SSIM/PSNR 从 **0.778/18.59** 升到 **0.791/19.26**。  
  这表明 GAF 可视作一个更通用的 dense flow regularization operator。

### 局限性

- **Fails when**: 遇到前视上衣分布之外的样本（如多视角、全身、下装、超宽松服装、极端遮挡）时，论文没有给出证据说明还能稳定工作；DensePose 或人体解析出错时，前后关系和边界也可能一起错。
- **Assumes**: 需要可用的商品服装图与服装 mask；依赖预训练的 DensePose 和 human parser 生成 UV / body-part / segmentation 先验；训练主要在 VITON 这类受控数据上完成；论文文本未给出代码链接，复现需自行实现。
- **Not designed for**: 真实物理试穿、任意视角 3D 服装模拟、视频连续试衣、移动端实时部署。

### 可复用组件

- **Gated Appearance Flow**：可作为任何 dense flow 预测任务的“层级候选 + 门控聚合”模块，论文已在 pose transfer 上验证。
- **IUV prior reconstruction 约束**：适合插入其他人像编辑/人像合成任务，在 fusion 阶段强化人体几何一致性。

## Local PDF reference

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/ICCV_2021/2021_ZFlow_Gated_Appearance_Flow_based_Virtual_Try_on_with_3D_Priors.pdf]]