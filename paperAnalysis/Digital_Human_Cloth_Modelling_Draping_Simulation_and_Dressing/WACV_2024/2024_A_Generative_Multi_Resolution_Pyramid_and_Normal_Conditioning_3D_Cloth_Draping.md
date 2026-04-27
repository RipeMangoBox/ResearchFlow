---
title: "A Generative Multi-Resolution Pyramid and Normal-Conditioning 3D Cloth Draping"
venue: WACV
year: 2024
tags:
  - Others
  - task/3d-cloth-draping
  - variational-autoencoder
  - multi-resolution-pyramid
  - normal-conditioning
  - dataset/CLOTH3D
  - dataset/CAPE
  - opensource/full
core_operator: "在 canonical UV 空间中先生成法线条件，再用多分辨率条件 VAE 从低频到高频逐级补充服装几何，最后回到 3D 并蒙皮得到悬垂结果。"
primary_logic: |
  SMPL姿态/形状参数 + 模板服装 → 去全局旋转、unpose/unshape并投影到UV空间 → 先采样法线UV图，再由金字塔条件VAE逐层生成服装坐标/残差 → 回投3D、重塑并蒙皮得到最终悬垂服装
claims:
  - "在 CLOTH3D 上，端到端金字塔 + 正则化把 combined V2V 从单层基线的 8.01 mm 降到 3.95 mm [evidence: ablation]"
  - "在论文报告的 CLOTH3D 协议下，该方法取得 3.95 mm V2V，优于 DeepCloth 的 19.23 mm 与 DeePSD 的 23.78 mm [evidence: comparison]"
  - "在 CAPE 上，第三层金字塔分别达到 female 3.11 mm、male 3.66 mm 的 V2V，优于作者引用的 CAPE 基线 3.61 mm 与 6.15 mm [evidence: comparison]"
related_work_position:
  extends: "CAPE (Ma et al. 2020)"
  competes_with: "CAPE (Ma et al. 2020); DeePSD (Bertiche et al. 2021)"
  complementary_to: "PBNS (Bertiche et al. 2021); TemporalUV (Xie et al. 2022)"
evidence_strength: strong
pdf_ref: "paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/WACV_2024/2024_A_Generative_Multi_Resolution_Pyramid_and_Normal_Conditioning_3D_Cloth_Draping.pdf"
category: Others
---

# A Generative Multi-Resolution Pyramid and Normal-Conditioning 3D Cloth Draping

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.02700), [Code](https://github.com/HunorLaczko/pyramid-drape)
> - **Summary**: 这篇工作把 3D 服装悬垂从“一次性高分辨率坐标回归”改写成 canonical UV 空间中的“法线引导 + 多分辨率残差生成”，从而同时提升随机性、褶皱细节和跨服装泛化。
> - **Key Performance**: CLOTH3D 上 V2V = 3.95 mm；CAPE 上最佳 V2V = 3.11 mm（female）/ 3.66 mm（male）。

> [!info] **Agent Summary**
> - **task_path**: SMPL 姿态/形状 + 模板服装 -> 悬垂后的 3D 服装网格
> - **bottleneck**: 服装悬垂是 one-to-many 问题，且高频褶皱难以通过单次 3D 坐标回归稳定学习，容易被平均化或过拟合到服装类别/姿态
> - **mechanism_delta**: 将直接 3D 回归改为 canonical UV 空间中的多层条件 VAE 逐级生成，并用可采样 normal UV 作为中间几何条件
> - **evidence_signal**: 两个公开数据集上的显著对比优势，加上金字塔训练方式与正则化消融共同支持核心设计
> - **reusable_ops**: [canonical-space unpose-unshape, UV-space pyramid residual generation]
> - **failure_modes**: [依赖高质量 garment-to-SMPL registration, 不支持多层或复杂附件服装]
> - **open_questions**: [能否学习 garment-specific blend weights 替代固定 SMPL 权重, 能否加入时序约束扩展到动态连续模拟]

## Part I：问题与挑战

这篇文章要解决的，不是“给定姿态回归一件衣服”这么简单，而是更难的版本：

1. **同一 pose + 同一模板服装，对应很多合理悬垂结果**，所以本质上是 one-to-many；
2. **细节主要体现在高频褶皱**，而 CNN/回归模型往往先学低频轮廓，容易把细节抹平；
3. **服装拓扑和类别差异大**，很多方法需要按单件衣服或单类衣服单独训练，泛化差；
4. **直接在 mesh / point / implicit 空间做生成**，实现复杂、成本高、对未见姿态/服装不稳。

### 真正瓶颈
真正瓶颈不是“模型不够大”，而是**表示空间和优化路径不对**：  
如果直接在 posed 3D 空间一次性预测全分辨率服装坐标，模型必须同时处理姿态变化、身体形状变化、拓扑差异和高频褶皱，结果往往收敛到“平均服装”。

### 为什么现在值得解决
作者抓住了两个成熟条件：

- **UV map** 让 3D 服装能借用高效的 2D CNN 生成范式；
- **多分辨率生成** 很适合把服装的低频形状和高频细节拆开学。

### 输入/输出与边界条件
- **输入**：SMPL pose/shape 参数 + canonical 模板服装
- **输出**：与目标姿态匹配的 3D draped garment mesh
- **边界条件**：
  - 服装尺寸已与身体匹配
  - 服装默认**低弹性**
  - 只考虑**单层表面**服装
  - 主要面向日常服装
  - 裙装与非裙装分开建模，总共两套模型
  - 强依赖 garment 到 SMPL 的 registration 预处理

## Part II：方法与洞察

### 方法主线

作者的整体策略可以概括为：**先把问题变简单，再分频率逐步生成**。

#### 1. 先把服装拉回 canonical 空间
他们先做三件事：

- 去掉全局旋转
- **unpose**
- **unshape**

这样，网络不再直接面对“穿在各种姿态和体型上的衣服”，而是面对一个更规整的 canonical cloth space。  
这一步的作用是把“姿态/体型变化”从“服装动力学细节”里剥离出去。

#### 2. 用 UV map 取代直接 mesh 生成
把 body、template garment、canonical garment 都展开成 UV 图像。这样有两个直接好处：

- 可以用成熟、便宜、强大的 2D ConvNet
- 不需要为不同 mesh 拓扑专门设计复杂网络

但 UV 也有代价：边界和 seam 会带来插值伪影。所以作者额外做了一个**专用 upscaling 网络**来补背景区域，避免普通双线性上采样在边缘处出错。

#### 3. 先生成 normal，再生成 cloth
作者没有只拿 body/template 去直接生成服装坐标，而是加了一个中间层：

- **VAEnorm**：根据 posed body + template garment 先生成 normal UV map
- **VAEdrape**：再用 template/body/normal 作为条件去生成 canonical garment UV

这一步很关键。因为**法线比 3D 坐标更容易学**，但又强烈编码了局部曲率和褶皱方向，所以能给后续服装生成一个更“几何化”的细节先验。

#### 4. 用金字塔逐层补细节
金字塔不是简单多尺度输入，而是：

- 最低分辨率层负责预测**低频整体形状**
- 更高层只预测相对前一层的**残差/offset**
- 各层逐步把细节补上去

而且作者发现：
- **incremental training** 稳，但低层错误难修
- **end-to-end training** 更好，因为高层可以反过来纠正低层的误差

#### 5. 回到 3D 并加几何正则
最后把 UV 结果还原为 3D mesh，做 reshaping + skinning，得到最终悬垂服装。  
训练时再加上碰撞、边长、法线和 3D 重建等正则，约束结果不要只“数值接近 GT”，也要**少穿模、少拉伸、少压缩**。

### 核心直觉

**作者真正拧动的因果旋钮是：**

> 把“高维、单次、posed 3D 坐标回归”  
> 改成  
> “canonical UV 空间中的低频形状生成 + 高频残差修正 + 法线几何条件”。

这带来三层变化：

1. **表示空间变了**  
   posed 3D -> canonical UV  
   姿态/体型干扰被减弱，网络更专注学服装本身的变形分布。

2. **信息瓶颈变了**  
   直接坐标预测 -> 法线中间表示 + 分层残差  
   高频褶皱不再和整体形状抢同一条优化通道。

3. **能力边界变了**  
   从“单解、平滑、按类过拟合”  
   变成“可采样、多解、细节逐级增强、少模型覆盖多种服装”。

一句话说：**先把大轮廓学稳，再让高层只负责补细节，法线则负责把‘褶皱该往哪长’这件事提前告诉网络。**

### 战略权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/妥协 |
|---|---|---|---|
| canonical unpose/unshape | 姿态/体型与服装形变纠缠 | 更好跨姿态泛化 | 依赖 SMPL 与配准质量 |
| UV-map 表示 | mesh 拓扑处理复杂、CNN 难直接用 | 高效卷积建模、推理快 | seam/边界插值需要专门处理 |
| normal conditioning | 高频褶皱直接学坐标太难 | 更强局部几何细节与采样多样性 | 推理时要先走一遍 VAEnorm |
| pyramid residual generation | 单层模型易学成平滑平均解 | 低频到高频逐层细化，细节可解释 | 模块更多，训练更复杂 |
| skirt / non-skirt 双模型 | 单一 UV 模板无法覆盖所有服装 | 用最少模型覆盖多数服装类型 | 仍不是完全统一单模型 |

## Part III：证据与局限

### 关键证据信号

- **信号 1｜ablation**  
  在 CLOTH3D 上，单层 baseline 的 combined V2V 是 **8.01 mm**，端到端金字塔 + 正则化降到 **3.95 mm**。  
  **结论**：性能提升不是“多堆几层网络”这么简单，而是来自“分频率建模 + 跨层纠错”的组合。

- **信号 2｜regularization diagnostics**  
  加正则后，边长误差和碰撞率都下降；例如 non-skirt collision 从 **5.49%** 降到 **1.98%**。  
  **结论**：它不只是更贴近 GT，也更保形、更少穿模。

- **信号 3｜benchmark comparison**  
  CLOTH3D 上报告 **3.95 mm** V2V，明显优于 DeepCloth 的 **19.23 mm** 和 DeePSD 的 **23.78 mm**。  
  CAPE 上最佳达到 **3.11 mm / 3.66 mm**（female/male），也优于作者引用的 CAPE 基线 **3.61 mm / 6.15 mm**。  
  **结论**：方法的能力跳跃体现在“泛化 draping + 细节恢复”而不只是单一数据集上的调参收益。

- **信号 4｜one-to-many sampling**  
  固定 body pose 和 template garment 后随机采样，局部褶皱会变化，但文中示例的面积偏差平均小于 **8%**。  
  **结论**：VAE 没有完全塌成确定性回归，确实保留了“多种合理悬垂”的生成能力。

- **信号 5｜实用性**  
  整体模型约 **40M** 参数，去掉 VAE encoder 后推理参数约 **26M**；3090 上推理约 **50 ms**。  
  **结论**：复现难点主要在预处理与配准，而不是超大模型算力。

### 局限性

- **Fails when**: 配准质量差、服装拓扑复杂（多层、口袋、附加结构）、或材料明显高弹性时；在 CAPE 这种分辨率较低、细节较少的数据上，最高层金字塔还有过拟合迹象。
- **Assumes**: 提供 SMPL pose/shape、与身体尺寸匹配的模板服装、高质量 garment-to-SMPL registration；服装是低弹性、单层表面；裙装与非裙装分开训练。
- **Not designed for**: 自动服装缩放、强物理一致性的长时序布料模拟、多层服装、复杂附件服装、无需模板的全新服装拓扑生成。

### 可复用组件

1. **canonical-space unpose/unshape**：适合任何“先去掉人体因素，再学服装自身变形”的任务。  
2. **normal-conditioned generative bridge**：把难学的 3D geometry 先拆成更容易学的中间几何信号。  
3. **UV-aware upscaling**：适用于所有带 mask/seam 的 UV 生成任务。  
4. **low-to-high residual pyramid**：对高频细节敏感的 3D surface generation 都有参考价值。

## Local PDF reference

![[paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/WACV_2024/2024_A_Generative_Multi_Resolution_Pyramid_and_Normal_Conditioning_3D_Cloth_Draping.pdf]]