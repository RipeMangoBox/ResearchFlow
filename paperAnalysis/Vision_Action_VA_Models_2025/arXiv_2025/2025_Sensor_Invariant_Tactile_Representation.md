---
title: "Sensor-Invariant Tactile Representation"
venue: ICLR
year: 2025
tags:
  - Embodied_AI
  - task/object-classification
  - task/pose-estimation
  - transformer
  - contrastive-learning
  - physics-based-simulation
  - "dataset/Simulated Tactile Dataset"
  - "dataset/Real-World Tactile Dataset"
  - repr/normal-map
  - opensource/no
core_operator: 用校准图像条件化Transformer，并以法向图重建和跨传感器监督对比学习共同逼出传感器不变的触觉几何表征
primary_logic: |
  触觉图像 + 该传感器的少量校准图像 → 将触觉token与校准token联合输入Transformer，并在多传感器仿真数据上用法向图几何监督和同接触跨传感器对比对齐 → 输出可零样本迁移到新传感器的触觉表示，用于分类、位姿估计与形状重建
claims:
  - "Claim 1: SITR在4传感器inter-sensor对象分类上达到81.94%平均准确率，显著高于最强ViT-Large预训练基线的54.34% [evidence: comparison]"
  - "Claim 2: 在inter-sensor 3-DoF位姿估计上，SITR将RMSE降至0.80 mm，相比ViT-Large预训练基线1.49 mm约降低46% [evidence: comparison]"
  - "Claim 3: 法向图监督与监督对比学习联合使用优于任一单独损失，transfer分类准确率由84.21%/78.86%提升到91.43% [evidence: ablation]"
related_work_position:
  extends: "T3 (Zhao et al. 2024)"
  competes_with: "T3 (Zhao et al. 2024); UniT (Xu et al. 2024)"
  complementary_to: "Sparsh (Higuera et al. 2024); Binding Touch to Everything (Yang et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Sensor_Invariant_Tactile_Representation.pdf
category: Embodied_AI
---

# Sensor-Invariant Tactile Representation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.19638)
> - **Summary**: 论文把“不同触觉传感器产生不同图像风格”这个跨域问题显式化，利用少量校准图像、几何法向图监督和大规模多传感器仿真对比预训练，学习可在新传感器上零样本迁移的触觉表示。
> - **Key Performance**: inter-sensor对象分类 81.94%（ViT-Large: 54.34%）；inter-sensor位姿估计 RMSE 0.80 mm（ViT-Large: 1.49 mm）

> [!info] **Agent Summary**
> - **task_path**: 视觉式触觉图像/双帧触觉图像 + 传感器校准图像 -> 传感器不变触觉表征 -> 分类、位姿估计、形状重建
> - **bottleneck**: 同一接触几何会被不同光学设计与制造偏差映射成截然不同的触觉图，导致模型把传感器风格误学成任务语义
> - **mechanism_delta**: 用校准图像显式提供传感器域信息，再用法向图几何监督和同接触跨传感器对比学习，把训练目标从“拟合某个传感器图像”改成“恢复跨传感器一致的接触几何”
> - **evidence_signal**: 7个真实传感器上的zero-shot transfer显著优于ViT/T3/UniT，且校准图像、SCL、损失组合与仿真规模都有ablation支撑
> - **reusable_ops**: [calibration-token conditioning, geometry-supervised cross-domain contrastive pretraining]
> - **failure_modes**: [unseen optical designs outside simulator coverage such as DIGIT-like gaps, tasks dominated by force-torque or marker dynamics rather than contact geometry]
> - **open_questions**: [how to extend invariance beyond flat GelSight-style optical sensors, how much simulator diversity is sufficient for open-world sensor transfer]

## Part I：问题与挑战

这篇论文要解决的核心不是“触觉模型不够大”，而是**传感器方差导致的表示纠缠**。

高分辨率视觉式触觉传感器（如 GelSight、DIGIT）会把接触信息转成图像，但不同传感器的光源布局、胶体材料、相机视场、反光层、制造误差都不同。结果是：**同一个物体、同一个接触位置，在不同传感器上会产生差异极大的触觉图像**。如果直接把这些图像当普通视觉输入，模型很容易学到“这是某种传感器的成像风格”，而不是“这是什么接触几何”。

### 真正瓶颈是什么？
- **域间差距大且不可预测**：不是简单颜色偏移，而是成像机理和物理接触响应都变了。
- **真实触觉数据昂贵**：每换一个传感器就重采大规模标注数据，成本太高。
- **已有方法多把传感器当离散类别**：这对未见传感器尤其无效，也难覆盖同类型内部制造差异。

### 为什么现在值得解决？
- 视觉式触觉传感器正在快速扩散，跨实验室、跨硬件版本复用模型和数据变得越来越重要。
- 机器人操作越来越依赖触觉，若表征不能迁移，真实部署成本会被硬件碎片化放大。
- 仿真器和Transformer表征学习已经成熟，开始具备“先在仿真里学跨传感器不变性，再落到真实传感器”的条件。

### 输入 / 输出接口
- **输入**：单帧触觉图像，外加该传感器的一组少量校准图像；位姿任务则输入前后两帧触觉图。
- **输出**：一个传感器不变的触觉表示 SITR；下游只训练小型任务头，用于对象分类、位姿估计、形状重建。

### 边界条件
- 主要面向**光学式、平面 gel pad** 的视觉触觉传感器。
- 论文默认**接触几何**是跨任务最稳定、最值得保留的公共信息。
- 这不是完全无条件迁移：每个新传感器仍需要少量校准图像。

## Part II：方法与洞察

SITR 的关键不是简单“多传感器预训练”，而是把**传感器特有信息**与**接触几何公共信息**在训练机制上分开。

### 方法主线

1. **用校准图像描述传感器本身**  
   作者对每个传感器采集一小组容易获取的校准图像：4mm 球和立方体角，各在 3×3 位置按压，共 18 张。  
   这些图像提供了该传感器的光照、空间不均匀性、边角变形等“域信息”。

2. **把触觉图和校准图一起送入Transformer**  
   触觉图 token 与校准 token 被共同编码。  
   - **patch token**：保留局部稠密接触细节  
   - **class token**：形成全局可迁移表示  
   论文将二者组合作为 SITR，用于不同下游任务。

3. **用法向图而不是原始RGB做监督**  
   训练时不是让模型重建传感器图像，而是重建接触表面的 normal map。  
   这等于把监督锚定到一个更接近物理几何、而不是传感器外观的目标上。

4. **用监督对比学习做跨传感器对齐**  
   在仿真中，同一接触几何会在不同传感器配置下被渲染成多种触觉图。  
   作者把“同几何、不同传感器”作为正样本，把不同几何或不同位置作为负样本，让 class token 学会跨域聚类。

5. **用大规模仿真提供对齐所需的覆盖度**  
   作者构建了 1M synthetic data：100 种传感器配置 × 10K 接触构型。  
   没有这种“同一接触跨很多传感器”的成对数据，传感器不变性其实很难直接从真实数据中学出来。

### 核心直觉

传统触觉表征学习的问题是：**模型只看一张触觉图时，无法判断图像变化到底来自接触几何，还是来自传感器成像差异**。  
SITR 通过三个联动改动改变了这个信息瓶颈：

- **加入校准图像**：把“这是谁的传感器”变成显式条件，而不是让模型从任务图里盲猜。
- **用法向图监督**：把学习目标从传感器外观转向接触几何。
- **用跨传感器正样本对齐**：让“同几何”成为跨域中唯一稳定的可聚合因素。

于是，训练分布从  
**“每个传感器各学各的图像模式”**  
变成  
**“同一几何在不同传感器下必须被编码成相近表示”**。

能力上的变化是：
- 以前：同传感器内效果好，换传感器就崩。
- 现在：只在一个真实传感器上训练下游头，也能在其他真实传感器上零样本迁移。

### 为什么这个设计因果上有效？
- **校准图**降低了域识别难度：模型不用把有限表征容量浪费在反推传感器光学属性。
- **normal map**比原始触觉RGB更接近跨传感器共享物理量，因此更适合做不变表示锚点。
- **SCL**主要对齐全局语义，**patch-level几何监督**保留局部细节；这解释了为什么它同时能支持分类和位姿估计。
- **仿真多样性**提供“可控跨域配对”，这是零样本泛化成立的前提，而不只是数据量堆砌。

### 策略取舍

| 设计选择 | 得到的能力 | 代价 / 风险 |
| --- | --- | --- |
| 用校准图像而不是离散 sensor ID | 能处理同类型内部制造差异，也能面向未见传感器 | 每个新传感器都要做一次标定 |
| 用 normal map 监督而不是重建原始触觉图 | 更强的几何不变性，减少外观过拟合 | 对力/扭矩/marker 等非几何信息保留不足 |
| 在 100 种仿真传感器上预训练 | 支持 zero-shot sensor transfer | 泛化受仿真覆盖限制，分布外光学设计仍可能掉点 |
| 冻结 encoder、只训下游头 | 下游数据需求低，迁移评估更干净 | 任务特化能力受限，不能充分适应特别新的任务 |

## Part III：证据与局限

### 关键信号

1. **跨传感器对象分类显著提升**  
   在 4 传感器 inter-sensor 设置下，SITR 达到 **81.94%**，明显高于最强 ViT-Large 预训练基线 **54.34%**。  
   这说明它学到的不是“某个传感器风格下的分类特征”，而是更稳定的接触结构特征。

2. **跨传感器位姿估计误差近乎减半**  
   inter-sensor 3-DoF pose estimation 上，SITR 的 RMSE 为 **0.80 mm**，而 ViT-Large 预训练是 **1.49 mm**。  
   这很关键，因为位姿回归比分类更依赖几何尺度与局部接触细节，说明其 patch-level 几何表示确实有效。

3. **ablation 证明“校准 + 几何监督 + 对比对齐”缺一不可**  
   - 加入更多校准图能持续提升性能，且加入**第二种校准物体**（球 + 立方体角）比只增加同类图片更有效。  
   - 无校准到标准 18 张校准图，inter-sensor 分类提升超过 **20%**。  
   - normal-only / SCL-only 都有效，但两者联合最好：**91.43% > 84.21% / 78.86%**。  
   - 更大的仿真覆盖也直接带来更强 transfer：100 种传感器配置明显优于 10 种。

4. **SCL 的作用更偏“跨域聚类”，不一定直接改善所有回归任务**  
   论文显示 SCL 对分类帮助明显，但对 pose RMSE 改善有限。  
   这说明它主要优化的是**跨传感器语义对齐**，而精细回归更依赖 normal map 带来的局部几何信息。

5. **分布外传感器仍是短板**  
   t-SNE 和详细 transfer 结果都显示 DIGIT 更难与其他传感器聚类。  
   作者给出的解释也很合理：DIGIT 的光学设计与训练仿真覆盖的 GelSight-style 设计差别更大。

### 1-2 个最值得记住的指标
- **inter-sensor 分类**：81.94% vs 54.34%（最强 ViT 基线）
- **inter-sensor 位姿估计**：0.80 mm vs 1.49 mm（最强 ViT 基线）

### 局限性
- **Fails when**: 目标传感器的光学设计明显超出仿真分布覆盖时，尤其像 DIGIT 这类与 GelSight 家族差异较大的设计；以及任务主要依赖力、扭矩、marker 运动而非接触几何时。
- **Assumes**: 每个新传感器可获取少量校准图像；PBR 仿真能较好近似目标传感器家族；下游任务主要受益于几何信息；训练时可承担 1M 级别仿真数据和 100 种传感器配置的预训练成本。
- **Not designed for**: 低分辨率阵列式触觉传感器、marker-based 多模态触觉建模、完全无校准即插即用部署、非平面/360°复杂光学结构传感器的直接统一建模。

### 复现与资源依赖
- 方法真正的门槛不只是模型，而是**仿真覆盖度 + 校准流程**。
- 论文正文未给出明确代码/项目链接，因此复现者需要自己重建 Blender/PBR 触觉仿真、校准采集和多传感器评测流程。
- 真实评测使用了 7 个不同传感器，硬件门槛不低；不过其核心思想——“显式传入传感器校准上下文，再把监督锚在几何上”——是高度可迁移的。

### 可复用组件
- **calibration-token conditioning**：任何跨硬件域迁移问题，只要有低成本标定，都可借鉴。
- **geometry-first supervision**：用稳定物理中间量替代原始观测重建，能减少域外观过拟合。
- **simulator-built positive pairs**：在仿真中构造“同物理状态、不同观测域”的正样本，是学不变表示的强套路。

## Local PDF reference
![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Sensor_Invariant_Tactile_Representation.pdf]]