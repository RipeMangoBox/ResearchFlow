---
title: "InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion"
venue: ICCV
year: 2023
tags:
  - Human_Object_Interaction
  - task/human-object-interaction
  - task/motion-prediction
  - diffusion
  - dataset/BEHAVE
  - repr/SMPL
  - opensource/full
core_operator: 物理信息扩散修正（Interaction Correction）：在扩散去噪过程中注入基于参考系变换的物理先验——接触点参考系下物体运动模式简单可预测，修正穿透和浮空
primary_logic: |
  历史人-物交互 + 物体形状
  → 交互扩散模型（Transformer）：建模未来交互分布
  → 每个去噪步：修正调度器判断是否需要修正（穿透/无接触）
  → 交互预测器（STGNN）：在接触点参考系下预测物体相对运动（简单模式）
  → 将修正后的物体运动注入去噪结果 → 物理可信的长期交互预测
claims:
  - "InterDiff首次提出基于mesh的3D人-物交互预测任务，联合建模全身运动和物体动态"
  - "交互修正步骤通过参考系变换将复杂物体运动转化为简单模式，无需物理仿真即可大幅减少穿透和浮空"
  - "在BEHAVE数据集上穿透率从228降至164（×10⁻²%），且可零样本泛化到GRAB数据集"
pdf_ref: paperPDFs/Human_Object_Interaction/ICCV_2023/2023_InterDiff_Generating_3D_Human_Object_Interactions_with_Physics_Informed_Diffusion.pdf
category: Human_Object_Interaction
---

# InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://sirui-xu.github.io/InterDiff/) · [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_InterDiff_Generating_3D_Human-Object_Interactions_with_Physics-Informed_Diffusion_ICCV_2023_paper.pdf)
> - **Summary**: InterDiff在扩散去噪过程中注入物理信息修正——利用接触点参考系下物体运动模式简单可预测的先验，通过STGNN预测相对运动并注入扩散过程，无需物理仿真即可生成物理可信的长期3D人-物交互预测。
> - **Key Performance**:
>   - BEHAVE数据集穿透率从228降至**164**（×10⁻²%），旋转误差从256降至**226**
>   - 支持自回归长期预测（100帧），且可零样本泛化到GRAB数据集

---

## Part I：问题与挑战

### 真正的卡点

3D人-物交互预测面临**物理可信性**和**长期稳定性**两大核心挑战：

- **扩散模型不懂物理**：标准扩散模型生成的交互运动存在接触浮空和穿透伪影，因为模型没有内置物理约束
- **长期误差累积**：自回归推理时，每步的小误差会累积，导致长期预测中物体运动越来越不合理
- **物理仿真成本高**：传统方法依赖后处理优化或物理仿真器来保证物理可信性，但仿真器需要注册物体（摩擦、刚度、质量等），成本高且不通用

### 输入/输出接口

- 输入：历史H帧人-物交互（SMPL参数 + 物体6DoF）+ 物体形状点云
- 输出：未来F帧人-物交互预测

---

## Part II：方法与洞察

### 整体设计

InterDiff由两个可独立训练、推理时组合的模块组成：

1. **交互扩散模型**：Transformer encoder-decoder架构，条件=历史运动编码+物体形状（PointNet），生成未来交互分布
2. **交互修正**：
   - 修正调度器：在每个去噪步检查当前结果是否存在穿透或无接触，决定是否修正
   - 交互预测器（STGNN）：将物体运动从全局坐标系变换到接触点参考系 → 在参考系下预测物体相对运动 → 变换回全局坐标系
   - 将修正后的物体运动与原始去噪结果加权融合

### 核心直觉

**什么变了**：从"在全局坐标系下预测复杂物体运动"到"在接触点参考系下预测简单相对运动"。

**哪些分布/约束/信息瓶颈变了**：
- 参考系变换将复杂的全局物体运动转化为简单的局部模式（如绕固定轴旋转、近似静止）→ 预测难度大幅降低
- 修正步骤在扩散去噪过程中注入物理先验 → 不需要后处理，物理约束与生成过程融合
- 两个模块独立训练 → 交互预测器在干净数据上训练，推理时直接应用于去噪中间结果，无需联合微调

**为什么有效**：人-物交互的核心物理约束是接触——在接触点参考系下，物体运动确实遵循简单模式（这是物理定律的直接推论）。利用这个先验，一个简单的STGNN就能准确预测相对运动。

**权衡**：修正步骤增加了推理时间；参考系选择依赖于接触检测的准确性；对无接触阶段的修正能力有限。

---

## Part III：证据与局限

### 关键实验信号

- **物理可信度提升**：BEHAVE上穿透率从228降至164，旋转误差从256降至226——仅通过修正步骤即获得显著改善
- **长期预测**：100帧自回归预测中，修正步骤的优势随时间增长更加明显（误差累积被持续抑制）
- **零样本泛化**：在BEHAVE上训练的模型直接泛化到GRAB数据集（小物体抓取），无需微调
- **用户研究**：67.8%的人类评估者认为完整模型比无修正版本更真实

### 局限与可复用组件

- **局限**：接触不一致仍偶尔出现；仅处理单人单物体；不支持可变形物体；骨架表示下接触定义不精确
- **可复用**：参考系变换+简单预测器的物理修正范式可迁移到任何需要物理可信性的扩散生成任务；修正调度器的设计（基于穿透/接触检测）适用于任何交互生成；STGNN在参考系下预测相对运动的思路可用于机器人操作规划

---

## 本地 PDF 引用

![[paperPDFs/Human_Object_Interaction/ICCV_2023/2023_InterDiff_Generating_3D_Human_Object_Interactions_with_Physics_Informed_Diffusion.pdf]]
