---
title: S5-ViT / S4D-ViT
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Object Detection
---

# S5-ViT / S4D-ViT

S5-ViT / S4D-ViT 是 CVPR 2024 提出的一类将状态空间模型（State Space Models, SSM）与视觉 Transformer（ViT）相结合的混合架构方法，专门面向事件相机（Event Camera）数据处理任务。

事件相机是一种新型视觉传感器，以异步、高时间分辨率的方式记录像素级亮度变化，与传统帧相机有本质差异。S5-ViT 和 S4D-ViT 分别基于 S5（Structured State Space Sequence Model）和 S4D（Structured State Space Model with Diagonalization）这两种状态空间模型变体，通过将其与 ViT 架构融合，旨在高效处理事件流数据的长程时序依赖。

**研究领域**: Object Detection

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__事件相机的状态空间模型检测器_S5-ViT_-_S4D-ViT]] | CVPR | 2024 |

