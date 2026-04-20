---
title: 'SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy'
type: paper
paper_level: B
venue: arXiv (Cornell University)
year: 2026
acceptance: null
cited_by: null
facets:
  modality:
  - Image
paper_link: https://arxiv.org/abs/2603.29962
---

# SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy

> **结构性改进**。先读 baseline，再看本文修改了哪些核心组件。

## 详细分析

# SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy

## Part I：问题与挑战

腹腔镜胆囊切除术等外科手术视频具有高度复杂性：视觉对比度低、解剖结构辨识困难、手术阶段跨越较长时间窗口，且任务层次从基础感知（器械/动作/解剖识别）延伸至高层评估（CVS安全标准、术中难度、技能熟练度、不良事件）。现有手术VQA研究主要聚焦于静态帧分析，忽视了丰富的时序语义信息。通用视频多模态大模型（如VideoGPT+、LLaVA-Video）在零样本场景下对基础感知任务表现尚可，但在评估级任务上严重不足——零样本模型往往生成表面合理但临床无效的回答（高答案率、低准确率）。此外，手术视频时长差异大，如何在保持计算效率的同时捕捉跨越不同时间粒度的关键安全线索，是一个尚未被充分解决的工程与建模挑战。现有数据集也缺乏覆盖完整任务层次的大规模手术视频QA基准，制约了该方向的系统性研究。
