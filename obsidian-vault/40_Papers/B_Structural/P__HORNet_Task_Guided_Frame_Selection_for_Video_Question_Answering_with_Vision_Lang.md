---
title: 'HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models'
type: paper
paper_level: B
venue: arXiv (Cornell University)
year: 2026
acceptance: null
cited_by: null
facets:
  modality:
  - Image
paper_link: https://arxiv.org/abs/2603.18850
---

# HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models

> **结构性改进**。先读 baseline，再看本文修改了哪些核心组件。

## 详细分析

# HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models

## Part I：问题与挑战

视频问答（VideoQA）系统在使用视觉语言模型（VLM）时面临一个根本性的效率瓶颈：视频包含大量冗余帧，而现有系统普遍采用均匀采样或启发式采样策略，无法根据具体问题的语义需求动态选择最相关的帧。这导致两个相互关联的问题：一是计算资源浪费，VLM需要处理数千帧输入（如ActivityNet平均3152帧），推理延迟极高；二是信息噪声干扰，无关帧可能分散VLM的注意力，降低答案质量。

现有帧选择方法存在明显局限：基于规则的方法（如均匀采样）不具备任务感知能力；基于学习的方法（如SeViLA、Frame-Voyager）通常需要修改VLM本身或依赖监督信号，无法在保持VLM冻结的同时通过下游奖励信号端到端优化帧选择策略。更深层的挑战在于，帧选择是一个组合优化问题——从T帧中选k帧的搜索空间为C(T,k)，且奖励信号（QA准确率）不可微，传统梯度方法难以直接应用。

此外，现有方法在四个关键属性上无法同时满足：（1）学习式帧选择、（2）下游奖励优化、（3）VLM保持冻结、（4）参数高效。这一空白使得在资源受限场景下部署高质量VideoQA系统面临实际障碍。
