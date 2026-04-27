---
title: Image Generation
type: task
domain: task
name_zh: 图像生成
---

# Image Generation (图像生成)

图像生成（Image Generation）是计算机视觉的核心生成任务，旨在从零或从条件输入（如文本描述、语义布局、草图等）创造逼真的图像内容。该任务要求模型学习复杂视觉数据的分布，并能够采样出高质量、多样化的图像。

近年来，扩散模型（Diffusion Models）和生成对抗网络（GAN）的演进使图像生成质量达到前所未有的高度，文本到图像生成（Text-to-Image）更是成为研究热点，催生了 DALL-E、Stable Diffusion、Midjourney 等影响力深远的应用。

## 代表方法

- SmartPhotoCrafter (1 篇)
- Balanced Rate-Distortion Optimization (1 篇)
- UNIFIEDREWARD-THINK (1 篇)
- Knowledge Bridger (1 篇)
- Unified-IO 2 (1 篇)
- ReImagine (1 篇)
- MMCORE (1 篇)
- DDPRISM (1 篇)

## 常用数据集

- ImageNet-1K (9 篇)
- VLRewardBench (4 篇)
- IU X-ray (2 篇)
- MM-IMDb (2 篇)
- Kodak (1 篇)
- Tecnick (1 篇)
- CLIC 2022 (1 篇)
- GRIT (1 篇)

## 分布

- 年份: 2024 (1) · 2025 (9) · 2026 (14)
- 会议: CVPR (9) · NeurIPS (4) · arXiv (2) · OpenMIND (1)

## 相关论文 (24)

| 论文 | 会议 | 年份 | 核心贡献 |
|------|------|------|----------|
| [[P__统一推理生成优化的摄影图像编辑框_SmartPhotoCrafte]] | Unknown | 2026 | — |
| [[P__学习型图像压缩的平衡率失真优化_Balanced_Rate-Di]] | CVPR | 2025 | — |
| [[P__1D有序Token赋能高效推理时_OTEETS]] | Unknown | 2026 | 1D coarse-to-fine有序token的核心优势在于：生成过程的中间状态（部分前缀）已编码全局语义信息，ver |
| [[P__多模态思维链奖励模型的统一强化微_UNIFIEDREWARD-TH]] | NeurIPS | 2025 | Incorporating explicit long chains of thought into reward re |
| [[P__免训练知识桥接缺失模态补全_Knowledge_Bridge]] | CVPR | 2025 | — |
| [[P__统一自回归多模态大模型：视觉语言_Unified-IO_2]] | CVPR | 2024 | — |
| [[P__图像优先的人体可控视频生成新范式_ReImagine]] | arXiv | 2026 | 核心直觉是「先验分离复用」：图像生成模型（FLUX.1 Kontext）已在海量数据上学会了人体外观的强先验，视频扩散模 |
| [[P__多模态轻量对齐生成框架MMCOR_MMCORE]] | Unknown | 2026 | — |
| [[P__扩散模型驱动的多视角源分离EM框_DDPRISM]] | NeurIPS | 2025 | Diffusion models can solve multi-view source separation with |
| [[P__建模千名标注者的文本行人重识别数_HAM-PEDES_(Human]] | CVPR | 2025 | — |
| [[P__MERGE_More_Than_Generation__Uni]] | NeurIPS | 2025 | MERGE demonstrates that pre-trained text-to-image models can |
| [[P__连续参数预测整数的Dalap分布_Predicting_integ]] | OpenMIND | 2026 | 核心洞察是：离散整数预测的瓶颈不在于分布形状，而在于均值参数的连续性。原始离散Laplace类比分布因均值为整数而无法反 |
| [[P__任意视图3D重建的视频扩散框架_AnyRecon]] | Unknown | 2026 | 现有视频扩散重建方法的核心瓶颈在于：条件化视角数量受限（1-2帧）导致全局上下文不足，以及生成与重建的解耦导致大场景重建 |
| [[P__Omni模型的上下文展开机制_CUOM]] | Unknown | 2026 | — |
| [[P__图像生成器即通用视觉理解器_IGGVL]] | Unknown | 2026 | 核心直觉是：图像生成预训练与LLM文本预训练在本质上同构——大规模生成训练迫使模型内化视觉世界的完整数据分布，从而隐式习 |
| [[P__WebGen-R1：基于强化学习_WebGen-R1]] | Unknown | 2026 | 核心洞察在于将「开放式项目生成」问题重新表述为「受约束的组件填充」问题。通过预验证的脚手架模板固定项目的结构不变量，LL |
| [[P__线性差分视觉Transforme_Visual-Contrast_]] | NeurIPS | 2025 | Visual-Contrast Attention (VCA) is a drop-in replacement for |
| [[P__UDM-GRPO：面向均匀离散扩_UDM-GRPO]] | arXiv | 2026 | UDM-GRPO的核心直觉是：**让RL训练的状态-动作分布尽可能贴近预训练时的分布**。动作重定义（$\hat{x}_ |
| [[P__扩散模型遗忘鲁棒性的幻觉揭示_TIU_(The_Illusio]] | CVPR | 2025 | — |
| [[P__统一3D网格理解与生成的UniM_UniMesh]] | Unknown | 2026 | — |
| [[P__揭示扩散模型的SNR-t偏差_ESBDPM]] | CVPR | 2026 | — |
| [[P__统一HOI生成与编辑的DiT框架_OneHOI]] | CVPR | 2026 | — |
| [[P__生成-检测协同进化的统一框架_UniGenDet]] | CVPR | 2026 | 核心直觉是：生成器对图像分布的内部表示（VAE潜变量）天然包含伪影的成因信息，而检测器对真实性的判断标准可以反向约束生成 |
| [[P__SoftVQ-VAE：高效1维连_SoftVQ-VAE]] | CVPR | 2025 | — |
