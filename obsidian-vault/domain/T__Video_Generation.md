---
title: Video Generation
type: task
domain: task
name_zh: 视频生成
---

# Video Generation (视频生成)

视频生成是极具挑战性的多模态生成任务，要求模型合成时间维度上连贯、视觉质量高且语义一致的动态影像，涉及空间内容生成与时间动态建模的双重复杂性。

近年来，扩散模型（如Sora、Runway Gen-2）在该领域取得突破性进展，实现了从文本描述生成高质量长视频的能力。

## 代表方法

- CityRAG (1 篇)
- OmniVCus (1 篇)
- Diffusion Latent Aligners (1 篇)
- Video-Bench (1 篇)
- DeVI (1 篇)
- MultiWorld (1 篇)
- Foley-Flow (1 篇)
- VAB (1 篇)

## 常用数据集

- VGGSound Video-to-Audio (1 篇)
- VGGSound Image-to-Audio (1 篇)
- Human alignment (1 篇)
- Video quality dimensions (1 篇)
- Video-Condition Alignment (1 篇)
- Inter-rater agreement (1 篇)
- Video (1 篇)
- Video-Condition Alignment - Video-text Consistency (1 篇)

## 分布

- 年份: 2024 (3) · 2025 (4) · 2026 (10)
- 会议: CVPR (5) · arXiv (3) · NeurIPS (2) · ICML (1)

## 相关论文 (17)

| 论文 | 会议 | 年份 | 核心贡献 |
|------|------|------|----------|
| [[P__CityRAG-_基于空间感知的_CityRAG]] | Unknown | 2026 | — |
| [[P__OmniVCus_OmniVCus__Feedforward_Sub]] | NeurIPS | 2025 | OmniVCus achieves feedforward multi-subject video customizat |
| [[P__扩散潜空间对齐的开放域视音频生成_Diffusion_Latent]] | CVPR | 2024 | — |
| [[P__Video-Bench：人类对齐_Video-Bench]] | CVPR | 2025 | — |
| [[P__Omni模型的上下文展开机制_CUOM]] | Unknown | 2026 | — |
| [[P__基于合成视频模仿的物理灵巧HOI_DeVI]] | Unknown | 2026 | DeVI的核心直觉是：对人体和物体采用「不对称维度」的参考信号——人体用3D（因为HMR技术相对成熟），物体用2D（因为 |
| [[P__多智能体多视角视频世界模型Mul_MultiWorld]] | Unknown | 2026 | — |
| [[P__视频到音频生成的动态条件流匹配_Foley-Flow]] | CVPR | 2025 | — |
| [[P__统一音视频表示与生成模型VAB_VAB_(Vision-Audi]] | ICML | 2024 | — |
| [[P__WorldMark：交互式视频世_WorldMark]] | NeurIPS | 2026 | 评测碎片化的根本原因不是缺乏指标，而是缺乏公共测试场地。WorldMark的核心洞察是：将动作接口标准化（统一映射层）与 |
| [[P__图像优先的人体可控视频生成新范式_ReImagine]] | arXiv | 2026 | 核心直觉是「先验分离复用」：图像生成模型（FLUX.1 Kontext）已在海量数据上学会了人体外观的强先验，视频扩散模 |
| [[P__SDVG：将投机解码从LLM扩展_SDAVG]] | arXiv | 2026 | 核心变化是将 LLM 投机解码中不可用的「token 概率比较」替换为「图像质量奖励信号」，使得块级路由决策在连续视频域 |
| [[P__视频编辑三维解耦评估基准VEFX_VEFX-Bench]] | Unknown | 2026 | 视频编辑评估的失败根源在于「维度混淆」——将指令执行、渲染质量和内容保留压缩为单一分数，导致评估信号模糊且无法指导优化。 |
| [[P__视频生成模型的VLM链式思考评估_VBench]] | CVPR | 2024 | — |
| [[P__快慢感知：学习视频中的时间流_SFSLFT]] | Unknown | 2026 | — |
| [[P__语义引导层次化视频预测Re2Pi_RBPSGH]] | arXiv | 2026 | Re2Pix提出一个两阶段层次化视频预测框架，将预测任务显式分解为语义表示预测（Stage 1）与表示引导的视觉合成（S |
| [[P__T2V组合生成综合评测基准_T2V-CompBench]] | CVPR | 2025 | — |
