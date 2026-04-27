---
title: Domain Adaptation
type: task
domain: task
name_zh: null
---

# Domain Adaptation

_任务节点_。本节点目前由 15 篇知识库论文支撑，主要来源：CVPR、ECCV、ICLR。

## 代表方法

- I-sampling based SFDA (1 篇)
- COMPASS (1 篇)
- Visual Fourier Prompt Tuning (1 篇)
- C-TPT (1 篇)
- Cascade Prompt Learning (1 篇)
- KMD (1 篇)
- RrED (1 篇)
- PerReg / PerReg+ (1 篇)

## 常用数据集

- Treatment Effect Estimation - ACIC (4 篇)
- Few-shot (4 篇)
- VisDA-17 (3 篇)
- AV2 (3 篇)
- MNIST (3 篇)
- CIFAR-10 (3 篇)
- Office-Home (2 篇)
- VTAB-1k (2 篇)

## 分布

- 年份: 2024 (5) · 2025 (8) · 2026 (2)
- 会议: NeurIPS (6) · CVPR (3) · ICLR (2) · arXiv (2) · TMLR (1) · 其他 (1)

## 相关论文 (15)

| 论文 | 会议 | 年份 | 核心贡献 |
|------|------|------|----------|
| [[P__无源域自适应的重要性采样新框架_I-sampling_based]] | CVPR | 2025 | — |
| [[P__多语言PEFT自适应语义采样持续_COMPASS]] | TMLR | 2026 | 核心直觉是：多语言负迁移的根源不在于语言之间的语言学距离，而在于训练数据与目标使用分布之间的语义覆盖缺口。通过在共享嵌入 |
| [[P__视觉傅里叶提示微调VFPT_Visual_Fourier_P]] | NeurIPS | 2024 | — |
| [[P__通过文本特征分散校准测试时提示微_C-TPT_(Calibrate]] | ICLR | 2024 | — |
| [[P__视觉语言模型级联提示学习_Cascade_Prompt_L]] | ECCV | 2024 | — |
| [[P__Koopman多模态分解的缺失模_KMD_(Koopman_Mul]] | CVPR | 2025 | — |
| [[P__扩散推理纠错的两阶段黑盒域适应_RrED_(Rectifying]] | NeurIPS | 2025 | RrED achieves superior BUDA performance by rectifying reason |
| [[P__自适应提示的双层轨迹预测表示学习_PerReg_-_PerReg+]] | CVPR | 2025 | — |
| [[P__手术域少样本SAM分割的循环一致_CycleSAM]] | arXiv | 2024 | 通用特征匹配在手术域失败的根本原因有两个：特征本身不适配（域间隙）和匹配过程不可靠（噪声对应）。CycleSAM用手术特 |
| [[P__EEG基础模型测试时适应的系统诊_TAEFMA]] | arXiv | 2026 | 本文的核心洞察是：EEG信号的高度非平稳性和复杂分布结构使得基于熵最小化的梯度类TTA方法容易陷入退化解——当预测熵被强 |
| [[P__图平滑贝叶斯黑盒标签偏移估计器_GS-B3SE_(Graph-S]] | NeurIPS | 2025 | GS-B3SE is a fully probabilistic black-box shift estimator t |
| [[P__非平衡最优传输的等效变换与KKT_ETM-Refine_+_MRO]] | NeurIPS | 2025 | The Equivalent Transformation Mechanism (ETM) can exactly de |
| [[P__CLIP免训练自适应的极简集成基_Alpha-CLIP_(trai]] | ICLR | 2024 | — |
| [[P__协变量偏移下KRR的伪标签选择方_Pseudo-Labeling_]] | NeurIPS | 2025 | A pseudo-labeling approach that splits source data for train |
| [[P__单神经元表示的对抗不变学习框架_model-agnostic_a]] | NeurIPS | 2025 | A model-agnostic adversarial training strategy with a gradie |
