---
title: Self-Supervised Learning
type: task
domain: task
name_zh: null
---

# Self-Supervised Learning

_任务节点_。本节点目前由 31 篇知识库论文支撑，主要来源：NeurIPS、CVPR、CUP。

## 代表方法

- MARCO (1 篇)
- Factorized Hahn-Grant Solver for Gromov-Wasserstein Distance (1 篇)
- SeTa (1 篇)
- CSR (1 篇)
- InfoBatch (1 篇)
- Visual Fourier Prompt Tuning (1 篇)
- Indra Representation (1 篇)
- Gradient Flow for Shallow ReLU Networks on Weakly Interacting Data (1 篇)

## 常用数据集

- CIFAR-10 (8 篇)
- ImageNet-1K (5 篇)
- CIFAR-100 (4 篇)
- Office-Home (3 篇)
- AV2 (3 篇)
- CINIC-10 (2 篇)
- VTAB-1k (2 篇)
- nuScenes (2 篇)

## 分布

- 年份: 2022 (1) · 2024 (4) · 2025 (17) · 2026 (9)
- 会议: NeurIPS (11) · CVPR (8) · ICLR (4) · arXiv (3) · CUP (1)

## 相关论文 (31)

| 论文 | 会议 | 年份 | 核心贡献 |
|------|------|------|----------|
| [[P__MARCO：探索语义对应的未知空_MARCO]] | CVPR | 2026 | MARCO在单DINOv2骨干（ViT-L/14）基础上引入两个核心训练机制，分别针对精细定位和语义泛化两个目标。

* |
| [[P__无平行数据的视觉-语言盲匹配_Factorized_Hahn-]] | CVPR | 2025 | — |
| [[P__大规模数据集的滑动窗口课程高效训_SeTa_(Scale_Effi]] | CVPR | 2025 | — |
| [[P__多样字典学习：集合论可识别性_DDL]] | CUP | 2026 | 核心直觉是：与其在强假设下追求完全可识别性，不如在通用设定下识别潜变量的集合论结构（交集、补集、对称差）。这一弱化目标在 |
| [[P__视觉语言模型的校准自奖励优化_CSR_(Calibrated_]] | NeurIPS | 2024 | — |
| [[P__自监督引导增强视觉指令微调_BVITSG]] | NeurIPS | 2026 | 标准指令微调数据中充斥着可被语言先验解答的样本，导致模型形成视觉懒惰习惯。V-GIFT的直觉是：通过注入少量「语言先验完 |
| [[P__无偏动态数据剪枝无损加速训练_InfoBatch]] | ICLR | 2024 | — |
| [[P__弱监督下LLM推理学习的诊断与干_WCLLRW]] | Unknown | 2026 | 弱监督下 RLVR 的成败不取决于 RL 算法本身，而取决于模型在 RL 开始前是否具备「推理忠实度」——即中间步骤真正 |
| [[P__视觉傅里叶提示微调VFPT_Visual_Fourier_P]] | NeurIPS | 2024 | — |
| [[P__基于Yoneda嵌入的关系型表示_Indra_Representa]] | NeurIPS | 2025 | Representations from unimodal foundation models implicitly r |
| [[P__弱相关数据下浅层ReLU梯度流收_Gradient_Flow_fo]] | NeurIPS | 2025 | For one-hidden-layer ReLU networks trained by gradient flow  |
| [[P__无监督视觉特征学习的DINOv2_DINOv2]] | ICLR | 2025 | — |
| [[P__快慢感知：学习视频中的时间流_SFSLFT]] | Unknown | 2026 | — |
| [[P__自适应提示的双层轨迹预测表示学习_PerReg_-_PerReg+]] | CVPR | 2025 | — |
| [[P__语义引导层次化视频预测Re2Pi_RBPSGH]] | arXiv | 2026 | Re2Pix提出一个两阶段层次化视频预测框架，将预测任务显式分解为语义表示预测（Stage 1）与表示引导的视觉合成（S |
| [[P__两层网络低秩梯度的尖峰协方差分析_Spiked_Covarianc]] | NeurIPS | 2025 | The gradient of inner-layer weights in two-layer networks is |
| [[P__无源域自适应的重要性采样新框架_I-sampling_based]] | CVPR | 2025 | — |
| [[P__纯加法脉冲自注意力Transfo_Spiking_Transfor]] | CVPR | 2025 | — |
| [[P__MoE专家上循环：低成本扩容新范_EUSCEF]] | Unknown | 2026 | — |
| [[P__离散扩散的信息论精确似然估计_Information-Theo]] | NeurIPS | 2025 | The I-MDSE and I-MDCE relations provide tight, principled es |
| [[P__运动预测的双任务自监督预训练框架_SmartPretrain]] | ICLR | 2025 | — |
| [[P__无配对数据的盲视觉-语言匹配_IBMTVL]] | CVPR | 2025 | — |
| [[P__对比等变学习：无监督有限群等变嵌_Equivariance_by_]] | NeurIPS | 2025 | Equivariance by Contrast (EbC) is the first general-purpose  |
| [[P__最优传输驱动的在线增量潜在空间学_AOTACL]] | arXiv | 2022 | 本文提出OTC（Optimal Transport-driven Centroid）框架，核心由两个模块构成：MMOT（ |
| [[P__单神经元表示的对抗不变学习框架_model-agnostic_a]] | NeurIPS | 2025 | A model-agnostic adversarial training strategy with a gradie |
| [[P__AptGCD：面向GCD的提示T_AptGCD]] | CVPR | 2025 | — |
| [[P__通过文本特征分散校准测试时提示微_C-TPT_(Calibrate]] | ICLR | 2024 | — |
| [[P__可识别性视角下的分布接近与表示相_d^λ_LLV_distance]] | NeurIPS | 2025 | Small KL divergence between model distributions does not gua |
| [[P__测试时训练的EM扩展框架TEMP_TEMPO]] | Unknown | 2026 | 现有TTT方法的失败根源在于用一个会随策略演化而漂移的自生成信号来训练策略本身——这是一个没有外部锚点的自我强化循环。T |
| [[P__温度自适应多模态对比学习的内在维_Temperature-adap]] | NeurIPS | 2025 | Temperature optimization in multi-modal contrastive learning |
