---
title: "T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations"
venue: CVPR
year: 2023
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - vq-vae
  - autoregressive
  - dataset/HumanML3D
  - dataset/KIT-ML
  - repr/HumanML3D-263d
  - opensource/full
core_operator: 简洁VQ-VAE + GPT自回归生成：用CNN-based VQ-VAE（EMA + Code Reset）获得高质量离散运动表示，GPT自回归生成运动token，以极简设计超越扩散方法
primary_logic: |
  文本描述 → CLIP文本嵌入作为条件
  → GPT自回归Transformer逐token预测运动序列（条件=CLIP嵌入 + 已生成token）
  → VQ-VAE解码运动token为连续运动序列
  → 训练技巧：corruption strategy缓解训练-测试不一致；EMA + Code Reset保持codebook利用率
claims:
  - "T2M-GPT以极简VQ-VAE+GPT设计在HumanML3D上FID 0.116，大幅优于MotionDiffuse(0.630)"
  - "简单的CNN-based VQ-VAE配合EMA和Code Reset即可获得高质量离散运动表示，无需复杂架构"
  - "Corruption strategy在GPT训练中引入随机token替换，有效缓解自回归生成的训练-测试分布偏移"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/CVPR_2023/2023_T2M_GPT_Generating_Human_Motion_from_Textual_Descriptions_with_Discrete_Representations.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://mael-zys.github.io/T2M-GPT/) · [CVPR 2023](https://arxiv.org/abs/2301.06052)
> - **Summary**: T2M-GPT以极简设计（CNN VQ-VAE + GPT自回归）在文本-运动生成上大幅超越扩散方法，证明VQ-VAE+GPT范式在运动生成中仍具强竞争力。关键技巧：EMA+Code Reset保持codebook利用率，corruption strategy缓解自回归训练-测试偏移。
> - **Key Performance**:
>   - HumanML3D FID **0.116** vs. MotionDiffuse(0.630)，5倍以上提升
>   - 极简架构：CNN VQ-VAE + 标准GPT，无复杂模块

---

## Part I：问题与挑战

### 真正的卡点

文本-运动生成面临**离散表示质量**和**自回归生成的训练-测试偏移**两大核心挑战：

- **VQ-VAE的codebook退化**：标准VQ-VAE训练中，大量code向量不被使用（codebook collapse），导致离散表示的信息容量远低于理论上限
- **自回归的误差累积**：GPT训练时使用ground truth token作为输入，但推理时使用自身预测的token——这种训练-测试分布偏移导致误差累积，长序列生成质量下降
- **扩散方法的主导地位**：当时扩散模型在运动生成上占据主导，VQ-VAE+GPT范式被认为不如扩散方法

### 输入/输出接口

- 输入：自然语言文本描述
- 输出：3D人体运动序列（HumanML3D 263维表示）

---

## Part II：方法与洞察

### 整体设计

T2M-GPT的核心贡献是证明简单方法做好就够了：

1. **CNN VQ-VAE**：
   - 简单的1D CNN编码器-解码器
   - EMA（指数移动平均）更新codebook：平滑更新避免剧烈跳变
   - Code Reset：定期将低利用率的code向量重置为当前batch中的编码向量，强制提高利用率
   - 两个技巧组合即可保持>90%的codebook利用率

2. **GPT自回归生成**：
   - CLIP文本嵌入作为条件前缀
   - 标准因果Transformer逐token预测运动序列
   - Corruption strategy：训练时随机将部分ground truth token替换为模型预测的token，模拟推理时的分布偏移

### 核心直觉

**什么变了**：从"复杂架构+复杂训练"到"简单架构+关键训练技巧"。

**哪些分布/约束/信息瓶颈变了**：
- EMA + Code Reset解决了codebook collapse → 离散表示的信息容量从理论值的30%提升到90%+，VQ-VAE的重建质量大幅提升
- Corruption strategy在训练时注入了推理时的分布偏移 → 模型学会了在不完美输入下仍能生成合理输出，误差累积被显著抑制
- 两个改进叠加使得简单的VQ-VAE+GPT框架的性能上限被大幅提高，超越了当时更复杂的扩散方法

**为什么有效**：运动生成的核心瓶颈不在模型架构的复杂度，而在离散表示的质量和自回归生成的稳定性。T2M-GPT精准地解决了这两个瓶颈，用最简单的架构达到了最好的效果。

**权衡**：VQ离散化仍有量化误差上限；CLIP文本嵌入不是运动最优的语义表示（后续LaMP改进了这一点）；数据规模是性能天花板。

---

## Part III：证据与局限

### 关键实验信号

- **FID大幅领先**：HumanML3D上FID 0.116 vs. MotionDiffuse 0.630，证明VQ-VAE+GPT范式在运动生成中的竞争力
- **R-Precision可比**：文本-运动一致性与扩散方法可比，说明离散表示未损失语义信息
- **消融验证**：去掉EMA/Code Reset后codebook利用率降至<30%，FID退化至>0.5；去掉corruption strategy后长序列生成质量显著下降
- **数据规模分析**：在HumanML3D子集上的实验表明，数据规模是当前方法的主要瓶颈

### 局限与可复用组件

- **局限**：数据规模限制了性能上限；CLIP文本嵌入对运动语义的编码不够精确；生成多样性不如扩散方法；长序列（>10秒）仍有退化
- **可复用**：EMA + Code Reset的VQ-VAE训练方案已成为运动生成领域的标准做法（被ScaMo、MotionGPT等广泛采用）；corruption strategy可迁移到任何自回归生成任务；整体框架为后续VQ+GPT运动生成工作（Being-M0、ScaMo等）奠定了基础

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/CVPR_2023/2023_T2M_GPT_Generating_Human_Motion_from_Textual_Descriptions_with_Discrete_Representations.pdf]]
