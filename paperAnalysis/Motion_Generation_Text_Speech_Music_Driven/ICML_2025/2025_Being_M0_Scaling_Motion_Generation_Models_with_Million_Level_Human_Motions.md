---
title: "Being-M0: Scaling Motion Generation Models with Million-Level Human Motions"
venue: ICML
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - vq-vae
  - autoregressive
  - dataset/MotionLib
  - dataset/HumanML3D
  - repr/joint-rotation
  - opensource/partial
core_operator: 2D Lookup-Free Motion Tokenizer（Motionbook）+ 百万级运动数据集MotionLib，首次在运动生成中验证数据与模型规模的Scaling行为
primary_logic: |
  文本描述 → LLM tokenizer编码
  → 自回归Transformer在Motionbook离散token空间中逐步预测运动token
  → Motionbook解码：2D lookup-free量化（时间×空间双维度FSQ）保留细粒度运动细节
  → 百万级MotionLib数据集（15×现有规模）+ 层次化文本标注 → 大规模训练验证Scaling Law
claims:
  - "MotionLib是首个百万级运动生成数据集，规模至少为现有数据集的15倍，包含层次化文本描述"
  - "Being-M0首次在运动生成领域验证了数据规模和模型规模的Scaling行为，性能随两者增长持续提升"
  - "2D Lookup-Free Motion Tokenizer通过时间×空间双维度FSQ量化，在扩大codebook容量的同时保留细粒度运动细节"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICML_2025/2025_Being_M0_Scaling_Motion_Generation_Models_with_Million_Level_Human_Motions.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# Being-M0: Scaling Motion Generation Models with Million-Level Human Motions

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://beingbeyond.github.io/Being-M0/) · [ICML 2025](https://arxiv.org/abs/2410.03311)
> - **Summary**: Being-M0构建了首个百万级运动数据集MotionLib（15×现有规模），提出2D Lookup-Free Motion Tokenizer（Motionbook），首次在运动生成中系统验证数据规模和模型规模的Scaling行为，展示了通向通用运动大模型的可行路径。
> - **Key Performance**:
>   - MotionLib包含百万级运动片段，规模为现有最大数据集的15倍以上
>   - Being-M0在广泛动作类别（含未见类别）上展现稳健生成能力，性能随数据和模型规模持续提升

---

## Part I：问题与挑战

### 真正的卡点

构建通用运动大模型面临**数据规模瓶颈**和**运动表示效率**两大核心挑战：

- **数据匮乏**：现有运动数据集（HumanML3D ~15K片段、MotionX ~80K片段）规模远不足以支撑大模型训练，导致模型泛化能力受限，无法覆盖长尾和未见动作类别
- **运动token化效率低**：传统VQ-VAE的1D量化方案在扩大codebook时面临利用率崩塌（codebook collapse），无法在保持细粒度运动细节的同时扩展表示容量
- **Scaling行为未知**：运动生成领域尚未有系统性的Scaling Law研究，模型/数据/计算的最优配比缺乏理论指导

### 输入/输出接口

- 输入：自然语言文本描述（层次化：粗粒度动作类别 + 细粒度动作描述）
- 输出：3D人体运动序列（关节旋转表示）

---

## Part II：方法与洞察

### 整体设计

Being-M0的核心贡献在数据和表示两个层面：

1. **MotionLib数据集**：百万级运动片段，来源包括MoCap和视频重建，配备层次化文本标注（粗粒度类别标签 + 细粒度自然语言描述）。数据清洗流程包括运动质量过滤、重复检测、标注一致性校验
2. **Motionbook（2D Lookup-Free Motion Tokenizer）**：
   - 紧凑无损特征：设计新的运动特征表示，减少冗余维度同时保留关键运动信息
   - 2D量化：将运动token在时间和空间两个维度上分别量化（FSQ，有限标量量化），避免1D VQ-VAE的codebook collapse问题
   - Lookup-Free：不维护显式codebook，直接通过标量量化映射，codebook容量可大幅扩展而不损失利用率
3. **自回归生成**：基于LLM架构的自回归Transformer，在Motionbook离散token空间中逐步预测运动序列

### 核心直觉

**什么变了**：从"小数据+1D VQ-VAE+小模型"到"百万级数据+2D Lookup-Free量化+大模型"。

**哪些分布/约束/信息瓶颈变了**：
- 数据规模15×提升 → 训练分布从窄域动作扩展到广域覆盖，包括长尾和未见类别
- 2D Lookup-Free量化打破了1D VQ-VAE的codebook容量瓶颈 → 运动token的信息保真度提升，细粒度运动细节（手指、面部微表情等）不再被量化误差抹平
- 两者叠加使得Scaling行为首次在运动生成中可观测：更多数据+更大模型 → 持续的性能提升，而非饱和

**为什么有效**：FSQ的标量量化天然避免codebook collapse（每个量化级别都被均匀使用），2D分解让时间和空间维度独立编码，减少了维度间的干扰。百万级数据提供了足够的分布覆盖，使大模型不会过拟合。

**权衡**：百万级数据的采集和清洗成本极高；视频重建数据的质量仍不及MoCap，引入噪声；2D量化增加了tokenizer的复杂度。

---

## Part III：证据与局限

### 关键实验信号

- **Scaling验证**：系统实验表明，固定模型增大数据、固定数据增大模型，FID和R-Precision均持续改善，未出现饱和——首次在运动生成中确认Scaling行为
- **未见类别泛化**：Being-M0在训练集未覆盖的动作类别上仍能生成合理运动，证明大规模数据带来的泛化能力
- **Motionbook消融**：2D Lookup-Free量化在不同codebook规模下均保持高利用率（>95%），而传统VQ-VAE在大codebook时利用率骤降至<30%
- **定量指标**：在HumanML3D测试集上FID和多样性指标优于同规模基线

### 局限与可复用组件

- **局限**：MotionLib数据集中视频重建部分质量受限，影响生成质量上限；百万级数据的标注一致性仍有改进空间；未充分探索模型规模的上限（最大模型仍远小于LLM）
- **可复用**：Motionbook的2D Lookup-Free量化方案可直接迁移到其他时序信号的离散化（语音、手势等）；MotionLib的层次化标注策略适用于任何大规模运动数据集构建；Scaling Law的实验方法论为运动生成领域的资源分配提供了参考框架

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICML_2025/2025_Being_M0_Scaling_Motion_Generation_Models_with_Million_Level_Human_Motions.pdf]]
