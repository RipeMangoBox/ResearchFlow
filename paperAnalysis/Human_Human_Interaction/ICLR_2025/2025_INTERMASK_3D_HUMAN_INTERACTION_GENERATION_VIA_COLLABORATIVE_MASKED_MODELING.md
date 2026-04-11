---
title: "InterMask: 3D Human Interaction Generation via Collaborative Masked Modeling"
venue: ICLR
year: 2025
tags:
  - Human_Human_Interaction
  - task/human-human-interaction
  - vq-vae
  - dataset/InterHuman
  - dataset/InterX
  - repr/HumanML3D-263d
  - opensource/full
core_operator: 协作掩码建模（Collaborative Masked Modeling）：2D离散运动token图 + 双人协作掩码预测，在离散空间中联合建模两人交互运动的时空依赖
primary_logic: |
  文本描述 → 条件嵌入
  → 两人运动各自通过2D VQ-VAE编码为2D离散token图（时间×空间维度）
  → 协作掩码建模：随机掩码两人token → Transformer联合预测被掩码token
  → 推理：从全掩码开始，渐进式填充两人token → 2D VQ-VAE解码 → 双人交互运动
  → 无需修改即支持反应生成（给定一人运动，生成另一人反应）
claims:
  - "InterMask在InterHuman上FID 5.154 vs. in2IN(5.535)，在InterX上FID 0.399 vs. InterGen(5.207)"
  - "2D离散token图比1D token序列更好地保留细粒度时空细节，空间维度编码促进身体部位间的协调"
  - "协作掩码建模天然支持反应生成（给定一人运动生成另一人反应），无需模型重设计或微调"
pdf_ref: paperPDFs/Human_Human_Interaction/ICLR_2025/2025_INTERMASK_3D_HUMAN_INTERACTION_GENERATION_VIA_COLLABORATIVE_MASKED_MODELING.pdf
category: Human_Human_Interaction
---

# InterMask: 3D Human Interaction Generation via Collaborative Masked Modeling

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://gohar-malik.github.io/intermask) · [ICLR 2025](https://arxiv.org/abs/2410.10010)
> - **Summary**: InterMask提出协作掩码建模框架，将两人运动编码为2D离散token图，通过Transformer联合掩码预测建模双人交互的时空依赖，在InterHuman和InterX上大幅超越扩散基线，且天然支持反应生成。
> - **Key Performance**:
>   - InterHuman FID **5.154** vs. in2IN(5.535)；InterX FID **0.399** vs. InterGen(5.207)
>   - 天然支持反应生成，无需模型修改或微调

---

## Part I：问题与挑战

### 真正的卡点

文本驱动双人交互生成面临**交互时空依赖建模**和**运动表示精度**两大核心挑战：

- **双人耦合建模**：两人交互运动高度耦合——一人的动作直接影响另一人的反应。现有扩散方法在连续空间中建模这种耦合，但生成结果常缺乏真实感和保真度
- **1D token的信息损失**：传统VQ-VAE将运动编码为1D token序列，丢失了空间维度信息（不同身体部位的独立变化），限制了细粒度交互建模
- **反应生成的额外成本**：给定一人运动生成另一人反应是重要应用场景，但现有方法通常需要为此重新设计或微调模型

### 输入/输出接口

- 输入：文本描述（交互生成）或 文本+一人运动（反应生成）
- 输出：两人3D运动序列（HumanML3D 263维表示）

---

## Part II：方法与洞察

### 整体设计

InterMask的核心创新在表示和建模两个层面：

1. **2D VQ-VAE**：
   - 将运动序列编码为2D token图（时间维度 × 空间维度）
   - 空间维度保留了不同身体部位的独立编码 → 每个token同时包含时间和空间信息
   - 比1D token序列更好地保留细粒度时空细节

2. **协作掩码建模**：
   - 训练时：随机掩码两人的2D token图，Transformer联合预测被掩码token
   - Transformer架构：专门设计的注意力模式捕捉人内（同一人不同部位/时间）和人间（两人之间）的时空依赖
   - 推理时：从全掩码开始，按置信度渐进式填充token（类似MaskGIT）

3. **反应生成**：推理时只掩码一人的token，另一人的token作为条件 → 天然支持，无需修改

### 核心直觉

**什么变了**：从"连续空间扩散建模双人交互"到"离散空间协作掩码建模"。

**哪些分布/约束/信息瓶颈变了**：
- 2D token图保留了空间维度 → 模型可以独立推理不同身体部位的交互（如手部接触vs脚步协调），而非将所有信息压缩到1D序列
- 掩码建模的双向注意力 → 每个token的预测可以同时参考两人的所有已知token，而非自回归的单向依赖
- 渐进式填充策略 → 高置信度token先填充，为后续低置信度token提供更多上下文，生成质量逐步提升

**为什么有效**：双人交互的核心是时空耦合——A的手在t时刻的位置影响B的手在t+1时刻的反应。2D token图天然编码了这种时空结构，掩码建模的双向注意力则允许模型在预测时同时考虑两人的全局上下文。

**权衡**：2D token图增加了token数量，推理时间较长；掩码建模的渐进式填充需要多次前向传播；VQ离散化仍有量化误差。

---

## Part III：证据与局限

### 关键实验信号

- **大幅超越扩散基线**：InterX上FID 0.399 vs. InterGen 5.207（13×提升），证明离散掩码建模在交互生成上的优势
- **2D vs 1D消融**：2D token图比1D token序列在FID和多样性上均有显著提升
- **反应生成**：无需微调即可生成高质量反应运动，FID与专门训练的反应生成模型可比
- **用户研究**：在真实感和文本一致性上均被评为优于基线方法

### 局限与可复用组件

- **局限**：仅支持双人交互，未扩展到多人；2D token图的token数量较大，推理效率有待优化；依赖HumanML3D 263维表示，限制了手部和面部细节
- **可复用**：2D VQ-VAE的时空离散化方案可迁移到其他需要保留空间结构的运动表示任务；协作掩码建模框架可用于任何多agent联合生成；渐进式填充策略适用于任何离散空间的条件生成

---

## 本地 PDF 引用

![[paperPDFs/Human_Human_Interaction/ICLR_2025/2025_INTERMASK_3D_HUMAN_INTERACTION_GENERATION_VIA_COLLABORATIVE_MASKED_MODELING.pdf]]
