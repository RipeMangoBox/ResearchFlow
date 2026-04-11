---
title: "ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model"
venue: CVPR
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - autoregressive
  - vq-vae
  - dataset/MotionX
  - dataset/HumanML3D
  - repr/SMPL-X
  - opensource/full
core_operator: Motion FSQ-VAE + Text-Prefix自回归Transformer，首次在运动生成中确认Scaling Law存在（对数损失-计算预算幂律关系）
primary_logic: |
  文本前缀 → Text-Prefix自回归Transformer逐token预测运动序列
  → Motion FSQ-VAE：有限标量量化将连续运动编码为离散token（无codebook collapse）
  → Scaling Law验证：归一化测试损失与计算预算呈对数关系；非词表参数、词表参数、数据token与计算预算呈幂律关系
  → 基于Scaling Law预测最优模型/词表/数据配置 → ScaMo-3B在MotionX上实现SOTA
claims:
  - "ScaMo首次在运动生成领域确认Scaling Law的存在：归一化测试损失与计算预算呈对数关系"
  - "ScaMo-3B在MotionX测试集上FID 0.027，显著优于此前最优方法MoMask(0.045)"
  - "Motion FSQ-VAE通过有限标量量化避免codebook collapse，在不同codebook规模下均保持稳定的重建质量"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/CVPR_2025/2025_ScaMo_Exploring_the_Scaling_Law_in_Autoregressive_Motion_Generation_Model.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://shunlinlu.github.io/ScaMo/) · [CVPR 2025](https://arxiv.org/abs/2412.14322)
> - **Summary**: ScaMo首次在运动生成领域确认Scaling Law的存在——归一化测试损失与计算预算呈对数关系，并基于此预测最优模型/词表/数据配置。Motion FSQ-VAE通过有限标量量化避免codebook collapse，ScaMo-3B在MotionX上实现SOTA。
> - **Key Performance**:
>   - ScaMo-3B在MotionX测试集FID **0.027** vs. MoMask(0.045)
>   - 首次确认运动生成中的Scaling Law：损失与计算预算呈对数关系，参数/数据与计算预算呈幂律关系

---

## Part I：问题与挑战

### 真正的卡点

运动生成领域面临**Scaling Law未知**和**离散化质量瓶颈**两大核心挑战：

- **Scaling行为未验证**：NLP和视觉领域已广泛验证Scaling Law（更多数据+更大模型→可预测的性能提升），但运动生成领域缺乏系统性的Scaling研究，模型设计和资源分配缺乏理论指导
- **运动token化的codebook collapse**：传统VQ-VAE在扩大codebook规模时，大量code向量不被使用（利用率<30%），导致表示容量无法有效扩展，限制了自回归模型的上限
- **自回归框架的适配**：运动序列的连续性和物理约束使得直接套用NLP的自回归范式效果不佳，需要专门设计的tokenizer和生成策略

### 输入/输出接口

- 输入：自然语言文本描述（作为prefix token序列）
- 输出：3D人体运动序列（SMPL-X格式，通过FSQ-VAE解码）

---

## Part II：方法与洞察

### 整体设计

ScaMo构建了一个可扩展的运动生成框架，包含两个核心组件：

1. **Motion FSQ-VAE**：
   - 用有限标量量化（Finite Scalar Quantization）替代传统VQ-VAE的向量量化
   - 每个潜变量维度被量化到有限个离散级别（如8级），无需维护显式codebook
   - 天然避免codebook collapse：每个量化级别都被均匀使用，利用率接近100%
   - 支持灵活扩展codebook容量（通过增加维度或量化级别）

2. **Text-Prefix自回归Transformer**：
   - 文本token作为prefix，运动token作为续写目标
   - 标准因果Transformer架构，支持从百万到数十亿参数的扩展
   - 训练目标：next-token prediction on motion tokens

### 核心直觉

**什么变了**：从"固定规模模型+固定数据"到"系统性探索模型/词表/数据的最优Scaling配比"。

**哪些分布/约束/信息瓶颈变了**：
- FSQ消除了codebook collapse瓶颈 → 词表容量可以随计算预算自由扩展，不再是性能天花板
- 系统性Scaling实验揭示了三组幂律关系：非词表参数 vs. 计算预算、词表参数 vs. 计算预算、数据token vs. 计算预算 → 给定计算预算可预测最优配置
- 归一化测试损失与计算预算呈对数关系 → 性能提升是可预测的，不存在突然的饱和点

**为什么有效**：FSQ的均匀量化保证了codebook的完全利用，使得扩大词表真正等价于扩大表示容量。Text-Prefix架构将文本理解和运动生成统一在同一个自回归框架中，使得NLP领域的Scaling经验可以直接迁移。

**权衡**：FSQ的均匀量化可能不如学习到的VQ codebook在特定分布上高效；3B模型的推理成本较高；当前数据规模仍有限，未观察到涌现能力。

---

## Part III：证据与局限

### 关键实验信号

- **Scaling Law确认**：归一化测试损失与计算预算呈对数关系（R²>0.99），非词表参数、词表参数、数据token与计算预算分别呈幂律关系——首次在运动生成中系统验证
- **最优配置预测**：基于Scaling Law预测1e18计算预算下的最优配置，实际训练结果与预测高度吻合
- **ScaMo-3B SOTA**：MotionX测试集FID 0.027 vs. MoMask 0.045，R-Precision Top-1显著提升
- **FSQ消融**：FSQ在所有codebook规模下利用率>95%，而VQ-VAE在大codebook时降至<30%；FSQ重建质量随codebook扩大持续改善

### 局限与可复用组件

- **局限**：数据规模仍有限（MotionX ~80K），未观察到涌现能力；视频重建数据质量影响生成上限；3B模型推理延迟较高
- **可复用**：Motion FSQ-VAE可直接用于其他运动生成框架的tokenizer替换；Scaling Law的实验方法论（系统性变化模型/词表/数据规模+拟合幂律）可迁移到任何新领域的Scaling研究；Text-Prefix自回归范式为运动生成提供了简洁统一的生成框架

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/CVPR_2025/2025_ScaMo_Exploring_the_Scaling_Law_in_Autoregressive_Motion_Generation_Model.pdf]]
