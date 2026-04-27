---
title: Speculative Decoding for Autoregressive Video Generation
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.17397
aliases:
- SDVG：将投机解码从LLM扩展到自回归视频生成的训练-free加速方法
- SDAVG
- 核心变化是将 LLM 投机解码中不可用的「token 概率比较」替换为
code_url: https://github.com/longxiang-ai/awesome-video-diffusions
modalities:
- Image
---

# Speculative Decoding for Autoregressive Video Generation

[Paper](https://arxiv.org/abs/2604.17397) | [Code](https://github.com/longxiang-ai/awesome-video-diffusions)

**Topics**: [[T__Agent]], [[T__Video_Generation]], [[T__Compression]], [[T__Benchmark_-_Evaluation]] | **Datasets**: MovieGenVideoBench

> [!tip] 核心洞察
> 核心变化是将 LLM 投机解码中不可用的「token 概率比较」替换为「图像质量奖励信号」，使得块级路由决策在连续视频域成为可能。有效性来源于两点：一是自回归视频生成的块级结构天然提供了离散的决策单元，每块可独立评估后再提交 KV cache；二是最差帧聚合能够捕捉平均分会掩盖的单帧伪影，使路由信号对视觉质量更敏感。本质上，这是一个「用感知质量代理分布匹配」的工程替换，而非理论上的精确等价。

## 基本信息

- **论文标题**: Speculative Decoding for Autoregressive Video Generation
- **方法名称**: SDVG (Speculative Decoding for Video Generation)
- **基准模型**: Wan2.1-T2V-14B (target), Wan2.1-T2V-1.3B (drafter), 经 Self-Forcing 蒸馏
- **评估基准**: MovieGenVideoBench (1003 prompts)
- **硬件环境**: NVIDIA RTX A6000 (48 GB) × 2
- **代码/数据**: 未在提取内容中明确提及开源链接
- **训练成本**: 训练-free (零训练开销)

## 核心主张

SDVG 首次将 LLM 领域的投机解码框架适配到自回归视频扩散模型，通过**图像质量路由器**（ImageReward + 最坏帧聚合）替代 token 级精确拒绝采样，在**无需训练**的前提下实现 **1.59× 加速**并保留 **98.1%** 的目标模型质量（VisionReward 0.0773 vs 0.0788）。

**关键证据**: (1) 主实验表显示 τ=-0.7 时 60.9s vs target-only 97.0s，接受率 73.1%；(2) 消融实验证实 min-frame 聚合优于 avg-frame（0.0773 vs 0.0755），随机路由严重退化（0.0706）。置信度：**高（0.95）**。

## 研究动机

视频扩散模型的自回归推理成本极高（14B 模型逐块生成），而现有加速方案存在明显缺口：

1. **LLM 投机解码无法直接迁移**: 视频块是连续时空张量，**没有 token 级分布**可供精确拒绝采样（Introduction, confidence 0.95）
2. **蒸馏/量化方案不足**: Self-Forcing 等蒸馏方法虽能加速，但单独使用 1.3B drafter 质量骤降（VisionReward 仅 0.0644）
3. **相关加速方法缺乏对比**: SRDiffusion、T-Stitch、HybridStitch、Phased Consistency Models 等方法在论文中仅被引用，**未进行数值对比**

核心缺口：**训练-free 的、保持目标模型质量的高倍加速机制**。

## 方法流程

```
[Prompt] → UMT5-XXL 文本编码 → 文本嵌入
                ↓
    ┌─────────────────────────┐
    │  1.3B Drafter (Wan2.1)   │ ← 4步去噪 [1000, 937, 833, 625, 0]
    │  生成候选 latent block   │
    └─────────────────────────┘
                ↓
         VAE 解码为像素帧
                ↓
    ┌─────────────────────────┐
    │  ImageReward 逐帧评分    │ ← 新组件：最坏帧聚合 min_{i=1}^F
    │  q_b = min R(f_i^(b), p) │
    └─────────────────────────┘
                ↓
    阈值比较 q_b ≥ τ?  ──否──→ VAE cache 恢复 → 14B Target 重生成
                ↓ 是
         KV cache 集成，接受草稿块
                ↓
    【首块强制拒绝】→ 始终用 Target 锚定场景构图
```

**关键创新**: 用**分布偏移容忍**替代 LLM 的**精确分布匹配**，接受 drafter 与 target 之间的质量差异。

## 关键公式

**1. 最坏帧聚合（核心创新）**
```latex
q_b = \min_{i=1}^{F} \mathcal{R}\left(\mathbf{f}_i^{(b)},\, p\right)
```
对第 $b$ 块的 $F$ 帧取 ImageReward 最小值，**单帧瑕疵即触发拒绝**。

**2. 二元接受策略**
```latex
\pi(q_b) = \mathbb{1}[q_b \geq \tau]
```
阈值 $\tau=-0.7$ 为默认质量-速度权衡点，探索范围 $[-1.0, -0.7]$。

**3. 去噪时间步（继承 Self-Forcing）**
```latex
\mathbf{t} = [1000, 937, 833, 625, 0]
```
4步确定性去噪，从纯噪声到干净数据。

**公式谱系**: 式1-2为**全新设计**（无 baseline 对应公式）；式3为**继承**自 Self-Forcing 蒸馏设置。

## 实验结果

| 配置 | VisionReward | 时间 | 加速比 |
|:---|:---|:---|:---|
| Target-only (14B) | **0.0788** | 97.0s | 1.00× |
| **SDVG (τ=-0.7)** | **0.0773** (-1.9%) | **60.9s** | **1.59×** |
| Draft-only (1.3B) | 0.0644 (-18.3%) | 25.7s | 3.77× |

**核心指标**: 接受率 **73.1%**（非首块），质量保留率 **98.1%**。

**消融验证**:
- min-frame → avg-frame: 0.0753（-0.0020），**支持核心设计**
- 奖励路由 → 随机路由: 0.0706（-0.0067），**严重退化**
- 强制首块拒绝: 0.0771 vs 无拒绝随机路由 0.0706，**场景锚定有效**

**证据强度**: 0.75/1.0。主要弱点：未与 SRDiffusion、T-Stitch 等数值对比；仅单阈值报告；固定随机种子 42。

## 相关工作

**方法来源（直接继承）**:
1. *Fast inference from transformers via speculative decoding* — 核心算法框架，draft-then-verify 结构
2. *Accelerating large language model decoding with speculative sampling* — 精确拒绝采样机制，SDVG 明确**不采用**其 token 级匹配

**基线模型**:
3. *Self-Forcing: Bridging the train-test gap in autoregressive video diffusion* — target (14B) 和 drafter (1.3B) 的蒸馏来源
4. *Wan: Open and advanced large-scale video generative models* — 基础架构 Wan2.1-T2V 系列

**未充分对比的加速方法**（仅引用无数值）:
- SRDiffusion [3], T-Stitch [9], HybridStitch [11], Phased Consistency Models [13], One-step diffusion with DMD [17]

**关系定位**: SDVG 是投机解码从**离散 token 域**到**连续视频域**的首个适配器，填补了 LLM 加速与视频生成之间的方法空白。

## 方法谱系

**父方法**: 投机解码（Speculative Decoding for LLMs）

| 槽位 | 父方法取值 | SDVG 修改 | 修改类型 |
|:---|:---|:---|:---|
| **verification_mechanism** | token-level 精确拒绝采样，分布匹配 $P_{target}(x) \geq P_{drafter}(x)$ | **ImageReward 图像质量路由器** + VAE 解码帧评分 + 最坏帧聚合 | **替换** |
| **inference_strategy** | 标准自回归单模型生成 | **draft-then-verify** 双模型 + 首块强制拒绝 + KV cache 集成 | **修改** |
| **reward_design** | 无（LLM 用概率比） | **ImageReward 最小帧聚合** + 固定阈值 τ | **新增** |

**继承保留**: 核心 draft-then-verify 结构、KV cache 机制、加速比计算逻辑。

**谱系意义**: SDVG 证明投机解码的**验证层可解耦为任意质量评估函数**，不局限于概率分布匹配，为向其他连续域（音频、3D）扩展提供范式。

## 局限与展望

**论文明确承认**:
1. ImageReward 存在**时间不一致性**问题（未针对视频时序训练）
2. 接受**分布偏移**——SDVG 不保证精确匹配 target 分布，与 LLM 投机解码的数学保证不同

**分析推断的额外局限**:
3. **阈值调参敏感**: 仅报告 τ=-0.7，完整 Pareto 曲线未展示；τ 范围窄（[-1.0, -0.7]）
4. **基准对比不足**: SRDiffusion、T-Stitch 等同类加速方法无数值比较，公平性存疑
5. **单点评估**: 固定随机种子 42，未报告方差；单基准 MovieGenVideoBench
6. **硬件依赖**: 需双 A6000 GPU 分离部署（GPU 0: DiT, GPU 1: VAE+ImageReward）

**未来方向**: 训练专用视频奖励模型替代 ImageReward；动态阈值自适应；扩展至多 GPU 并行；向实时交互式视频生成演进。

## 知识图谱定位

**任务节点**: 
- `autoregressive video generation`（自回归视频生成）— 核心任务
- `video diffusion acceleration`（视频扩散加速）— 优化目标，训练-free 属性

**方法节点**:
- `SDVG` — **新建方法节点**，连接投机解码与视频生成两大子图
- `speculative decoding`（机制节点）— 被扩展的父方法
- `Self-Forcing` + `Wan2.1` — 基础模型节点
- `ImageReward`, `worst-frame aggregation`, `force-reject first block` — 新增机制组件

**数据集节点**: `MovieGenVideoBench` — 评估锚点

**图谱贡献**: SDVG 建立了 **LLM 高效推理 → 视频生成高效推理** 的跨域桥接边，证明**验证层的可替换性**（verification_mechanism 槽位可插拔）。这为后续工作开辟了"连续域投机解码"子图，潜在扩展至音频扩散、3D 生成、多模态序列等节点。其**训练-free** 特性使其在"无需重训练的推理优化"簇中具有独特定位。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/69923205-fb3e-4542-86b1-c85b9b334891/figures/Figure_1.png)
*Figure 1 (qualitative): Qualitative comparison on MovieGenVideoBench.*


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/69923205-fb3e-4542-86b1-c85b9b334891/figures/Table_1.png)
*Table 1 (quantitative): Main results on 1003 MovieGenVideoBench prompts (832×480).*


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/69923205-fb3e-4542-86b1-c85b9b334891/figures/Figure_3.png)
*Figure 3 (result): Quality–speed Pareto curve.*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/69923205-fb3e-4542-86b1-c85b9b334891/figures/Figure_2.png)
*Figure 2 (pipeline): SDVG inference pipeline.*


