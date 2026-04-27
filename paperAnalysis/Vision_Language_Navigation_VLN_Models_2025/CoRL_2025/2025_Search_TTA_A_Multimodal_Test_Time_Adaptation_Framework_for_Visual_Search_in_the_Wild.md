---
title: "Search-TTA: A Multimodal Test-Time Adaptation Framework for Visual Search in the Wild"
venue: CoRL
year: 2025
tags:
  - Embodied_AI
  - task/visual-search
  - test-time-adaptation
  - contrastive-learning
  - reinforcement-learning
  - dataset/AVS-Bench
  - opensource/no
core_operator: "将卫星图像编码到共享多模态嵌入空间，并用基于空间泊松点过程的覆盖率加权在线梯度更新持续修正搜索概率图。"
primary_logic: |
  卫星图像 + 查询模态（图像/文本/声音）+ 搜索中的正负检测反馈
  → 共享嵌入生成初始目标概率图并把卫星 patch 聚成语义区域
  → 用覆盖率感知的 SPPP 式测试时更新调整卫星编码器
  → 输出随搜索迭代改进的概率图，驱动规划器更高效地找到目标
claims:
  - "在 AVS-Bench 上，Search-TTA 对差先验最有效：在训练数据受限时，底部 2% 初始预测样本的目标发现率提升最高达 30.0% [evidence: comparison]"
  - "在相同 RL 规划器下，轻量 CLIP+Search-TTA 的发现率整体上优于或接近 LISA、LLM-Seg、Qwen2+GroundedSAM 与 LLaVA+GroundedSAM，同时单次推理/TTA 更快 [evidence: comparison]"
  - "无需对卫星编码器做文本或声音联合微调，文本查询与图像查询的性能差距至多 0.9%，声音差距至多 2.4%，表明共享嵌入带来零样本模态泛化 [evidence: comparison]"
related_work_position:
  extends: "PSVAS (Sarkar et al. 2023)"
  competes_with: "VAS (Sarkar et al. 2024); LISA (Lai et al. 2024)"
  complementary_to: "GroundedSAM (Ren et al. 2024); LoRA (Hu et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Language_Navigation_VLN_Models_2025/CoRL_2025/2025_Search_TTA_A_Multimodal_Test_Time_Adaptation_Framework_for_Visual_Search_in_the_Wild.pdf
category: Embodied_AI
---

# Search-TTA: A Multimodal Test-Time Adaptation Framework for Visual Search in the Wild

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.11350), [Project](https://search-tta.github.io)
> - **Summary**: 这篇工作把“由卫星图生成的一次性静态搜索先验”改造成“可被机载观测在线纠偏的多模态概率图”，从而提升野外自主视觉搜索效率。
> - **Key Performance**: AVS-Bench 上目标发现率最高提升 30.0%；概率图 RMSE 最高改善 8.5%。

> [!info] **Agent Summary**
> - **task_path**: 卫星图像 + 查询模态（图像/文本/声音）+ 在线检测反馈 -> 目标概率图 -> 搜索轨迹
> - **bottleneck**: 域外卫星图像上的 VLM 先验会 hallucinate，而传统搜索框架无法用在线观测去持续修正这些错误先验
> - **mechanism_delta**: 把静态 CLIP 概率图改成基于语义区域覆盖率加权的 SPPP 式闭环更新，使正负检测都能反向重塑卫星编码器输出
> - **evidence_signal**: 收益显著集中在最差先验样本，底部 2% 样本目标发现率提升最高达 30.0%
> - **reusable_ops**: [patch-level satellite-query alignment, coverage-weighted online score-map refinement]
> - **failure_modes**: [长时间只有负样本时纠偏速度有限, 检测器误检漏检或区域聚类失真会污染更新信号]
> - **open_questions**: [如何在非完美检测器下做稳定 TTA, 如何支持多目标搜索而不发生灾难性遗忘]

## Part I：问题与挑战

这篇论文解决的是 **outdoor autonomous visual search**：机器人要在预算有限的情况下，借助卫星图在野外寻找目标，但目标本身通常 **并不能直接从卫星图中被看见**。因此系统只能依赖间接语义线索，例如植被、水域、城市区域、海岸线等，来推断“哪里更可能有目标”。

**输入/输出接口**大致是：
- **输入**：卫星图像 `S`，以及查询 `G`（地面图像 / 文本 / 声音之一），再加上搜索过程中的机载检测反馈。
- **输出**：一个可供规划器使用的目标概率图，以及在预算约束下的搜索轨迹。

**真正的瓶颈**不是“有没有先验”，而是：  
**先验一旦错了，后续搜索会被持续误导，而现有 VLM/规划器组合通常没有在线纠错回路。**

具体来说有三层难点：
1. **卫星域偏移**：现成 VLM 主要在自然图像上预训练，面对遥感视角和生物 taxonomy 时容易 hallucinate。
2. **数据稀缺**：缺少大规模、野外、目标不可直视的卫星搜索数据。
3. **负反馈难用**：搜索时往往先收集到大量“没找到”的负样本；如果直接强烈惩罚，很容易把真正有价值的区域也过早压低。

**为什么现在值得解决**：
- 野外搜索受电量、FOV、航时强约束，错误先验会直接变成飞行浪费。
- foundation models 已经足够强，能提供“粗先验”；现在缺的是 **在线适配机制**，不是再堆一个更大的静态模型。
- 作者同步构建了 **AVS-Bench**（380k 训练图像，8k 验证图像，含 in-/out-domain taxonomy），填补了这一任务的数据空缺。

**边界条件**：
- 搜索空间被离散成 24×24 网格。
- 传感器模型很窄，每步只观测当前位置。
- 论文主要在 **完美二值检测器** 假设下评估。
- 任务目标是单类目标搜索，而不是多目标同时持续学习。

## Part II：方法与洞察

Search-TTA 本质上是一个 **“多模态先验生成器 + 规划器无关接口 + 在线纠偏回路”**。

### 1) 共享嵌入的多模态概率图

作者先把卫星图编码器对齐到一个共享多模态空间里：
- 选用 **BioCLIP** 作为地面图像/文本空间；
- 再把一个 **CLIP 卫星图编码器** 通过 **patch-level contrastive learning** 对齐到这个空间；
- 推理时，对卫星图每个 patch 与查询 embedding 做余弦相似度，得到 24×24 概率图。

这一步的关键不是“看见目标”，而是学会：
**某类环境语义 ↔ 某类目标更可能出现** 的弱对应关系。  
因此即使目标本体不可见，也能形成搜索先验。

### 2) 规划器解耦接口

Search-TTA 不把视觉和规划做成端到端黑盒，而是输出一个可插拔的概率图，接到：
- RL planner
- Information Surfing (IS)
- 其他能消费 heatmap 的规划器

这样做的意义是：**视觉 backbone 和 planner 可以分别替换**，避免每换一个模块都重新训练整套系统。

### 3) 基于 SPPP 的 Test-Time Adaptation

在线适配是这篇论文的核心。

作者先对卫星 patch embedding 做 **k-means 聚类**，得到若干“语义区域”。然后在搜索过程中：
- 如果找到目标，提升对应区域的强度；
- 如果没找到目标，不是立即强烈惩罚，而是根据该区域 **已覆盖比例** 去加权负更新。

其灵感来自 **Spatial Poisson Point Process (SPPP)**，但作者把它改造成适合在线搜索的版本。  
核心思想是：

- **正样本**是强证据；
- **负样本**只有在“这个区域已经被看得足够多”时，才是强证据；
- 因此负更新权重要随区域覆盖率上升而增大。

此外，作者每轮 TTA 都从 base encoder 权重重新开始，再结合逐步增强的学习率，避免在线更新累积漂移。

### 核心直觉

这篇论文真正调的“因果旋钮”不是换更大的 VLM，而是改变了 **观测如何进入视觉先验**：

- **原来**：观测只影响 planner 的状态，概率图基本静态。
- **现在**：观测直接回流到卫星编码器，概率图在搜索中持续重估。
- **关键约束变化**：把“负样本=强否定”改成“负样本=与区域覆盖率相关的弱证据”。
- **能力变化**：系统不再被初始 hallucinated prior 长期绑死，尤其能修复最差那批先验。

为什么这会有效？  
因为野外搜索里的负样本高度不可靠：你只看了森林的一小块，没有发现熊，并不代表整片森林都没熊。  
覆盖率加权后，模型不会因为少量早期负样本就把高价值语义区域压塌；一旦出现正样本，相关区域又能被迅速放大，于是 planner 会更快转向真正高概率区域。

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| patch-level 卫星-地面对齐 | 卫星图中目标不可见，只能靠环境语义 | 能从图像/文本/声音统一生成先验 | 需要大量配对数据，学习的是相关性而非显式因果 |
| k-means 语义区域 | 单点负样本太噪声 | 能按区域覆盖率解释“没找到”的可信度 | 聚类错了会误导更新 |
| 覆盖率加权 SPPP 更新 | 早期负样本会造成模式坍塌 | 坏先验可被在线修复，worst-case 收益大 | 仍需在线反传，模型越大越慢 |
| vision/planner 解耦 | 端到端方案替换模块成本高 | 可插 RL/IS 等现成规划器 | 不能联合最优表示与策略 |

## Part III：证据与局限

### 关键证据

**1. 最强信号：收益集中在“差先验”而不是平均样本。**  
这是论文最重要的实验结论。  
在 out-domain taxonomy 上，RL+TTA 相比 RL 无 TTA 的整体平均提升不算巨大，但在 **底部 2% 初始差先验** 上提升非常明显；作者还报告在训练数据更少时，这个提升可高达 **30.0%**。这说明 Search-TTA 真正在修复的是“先验错误”这个核心瓶颈，而不是只给强样本锦上添花。

**2. 概率图本身变好了，而不只是 planner 碰巧走对。**  
作者跟踪了 score map 的 RMSE，报告最高 **8.5%** 改善。  
这很关键，因为它说明：
- 改善不是单纯来自 RL 随机性；
- TTA 确实把视觉先验往更合理的分布推了。

**3. 轻量 CLIP + TTA 能和更大 VLM 打。**  
在相同 RL planner 下，CLIP+TTA 的发现率整体优于或接近 LISA、LLM-Seg、Qwen2+GroundedSAM、LLaVA+GroundedSAM，而且速度明显更快：A5000 上单次 CLIP+TTA 约 **0.15s**，而较大基线最高到 **3.48s**。  
这意味着作者的能力提升主要来自 **闭环纠偏机制**，而不只是堆参数。

**4. 零样本多模态泛化不是口号。**  
未对卫星编码器做文本/声音联合微调时：
- 文本查询与图像查询差距至多 **0.9%**
- 声音查询差距至多 **2.4%**

说明共享嵌入对齐确实带来了 **emergent alignment**。

**5. 硬件证据有价值，但仍是 case study。**  
Crazyflie + Gazebo 的 hardware-in-the-loop 实验中，TTA 找到 **5 个目标**，无 TTA 找到 **3 个**。这证明部署可行，但样本量很小，更适合作为可行性信号，而不是强统计证据。

**最该记住的两个指标**：
- **Targets Found**：直接衡量搜索任务收益
- **Score-map RMSE**：衡量先验是否真的被纠正

### 局限性

- **Fails when**: 检测器存在明显误检/漏检、目标极稀疏且长时间没有正样本、或 k-means 语义区域无法对应真实环境语义时，在线更新可能变慢、漂移，甚至误修正。
- **Assumes**: 窄 FOV、单格观测、完美二值检测器，以及“即使目标被卫星图遮挡，机载传感器仍能发现它”的假设；训练还依赖 AVS-Bench，以及部分由 GSNet + GPT4o + 人工示例生成的 pseudo score maps；全参数在线更新需要可用反向传播算力。
- **Not designed for**: 多目标同时搜索、显式建模物种交互/食物链/风险源等更深生态关系、以及超大模型上的高频全参数 TTA。

此外，有几个会影响复现和外推的现实约束需要明确指出：
- **RMSE 的“真值”不是人工密度图，而是 pseudo score map**，因此这个指标会继承数据生成管线的偏差。
- **硬件实验仅有少量案例**，证据强度应保守看待。
- 文中给出 project page，但**未明确给出代码发布信息**。
- 训练成本并不低：卫星编码器微调使用 **2×A6000，约 3.5 天**；虽然 Orin AGX 上推理/在线更新仍可运行（约 0.14s / 0.37s），但模型更大时 full-weight TTA 的部署成本会迅速上升。

### 可复用组件

- **共享嵌入式卫星先验生成器**：把卫星 patch 对齐到图像/文本/声音共同空间。
- **覆盖率加权负反馈机制**：把“未发现目标”转成不确定性敏感的弱监督。
- **planner-agnostic heatmap 接口**：适合接入不同 IPP / RL 搜索器，而不必端到端重训整套系统。

## Local PDF reference

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/CoRL_2025/2025_Search_TTA_A_Multimodal_Test_Time_Adaptation_Framework_for_Visual_Search_in_the_Wild.pdf]]