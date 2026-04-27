---
title: "Grasp What You Want: Embodied Dexterous Grasping System Driven by Your Voice"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/language-conditioned-grasping
  - task/referring-expression-segmentation
  - vision-language-model
  - skeleton-based-sampling
  - grasp-wrench-space
  - dataset/GraspNet-1Billion
  - opensource/no
core_operator: 用VLM先把模糊语音指令补全成可分割的目标描述，再以骨架/PCA约束的抓取采样配合力闭合与GWS筛选，输出可执行的灵巧抓取。
primary_logic: |
  语音指令 + RGB-D杂乱场景 → VLM进行目标属性补全与语义-目标对齐，分割得到目标点云 → 骨架/PCA提取物体方向并约束灵巧手抓取候选采样 → 力闭合、GWS质量与轨迹代价联合筛选 → 输出稳定抓取动作与执行路径
claims:
  - "Claim 1: 在 GraspNet-1Billion 上，RERE 将 Grounded SAM 的 IoU 从 44.9 提升到 64.4，并同时将 SEEM/Florence-2 提升到 50.4/55.8 [evidence: comparison]"
  - "Claim 2: 在 Grounded SAM 上逐步加入实例类别、纹理、形状、材质和位置属性时，IoU 最终提升到 64.4，说明属性级表达补全是主要增益来源 [evidence: ablation]"
  - "Claim 3: 在论文报告的多物体逐个抓取设置中，EDGS 取得 95.5% 成功率，高于 DexGraspNet 2.0 的 90.7% 和 ISAGrasp 的 54.8% [evidence: comparison]"
related_work_position:
  extends: "N/A"
  competes_with: "DexGraspNet 2.0 (Zhang et al. 2024); ISAGrasp (Chen et al. 2022)"
  complementary_to: "Grounded SAM (Ren et al. 2024); SEEM (Zou et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2024/arXiv_2024/2024_Grasp_What_You_Want_Embodied_Dexterous_Grasping_System_Driven_by_Your_Voice.pdf
category: Embodied_AI
---

# Grasp What You Want: Embodied Dexterous Grasping System Driven by Your Voice

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2412.10694)
> - **Summary**: 该文提出 EDGS，把“模糊语音找物”拆成先做跨模态指代表达补全、再做受人手启发的受约束灵巧抓取规划，从而提升杂乱场景中的目标对齐与稳定抓取能力。
> - **Key Performance**: Grounded SAM 的 IoU 从 44.9 提升到 64.4；多物体逐个抓取成功率达到 95.5%。

> [!info] **Agent Summary**
> - **task_path**: 语音指令 + RGB-D杂乱场景 -> 目标指代表达补全 -> 目标分割点云 -> 灵巧手抓取姿态/轨迹 -> 抓取执行
> - **bottleneck**: 杂乱场景中口语指令通常不足以唯一定位目标，同时灵巧手抓取候选空间大、可执行稳定抓取难以直接搜索
> - **mechanism_delta**: 先用 VLM 将原始指令补全为包含类别/颜色/形状/材质/位置的表达，再用骨架+PCA导向的受限采样和力闭合/GWS/轨迹代价三级筛选生成抓取
> - **evidence_signal**: RERE 在 GraspNet-1Billion 上将 Grounded SAM IoU 提升 19.5 点，整系统在多物体抓取中达到 95.5% 成功率
> - **reusable_ops**: [属性级语义补全, 骨架+PCA约束采样]
> - **failure_modes**: [严重遮挡或重叠时错分相似完整物体, 大尺寸或需精细接触的物体容易因单手力闭合不足或转运滑移失败]
> - **open_questions**: [能否用触觉闭环替代静态摩擦先验, 能否扩展到双臂与in-hand reorientation]

## Part I：问题与挑战

这篇论文要解决的不是单纯“抓住一个物体”，而是更难的 **voice-driven embodied grasping**：用户用自然口语说出目标，机器人要在杂乱场景里先找对，再稳定抓起。

**真正瓶颈有两个串联层次：**

1. **语义 grounding 瓶颈**  
   人类口语往往是不完整的，如“把那个红色的给我”“拿桌上的鸭子”。在 clutter 场景里，仅靠原始文本很难唯一指向目标，导致后续分割和抓取都建立在错误目标上。

2. **灵巧抓取搜索瓶颈**  
   即便目标分出来了，多指灵巧手的接触组合、手部姿态和机械臂轨迹空间仍然很大。全空间搜索既低效，也容易采到物理上不稳或运动学上不可执行的抓法。

**为什么现在值得做：**
- VLM/开放词汇分割让“看图补全用户意图”成为可行前端；
- 纯数据驱动灵巧抓取仍受数据规模、sim-to-real、接触不稳定限制；
- 因此，把 **VLM语义补全** 和 **解析式物理筛选** 结合，是一个更贴近真实部署的折中路线。

**输入 / 输出接口：**
- **输入**：语音指令 + RGB-D 桌面场景
- **输出**：目标 mask / 点云 + 灵巧手抓取动作 + 机械臂执行轨迹

**边界条件：**
- 单臂 UR5 + Inspire dexterous hand
- 面向桌面 clutter 抓取
- 不依赖 CAD 模型
- 重点是“抓起来”，不是复杂的在手重定向或双臂协作操作

## Part II：方法与洞察

EDGS 可以概括为三段式流水线：

1. **ERGS / RERE：先把话说清楚，再做分割**  
   系统先把语音转成文本，再用 VLM 结合图像信息，对原始 referring expression 做属性级补全。补全维度包括：
   - 实例类别
   - 颜色
   - 形状
   - 材质/纹理
   - 相对位置

   核心不是“多加点词”，而是把原本模糊的目标描述变成对分割模型真正有判别力的描述。若跨模态对齐不足，系统还会要求进一步澄清。

2. **DGCG：先缩小抓取搜索空间，再采样**  
   对分割出的目标点云，不直接暴力搜索抓取，而是先提取物体几何方向：
   - 用 skeleton 处理不规则物体形态
   - 用 PCA 稳定主方向估计
   - 在此基础上构造物体特征向量

   然后把该向量与手部固定的“拇指-其他手指”捏合轴对齐，只在这个受约束的局部空间中采样抓取候选。这样做等价于把“高维盲采样”变成“围绕 plausible opposition axis 的定向采样”。

3. **DGR：从能接触，到能稳抓，再到能执行**  
   论文对候选抓取做三级筛选：
   - **Force Closure**：先滤掉物理上不稳定的接触组合
   - **GWS**：再按抗扰动能力排序
   - **运动代价**：最后用 STOMP/MoveIt 选出关节变化更小、路径更顺的可执行方案

### 核心直觉

这篇文章真正改变的，不是某个单点网络结构，而是把系统从：

**“原始短指令 → 直接分割 → 高维抓取搜索”**

改成了：

**“属性充分的目标表达 → 更可靠的目标点云 → 受约束的可行抓取搜索 → 物理与运动联合筛选”**

这带来两类因果变化：

- **信息瓶颈变化**：  
  原来分割模型接收到的是低信息密度指令；现在接收到的是包含类别、外观和空间关系的 enriched expression。  
  **结果**：候选目标后验熵下降，错选/混淆减少。

- **约束瓶颈变化**：  
  原来抓取搜索空间过大且盲；现在被压缩到围绕物体方向与手部捏合几何的局部子空间。  
  **结果**：采样命中物理可行抓取的概率提高，执行效率更高。

- **执行瓶颈变化**：  
  原来“看起来能抓”不等于“真实机器人能稳定抓并顺利到达”；现在用 force closure + GWS + trajectory cost 连续过滤。  
  **结果**：最终输出更接近真实硬件可执行动作，而不是仅几何上好看的姿态。

### 战略权衡

| 设计选择 | 改变了什么瓶颈 | 收益 | 代价 / 风险 |
|---|---|---|---|
| 先做 RERE 语义补全 | 降低口语指令歧义 | 分割与目标对齐更稳 | 依赖 VLM 常识和提示设计，非标准表达仍会失败 |
| 骨架 + PCA 约束采样 | 压缩高维抓取搜索空间 | 提高采样效率和可行性 | 可能遗漏少见但有效的非常规抓法 |
| Force Closure + GWS + 轨迹代价联合筛选 | 从“能接触”提升到“能稳定执行” | 提升真实抓取成功率 | 依赖摩擦先验、规划时间和硬件建模质量 |

## Part III：证据与局限

### 关键证据信号

- **比较信号（感知前端）**  
  在 GraspNet-1Billion 上，RERE 对三种分割器都带来提升，其中 Grounded SAM 从 44.9 提升到 64.4，说明增益不是单一 backbone 特例，而是“表达补全”本身有效。

- **消融信号（机制归因）**  
  随着实例类别、纹理、形状、材质、位置等属性逐步加入，IoU 持续提高，完整属性组合最好。这个结果支持论文的核心判断：**问题首先出在表达不充分，而非仅仅出在分割模型不够强。**

- **系统级比较信号（抓取执行）**  
  在多物体逐个抓取实验中，EDGS 达到 95.5%，高于 DexGraspNet 2.0 的 90.7% 和 ISAGrasp 的 54.8%。这表明前端语义对齐与后端抓取规划的组合，确实转化成了真实执行收益。

- **应用场景信号（部署可用性）**  
  在 fruits / households / vegetables 三类 360 次尝试中，总体成功率 96.1%，说明系统对常见家庭物体和 clutter 场景有一定泛化与实用性。

**但证据也有边界：**
- RERE 有明确消融，但 **DGCG / DGR 本身缺少独立 ablation**，因此我们知道“整系统有效”，但不完全知道每个抓取模块各自贡献多少；
- 部分基线比较并非严格统一硬件、训练数据和评测协议，所以“领先趋势”可信，但“领先幅度”应谨慎解读。

### 局限性

- **Fails when:** 目标严重遮挡、密集重叠或外观极其相似时，RERE 与分割器仍可能出现类混淆、边界错误、目标合并或漏检；对不规则物体、需要轻柔接触的物体，转运阶段也更容易滑移失败。
- **Assumes:** 假设有可用的 RGB-D 观测、可工作的语音识别、VLM/GPT 类模块做表达补全与纹理到摩擦的先验映射；系统依赖单臂单手硬件、MoveIt/STOMP 规划栈，且缺少细粒度触觉反馈。论文也未提供开源实现，复现实用性受限。
- **Not designed for:** 双臂/双手协作、大尺寸物体抓取、精细 in-hand manipulation、功能部件级语义理解，以及强动态场景中的闭环操作。

### 可复用组件

- **RERE 语义补全前端**：适合作为 referring-expression segmentation 或 open-vocab manipulation 的前置模块。
- **骨架 + PCA 物体方向抽取**：适合不规则物体的几何定向抓取候选生成。
- **Force Closure → GWS → Motion Cost 三级筛选链**：适合作为真实机器人抓取的执行层过滤器。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2024/arXiv_2024/2024_Grasp_What_You_Want_Embodied_Dexterous_Grasping_System_Driven_by_Your_Voice.pdf]]