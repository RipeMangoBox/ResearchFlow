---
title: "Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - simulation-based-benchmark
  - counterfactual-evaluation
  - atomic-evaluation
  - dataset/WM-ABench
  - opensource/partial
core_operator: 用认知启发的双阶段原子任务与反事实候选状态，在多模拟器中拆解评测VLM的世界模型能力
primary_logic: |
  评测目标（VLM是否具备内部世界模型） → 将能力分解为感知/预测两阶段与23个原子维度 → 在6个模拟环境中构造单变量控制与反事实选项并加入人类对照 → 输出细粒度能力边界与失效模式
claims:
  - "在WM-ABench的660组实验中，当前VLM在世界模型相关预测任务上显著落后于人类，最佳平均准确率仅为47.5% [evidence: analysis]"
  - "几乎所有被测模型在运动轨迹辨别任务上接近随机水平，说明其动态场景表征能力明显不足 [evidence: analysis]"
  - "即使筛选到当前状态已被正确感知的样本，直觉物理预测表现也只小幅改善甚至下降，表明瓶颈不只来自感知误差而是机制知识缺失 [evidence: analysis]"
related_work_position:
  extends: "N/A"
  competes_with: "Perception Test (Pătrăucean et al. 2023); CLEVRER (Yi et al. 2020)"
  complementary_to: "MMBench (Liu et al. 2024); BLINK (Fu et al. 2024b)"
evidence_strength: strong
pdf_ref: paperPDFs/Evaluating_World_Models/arXiv_2025/2025_Do_Vision_Language_Models_Have_Internal_World_Models_Towards_an_Atomic_Evaluation.pdf
category: Survey_Benchmark
---

# Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.21876), [Project](https://wm-abench.maitrix.org/)
> - **Summary**: 论文提出 WM-ABench，用“感知→预测”的认知启发双阶段框架，把 VLM 的世界模型能力拆成 23 个原子维度并进行反事实评测，发现当前模型离真正可用的内部世界模型仍有明显差距。
> - **Key Performance**: 最佳感知平均准确率为 67.7%（Qwen2-VL）；最佳预测平均准确率仅 47.5%，且多数模型在运动轨迹判断上接近随机。

> [!info] **Agent Summary**
> - **task_path**: 多视角图像/连续帧/动作条件 -> 候选当前/未来状态选择 -> 世界模型原子能力评分
> - **bottleneck**: 现有评测把视觉、时空、物理预测分开测，难以区分模型是“没看对当前状态”还是“不会模拟状态转移”
> - **mechanism_delta**: 用“感知→预测”双阶段 taxonomy + 单变量控制 + 反事实候选项，把世界模型能力拆成可定位误差来源的 23 个原子维度
> - **evidence_signal**: 15 个 VLM、660 组实验、多人类基线与“感知正确后再看预测”的过滤分析共同显示机制性短板
> - **reusable_ops**: [dual-stage-taxonomy, counterfactual-state-action-generation]
> - **failure_modes**: [multi-view-3d-reasoning, transitive-compositional-prediction]
> - **open_questions**: [真实开放场景上的大规模泛化是否成立, 如何向VLM注入稳定的3D与物理机制表征]

## Part I：问题与挑战

这篇论文真正想回答的，不是“VLM 会不会做视觉问答”，而是更底层的问题：**VLM 是否已经形成了可用于推演世界变化的内部世界模型**。

### 现有评测缺什么
过去很多 benchmark 会分别测试：
- 颜色、形状、材质等静态视觉属性；
- 时序排序、视频理解；
- 直觉物理或下一状态预测。

但这些评测通常有两个缺口：

1. **缺少统一框架**  
   它们把“看见当前世界”和“预测世界如何变化”分开讨论，难以判断模型失败到底卡在哪一层。

2. **缺少原子化与可归因性**  
   如果一道题同时混合了空间、时间、动作、物理规律等多个因素，那么模型答错后很难知道是：
   - 3D 表征没建起来，
   - 运动轨迹没跟住，
   - 还是根本不懂碰撞/操控机制。

### 真正瓶颈是什么
论文认为，VLM 作为世界模型的真实瓶颈是：

- **表征不稳健**：对空间、时间、运动等维度的内部表示不够独立；
- **机制知识不足**：即便当前状态看对了，也不会正确外推未来状态；
- **容易走捷径**：依赖颜色、外观共现或训练分布偏差，而不是因果机制。

这也是作者强调“atomic evaluation”的原因：**先把世界模型拆开，再逐项定位短板**。

### 为什么现在必须做
因为 frontier VLM 已开始被当作：
- 通用视觉代理，
- 具身系统的认知核心，
- 甚至“语言化世界模型”。

如果没有这种细粒度诊断，模型在 robotics、navigation、planning 中的失败会显得“偶然”，但其实可能是结构性的。

### 输入/输出接口与边界
这套 benchmark 的接口很明确：

- **输入**：多视角图像、连续帧、起始状态 + 动作条件
- **输出**：从若干候选状态中选出正确的当前/未来状态

它评测的是：
- **可语言化的世界状态理解**
- **基于状态的转移外推**

它**不直接评测**：
- 视频生成式 world model，
- 在线交互式决策闭环，
- 开放式自由生成未来场景。

---

## Part II：方法与洞察

这篇论文的关键贡献，不是又造了一个“更大题库”，而是把评测方式从“混合能力打分”改成了“可因果定位的原子诊断”。

### 设计拆解

作者提出一个认知启发的双阶段框架：

#### 1. 感知阶段
把当前世界状态的形成拆成 5 类能力：
- **Spatial**：空间关系、位置、占据/空隙
- **Temporal**：先后、持续时间
- **Motion**：运动方向、速度、轨迹、运动体识别
- **Quantity**：离散数量、连续量、相对数量
- **Vision**：颜色、形状、材质

#### 2. 预测阶段
把未来状态外推拆成 3 类能力：
- **Mechanistic simulation**：直觉物理、导航、操控
- **Transitive inference**：多步动作链式推演
- **Compositional inference**：多物体/多智能体并发组合推理

总计覆盖 **23 个细粒度维度**，并用 **10 万+ 样本**、**多模拟环境** 生成测试数据。

#### 3. 反事实候选项
这是 benchmark 设计里最关键的一步。

作者不是简单生成一个正确答案，而是用两种方式制造“高相似错误选项”：

- **counterfactual action**：固定当前状态，改动作
- **counterfactual previous state**：固定动作，改前一状态

这样错误选项在视觉上常常很像正确答案，模型如果只靠表面匹配，很容易中招。  
要答对，必须真的理解：
- 当前世界状态；
- 动作/机制如何导致转移。

#### 4. 受控实验与人类基线
每个数据点尽量只改变一个因素，其余保持不变，从而增强归因性。  
同时作者还做了：
- **人类评测**：验证任务可解且公平；
- **真实数据改写测试**：检验模拟环境结论是否大致能迁移。

### 核心直觉

过去的评测往往是：

**自然数据混杂多个因素 → 模型可利用共现偏差/语义先验/表面匹配 → 你只能看到“答对/答错”，却看不出它有没有真正的世界模型**

这篇工作的改动是：

**把世界模型拆成感知与预测两阶段 + 对每个维度做单变量控制 + 用反事实选项压缩捷径空间**

于是带来的能力变化是：

**从粗粒度排行榜，变成能定位“表征缺失、机制缺失、组合推理缺失”的诊断仪表盘。**

更因果地说：

- **设计改变了什么**：题目不再允许多个隐变量同时漂移；
- **约束改变了什么**：模型无法轻易靠颜色、频次、模板匹配过关；
- **能力揭示了什么**：一旦模型仍然失败，就更能说明它缺的是世界状态表征或状态转移机制，而不是“题没见过”。

### 为什么这套设计有效
因为世界模型的核心不是识别物体标签，而是：
1. **构建当前状态表示**
2. **依机制推演下一状态**

WM-ABench 把这两步显式拆开后，能回答两个过去很难回答的问题：
- 错误来自 perception 还是 prediction？
- 模型学到的是机制，还是 spurious correlation？

### 战略权衡

| 设计选择 | 得到的好处 | 付出的代价 |
|---|---|---|
| 双阶段原子分解 | 能明确区分感知错误与预测错误 | 任务更离散，离真实开放任务更远 |
| 模拟器生成 + 单变量控制 | 可做强归因分析，规模大且便宜 | 图像真实性有限，可能与训练分布不完全匹配 |
| 反事实候选项 | 抑制表面匹配和捷径 | 仍是多选评测，不是开放生成 |
| 多环境 + 人类基线 | 降低单一环境偏差，验证题目可解性 | 评测成本高，frontier API 只能测子集 |

---

## Part III：证据与局限

### 关键证据

#### 1. 大规模比较信号：VLM 离“世界模型”还远
作者在 15 个最新开源/闭源 VLM 上做了 660 组实验，核心结论很直接：

- **感知任务**：最好模型平均也只有 **67.7%**
- **预测任务**：最好模型平均只有 **47.5%**
- **人类**：大多数子任务显著更高，很多接近满分

这说明问题不是某个单独模型不行，而是**当前 VLM 范式整体还没有稳定跨过世界模型门槛**。

#### 2. 最强失败信号：动态世界表征很弱
论文最有冲击力的结果之一是：

- **运动轨迹（Motion Trajectory）几乎接近随机**

这比“看不懂颜色/形状”更关键，因为轨迹理解要求模型把多帧变化整合成动态状态表示。  
换句话说，模型可能“看得见运动”，但**不真正理解运动是如何沿时间展开的**。

#### 3. frontier 模型有提升，但提升偏静态
o3、GPT-4.5、Gemini-2.5-Pro 在一些静态感知任务上已接近甚至达到人类水平。  
但真正卡脖子的点仍在：

- 多视角空间定位（SP）仍明显低于人类；
- Temporal Extension 提升有限；
- 高级预测，尤其是 transitive / compositional，依旧弱。

也就是：**它们更像“更强的视觉答题器”，还不是稳定的机制模拟器。**

#### 4. 过滤分析：问题不只是“没看清”
作者把那些“所有模型都能正确感知当前状态”的样本拿出来，再测物理预测。结果发现：
- slide / drop 只小幅提升；
- collision 甚至会下降。

这很关键。  
它直接支持一个更强的结论：**预测失败并不只是 perception error 传导，模型本身就缺少足够的物理机制知识。**

#### 5. 表征纠缠：模型把不该相关的属性绑在一起
作者还做了 entanglement 分析，发现颜色、形状等属性会不合理地干扰：
- 数量判断，
- 速度判断，
- 其他本应正交的世界属性。

比如某些模型会倾向于“觉得蓝色更快”。  
这意味着它的内部表示不是可组合、可独立操纵的世界状态，而更像统计共现堆出来的表征。

#### 6. 真实数据改写：趋势基本一致
作者把真实世界数据集改写成同类评测格式后，发现大趋势与模拟环境一致：
- 颜色/形状仍容易；
- 空间定位、运动轨迹仍难；
- Temporal Positioning 普遍强于 Temporal Extension。

这给 benchmark 的外部有效性提供了一个积极信号，尽管规模还不大。

### 局限性

- **Fails when**: 需要高度写实视觉、开放词汇真实视频、超长时程因果链、或开放式自由生成评测时；当前模拟图像与多选接口无法完整代表真实部署条件。
- **Assumes**: 世界模型能力可被拆成封闭集原子维度；依赖模拟器构造受控样本与反事实状态；部分 frontier 模型受 API/成本限制仅在每任务约 100 个样本上评测；人类基线也只是在每子任务 50 题规模上验证可解性。
- **Not designed for**: 端到端 agent control 成功率、在线规划闭环表现、视频生成式 world model 质量，或广义多模态知识问答能力。

### 复现与资源依赖
这篇工作虽然有项目页，但从正文看，**完整代码/数据开放范围并未被非常明确地系统说明**；同时它依赖：
- 多个模拟器环境，
- 闭源模型 API，
- 较高评测成本。

因此它的结论很强，但复现实操仍带有一定工程门槛。

### 可复用组件
这篇论文最值得复用的，不只是 benchmark 本身，而是几种“评测操作符”：

- **双阶段 taxonomy**：先分 perception，再分 prediction
- **反事实选项生成**：改动作或改前态，构造高相似 hard negatives
- **属性纠缠诊断**：测表征是否 disentangled
- **感知正确后再测预测**：把 perception noise 与 mechanism failure 分离

这些设计可以直接迁移到其他 MLLM benchmark、robotics evaluation，甚至训练后诊断流程中。

## Local PDF reference

![[paperPDFs/Evaluating_World_Models/arXiv_2025/2025_Do_Vision_Language_Models_Have_Internal_World_Models_Towards_an_Atomic_Evaluation.pdf]]