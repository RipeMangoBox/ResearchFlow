---
title: "MetaFold: Language-Guided Multi-Category Garment Folding Framework via Trajectory Generation and Foundation Model"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/garment-folding
  - task/language-conditioned-manipulation
  - cvae
  - point-cloud-trajectory
  - closed-loop-control
  - dataset/MetaFold
  - dataset/CLOTH3D
  - opensource/no
core_operator: 用语言条件点云轨迹生成先规划衣物中间状态，再将相邻状态交给 ManiFoundation 预测接触点与施力方向，并通过闭环重规划完成多类别折衣。
primary_logic: |
  当前衣物点云 + 用户语言指令 → PointNet++/LLaMA 编码并由 CVAE 生成单阶段折叠点云轨迹 → 将相邻轨迹帧送入 ManiFoundation 得到接触点与运动方向 → 机器人执行后重新感知并继续生成后续轨迹，直到达到目标折叠状态
claims:
  - "MetaFold 在 MetaFold 数据集四类服装上的 success rate 分别达到 0.88/0.86/0.90/0.97，整体优于 UniGarmentManip、DP3 和 GPT-Fabric 的大多数对应结果 [evidence: comparison]"
  - "在 CLOTH3D 的语言指导评测中，MetaFold 对 seen/unseen 指令的 success rate 达到 0.97/0.93，明显高于图动态基线方法的 0.56/0.47 [evidence: comparison]"
  - "去掉 ManiFoundation 或闭环控制会将 success rate 从 0.86 分别降至 0.27 和 0.07，说明动作桥接与反馈重规划是性能关键 [evidence: ablation]"
related_work_position:
  extends: "ManiFoundation (Xu et al. 2024)"
  competes_with: "UniGarmentManip (Wu et al. 2024); GPT-Fabric (Raval et al. 2024)"
  complementary_to: "SAM 2 (Ravi et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_MetaFold_Language_Guided_Multi_Category_Garment_Folding_Framework_via_Trajectory_Generation_and_Foundation_Model.pdf
category: Embodied_AI
---

# MetaFold: Language-Guided Multi-Category Garment Folding Framework via Trajectory Generation and Foundation Model

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.08372), [Project](https://meta-fold.github.io/)
> - **Summary**: 这篇论文把“衣物应折成怎样的中间形态”和“机器人此刻该如何抓取施力”拆开学习：先生成语言条件的点云折叠轨迹，再用基础操作模型把轨迹翻译成接触动作，从而提升多类别服装与多语言指令下的泛化。
> - **Key Performance**: MetaFold 数据集四类服装 success rate 为 0.88/0.86/0.90/0.97；CLOTH3D 上未见语言指令 success rate 达 0.93，真实机器人成功数为 8/10–10/10。

> [!info] **Agent Summary**
> - **task_path**: 语言指令 + 当前衣物点云 -> 单阶段折叠点云轨迹 -> 接触点/施力方向 -> 闭环执行后的新衣物状态
> - **bottleneck**: 高维可变形衣物的长程折叠规划与局部接触控制耦合在一起，导致动作分布难学、误差易累积、跨类别泛化差
> - **mechanism_delta**: 用语言条件 CVAE 先预测可控的中间状态轨迹，再把相邻状态交给微调后的 ManiFoundation 做局部接触合成，并在执行后重感知重规划
> - **evidence_signal**: 双数据集对比 + 语言泛化评测 + 消融中去掉 ManiFoundation/闭环后 success rate 从 0.86 降到 0.27/0.07
> - **reusable_ops**: [语言条件点云轨迹生成, 轨迹切片到接触动作映射]
> - **failure_modes**: [开环执行时误差快速累积, 点云分割或深度噪声会导致接触点偏移]
> - **open_questions**: [如何支持真正自由语言而非映射到预定义描述, 如何减少 160-seed 集成带来的推理成本]

## Part I：问题与挑战

MetaFold解决的核心不是“找两个抓点”这么局部，而是**如何在高维、可变形、接触敏感的衣物状态空间里，把语言指定的折叠意图稳定落到机器人动作上**。

### 真正难点是什么
1. **长程规划难**：从任意皱褶初态直接输出动作序列，模型要同时理解衣物类别、折叠顺序、未来几步的形变趋势。
2. **低层执行难**：即使知道“应该折成什么样”，把这个目标转成稳定的接触点与施力方向仍依赖局部物理、抓取姿态与误差控制。
3. **泛化难**：同一衣物可对应多种折法，同一折法又可能被多种语言表述；如果直接学 end-to-end policy，很容易把“语言、类别、动力学”纠缠在一起。

### 论文的判断
作者认为：**衣物折叠中的“状态轨迹”比“任意状态下的下一动作”更容易学习。**  
原因是折衣的中间状态演化相对规整，而直接动作分布隐含更多不可见物理与控制细节。

### 输入 / 输出接口
- **输入**：当前衣物点云 + 用户语言指令
- **中间接口**：单阶段折叠的点云轨迹
- **输出**：机器人接触点与运动方向，循环执行直到达到目标折叠状态

### 为什么是现在
这件事现在更可行，主要因为三点同时成熟：
- 点云生成模型可以学“状态演化”而不仅是离散关键点；
- ManiFoundation 这类操作基础模型提供了“从状态差异到接触动作”的通用桥梁；
- DiffClothAI / Isaac Sim 使大规模可变形物体轨迹数据生成成为可能。

### 边界条件
- 一次只生成**一个 folding stage**，不是一口气规划完整多阶段任务。
- 依赖较干净的点云观测；真实世界中还需要 RGB-D 和 SAM2 做分割。
- 主要验证了无袖、短袖、长袖、裤子四类服装。

## Part II：方法与洞察

MetaFold 的整体结构可以概括成三段：  
**语言条件轨迹生成器 → 轨迹到动作的基础模型桥接 → 闭环执行与重规划。**

### 核心直觉

作者引入的关键因果旋钮是：**把“观测直接到动作”的学习目标，改成“先预测未来衣物状态轨迹，再局部合成动作”。**

这带来的变化是：

- **什么变了**：监督目标从稀疏、物理耦合强的动作标签，变成更结构化的点云状态演化。
- **哪个瓶颈被改写了**：全局折叠顺序和语义约束交给轨迹模型；局部接触物理由 ManiFoundation 负责。一个模型不再同时承担语言理解、长程规划和局部接触控制三件事。
- **能力如何变化**：模型更容易跨服装类别复用，也更能按语言要求改变折叠顺序；闭环重规划还能抵抗执行偏差。

**为什么这能工作：**
1. 点云轨迹是对“未来形状”的直接监督，比直接学动作更规则。
2. 相邻轨迹帧天然定义了局部 flow，适合作为低层动作模型的输入。
3. 闭环执行允许系统在每段操作后重新对齐当前状态，减少误差累积。

### 1）语言条件点云轨迹生成

作者先构建了一个新的折衣轨迹数据集：
- 基于 **ClothesNetM** 服装网格；
- 用 **DiffClothAI** 通过启发式抓点/目标点生成折叠过程；
- 抽取网格顶点并下采样为点云；
- 为不同折法补充语言描述。

最终得到 **1210 件衣物、3376 条轨迹**。

模型上：
- 用 **PointNet++** 提取当前点云空间特征；
- 用 **Meta-Llama-3.1-8B-Instruct** 提取语言语义；
- 先把高维语言特征降到 128 维，避免语义特征压过几何特征；
- 用 **CVAE** 生成后续多帧点云轨迹。

这里的重点不是“预测下一步”，而是**预测一个单阶段的完整折叠轨迹**。这让模型直接学习“这一段应该怎么折”，而不是只学局部反应。

### 2）语言泛化策略

这篇论文的语言泛化并不是完全自由的 open-ended instruction following。  
它更像是：

**用户输入 → LLaMA 编码 → 映射到一组预定义折叠描述附近的语义槽位。**

好处是：
- 显著降低训练时的语言条件复杂度；
- 对未见表述有一定泛化能力。

代价是：
- 泛化更像“语义归并”而不是完全开放词汇；
- 超出预定义折法空间的指令，能力边界会比较明显。

### 3）从轨迹到动作：ManiFoundation + 闭环

轨迹生成后，系统把相邻点云帧切片成若干局部状态对，送入 **ManiFoundation**：
- 输入：两帧点云间的 flow
- 输出：接触点 + 运动方向

论文还对 ManiFoundation 在服装数据上做了微调，以提高折衣场景下的动作预测质量。

一个很实际的工程细节是：作者发现 ManiFoundation 输出受随机种子影响，因此用了 **160 个 random seeds 集成**，再按接近程度聚类，取主模态结果作为最终接触动作。  
这提高了稳定性，但也明显增加了推理成本。

随后系统执行动作、重新采集点云、再次生成后续轨迹，形成闭环。消融表明：**每 10 帧做一次闭环重规划最好**，过短会破坏长程一致性，过长又会导致误差积累。

### 战略权衡

| 设计选择 | 收益 | 代价 / 风险 |
|---|---|---|
| 规划与执行解耦 | 更容易跨类别泛化，训练更稳定 | 需要额外轨迹数据和桥接模块 |
| 点云轨迹而非关键点 | 保留完整形状与中间状态 | 生成维度更高，训练更重 |
| 全轨迹预测而非 next-step | 保持折叠顺序和中间态一致性 | 远期预测误差仍可能累积 |
| 闭环重规划而非开环 | 更抗扰动、更适合 sim-to-real | 感知和推理成本更高 |
| LLaMA 语义匹配模板 | 提高未见表述鲁棒性 | 真正自由语言的覆盖有限 |

## Part III：证据与局限

这篇论文最有说服力的地方，不是单个数值，而是**比较、消融、实机**三类证据相互支撑。

### 关键证据信号

- **比较信号：多类别折衣有效**
  - 在 MetaFold 数据集上，四类服装 success rate 为 **0.88 / 0.86 / 0.90 / 0.97**。
  - 相比 UniGarmentManip、DP3、GPT-Fabric，MetaFold 在大多数类别和指标上更优。
  - 除了成功率，Area Ratio 在长袖/裤子等类别上也更低，说明结果更紧凑。

- **泛化信号：跨数据集与未见语言仍成立**
  - 在 **CLOTH3D 零样本**设置下，长袖与裤子 success rate 仍达到 **0.97 / 0.97**。
  - 在语言指导任务上，CLOTH3D 的 seen/unseen 指令 success rate 为 **0.97 / 0.93**，显著高于语言基线 **0.56 / 0.47**。
  - 这说明提升不只是“更会折”，而是**更会按语言改变折叠顺序**。

- **因果信号：关键模块缺一不可**
  - 去掉 ManiFoundation：success rate **0.86 → 0.27**
  - 去掉闭环控制：**0.86 → 0.07**
  - 只预测 next-step：**0.86 → 0.41**
  - 这直接支持作者的核心论点：性能提升来自“轨迹规划 + 动作桥接 + 闭环纠偏”的组合，而不是某个单独组件。

- **实机信号：具备一定落地性**
  - 在 xArm6 + RealSense D435 上，四类衣物 10 次实验成功数分别为 **10/10、8/10、9/10、9/10**。
  - 说明该方法至少在点云驱动的一臂折衣场景中具备可迁移性。

### 局限性

- **Fails when**: 衣物拓扑或皱褶状态明显超出训练分布、点云分割/深度质量差、需要强双臂协同或严重自遮挡时，轨迹预测和接触动作都会失稳。
- **Assumes**: 依赖可获得的点云观测、DiffClothAI/Isaac Sim 生成的大规模启发式轨迹、LLaMA 语义编码、微调后的 ManiFoundation，以及 160-seed 集成来稳定动作输出。
- **Not designed for**: 任意开放语义的服装操作、一次性全流程长程规划、双臂复杂折衣、熨平/抚平/打结等非折叠型布料操作。

### 复现与扩展时要注意的依赖

这篇方法的可扩展性不差，但复现门槛不低：
- 训练数据需要 **可变形仿真 + 启发式轨迹生成**；
- 真实部署需要 **RGB-D + SAM2**；
- 动作层依赖 **ManiFoundation**；
- 稳定推理还用了 **160 个随机种子的集成聚类**。

所以它不是“一个单模型端到端就能跑起来”的方案，而是一个**模块化系统工程**。

### 可复用组件

- **语言条件点云轨迹生成器**：可迁移到其他可变形物体的中间状态规划。
- **轨迹切片 → 接触动作映射接口**：为“世界模型 + 操作基础模型”组合提供通用桥梁。
- **闭环重规划范式**：适合处理可变形物体执行中的累积误差。
- **轨迹数据生成流程**：对其他布料/软体操作任务也有参考价值。

**一句话总结“so what”**：  
MetaFold 的真正能力跃迁，不在于更准地找抓点，而在于把折衣从“难学的动作策略问题”改写成“可生成的状态轨迹问题”，再让基础操作模型只负责它更擅长的局部接触合成。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_MetaFold_Language_Guided_Multi_Category_Garment_Folding_Framework_via_Trajectory_Generation_and_Foundation_Model.pdf]]