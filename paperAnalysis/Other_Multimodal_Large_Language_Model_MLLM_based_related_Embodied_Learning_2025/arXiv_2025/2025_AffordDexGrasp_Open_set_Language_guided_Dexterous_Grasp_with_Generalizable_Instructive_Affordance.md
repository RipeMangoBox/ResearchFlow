---
title: "AffordDexGrasp: Open-set Language-guided Dexterous Grasp with Generalizable-Instructive Affordance"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/language-guided-grasping
  - task/open-set-grasping
  - flow-matching
  - affordance-learning
  - test-time-optimization
  - dataset/OakInk
  - opensource/no
core_operator: "把同一语义抓取组的接触区域并成类别无关的局部affordance，再用“语言→affordance→手姿”的双阶段flow matching与测试时优化实现开集灵巧抓取。"
primary_logic: |
  场景点云/RGB/用户命令
  → MLLM提取类别、意图、接触部位与离散方向
  → Affordance Flow Matching生成与语义对齐的局部可抓affordance
  → Grasp Flow Matching在affordance条件下生成高自由度手部姿态
  → affordance-guided优化减少穿透并保持意图一致
  → 输出开集类别上的灵巧抓取参数
claims:
  - "在两个open-set split上，AffordDexGrasp相对DexGYS、SceneDiffuser等基线取得更低FID/CD与更高Top-1 R-Precision，显示更强的跨类别语言-抓取对齐能力 [evidence: comparison]"
  - "将中间表示替换为作者提出的generalizable-instructive affordance后，open-set Top-1与CD均优于object part和contact map条件，支持其更好平衡‘可泛化性/可指导性’ [evidence: ablation]"
  - "在Leap Hand + Kinova Gen3真实实验中，方法在8组任务上累计66/80次成功，高于DexGYS的39/80，且未使用额外真实训练数据 [evidence: comparison]"
related_work_position:
  extends: "DexGYS / Grasp as You Say (Wei et al. 2025)"
  competes_with: "DexGYS / Grasp as You Say (Wei et al. 2025); ContactGen (Liu et al. 2023)"
  complementary_to: "ReKep (Huang et al. 2024); PartSLIP (Liu et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_AffordDexGrasp_Open_set_Language_guided_Dexterous_Grasp_with_Generalizable_Instructive_Affordance.pdf
category: Embodied_AI
---

# AffordDexGrasp: Open-set Language-guided Dexterous Grasp with Generalizable-Instructive Affordance

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.07360) · [Project](https://isee-laboratory.github.io/AffordDexGrasp/)
> - **Summary**: 论文把“语言到高自由度灵巧手动作”的直接映射，拆成“语言到可泛化affordance，再到抓取姿态”的两段式过程，从而显著提升未见类别上的意图一致抓取。
> - **Key Performance**: Open Set B 上达到 **FID 0.162 / CD 2.95 / Top-1 0.532**；真实世界 8 组任务累计 **66/80** 成功，高于 DexGYS 的 **39/80**。

> [!info] **Agent Summary**
> - **task_path**: 场景点云+RGB+语言指令 -> 语义一致的灵巧手抓取姿态 $(r,t,q)$
> - **bottleneck**: 高层语言语义与高自由度抓取动作之间缺少可泛化且可执行的中间表示，导致 unseen category 上语义理解和动作生成一起失稳
> - **mechanism_delta**: 把直接 language-to-grasp 改成 MLLM 关键线索压缩 + affordance-to-grasp 两段式生成，并把几何修正移到测试时非参数优化
> - **evidence_signal**: open-set A/B 对比全面领先 + affordance 消融明确优于 object part/contact map
> - **reusable_ops**: [MLLM关键线索规范化, 语义群组接触并集生成affordance]
> - **failure_modes**: [六向离散方向限制了细粒度朝向表达, 级联架构会把affordance误差传到最终抓取]
> - **open_questions**: [如何摆脱闭源GPT-4o仍保持开放表达泛化, 如何扩展到可变形/铰接物体和长时序操作]

## Part I：问题与挑战

这篇文章研究的是一个很具体但也很难的问题：**open-set language-guided dexterous grasp**。  
输入不是单纯物体，而是：

- 场景点云 \(O\)
- RGB 图像 \(I\)
- 用户语言命令 \(C\)

输出是灵巧手抓取参数：

- 全局旋转 \(r\)
- 平移 \(t\)
- 关节角 \(q\)

### 这个问题真正难在哪？

不是“怎么抓稳”本身，而是：

1. **语言是高层意图**
   - 比如“use the mug”“hold the bottle from the left”
   - 里面隐含了用途、部位、接近方向

2. **灵巧手动作是低层高维控制**
   - 比 parallel gripper 多得多的自由度
   - 接触模式非常多样

3. **open-set 要求跨类别泛化**
   - 训练里没见过 lotion pump、kettle，也要能从“trigger sprayer / mug / frying pan”这类经验迁移过去
   - 但数据驱动模型往往学到的是**类别相关的接触模式**，不是可迁移的语义-几何对应

所以，作者认为真正瓶颈不是生成器不够强，而是：

> **语言语义和抓取动作之间缺少一个既能跨类别泛化、又足够指导高自由度手部姿态的中间层。**

### 为什么现在值得做？

因为已有语言引导灵巧抓取方法（如 DexGYS）在**seen category 内泛化**还可以，但在 open-set 下会明显掉点；同时真实灵巧手数据昂贵，不可能靠不断扩充类别去补。  
也就是说，问题已经从“能不能学会抓”转向“能不能学会**跨类别理解意图并抓**”。

### 任务边界

这篇工作有清晰边界，不是通用操作系统：

- 主要是**桌面场景**
- 方向被离散成 **6 个坐标轴方向**
- 主要处理的是**单次抓取姿态生成**
- 关注“use / hold / lift / hand over”等意图
- 不直接解决长时序 manipulation policy

---

## Part II：方法与洞察

整体设计思想很明确：

> 不再强迫模型直接把自然语言翻译成高自由度手部参数，而是先把语言压缩成“局部可抓区域”的几何语义，再从这个中间表征生成具体抓取。

方法由四部分组成：

1. **MLLM 预理解**
2. **Generalizable-Instructive Affordance**
3. **Affordance Flow Matching (AFM)**
4. **Grasp Flow Matching (GFM) + affordance-guided optimization**

### 核心直觉

作者的关键改变是：

- **原来**：语言 → 直接抓取姿态
- **现在**：语言 → 可泛化 affordance → 抓取姿态

这带来的因果变化是：

- **改变了什么**：把高熵、类别敏感的 hand-object contact 预测，改成低一层、更类别无关的“可抓局部区域”预测
- **改变了哪个瓶颈**：把“语义到动作”的跨度，降成“语义到局部结构” + “局部结构到动作”两段
- **带来什么能力变化**：对于 unseen category，只要局部结构和语义属性相似（handle/body/left/right/use），模型就更容易迁移

更直白地说：

- **contact map** 太细，泛化差  
- **object part** 太粗，指导弱  
- 作者想找一个中间甜点：  
  **既不要精确到每根手指的接触模式，也不能粗到只知道是“handle/body”**

### 1) Generalizable-Instructive Affordance

这是论文最核心的点。

作者不是直接把某个抓取的 contact map 当监督，而是先把**同语义组**的多个抓取合并：

- 同一意图
- 同一接触部位
- 同一抓取方向

然后把这些抓取的接触区域做并集，再做平滑，得到一个**语义组级别**的 affordance map。

它表达的不是：

- “这只手此时此刻该接触哪里”

而是：

- “在这种语义下，这个物体上哪些局部区域通常可抓”

这一步很关键，因为它把表示从**手依赖**变成了更**对象局部结构依赖**。

### 2) MLLM 预理解：先把用户话说“规整”

作者用 GPT-4o 做预理解，把原始语言和图像输入进去，抽取：

- 物体类别
- 用户意图
- 接触部位
- 抓取方向

最后组织成紧凑句子，例如：

- “use the mug from the left by contacting the handle”

这一步的作用不是“让 LLM 直接规划抓取”，而是把用户表达中的自由变体先归一化，减轻后续模型面对开放表达时的条件分布漂移。

### 3) AFM：从语言生成 affordance

AFM 的输入是：

- 点云特征
- 语言特征
- 离散方向特征
- 噪声化 affordance

输出是 affordance 的流场更新。

这里的重点不是 flow matching 本身，而是它学的是：

> **在给定语言语义和物体局部几何时，哪些区域应该被高亮为“可抓且符合意图”的区域。**

也就是说，AFM 负责把高层语义落到物体表面局部结构上。

### 4) GFM：从 affordance 生成灵巧手姿态

GFM 再把：

- affordance
- 语言
- 方向

作为条件，生成高自由度抓取姿态。

这意味着高维手姿不需要直接从语言“硬猜”，而是在 affordance 先验下做条件生成，难度更低。

### 5) 为什么把 penetration 处理放到测试时？

作者显式指出：训练时加 penetration loss 会损害

- 意图对齐
- 生成多样性

所以他们把几何可行性修正挪到测试阶段，做**non-parametric optimization**。  
优化目标包括：

- 接触点贴近 affordance 区域
- 保持关键指尖接触不漂移
- 减少物体穿透
- 减少手自身穿透
- 关节限位约束

这个决策背后的逻辑是：

> 训练阶段优先学“语义正确的抓”，推理阶段再补“几何更干净的抓”。

这是一个典型的“语义生成 / 几何修正”解耦策略。

### 表征层面的战略权衡

| 表示方式 | 泛化性 | 对灵巧手的指导性 | 主要问题 |
|---|---:|---:|---|
| 直接 language-to-grasp | 低 | 中 | 语言到动作跨度太大 |
| Contact map | 低 | 高 | 过细，强依赖具体接触模式，unseen category 容易崩 |
| Object part | 高 | 低 | 过粗，无法约束高自由度手姿 |
| 本文 affordance | 中高 | 中高 | 需要额外语义分组与中间生成模块 |

### 模块级 trade-off

| 模块 | 带来的好处 | 代价 |
|---|---|---|
| GPT-4o 预理解 | 提升开放表达鲁棒性 | 依赖闭源 MLLM |
| AFM + GFM 两段式 | 降低条件熵、增强跨类别迁移 | 级联误差可能传播 |
| 测试时优化 | 提升抓取质量且保留意图 | 推理更慢，需要几何计算 |

---

## Part III：证据与局限

### 关键证据信号

- **对比实验（open-set A/B）**：  
  作者方法在两个 open-set split 上都优于 ContactGen、Contact2Grasp、SceneDiffuser、DexGYS。最核心信号不是某一个指标，而是**意图一致性指标和抓取质量指标同时提升**。  
  例如 Open Set B 上，方法达到 **FID 0.162、CD 2.95、Top-1 0.532**，说明它不仅生成得像，而且更符合语言意图。

- **affordance 消融**：  
  这是最能支撑论文主张的证据。  
  用 predicted object part / predicted contact map / predicted our affordance 做条件时，作者的 affordance 在 open-set 上 consistently 更好。  
  这说明论文的贡献不只是“换了个生成器”，而是**中间表示本身确实改对了瓶颈**。

- **预理解与优化的消融**：  
  去掉 key cues extraction 或 direction，性能会下降；  
  用训练时 penetration loss 或学习式 refinement，效果也不如作者的 affordance-guided optimization。  
  这支持了两点：
  1. 开放表达需要先规整语义  
  2. 几何修正最好与语义生成解耦

- **真实世界实验**：  
  在 Leap Hand + Kinova Gen3 上，8 组任务累计 **66/80** 成功，显著高于 DexGYS 的 **39/80**。  
  更重要的是，作者强调**未额外使用真实训练数据**，说明 sim-to-real 泛化有一定说服力。

- **one-shot 泛化**：  
  在 novel categories 上，加入每类一个样本后，性能提升明显，说明这套 affordance 框架不仅适合 zero-shot，也适合低样本扩展。

### 1-2 个最值得记住的数字

1. **Open Set B：Top-1 0.532 / FID 0.162**  
   代表 open-set 语言-抓取对齐明显更强。

2. **真实世界：66/80 vs 39/80**  
   代表不是只在仿真里有效。

### 局限性

- **Fails when**: 指令需要连续、精细的接近方向或手指级接触约束时，六向离散方向和区域级 affordance 可能不够；当点云严重遮挡、目标局部结构缺失，或未见类别的局部几何与训练分布差异过大时，affordance 与最终抓取都会变差。

- **Assumes**: 依赖 GPT-4o 做预理解；依赖较高质量的 RGB 与场景点云；依赖“意图-部位-方向”语义组构造监督；当前主要围绕 Shadow Hand / Leap Hand 和桌面场景设计。测试时还有 200 步优化，存在额外推理成本。

- **Not designed for**: 双手协作、长时序 manipulation policy、可变形物体、强铰接物体、无相机位姿支持的自由方向推理，以及完全开放词汇的复杂任务规划。

### 复现与可扩展性的现实约束

- 使用 **GPT-4o** 做预理解，属于闭源依赖
- 真实部署需要多视角或机器人扫描来获得较完整点云
- 数据构造本身需要语义分组和高质量 grasp annotation
- 虽然训练只用单张 RTX 4090，但整个系统的数据制作与推理链条并不轻

### 可复用组件

1. **语义群组 contact-union → affordance GT 构造**  
   适合任何“语义比接触更稳定”的 embodied generation 任务。

2. **MLLM 关键线索压缩**  
   先把自然语言整理成结构化 cue，再喂给控制模型，是非常通用的工程范式。

3. **语义生成与几何修正解耦**  
   训练阶段学语义一致性，推理阶段补几何可行性，这个思路可迁移到 grasp / pose / manipulation 等多类任务。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_AffordDexGrasp_Open_set_Language_guided_Dexterous_Grasp_with_Generalizable_Instructive_Affordance.pdf]]