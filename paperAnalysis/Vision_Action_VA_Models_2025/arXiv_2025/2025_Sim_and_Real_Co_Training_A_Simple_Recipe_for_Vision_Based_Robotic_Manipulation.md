---
title: "Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - co-training
  - diffusion
  - digital-cousin
  - dataset/RoboCasa
  - opensource/no
core_operator: 以可调采样比α把真实演示、任务感知digital cousin仿真和任务无关先验仿真混合到同一行为克隆训练流中，用真实数据锚定部署分布、用仿真数据扩展覆盖度
primary_logic: |
  少量真实视觉操控演示 + 大量仿真演示（Prior/DC） → 按采样比α进行混合批次行为克隆，并近似对齐任务语义与关键相机视角 → 直接部署到真实机器人，提升成功率与对新物体/新位置的泛化
claims:
  - "在两个机器人平台的6个真实任务上，Real+DC+Prior 的平均成功率从 Real-only 的 45.3% 提升到 83.2% [evidence: comparison]"
  - "即使不为目标任务专门构建仿真环境，Real+Prior 也在全部6个任务上优于 Real-only，平均成功率从 45.3% 提升到 76.8% [evidence: comparison]"
  - "仿真共训不仅提升域内成功率，也提升真实世界泛化：CounterToSinkPnP/CupPnP 在未见物体上由 33%/10% 提升到 50%/80%，在未见位置上由 11%/43% 提升到 28%/100% [evidence: comparison]"
related_work_position:
  extends: "RoboCasa (Nasiriany et al. 2024)"
  competes_with: "Domain Randomization (Tobin et al. 2017); Reconciling Reality through Simulation (Torne et al. 2024)"
  complementary_to: "RL-CycleGAN (Rao et al. 2020); Active Domain Randomization (Mehta et al. 2020)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Sim_and_Real_Co_Training_A_Simple_Recipe_for_Vision_Based_Robotic_Manipulation.pdf
category: Embodied_AI
---

# Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.24361), [Project](https://co-training.github.io)
> - **Summary**: 这篇工作证明：视觉机器人操控不一定要先把仿真调成“几乎真实”，只要用少量真实演示作锚点、再把大量仿真数据按合适比例混入训练，就能稳定提升真实世界操作成功率。
> - **Key Performance**: 6个真实任务平均成功率由 45.3% 提升到 83.2%；GR-1 CupPnP 在未见位置上的成功率由 43% 提升到 100%。

> [!info] **Agent Summary**
> - **task_path**: RGB视觉 + 本体状态 + 少量真实演示/大量仿真演示 -> 真实机器人连续动作序列与任务成功
> - **bottleneck**: 真实演示昂贵，而传统sim-to-real通常要求高成本视觉/动力学对齐；真正瓶颈是如何让“不完美但海量”的仿真数据对真实部署产生净增益
> - **mechanism_delta**: 将仿真数据从“单独预训练后再迁移”的辅助源改为与真实数据按采样比α联合训练，并只近似对齐任务语义与相机视角
> - **evidence_signal**: 6个真实任务上 Real+DC+Prior 平均成功率 83.2%，明显高于 Real-only 的 45.3%，且相机对齐、α调节、仿真数据量都有消融支持
> - **reusable_ops**: [mixed-batch-co-training, approximate-camera-alignment, digital-cousin-data-generation]
> - **failure_modes**: [camera-misalignment-reduces-gains, prior-sim-behavior-mismatch-can-bias-policy]
> - **open_questions**: [how-to-auto-tune-alpha, how-to-extend-to-deformable-liquid-and-long-horizon-tasks]

## Part I：问题与挑战

这篇论文要解决的不是“如何做一次漂亮的 sim-to-real 迁移”，而是一个更现实的问题：

**真实机器人数据太贵，仿真数据很多，但两者不完全一致时，怎样让仿真数据真正帮到真实部署？**

### 1. 真正的难点是什么
传统路线有两条：

1. **只用真实数据训练**  
   问题是数据采集慢、贵、难扩展，尤其是多任务、多场景、多物体时。

2. **先在仿真训练再迁移到真实**  
   问题是往往需要大量人工去缩小 reality gap，比如：
   - 精调动力学
   - 做 domain randomization
   - 构建高保真 digital twin

作者指出，真正卡住社区的瓶颈不是“仿真不够真实”本身，而是：

> **哪些对齐是必须的，哪些不需要；以及如何让少量真实数据去“校准”大量仿真数据的价值。**

### 2. 为什么现在值得做
因为现在已经有两类工具成熟了：

- **大规模真实机器人数据**开始出现，但采集成本依旧很高；
- **自动化仿真数据生成工具**（如 MimicGen / DexMimicGen）可以从几十条源示范扩展到上千上万条轨迹。

所以现在的关键问题变成：  
**既然仿真数据获取便宜，能否不用做重型 sim-to-real 工程，而直接把它和真实数据一起训练？**

### 3. 输入/输出接口
论文研究的是**vision-based robotic manipulation**：

- **输入**：RGB 图像 + 机器人本体状态 + 示范轨迹（真实/仿真）
- **输出**：机器人连续动作序列
- **训练目标**：行为克隆式策略学习
- **部署方式**：训练完直接上真实机器人执行

### 4. 边界条件
这篇工作虽然强调“简单 recipe”，但并不是无条件适用：

- 主要研究 **同一或兼容 embodiment / action space** 下的共训；
- 任务以 **pick-and-place、关门、倒球** 等视觉操控为主；
- 大多是 **短时程操作**；
- 不主张 **zero-real-data** 的纯仿真迁移；
- 对 **相机视角** 和 **任务语义** 的近似一致仍然比较重要。

---

## Part II：方法与洞察

### 方法骨架

作者提出的 recipe 很简单，核心是把训练数据分成三类：

- **Real**：目标真实任务上的少量示范
- **Prior**：任务无关的既有仿真数据
- **DC (Digital Cousin)**：为目标任务专门搭的、但并不追求完美一致的“数字近亲”仿真数据

然后做一件事：

> **用采样比 α 把仿真和真实数据混合到同一个 batch 流里，训练一个视觉行为克隆策略。**

具体上：

1. 收集少量真实示范；
2. 若有条件，则为目标任务构建 **task-aware digital cousin**；
3. 用 MimicGen / DexMimicGen 将少量仿真源示范扩增成大规模数据；
4. 用固定概率 α 从仿真集采样、1-α 从真实集采样；
5. 训练 Diffusion Policy；
6. 直接部署到真实机器人。

### 两类仿真数据的分工

#### 1) Task-agnostic Prior
优点：
- 开箱即用
- 数据量大
- 场景多样

缺点：
- 和目标任务可能差很多
- 行为模式不一定一致

#### 2) Task-aware Digital Cousin
作者给了一个很实用的定义：DC 至少尽量保留四件事：

1. 同样的机器人和动作空间  
2. 同样的任务目标/成功判定  
3. 同类物体类别  
4. 同类环境装置类别

注意：**不是 digital twin**。  
他们并不追求完美几何、完美纹理、完美动力学，而是追求“任务语义上足够像”。

### 核心直觉

**变化了什么：**  
从“仿真里学完再迁移”变成“真实与仿真共同定义训练分布”。

**改变了哪个瓶颈：**  
把瓶颈从“仿真是否足够逼真”转成“仿真是否提供了真实数据缺少的覆盖度，而真实数据是否足以提供部署锚点”。

**带来了什么能力变化：**  
策略不再依赖完美 sim-to-real 对齐，也能从仿真的多样性里学到对真实世界有用的不变性与动作先验，从而提升：
- 数据效率
- 真实任务成功率
- 对未见物体/位置的泛化

更因果地说，这个设计之所以有效，是因为：

1. **真实数据提供 grounding**  
   告诉模型：真实部署时哪些视觉线索和动作映射是必须保真的。

2. **仿真数据提供 support expansion**  
   扩大物体、场景、初始位姿、动作模式的覆盖面。

3. **α 是“梯度权重旋钮”**  
   太低：仿真贡献不够。  
   太高：真实分布被淹没。  
   合适时：既保留真实锚点，又最大化利用仿真规模。

4. **相机对齐比完美物理对齐更关键**  
   对视觉操控来说，观察空间对齐直接影响“看见什么再做什么”；论文甚至发现某些任务里动力学精调不是必要条件。

### 策略性取舍表

| 设计选择 | 改变的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 只用 Real | 只依赖真实覆盖 | 最贴近部署分布 | 数据稀缺，泛化差 |
| Real + Prior | 用现成仿真补覆盖 | 开箱即用也能涨点 | 任务/行为不一致时可能学偏 |
| Real + DC | 提高任务语义对齐 | 主任务收益通常更大 | 需要搭建 digital cousin |
| 提高 α（更多仿真） | 放大仿真先验 | 小数据时效果强 | α 过大真实信号被冲淡 |
| 近似相机对齐 | 降低视觉映射误差 | 明显提升视觉策略迁移 | 仍需一些环境工程 |
| 增加仿真多样性 | 扩展状态分布 | 提升 OOD 泛化 | 若行为模式不一致会引入噪声 |

### 这篇 paper 最重要的洞察
作者最有价值的结论不是“仿真有用”，而是：

> **仿真不需要和真实完美一致，只要在任务语义、视角等关键维度上不过分偏离，并且由少量真实数据做分布锚定，就能产生稳定增益。**

这使得方法从“重工程 sim-to-real”转向“轻对齐 co-training”。

---

## Part III：证据与局限

### 关键证据

- **比较信号｜主结论成立**  
  在 2 个平台、6 个真实任务上：
  - Real-only：**45.3%**
  - Real + DC：**81.1%**
  - Real + Prior：**76.8%**
  - Real + DC + Prior：**83.2%**  
  这说明不管是任务感知仿真还是任务无关仿真，都能提供真实部署收益，而两者结合最好。

- **比较信号｜“开箱即用仿真”也有价值**  
  最有说服力的一点是 **Prior** 数据并不是为目标任务量身做的，但仍在全部 6 个任务上优于 Real-only。  
  这直接支持了论文的核心判断：**不必先做高成本精确对齐，仿真也能帮忙。**

- **比较信号｜泛化确实增强**  
  在未见物体与未见初始位置上，共训策略显著优于只用真实数据：
  - 未见物体：33%→50%，10%→80%
  - 未见位置：11%→28%，43%→100%  
  这说明仿真数据的核心价值之一是 **补真实数据覆盖不到的变化因素**。

- **消融信号｜recipe 不是“随便混”**  
  作者做了几个关键消融：
  - **仿真数据量**减少会明显掉点；
  - **α 需要调**，1:1 并不好，CupPnP 上 99% 最好，但再推到 99.5%/99.9% 会明显变差；
  - **相机对齐重要**，未对齐会显著掉点；
  - 即使真实数据增多到 **400 demos**，共训仍然持续优于 real-only。  
  所以这个方法不是“仿真越多越好”，而是要守住几个关键操作位。

### 1-2 个最值得记的指标
1. **平均真实任务成功率：45.3% → 83.2%**  
2. **GR-1 CupPnP 未见位置泛化：43% → 100%**

### 局限性

- **Fails when:** 仿真与真实的行为模式不一致时，先验仿真可能反而带偏策略；论文附录里提到双手任务若只混入单臂 prior 数据，会学出错误的单臂行为。严重相机错位、难以模拟的可变形物/液体任务也会削弱收益。
- **Assumes:** 需要至少少量真实示范作锚点；通常假设相同或兼容的机器人 embodiment / action space；需要能构建基本 digital cousin、重渲染相机视角、并用 MimicGen/DexMimicGen 扩增数据；还需要调 co-training ratio α。
- **Not designed for:** 零真实数据的纯 sim-to-real、长时程复杂操作、高精度插入、以及高度依赖真实流体/软体物理的任务。

### 可复用组件

- **α 控制的 mixed-batch co-training**：最直接、最通用
- **task-aware digital cousin 定义**：不做 twin，只保关键语义
- **近似相机对齐**：低成本但高收益
- **仿真多样性扩增**：物体类别、初始位姿、场景纹理
- **MimicGen / DexMimicGen 式轨迹放大**：把少量源示范变成大规模训练集

### 可复现性/资源依赖提醒
论文没有在正文中明确给出代码仓库链接；虽然方法本身概念简单，但真实复现依赖：
- 真实机器人平台与遥操作采集
- RoboCasa / MimicGen / DexMimicGen 工具链
- 为目标任务做一定程度的 digital cousin 环境工程

所以它的“简单”更准确地说是：**比精细 sim-to-real 更轻量，而不是零工程成本。**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Sim_and_Real_Co_Training_A_Simple_Recipe_for_Vision_Based_Robotic_Manipulation.pdf]]