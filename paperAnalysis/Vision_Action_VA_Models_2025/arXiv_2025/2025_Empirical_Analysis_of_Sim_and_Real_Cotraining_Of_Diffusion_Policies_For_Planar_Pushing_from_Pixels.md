---
title: "Empirical Analysis of Sim-and-Real Cotraining of Diffusion Policies for Planar Pushing from Pixels"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - diffusion
  - behavior-cloning
  - data-reweighting
  - dataset/PlanarPushing
  - opensource/partial
core_operator: 通过 α 重加权把仿真示范与真实示范联合送入扩散策略训练，并保留可辨别的域信号，让仿真补覆盖、真实定上限、域标签定动作。
primary_logic: |
  少量真实示范 DR + 大量仿真示范 DS + 混合权重 α + 域差异设定
  → 训练像素到动作的 Diffusion Policy，并系统扫掠数据规模、混合比例、物理/视觉/任务 gap
  → 在真实平面推动作上获得更高成功率，并揭示“仿真补状态覆盖、真实决定性能天花板、域可辨别性决定是否正迁移”
claims:
  - "在真实示范仅 10 条时，sim-real cotraining 可将真实世界成功率从 2/20 提升到 14/20；在 50 条时可从 10/20 提升到 19/20 [evidence: comparison]"
  - "扩展仿真数据通常能提升或维持性能，但在固定真实数据规模下会出现平台；增加真实数据会抬高这一平台，最佳共训练效果约等价于把目标域数据增至 2-3 倍 [evidence: comparison]"
  - "当仿真与目标域仍存在 physics gap 时，完全消除 visual gap 会降低性能；加入环境 one-hot 编码可恢复性能，说明高性能策略需要识别所处域 [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2024)"
  competes_with: "Sim-only pretraining + finetuning; cotraining + finetuning"
  complementary_to: "Domain Randomization (Tobin et al. 2017); Scalable Real2Sim (Pfaff et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Empirical_Analysis_of_Sim_and_Real_Cotraining_Of_Diffusion_Policies_For_Planar_Pushing_from_Pixels.pdf
category: Embodied_AI
---

# Empirical Analysis of Sim-and-Real Cotraining of Diffusion Policies for Planar Pushing from Pixels

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.22634), [Project](https://sim-and-real-cotraining.github.io/), [Code-Policy](https://github.com/sim-and-real-cotraining/diffusion-policy), [Code-Sim/Eval](https://github.com/sim-and-real-cotraining/planning-through-contact)
> - **Summary**: 这篇论文系统研究了“少量真实示范 + 大量仿真示范”联合训练扩散控制策略时，性能提升究竟来自哪里，并指出关键不只是缩小 sim2real gap，而是让策略既能共享覆盖，又能识别自己当前身处哪个域。
> - **Key Performance**: 真实世界 10 条示范时成功率从 2/20 提升到 14/20，50 条时从 10/20 提升到 19/20；最佳 cotraining 效果约等价于把目标域数据增加 2–3 倍。

> [!info] **Agent Summary**
> - **task_path**: 双视角 RGB 图像 + 末端位姿 / 仿真与真实混合示范 -> 平面推任务动作序列与真实世界成功执行
> - **bottleneck**: 真实示范稀缺，且仿真与真实在物理、视觉和动作分布上不一致，导致“仿真补数据”与“错误迁移”同时发生
> - **mechanism_delta**: 用 α 重加权的 sim-real 联合训练替代“先仿真后微调”，并保留或显式提供域标签，使策略共享覆盖但按域选择动作
> - **evidence_signal**: 低数据真实实验最高 7x 提升，且“去掉 visual gap 但保留 physics gap”会掉点，加入 one-hot 域标签可恢复性能
> - **reusable_ops**: [alpha-mixture-reweighting, environment-one-hot-conditioning]
> - **failure_modes**: [fixed-real-data 下单纯扩展仿真数据会平台化, physics-gap 存在时若策略无法区分 sim/real 会输出折中的错误动作]
> - **open_questions**: [该规律能否迁移到多任务与抓取类操作, 是否存在既保留域可辨别性又能更强对齐表征的训练目标]

## Part I：问题与挑战

这篇论文要回答的不是“Diffusion Policy 能不能做平面推”，而是一个更基础的问题：**当真实机器人示范很贵时，如何把大量仿真示范真正转化成真实世界性能，而不是变成额外噪声？**

### 真实瓶颈是什么
核心瓶颈不是单纯的数据量不足，而是两个更具体的问题叠在一起：

1. **真实数据覆盖不足**：少量真实 demo 很容易让行为克隆过拟合，策略在未覆盖状态上失效。
2. **跨域动作不兼容**：sim 和 real 不仅视觉不同，物理也不同；更麻烦的是，文中 sim 数据来自近最优规划器，real 数据来自人类 teleop，本身还带有 **action gap**。  
   所以如果直接把两域数据拼起来，模型可能学到一个“平均动作”，而不是每个域都正确的动作。

### 输入/输出接口
- **输入**：顶视角 RGB、腕部 RGB、末端执行器位姿
- **输出**：未来一段 pushing 动作序列（末端 x-y 目标位置）
- **任务**：从像素完成 planar pushing，目标是 **真实世界成功率**

### 为什么现在值得研究
因为三个条件同时成熟了：
- Diffusion Policy 已是强 imitation learning 基线；
- 仿真数据生成基础设施更成熟；
- 机器人界越来越依赖“真实大数据 + 仿真扩展”的路线。  

这使得“sim-and-real cotraining 的经验规律”本身变成了高价值问题。

### 边界条件
论文刻意只研究一个任务：**planar pushing from pixels**。  
好处是能做非常彻底的 sweep；代价是结论的外推范围有限，尤其对多任务、抓取、长时程操作未必直接成立。

---

## Part II：方法与洞察

这篇工作的方法创新不在新 backbone，而在于把 cotraining 拆成几个可控旋钮，并逐个验证其因果作用：

- 真实/仿真数据规模：\(|D_R|, |D_S|\)
- 混合权重：\(\alpha\)
- gap 类型：visual / physics / task
- 域可辨别性：视觉差异或显式 one-hot 域标签

### 方法框架

#### 1. Vanilla cotraining
作者用同一个 Diffusion Policy 联合训练 sim 与 real 数据。  
训练时以概率 **α** 采样真实数据，以 **1-α** 采样仿真数据。直观上：

- **α 大**：更贴近真实部署域，但更容易在小样本真实数据上过拟合
- **α 小**：更吃到 sim 的规模优势，但风险是被错误物理/动作分布带偏

这使 α 成为控制“真实锚点 vs 仿真覆盖”的关键旋钮。

#### 2. Sim-and-sim 受控替身实验
真实世界 sweep 很贵，所以作者再构造一个 **target sim** 来模拟 real 域。  
这样能做两件事：

- 用更多试验做高置信度评估
- 人工控制 visual / physics / task gap，单独看哪种 gap 最伤 cotraining

#### 3. 机制诊断
作者没有停在“性能变好”这一层，而是继续问：**为什么会变好？**

他们用了三种诊断：
- **binary probe**：看策略内部表征是否保留了域标签
- **kNN rollout attribution**：分析策略当前动作更像是从 real 数据还是 sim 数据学来的
- **power-law 拟合**：看目标域 test loss / action MSE 是否随 sim 数据规模呈规律下降

### 核心直觉

**真正起作用的变化不是“让 sim 完全长得像 real”，而是：让 sim 去补真实数据没覆盖到的状态，同时让策略知道当前到底在 sim 还是 real，从而输出不同动作。**

更具体地说：

- **what changed**：从“只靠 real 学”或“先 sim 预训练再 real 微调”，变成“带 α 重加权的联合训练 + 域可辨别信号”
- **which bottleneck changed**：
  - 数据覆盖瓶颈被大量 sim demo 缓解
  - 域错配引起的动作歧义，被域标签/视觉线索缓解
  - 真实数据的稀缺性，通过 α 保住了训练目标仍以真实域为锚
- **what capability changed**：
  - 低数据真实场景下，成功率显著提升
  - 策略不会被迫在两个域之间学“平均动作”
  - sim 数据带来的收益可预测，但不会无限增长；最终上限仍由 real 数据决定

### 为什么这个设计有效
因果上最关键的一点是：

> **如果两个域的 physics 不同，那么相似观测并不对应同一个最优动作。**

这时：
- 如果把 visual gap 完全抹平，又不给域标签，模型就很难知道该输出哪套动作；
- 如果保留一些视觉差异，或直接给 one-hot 域标签，策略就能“路由”到对应域的动作模式；
- sim 数据因此不再是“替代 real”，而是“在 real 数据空白处补 coverage”。

### 战略取舍

| 设计选择 | 改变了什么 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| 增加仿真数据 \(|D_S|\) | 补充状态覆盖 | 低/中真实数据区间显著增益 | 固定真实数据时最终会平台化 |
| 增加真实数据 \(|D_R|\) | 提供正确部署域动作锚点 | 抬高性能天花板 | 采集昂贵、慢 |
| 提高物理保真度 | 缩小 contact dynamics gap | 对 contact-rich 任务最重要 | 建模成本高 |
| 完全消除视觉 gap（但 physics gap 仍在） | 降低域可辨别性 | 无稳定收益 | 反而可能掉点 |
| 加入 environment one-hot | 显式提供域标签 | 恢复/提升性能 | 部署时需知道域身份 |
| 先 sim 后 finetune | 分阶段训练 | 直觉简单 | 文中未见稳定优于混合训练 |
| 表征对齐损失（MMD/对抗） | 压缩域差异 | 理论上更 invariant | 可能抹掉动作所需的域信息 |

### 额外发现：什么没用
两类看起来“应该有效”的改动，在文中都没有稳定胜出：

1. **Classifier-Free Guidance 放大动作差异**：不如直接 one-hot + 常规采样实用。
2. **对抗/MMD 表征对齐**：没有可靠优于 vanilla cotraining。  
   这说明这里的关键不是“强行域不变”，而是**保留与控制相关的域差异**。

---

## Part III：证据与局限

### 关键证据信号

- **比较实验 → 低数据真实收益很大**  
  在真实 10 条 demo 时，成功率从 **2/20 提升到 14/20**；50 条时从 **10/20 提升到 19/20**。这说明 sim 数据在真实样本稀缺时确实能显著补位。

- **比较实验 → sim 扩展有收益，但有天花板**  
  随着 \(|D_S|\) 增加，性能通常提升或持平，但会出现平台；加入更多真实数据会抬高这个平台。最佳 cotraining 约等价于把目标域数据增至 **2–3x**。

- **消融实验 → physics gap 比纯视觉保真更关键**  
  去掉 physics gap 可带来 **15.5%** 的成功率提升；说明对接触丰富任务，物理一致性比单纯渲染更重要。

- **消融 + 分析 → 域可辨别性是必要条件**  
  当 physics gap 存在时，完全去掉 visual gap 反而更差；加 one-hot 域标签又能恢复。  
  binary probe 还能在最终激活上以 **74%–93%** 的准确率分出域，说明高性能策略会主动保留域信息。

- **分析实验 → 正迁移来自“补 coverage”，不是“覆盖 real”**  
  kNN rollout attribution 显示：真实部署时，策略大多数行为仍更接近真实数据，只在真实数据空白处借助 sim 数据。  
  同时，目标域 test loss / action MSE 随 sim 数据规模近似 obey power law，表明 sim 扩展带来的收益是系统性的。

### 局限性
- **Fails when**: 真实数据过少到不足以锚定真实域策略、或 physics/task gap 很大但策略又无法区分域时；此时会出现性能平台甚至负迁移。
- **Assumes**: 有可用仿真器与数据生成管线；sim 侧可由规划器生成高质量 demo；real 侧有 teleop 数据；部署时能通过视觉或显式标签识别域；训练依赖 Diffusion Policy、ResNet18、较大量 sweep 与评测算力。
- **Not designed for**: 多任务通用操作、抓取/灵巧手/装配等不同接触机制任务、无视觉输入设置、以及“完全靠 sim 替代 real 数据”的场景。

### 资源与复现依赖
虽然代码开放，但完整复现实验仍依赖：
- KUKA LBR iiwa 7 硬件
- Drake 仿真与规划器
- 人类 teleop 真实示范
- 大规模 sweep/评测计算  
作者共训练了 **50+ real policies**、**250 simulated policies**，评测超过 **1000+ real trials** 与 **50,000+ simulated trials**。因此这是一篇**开源但高基础设施门槛**的工作。

### 可复用组件
- **α 混合重加权**：低成本、强效果的 sim-real 联合训练基线
- **environment one-hot conditioning**：当 physics 不一致时，显式域标签很实用
- **target-sim surrogate**：用于高置信度 sweep 的分析范式
- **probe + kNN + scaling-law**：诊断“为何正迁移成立”的通用工具链

## Local PDF reference
![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Empirical_Analysis_of_Sim_and_Real_Cotraining_Of_Diffusion_Policies_For_Planar_Pushing_from_Pixels.pdf]]