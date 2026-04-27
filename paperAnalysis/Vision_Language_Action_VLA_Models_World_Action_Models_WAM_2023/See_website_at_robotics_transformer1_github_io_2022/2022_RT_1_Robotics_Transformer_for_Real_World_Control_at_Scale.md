---
title: "RT-1: Robotics Transformer for Real-World Control at Scale"
venue: arXiv
year: 2022
tags:
  - Embodied_AI
  - task/robotic-control
  - task/language-conditioned-manipulation
  - transformer
  - token-learner
  - action-tokenization
  - dataset/ImageNet
  - dataset/QT-Opt
  - opensource/partial
core_operator: 将语言条件的图像历史压缩成少量视觉-语言 token，再用解码式 Transformer 预测离散动作 token，以 3Hz 实时闭环执行多任务机器人控制。
primary_logic: |
  自然语言指令 + 6 帧图像历史 → FiLM 条件化 EfficientNet 提取任务相关视觉特征并经 TokenLearner 压缩 → 解码式 Transformer 建模短时序上下文并输出离散化 arm/base/mode 动作 token → 真实机器人实时闭环执行
claims:
  - "RT-1 在真实机器人评测中达到 97% 的 seen-task 成功率、76% 的 unseen-task 成功率，且均显著优于同数据训练的 Gato 与 BC-Z 基线 [evidence: comparison]"
  - "加入仿真数据后，RT-1 在真实对象任务上的性能几乎不下降（92%→90%），但对仅在仿真中出现对象的真实世界成功率从 23% 提升到 87% [evidence: comparison]"
  - "在保留 97% 数据量的前提下移除 25% 任务种类，会带来接近将总数据量减半的泛化退化，说明数据多样性比单纯数据量更关键 [evidence: ablation]"
related_work_position:
  extends: "Gato (Reed et al. 2022)"
  competes_with: "Gato (Reed et al. 2022); BC-Z (Jang et al. 2021)"
  complementary_to: "SayCan (Ahn et al. 2022); MT-Opt (Kalashnikov et al. 2021a)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2023/See_website_at_robotics_transformer1_github_io_2022/2022_RT_1_Robotics_Transformer_for_Real_World_Control_at_Scale.pdf
category: Embodied_AI
---

# RT-1: Robotics Transformer for Real-World Control at Scale

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2212.06817), [Project](https://robotics-transformer1.github.io), [Code](http://github.com/google-research/robotics_transformer)
> - **Summary**: RT-1 把“语言 + 视觉到机器人动作”的控制问题重写为紧凑 token 序列建模问题，使 Transformer 能在真实机器人上实时运行，并从大规模多任务演示中学到可泛化的操作能力。
> - **Key Performance**: Seen/Unseen 任务成功率 97%/76%；接入 SayCan 后在两个真实厨房里的长程任务执行成功率均为 67%

> [!info] **Agent Summary**
> - **task_path**: 自然语言指令 + 6 帧 RGB 图像历史 / 移动操作机器人场景 → 11 维离散 arm/base/mode 控制动作
> - **bottleneck**: 机器人需要高容量多任务策略来吸收稀缺且异构的真实世界数据，但标准 Transformer 在实时控制场景下算得太慢，且任务无关视觉表征会削弱泛化
> - **mechanism_delta**: 用 FiLM 做早期语言-视觉融合，再用 TokenLearner 和离散动作 token 压缩输入输出序列，使高容量 Transformer 进入可实时部署区间
> - **evidence_signal**: 3000+ 次真实机器人试验中，RT-1 在 seen/unseen/robustness/long-horizon 上均超过 Gato 与 BC-Z
> - **reusable_ops**: [FiLM 早期语言条件化视觉编码, TokenLearner 令牌压缩与滑窗特征复用]
> - **failure_modes**: [完全未见过的新动作原语, 大幅背景与场景分布偏移]
> - **open_questions**: [如何突破 imitation learning 的示范者上限, 如何把跨机器人/跨仿真的吸收能力扩展到更广泛形态与任务]

## Part I：问题与挑战

### 1) 真问题是什么
这篇论文要解的，不是单个抓取或单个放置任务，而是一个更硬的问题：

**能否训练一个单一的、语言条件化的真实机器人通用策略，把大规模多任务数据“吸进去”，并在新任务、新物体、新环境里仍然有效？**

作者认为机器人领域长期卡住的核心，不是“没有更复杂的网络”，而是两个耦合瓶颈：

1. **数据瓶颈**：真实机器人数据贵、慢、难标注，传统做法往往按任务单独采集，导致每个技能都是“数据孤岛”。
2. **系统瓶颈**：高容量模型通常更能吃下多样化数据，但机器人控制要求低延迟、稳定频率；如果模型很强但跑不动，就无法落地。

### 2) 为什么现在要解决
这件事“现在可做”的原因有两个：

- **数据规模第一次接近可验证区间**：作者用 13 台机器人、17 个月，采了约 130k 演示、744 个指令任务。
- **序列模型已足够成熟**：Transformer 在 NLP/视觉里已经证明了“容量 + 开放式训练”能带来泛化，但机器人里还缺一个能实时运行的版本。

所以，RT-1 的切入点很明确：  
**不是简单把 Transformer 搬到机器人上，而是把它压缩到机器人能跑的实时控制预算内，同时保住多任务泛化能力。**

### 3) 输入/输出接口与边界条件
- **输入**：自然语言指令 + 最近 6 帧 300×300 图像
- **输出**：11 维离散动作
  - 7 维机械臂动作：x, y, z, roll, pitch, yaw, gripper
  - 3 维底盘动作：x, y, yaw
  - 1 维模式：arm / base / terminate
- **控制频率**：3 Hz 闭环控制
- **训练方式**：behavior cloning / imitation learning
- **场景边界**：以办公室厨房中的移动操作为主，强调多任务泛化，不是通用全机器人世界模型

换句话说，这篇论文的边界很清楚：  
它证明的是**“真实世界多任务操控 backbone 可以规模化”**，而不是“机器人已经获得开放世界通用智能”。

## Part II：方法与洞察

### 1) 方法骨架
RT-1 的系统链路可以概括为：

1. 用 **USE** 编码语言指令；
2. 用 **FiLM 条件化的 EfficientNet-B3** 提取任务相关视觉特征；
3. 把每帧图像的 81 个 token 用 **TokenLearner** 压成 8 个；
4. 将 6 帧共 48 个 token 喂给 **decoder-only Transformer**；
5. 输出离散化动作 token，执行机器人动作。

这里最关键的不是“用了 Transformer”，而是**把输入侧和输出侧都 token 化，并把序列长度压到机器人可承受范围**。

### 核心直觉

#### 直觉一：把语言提前注入视觉编码，改变的是“信息瓶颈”
**改变了什么**：语言不再只在高层决策时起作用，而是在视觉 backbone 内部通过 FiLM 直接调制特征提取。  
**改变了哪个瓶颈**：从“任务无关视觉特征”变成“任务相关视觉特征”。  
**带来什么能力变化**：模型更容易在同一场景里聚焦与当前指令有关的物体、部件和交互区域，因此更容易做组合式泛化。

这也是它比“先看图、后拼语言”的方案更像一个真正的条件控制器，而 არა一个后接文本头的视觉模型。

#### 直觉二：把视觉 token 大幅压缩，改变的是“实时性约束”
**改变了什么**：每帧 81 个视觉 token 被压到 8 个，再配合滑窗特征复用。  
**改变了哪个瓶颈**：Transformer 的序列长度和推理延迟。  
**带来什么能力变化**：35M 级别模型能以 3Hz 在真实机器人上稳定闭环运行。

这一步非常关键。  
没有它，Transformer 在机器人上往往“能训、不能控”。

#### 直觉三：把连续控制改成离散动作 token，改变的是“输出分布约束”
**改变了什么**：每个动作维度离散成 256 个 bin。  
**改变了哪个瓶颈**：从单峰连续回归，转为更适合多模态演示数据的分类式预测。  
**带来什么能力变化**：在多任务、多人示范、异构数据下，策略不容易退化为“平均动作”。

这点和论文后续 ablation 的逻辑是一致的：  
真实机器人演示数据常常天然多峰，简单高斯回归容易学成折中解。

#### 直觉四：用短历史而不是单帧，改变的是“部分可观测性”
**改变了什么**：输入不是单张图，而是 6 帧短时序。  
**改变了哪个瓶颈**：单帧无法反映动态状态、机器人相对位姿和最近交互轨迹。  
**带来什么能力变化**：更稳的闭环修正能力，以及对底盘位置、遮挡和扰动更好的鲁棒性。

### 2) 为什么这个设计有效
RT-1 的成功不是某一个模块单独起作用，而是三件事同时成立：

- **早融合**保证模型看到的是“与任务有关的视觉信息”；
- **压 token**保证模型“来得及算”；
- **离散动作**保证模型“学得到多模态行为”。

这三者合在一起，才把“高容量 + 可实时 + 可泛化”放到了同一个系统里。

### 3) 战略权衡

| 设计选择 | 解决的瓶颈 | 主要收益 | 代价/风险 |
|---|---|---|---|
| FiLM 条件化 EfficientNet | 任务无关视觉表征 | 更早聚焦任务相关物体与区域，提升组合泛化 | 依赖较强的预训练视觉 backbone |
| TokenLearner 压缩 + 滑窗复用 | Transformer 推理过慢 | 81→8 token/帧，支持 3Hz 实时控制 | 可能丢失细粒度空间细节 |
| 离散动作 token | 连续回归难表示多模态策略 | 更适合 imitation learning 下的复杂动作分布 | 动作范围与分辨率需预先设定 |
| 6 帧历史 + decoder-only Transformer | 单帧部分可观测 | 更稳的闭环控制与短时记忆 | 上下文长度仍有限，不等于长期记忆 |
| 大规模多任务数据 | 任务孤岛导致泛化差 | 学到跨任务共享结构 | 数据采集成本极高，复现门槛高 |

## Part III：证据与局限

### 1) 关键证据：能力跳跃到底在哪里

#### 信号 A：真实机器人大规模对比评测
最直接的证据来自 3000+ 次真实世界试验。

结论不是“RT-1 能做很多事”，而是：  
**在同样使用作者数据训练时，RT-1 的架构比 Gato / BC-Z 更能把这些数据转成泛化能力。**

关键结果：
- seen tasks：**97%**
- unseen tasks：**76%**
- distractors：**83%**
- backgrounds：**59%**

最重要的含义是：  
RT-1 的优势不只体现在训练内任务上，也体现在**新指令组合、干扰物和新环境**上。

#### 信号 B：长程任务中，RT-1 没有像基线那样在分布偏移下崩掉
接入 SayCan 后：
- RT-1 在 Kitchen1 / Kitchen2 的长程执行成功率都是 **67%**
- BC-Z 在 Kitchen2 掉到 **13%**
- Gato 在 Kitchen2 是 **0%**

这说明 RT-1 的改进不只是短程抓取准确率，而是**足够可靠，能被更高层规划系统反复调用**。  
对 embodied system 来说，这是更关键的“系统级证据”。

#### 信号 C：它确实能“吸收”异构数据
加入仿真数据后：
- 真实对象任务几乎不掉：**92% → 90%**
- 仅在仿真出现的对象，真实世界成功率：**23% → 87%**
- 仿真见过对象但真实没见过技能组合：**7% → 33%**

这说明 RT-1 不只是“在单一数据分布上拟合得更好”，而是具备一定的**数据吸收性**：  
能把异构来源的数据转成真实世界能力，而不是被分布差异拖垮。

#### 信号 D：数据多样性比纯数量更值钱
论文一个非常重要的 ablation 结论是：

- 减少总数据量，性能会下降；
- 但**减少任务种类**，泛化下降更快。

也就是说，对通用机器人策略而言，**breadth > sheer volume**。  
这比“再多采点同类数据”更有方法论价值。

### 2) 局限性

- **Fails when**: 需要执行训练集中从未出现过的全新动作原语；背景/场景变化过大时，尤其是超出办公室厨房分布的环境；更高精度或更灵巧的操作任务中容易失效。
- **Assumes**: 大规模人工遥操作演示与文本标注；特定移动操作机器人平台；预训练视觉模型；固定动作边界与离散化分辨率；3Hz 控制足以覆盖任务节奏。
- **Not designed for**: 纯强化学习探索、超高灵巧手操作、人与机器人近距离交互安全场景、完全零样本跨机器人形态迁移。

### 3) 复现与可扩展性的现实约束
这篇论文最强，也最难复现的地方，不在网络，而在系统资源：

- **13 台机器人、17 个月、130k 演示** 的采集成本非常高；
- 评测用了 **3000+ 次真实机器人 rollouts**；
- 依赖 Everyday Robots 平台和专门搭建的数据采集环境；
- 代码开源，但**数据、硬件、完整训练条件并不完全开放**。

因此，这篇论文的真正贡献更像是一个**系统 recipe**：  
告诉你什么样的模型结构，才能把大规模真实机器人数据转成可部署的控制策略。

### 4) 可复用组件
如果把论文拆成可迁移的工程部件，最值得复用的是：

1. **FiLM 早期语言-视觉融合**：适合任何语言条件控制任务；
2. **TokenLearner + 特征复用**：适合实时 embodied Transformer；
3. **离散动作 token 化**：适合多模态示范数据上的 behavior cloning；
4. **多源数据混训范式**：真实 + 仿真 + 多机器人数据的统一吸收框架；
5. **“多样性优先于数量” 的采数原则**：对后续数据引擎设计很重要。

### 5) 一句话总结 So What
RT-1 的意义不只是“又一个机器人 Transformer”，而是它首次比较完整地证明了：

**只要把任务相关感知、token 压缩和离散动作建模组合起来，Transformer 可以在真实世界机器人上成为一个可扩展、可部署、可泛化的多任务控制 backbone。**

![[paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2023/See_website_at_robotics_transformer1_github_io_2022/2022_RT_1_Robotics_Transformer_for_Real_World_Control_at_Scale.pdf]]