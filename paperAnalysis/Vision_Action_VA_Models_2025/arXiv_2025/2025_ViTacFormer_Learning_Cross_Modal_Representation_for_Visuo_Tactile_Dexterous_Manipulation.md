---
title: "ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/dexterous-manipulation
  - task/visuo-tactile-manipulation
  - cross-attention
  - autoregressive-modeling
  - curriculum-learning
  - "dataset/Peg Insertion"
  - "dataset/Cap Twist"
  - "dataset/Vase Wipe"
  - "dataset/Book Flip"
  - "dataset/Make Hamburger"
  - opensource/no
core_operator: 用视觉-触觉双向交叉注意力学习共享表征，并通过自回归未来触觉预测把接触动态回注到动作生成中。
primary_logic: |
  多视角视觉 + 指尖触觉 + 机器人本体状态 → 双向交叉注意力对齐视觉与触觉语义 → 先预测未来触觉并将其作为接触先验回注策略解码 → 生成双臂灵巧手动作序列
claims:
  - "在四个短时真实机器人任务上，ViTacFormer分别达到10/10、10/10、9/10、9/10成功，而ACTw/T为6/10、6/10、4/10、4/10，显著优于无触觉和naive触觉融合基线 [evidence: comparison]"
  - "在11阶段Make Hamburger长时任务上，ViTacFormer的整体HNS为0.88，高于无触觉基线的0.61；论文同时报告完整任务成功率超过80% [evidence: comparison]"
  - "消融实验表明，交叉注意力、未来触觉预测和自回归使用预测触觉三项设计都能相对naive触觉拼接持续提升HNS与成功率，说明增益来自跨模态表征与接触动态建模而非仅增加输入模态 [evidence: ablation]"
related_work_position:
  extends: "ACT (Zhao et al. 2023)"
  competes_with: "HATO (Lin et al. 2025); Diffusion Policy (Chi et al. 2023)"
  complementary_to: "SPARSH (Higuera et al. 2024); π0 (Black et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ViTacFormer_Learning_Cross_Modal_Representation_for_Visuo_Tactile_Dexterous_Manipulation.pdf
category: Embodied_AI
---

# ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.15953)
> - **Summary**: 这篇工作把灵巧操作中的触觉从“当前附加观测”升级为“未来接触先验”，用视觉-触觉交叉注意力 + 自回归未来触觉预测来提升精细操作和长时操作稳定性。
> - **Key Performance**: 四个短时任务成功率分别为 100%、100%、90%、90%；11 阶段 Make Hamburger 的整体 HNS 为 0.88（无触觉基线为 0.61），论文报告完整任务成功率 >80%。

> [!info] **Agent Summary**
> - **task_path**: 多视角视觉 + 指尖触觉 + 本体状态 / 双臂模仿学习 -> 双臂灵巧手动作序列
> - **bottleneck**: 触觉常被当作当前时刻的附加 token，而不是接触变化的预测性表征，导致插入、旋拧、翻页等任务在关键接触瞬间不稳定
> - **mechanism_delta**: 把视觉与触觉的融合从简单拼接改为双向交叉注意力，并先预测未来触觉再条件化动作生成，同时用两阶段 curriculum 缓解早期预测触觉噪声
> - **evidence_signal**: 四个短时任务全面超越 DP/ACT/HATO/ACTw/T，且消融逐项验证 cross-attention、next-touch prediction、autoregressive feedback 的贡献
> - **reusable_ops**: [双向视觉-触觉交叉注意力, 未来触觉预测后再生成动作的两步解码]
> - **failure_modes**: [触觉依赖弱或触觉噪声较大时稳定性下降, 超出演示分布的新任务难以泛化]
> - **open_questions**: [能否减少对人工遥操作演示的依赖, 未来触觉预测能否迁移到更大规模通用机器人策略]

## Part I：问题与挑战

这篇论文针对的是**真实世界双臂多指灵巧操作**：包括 peg insertion、cap twist、vase wipe、book flip，以及 11 阶段的汉堡制作。  
输入接口是：

- 多视角视觉：顶视双目 + 左右腕部相机
- 触觉：10 个指尖触觉传感器的时序信号
- 本体状态：双臂、双手、颈部关节状态

输出接口是：

- 高频动作序列（action chunk），驱动双臂和多指手完成接触密集操作

### 真正的问题是什么？

真正的瓶颈不是“机器人没有触觉”，而是**触觉没有被建模成对控制真正有用的表示**。具体有三层：

1. **融合方式太浅**  
   许多方法只是把视觉 token 和触觉 token 直接拼接。这样模型知道“有两种模态”，但不知道**哪段视觉信息和哪段触觉变化是相关的**。

2. **只看当前，不看接下来**  
   精细 manipulation 成败往往取决于“接触将如何变化”，而不是“此刻是否接触到”。  
   - peg 接近孔时，关键是接触变化趋势  
   - twist cap 时，关键是盖子是否已经松开  
   - flip page 时，关键是页边是否被正确勾起

3. **低数据 regime 下表征容易学歪**  
   论文每个任务只用 50 条演示。对这种小样本真实机器人设定，直接端到端从高维视觉+触觉学动作，很容易学到脆弱相关性，而不是稳定的接触规律。

### 为什么现在值得做？

因为两件事同时成熟了：

- **视觉 BC/ACT/DP 这类策略框架已足够强**，瓶颈从“能不能做动作”转向“能不能稳定处理接触细节”
- **高分辨率触觉 + 遥操作采集系统已可用**，使得同步的视觉-触觉演示数据成为现实

### 边界条件

这项工作不是在“任意机器人、任意任务、任意传感器”上证明通用性，而是在一个明确边界内验证：

- 双臂 Realman + SharpaWave anthropomorphic hands
- 自建触觉指尖和自建遥操作系统
- 同一平台上的 5 类任务
- 短时任务只测试小幅位置扰动（2.5cm 级），长时任务约 3cm 级扰动

所以它更像是在回答：**在真实多指手、小样本示教、强接触依赖场景下，怎样把视觉和触觉真正融合成“可控”的策略表示？**

---

## Part II：方法与洞察

ViTacFormer 本质上是建立在 **ACT 风格 conditional VAE / action chunking** 之上的一个 visuo-tactile policy，但作者把策略链路从：

**当前观测 → 动作**

改成了：

**当前观测 → 未来触觉 → 动作**

这就是它最重要的因果改动。

### 核心直觉

- **What changed**：从“静态地融合当前视觉和当前触觉”变成“先对齐视觉-触觉，再预测未来接触，再用这个未来接触指导动作”。
- **Which bottleneck changed**：把 latent space 从“描述当前接触”转成“编码接触将如何变化”；同时用 cross-attention 代替 naive token fusion，减少跨模态信息稀释。
- **What capability changed**：模型更擅长在关键接触瞬间提前修正手部姿态与施力，因此插入、旋拧、翻页、长链路装配等任务更稳。

**为什么这个设计有效？**

1. **交叉注意力解决的是“相关性定位”问题**  
   触觉本身只告诉你“哪里有接触/力在变”，视觉本身只告诉你“物体和几何在哪里”。  
   双向 cross-attention 让两者互相查询后，得到的是“与控制有关的对应关系”，而不是简单把两个模态堆在一起。

2. **未来触觉预测解决的是“部分可观测性”问题**  
   接触密集操作中，真正关键的是下一步接触如何变。  
   一旦 latent 被迫去预测未来触觉，它就必须编码：
   - 物体相对位姿
   - 接触状态转移
   - 当前动作会导向什么接触结果  
   这比只感知当前触觉更接近控制所需的信息。

3. **两阶段 curriculum 解决的是“自回归自污染”问题**  
   早期模型预测的 future touch 很噪，若直接拿它再去生成动作，会把跨模态表示带偏。  
   所以作者先用 ground-truth future touch 训练大部分过程，再切换到 predicted touch，让模型逐步适应自回归闭环。

### 机制拆解

#### 1. 视觉-触觉双向交叉注意力

论文不是把视觉和触觉直接 concat，而是做双向 cross-attention：

- 视觉作为 K/V，触觉作为 Q
- 触觉作为 K/V，视觉作为 Q

这样做的目的不是“更复杂”，而是让模型学会：

- 当前哪块视觉区域对应当前哪段触觉变化
- 哪些触觉变化应该反过来聚焦哪些视觉细节

这个 cross-attention 被用**两次**：

- 一次用于未来触觉预测
- 一次用于动作生成

也就是说，作者认为视觉-触觉对齐不是一次性的前处理，而是整个决策过程中的持续约束。

#### 2. 未来触觉预测头

模型先基于：

- 当前视觉
- 当前触觉
- 本体状态
- style latent \(z\)

去预测未来 tactile tokens。然后再把这个预测结果拼回输入，用于生成动作。

这里的关键不是“多一个辅助任务”，而是把 future touch 变成**动作生成前的中间变量**。  
这意味着策略会更像：

> “如果我这样动，接下来触觉会怎么变；为了达到这个接触变化，我现在该怎么出动作”

这比直接从 observation 映射 action 更接近真实接触控制。

#### 3. 自回归地利用预测触觉

作者不是只预测一下 future touch 做 regularization，而是**真的把预测触觉反馈给动作解码器**。  
这个设计让动作推理不再只依赖当前观测，而是依赖“当前观测 + 预测到的近期接触先验”。

这尤其适合：

- 插入类任务：接近孔时依赖细微力变化
- 旋拧类任务：判断是否已经松脱
- 翻页类任务：判断页边是否勾住

#### 4. 两阶段 curriculum

训练日程是：

- 前 75%：动作生成使用 **ground-truth future touch**
- 后 25%：动作生成切换到 **predicted future touch**

这不是普通的 teacher forcing，而是为了避免：

- 早期 predicted touch 偏离真实分布
- cross-attention 被 noisy touch 牵着跑
- 整体 latent 表征崩掉

#### 5. 基于 ACT 的动作块生成 + 末端辅助监督

作者选择 ACT 而不是 diffusion policy，理由是当前数据量很小。  
此外，除了关节动作监督，还增加了**手臂末端位姿监督**，帮助手-臂协同更稳定。这个点不是本文主创新，但对真实机器人很重要。

### 战略性取舍

| 设计 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 双向 cross-attention | naive multimodal fusion 太粗 | 更好地对齐视觉几何与接触语义 | 模型更复杂，要求多模态同步更严格 |
| 未来触觉预测 | 当前触觉不足以支持精细控制 | 获得接触变化先验，提升 anticipatory control | 预测误差会传递到动作生成 |
| 预测触觉回注动作解码 | 动作只依赖当下观测，容易反应式而非预见式 | 更强的插入/旋拧/翻页稳定性 | 训练初期容易自回归失稳 |
| 两阶段 curriculum | 早期 predicted touch 噪声破坏表示学习 | 稳住 latent，再逐步闭环 | 多了 schedule 超参数 |
| 末端位姿辅助监督 | 仅关节监督难以稳定协调手臂与手指 | 真实平台上动作更稳 | 依赖较准确的运动学建模 |

---

## Part III：证据与局限

### 关键信号：能力跃迁到底在哪里？

#### 1. 比较信号：短时任务几乎被“解掉”

在四个短时任务上，ViTacFormer 的成功率分别为：

- Peg Insertion: **100%**
- Cap Twist: **100%**
- Vase Wipe: **90%**
- Book Flip: **90%**

而最强的 naive tactile baseline ACTw/T 只有：

- 60%
- 60%
- 40%
- 40%

**结论**：  
性能提升不是因为“加了触觉就行”，而是因为**触觉被以更合理的方式表征和利用**。  
尤其是 Book Flip 和 Cap Twist 这类任务，说明仅把 tactile 当附加 token 远远不够。

#### 2. 比较信号：长时任务出现明显能力跃迁

在 11 阶段的 Make Hamburger 任务上：

- 无触觉基线整体 HNS：**0.61**
- ViTacFormer 整体 HNS：**0.88**

论文还报告：

- 完整长链路成功率 **>80%**
- 可持续执行 **约 2.5 分钟**
- 成功完成 **11 个连续阶段**

**结论**：  
这不是只提高单步动作精度，而是降低了长时序中的误差累积。  
也就是说，future touch prior 对长链条 manipulation 不是“局部小修小补”，而是实质上改善了闭环稳定性。

#### 3. 消融信号：三项核心设计都有贡献

消融显示：

- **CrossAttention**：显著提升视觉-触觉对齐
- **NextTouchPred**：改善接触变化建模
- **AutoRegressive**：让动作利用预测到的 future touch

例如在 Book Flip 上，HNS 从 naive tactile fusion 的 **0.53**，提升到：

- w/ CrossAttention: **0.77**
- w/ NextTouchPred: **0.80**
- w/ AutoRegressive: **0.80**
- Full model: **0.93**

**结论**：  
完整增益来自“对齐 + 预测 + 回注”的链式设计，不是单一 trick。

#### 4. 失败案例与机制一一对应

作者给的失败分析有一定解释力：

- **Peg insertion 失败**：不知道孔附近接触变化何时出现  
  → 对应 future touch prediction 的必要性
- **Cap twist 失败**：不知道瓶盖是开是关  
  → 对应自回归利用预测接触的重要性
- **Book flip 失败**：不知道页边位置与接触点如何对齐  
  → 对应 cross-attention 融合的重要性

这类 failure mode 和方法组件之间有较清晰的因果对应，说明方法设计不是拍脑袋拼装。

### 局限性

- **Fails when**: 触觉在任务中不关键、或者触觉信号本身噪声较大时，方法稳定性会下降；论文也明确承认在“低触觉依赖”场景下可能受传感器噪声和表征学习限制影响。对更大幅度位姿扰动、不同物体分布、跨平台迁移的证据也不足。
- **Assumes**: 依赖任务级人工遥操作演示（每任务 50 条）、定制双臂灵巧手与指尖触觉硬件、同步多相机输入、以及人工定义阶段与权重的 HNS 评分协议；训练还使用 2×NVIDIA H20，复现门槛不低。论文中未见公开代码/权重/硬件实现细节发布。
- **Not designed for**: 零样本新任务泛化、跨机器人形态迁移、无触觉硬件平台部署、或大规模通用机器人策略/语言条件操控。

### 额外的可复用组件

即使不完全照搬整套系统，下面几个部件仍然很值得复用：

- **视觉-触觉双向 cross-attention 融合块**  
  适合任何“局部几何 + 接触反馈”联合控制任务
- **future touch prediction 头**  
  可作为接触型 manipulation policy 的中间监督
- **ground-truth future touch → predicted future touch 的 curriculum**  
  适合所有带自条件化/自回归感知中间变量的机器人策略
- **末端位姿辅助监督**  
  对真实双臂/灵巧手系统往往比纯关节监督更稳

### 总结判断

这篇论文最有价值的地方，不是简单证明“触觉有用”，而是更明确地回答了：

> **对灵巧操作真正有用的触觉，不是当前触觉本身，而是“未来接触如何变化”的预测性表征。**

相对于 prior work，它的能力跳跃主要体现在：

- 从静态融合走向预测式接触建模
- 从短时精细操作走向长时多阶段连续 manipulation
- 从“加触觉输入”走向“让触觉真正成为动作决策中的中间变量”

但从证据强度看，仍应保守视为 **moderate**：

- 结果很强
- 消融也完整
- 但评测仍主要集中在自建平台与自定义任务，开放复现性不足

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ViTacFormer_Learning_Cross_Modal_Representation_for_Visuo_Tactile_Dexterous_Manipulation.pdf]]