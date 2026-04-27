---
title: "OPG-Policy: Occluded Push-Grasp Policy Learning with Amodal Segmentation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/goal-oriented-grasping
  - task/object-retrieval
  - reinforcement-learning
  - deep-q-learning
  - amodal-segmentation
  - dataset/YCB
  - opensource/no
core_operator: 用amodal分割补全被遮挡目标形状，再以该补全掩码条件化Q学习与动作协调，实现更高效的推抓切换。
primary_logic: |
  RGB-D场景观测 + 目标部分可见信息 → UOAIS预测目标amodal mask并构建RGB/Depth/Mask heightmap，双Q网络评估各位置/角度的push与grasp价值，协调器融合Q值、遮挡结构和失败历史选择动作类型 → 以更少动作取出被遮挡目标
claims:
  - "在30 objects (hard)模拟场景中，OPG-Policy取得68%成功率和3.69次平均尝试，优于GE-GRASP的57%/3.98和GTI的33%/4.26 [evidence: comparison]"
  - "在无需真实世界微调的条件下，OPG-Policy在真实机器人 Challenging Test 和 Generalization Test 上分别达到85%与90%成功率，均高于GTI与GE-GRASP [evidence: comparison]"
  - "移除协调器后，平均成功率从82.66%降至77.00%，说明仅靠最高Q值选动作不足以稳定协调push与grasp [evidence: ablation]"
related_work_position:
  extends: "GTI (Yang et al. 2020)"
  competes_with: "GTI (Yang et al. 2020); GE-GRASP (Liu et al. 2022)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_OPG_Policy_Occluded_Push_Grasp_Policy_Learning_with_Amodal_Segmentation.pdf
category: Embodied_AI
---

# OPG-Policy: Occluded Push-Grasp Policy Learning with Amodal Segmentation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.04089)
> - **Summary**: 该文把目标的 amodal 补全掩码直接纳入 push-grasp 策略学习，让机器人在只看到目标一部分时也能更稳定地决定“先推还是直接抓”，从而减少无效动作。
> - **Key Performance**: 模拟平均 82.66% 成功率 / 2.61 次尝试；真实机器人 Generalization Test 达到 90.0% 成功率 / 6.65 次尝试

> [!info] **Agent Summary**
> - **task_path**: RGB-D桌面场景 + 被遮挡目标 → push/grasp动作序列 → 目标取出
> - **bottleneck**: 仅依赖 visible mask 时，目标隐藏几何与受阻关系不可见，导致 push/grasp 价值估计偏差和动作切换低效
> - **mechanism_delta**: 用 amodal mask 补足目标完整占据，再结合遮挡结构特征与Q值由协调器决定推或抓
> - **evidence_signal**: 重遮挡模拟与零微调真实机器人测试均优于 GTI / GE-GRASP，且去掉协调器后性能明显下降
> - **reusable_ops**: [amodal-mask-conditioned Q-map prediction, occlusion-aware push-grasp coordinator]
> - **failure_modes**: [amodal mask 预测错误会误导Q值估计, 极端拥挤时可能在有限步数内无法清出目标]
> - **open_questions**: [如何降低对目标族专门amodal训练的依赖, 如何扩展到更丰富动作原语与主动观察]

## Part I：问题与挑战

这篇工作解决的是**密集堆叠场景中的目标导向抓取**。机器人知道要取哪个目标，但该目标往往只露出一部分，还可能被上方和周围物体同时遮挡。系统输入是固定 RGB-D 相机观测与目标相关分割信息，输出是一串离散的 push / grasp 原语，直到目标被取出。

真正的瓶颈不是“找一个抓点”本身，而是**部分可观测导致的状态别名**：

- 同样的可见区域，可能对应完全不同的隐藏轮廓；
- 同样看起来“能抓”的位置，实际上可能被上方物体压住；
- 同样一次 push，可能是在清边界，也可能只是在无效扰动场景。

所以先前基于 visible segment 学的策略，容易出现两类问题：
1. **grasp Q 高估**：看到一点目标就以为能抓；
2. **push Q 低估或错估**：看不到完整边界，不知道该推哪一侧最能释放目标。

为什么现在值得做：因为 amodal segmentation 已经能提供“目标完整轮廓”的视觉假设，它正好补上了目标导向抓取里最关键但长期缺失的隐藏状态信息。

**边界条件**也很明确：论文聚焦桌面刚体场景、俯视 RGB-D、正交投影 heightmap、16 个离散旋转角、以及仅含 push / grasp 两类 primitive；并不讨论 6-DoF 抓取、开放词汇目标检索或复杂工具操作。

## Part II：方法与洞察

### 方法结构

OPG-Policy 的核心不是单纯“把 amodal 分割加进来”，而是把它变成**策略状态的一部分**，再围绕它重做 critic、奖励和动作协调。

1. **Amodal segmentation 模块**
   - 采用 UOAIS 预测目标的 amodal mask。
   - 标注来自仿真中的“场景图 + 单物体对照图”配对，再借助 SAM 生成可见/完整 mask 并匹配，得到 amodal 标注。
   - 这一步的作用是：把“目标真实占据范围”的估计显式化。

2. **统一状态表示**
   - 将 RGB、深度、amodal mask 都投影成 top-down heightmap。
   - 对 heightmap 进行 16 个角度旋转，编码动作方向。
   - 然后用 DenseNet 编码，再分别送入 push Q net 与 grasp Q net。

3. **双 Q 图评估动作**
   - push 和 grasp 各输出一张像素级 Q map。
   - 每个像素/角度位置对应一个动作原语的未来回报估计。
   - 这让系统能在统一表示下比较“哪里推最好”“哪里抓最好”。

4. **协调器决定动作类型**
   - 不直接用“哪个 Q 值大就选哪个”。
   - 而是把最佳 push/grasp Q 值，加上目标遮挡率、边界拥挤度、抓取失败历史等特征，输入一个 MLP 二分类器，判断当前更该 push 还是 grasp。

5. **训练上的关键改动**
   - grasp 奖励不仅奖励抓出目标，也奖励抓点落在目标 amodal 区域内；
   - push 奖励不仅看遮挡是否减少，还看它是否提升了“下一步最好 grasp”的价值；
   - 训练采用由易到难的课程设计，早期先用 ε-greedy，后期再启用协调器。

### 核心直觉

OPG-Policy 真正拧动的因果旋钮有三个：

1. **从 visible-only 到 amodal-conditioned**
   - 改变了什么：策略看到的不再只是目标露出来的部分，而是“可见部分 + 被遮挡部分的估计”。
   - 改变了哪个瓶颈：减少了部分可观测带来的状态别名。
   - 带来什么能力：在重遮挡时，Q 网络更容易判断目标边界、受阻方向和潜在抓取区域。

2. **从“推开一点东西”到“让下一步更容易抓”**
   - 改变了什么：push 的奖励不只对局部位移给分，而是跟“未来 grasp 变得更可行”对齐。
   - 改变了哪个瓶颈：缓解了 push 动作的长期信用分配问题。
   - 带来什么能力：push 不再只是搅动场景，而是更像有目的地创造抓取机会。

3. **从硬比较 Q 值到显式协调 push/grasp**
   - 改变了什么：加入了遮挡率、边界拥挤度、失败历史等上下文。
   - 改变了哪个瓶颈：减少了 push 和 grasp 切换时的决策抖动。
   - 带来什么能力：避免“明明该先推却反复抓”或“已经可抓却还在继续推”。

换句话说，这篇论文不是单纯提高感知精度，而是把**隐藏几何先验**注入到**动作价值估计**和**动作类型切换**两个关键环节里，所以收益主要体现在**高遮挡下的动作效率**，而不只是低难度场景的成功率。

### 策略权衡

| 设计选择 | 缓解的约束 | 能力收益 | 代价 / 风险 |
|---|---|---|---|
| amodal mask 作为状态输入 | 目标隐藏部分不可见 | 更准确估计目标边界与遮挡关系 | 依赖 amodal 分割泛化，错误 mask 会误导策略 |
| push / grasp 双 Q map | 位置与方向联合搜索难 | 可直接比较不同像素/角度的动作价值 | 动作空间仍局限于离散俯视 primitive |
| 协调器融合 Q 值 + 遮挡特征 | push/grasp 切换不稳定 | 减少重复抓取，提高动作效率 | 需要额外监督与手工设计特征 |
| 自适应 push 奖励 | push 长期回报难评估 | 学会“为下一抓创造条件”的 push | 依赖 grasp Q 的校准稳定性 |
| 课程式训练 | 高难 clutter 直接训练不稳 | 更平滑学到推抓协同 | 训练流程更复杂，更依赖场景设计 |

## Part III：证据与局限

### 关键证据信号

- **比较信号｜困难模拟场景**
  - 在 30 objects (hard) 上，OPG-Policy 达到 **68% 成功率 / 3.69 次尝试**，优于 GE-GRASP 的 **57% / 3.98** 和 GTI 的 **33% / 4.26**。
  - 这说明能力提升主要出现在“目标真的被重遮挡”时，而不是简单场景刷分。

- **遮挡分层信号｜按 occlusion ratio 分组**
  - 在 0.6-0.8 的重遮挡组，作者报告其相对次优方法**至少提高 1% 成功率、减少至少 0.61 次总尝试**。
  - 这支持论文的核心论点：amodal 信息主要改善的是**动作效率**，特别是“先推哪里更值”的判断。

- **消融信号｜协调器不可省**
  - 去掉协调器后，平均成功率从 **82.66%** 降到 **77.00%**，平均尝试数也变差。
  - 说明 amodal mask 本身不够，**push/grasp 何时切换**也是独立瓶颈。

- **反例信号｜amodal 不是可即插即用插件**
  - GTI-amodal 反而比 GTI 更差。
  - 这很关键：仅把 amodal mask 塞进旧框架并不会自动变强，必须连同奖励设计和动作协调一起重构。

- **真实部署信号｜零微调 sim-to-real**
  - 不做真实世界微调，真实机器人 Challenging Test / Generalization Test 分别达到 **85% / 90%** 成功率。
  - 说明方法不只是仿真内有效，至少在其设定的对象族与桌面场景中有可迁移性。

### 局限性

- **Fails when**: amodal 分割对未见形状、透明/反光物体或极端遮挡给出错误完整轮廓时，后续 Q 估计会被系统性误导；如果场景需要多步复杂重排而不仅是局部 push/grasp，在有限动作预算内仍可能失败。
- **Assumes**: 目标对象属于已训练的 amodal 分割对象族或其近邻分布；依赖固定俯视 RGB-D、正交 heightmap、离散 16 方向动作；协调器依赖遮挡率、边界拥挤度、抓取失败计数等人工先验；训练与标注流程还依赖 UOAIS、SAM 和仿真对照生成管线，且论文未声明开源实现。
- **Not designed for**: 开放集目标检索、6-DoF 抓取、可变形/透明物体、移动视角主动感知、以及超出 push/grasp primitive 的通用操作规划。

### 可复用组件

- **amodal-mask-conditioned state**：把目标完整占据估计并入策略状态，适合其他遮挡操控任务。
- **future-graspability push reward**：用“是否让下一步更易抓”来定义 push 的价值，适合推抓协同或主动清障。
- **occlusion-aware coordinator**：把 Q 值与遮挡几何特征联合起来做动作类型切换，适合多 primitive 操作系统。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_OPG_Policy_Occluded_Push_Grasp_Policy_Learning_with_Amodal_Segmentation.pdf]]