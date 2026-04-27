---
title: "UniGoal: Towards Universal Zero-shot Goal-oriented Navigation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/goal-oriented-navigation
  - task/embodied-navigation
  - scene-graph
  - graph-matching
  - multi-stage-exploration
  - dataset/Matterport3D
  - dataset/HM3D
  - dataset/RoboTHOR
  - opensource/no
core_operator: "将目标与在线场景统一编码为图，并用图匹配分数驱动零匹配、部分匹配、完美匹配三阶段探索与验证。"
primary_logic: |
  目标类别/实例图像/文本描述 + 机器人RGB-D观测与位姿
  → 构建目标图与在线场景图，计算节点/边/拓扑匹配并据此切换探索、锚点对齐、图校正与目标验证
  → 输出长期导航目标与逐步动作，最终在目标附近停止
claims:
  - "UniGoal 作为单一训练免费模型，在 HM3D 实例图像目标导航上达到 60.2 SR / 23.7 SPL，超过零样本专用方法 Mod-IIN 的 56.1 / 23.3 [evidence: comparison]"
  - "UniGoal 以同一框架覆盖 object-goal、instance-image-goal 和 text-goal 三类任务，并在 HM3D 文本目标导航上以 20.2 SR / 11.4 SPL 超过监督通用方法 GOAT 的 17.0 / 8.8 [evidence: comparison]"
  - "去掉 blacklist 后，HM3D IIN 从 60.2/23.7 降到 50.6/17.3；去掉阶段 2 后降到 59.0/23.2，说明匹配状态驱动的探索切换和失败记忆对性能关键 [evidence: ablation]"
related_work_position:
  extends: "SG-Nav (Yin et al. 2024)"
  competes_with: "SG-Nav (Yin et al. 2024); GOAT (Chang et al. 2023)"
  complementary_to: "ConceptGraphs (Gu et al. 2023); OVSG (Chang et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_UniGoal_Towards_Universal_Zero_shot_Goal_oriented_Navigation.pdf
category: Embodied_AI
---

# UniGoal: Towards Universal Zero-shot Goal-oriented Navigation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.10630), [Project](https://bagh2178.github.io/UniGoal/)
> - **Summary**: 这篇工作把类别目标、实例图像目标和文本目标统一成“目标图”，再与在线维护的场景图做显式匹配，并用匹配状态驱动三阶段探索，从而实现单模型、免训练的通用目标导航。
> - **Key Performance**: HM3D IIN 上 60.2 SR / 23.7 SPL；HM3D TextNav 上 20.2 SR / 11.4 SPL

> [!info] **Agent Summary**
> - **task_path**: goal(category/image/text) + posed RGB-D stream -> long-horizon navigation actions -> stop near referred object
> - **bottleneck**: 不同目标类型缺少统一表示，导致零样本导航推理链条任务专用化，难以复用场景-目标关系做稳定定位
> - **mechanism_delta**: 把 scene 和 goal 都变成图，并把“匹配程度”变成策略切换信号，而不是为每类 goal 单独写一套 LLM 推理流程
> - **evidence_signal**: 单一训练免费模型同时在 ObjNav/IIN/TextNav 三类任务上取得竞争性或最优结果，且 blacklist 与 stage-2 对齐的消融明显掉点
> - **reusable_ops**: [online-scene-graph, match-state-conditioned-exploration]
> - **failure_modes**: [goal-graph parsing error from text/image, ambiguous repeated objects causing wrong anchor alignment]
> - **open_questions**: [how robust is graph matching in dynamic scenes, can stage switching be learned/adaptive instead of thresholded]

## Part I：问题与挑战

这篇论文研究的是 **universal zero-shot goal-oriented navigation**：  
在未知室内环境里，智能体接收三种任一形式的目标——

- **Object-goal**：一个物体类别；
- **Instance-image-goal**：目标实例图像；
- **Text-goal**：关于目标及其上下文关系的文字描述；

然后基于 **posed RGB-D** 连续决策，最终在目标附近停止。

### 真正的问题是什么

作者要解决的，不只是“怎么导航”，而是：

1. **目标接口高度异构**  
   object、image、text 三类 goal 的信息形态完全不同。现有零样本方法通常针对单任务定制：
   - ObjNav：更像 open-vocabulary object search；
   - IIN：更像实例匹配；
   - TextNav：更像语言 grounding + reasoning。  
   结果就是每换一种 goal，整条推理 pipeline 都要重写。

2. **零样本方法虽强，但不通用**  
   LLM-based zero-shot navigation 已经能在单一任务上工作，但大多是 task-specific workflow，无法自然迁移到更广泛 goal 类型。

3. **训练式通用方法不够“零样本”**  
   像 GOAT、PSL 通过共享 embedding 或统一策略做到通用，但仍依赖大规模 RL / 训练，且容易受模拟器分布约束。

### 真正瓶颈

**真正瓶颈不是动作控制，而是“场景表示”和“目标表示”不一致，导致无法做统一、显式、可迁移的推理。**

以前的方法常把 scene/goal 都转成文本喂给 LLM，但文本化会丢失结构，尤其是：
- 目标内部关系；
- 目标与已观察场景的局部重叠；
- “我已经看到一部分目标上下文，还差哪一块”的推理信号。

### 输入/输出与边界条件

- **输入**：目标（类别 / 图像 / 文本）、RGB-D 视频流、位姿
- **输出**：动作序列 `{move_forward, turn_left, turn_right, stop}`
- **成功条件**：在步数上限内停在中心目标附近
- **场景边界**：室内未知环境，依赖 occupancy map 与在线 scene graph
- **零样本含义**：不训练/微调导航策略，但**并非无先验**，仍依赖预训练 LLM/VLM/CLIP 等基础模型

### 为什么是现在

因为现在 LLM/VLM 已经足够强，**缺的不是“推理能力”，而是“统一的结构化接口”**。  
UniGoal 的判断是：如果能把 scene 和 goal 都压缩成同一种图结构，那么不同任务之间共享的就不是 policy weight，而是 **推理语义空间**。

---

## Part II：方法与洞察

UniGoal 的核心策略不是学习一个共享 embedding policy，而是：

> **统一 goal/scene 表示 → 显式图匹配 → 根据匹配状态切换探索策略**

整体分成四个关键部件。

### 1. 统一图表示：把不同 goal 变成同一种“目标图”

作者定义图 `G=(V,E)`：

- **节点**：物体
- **边**：空间或语义关系
- **节点/边内容**：文本描述

不同 goal 的处理方式：

- **Object-goal**：退化成单节点图
- **Instance-image-goal**：从目标图像中识别物体，再用 VLM 推断物体关系，构图
- **Text-goal**：让 LLM 从文字描述中抽取物体与关系，构图

与此同时，agent 在探索过程中持续维护一个 **在线 scene graph**。

这一步的关键价值是：  
不是把所有目标都硬塞进同一个向量，而是把它们都转成 **同一结构语言**。

### 2. 图匹配：用匹配程度决定“现在该怎么搜”

UniGoal 不只做“找到/没找到”的二分类，而是对 goal graph 和 scene graph 做三种相似度衡量：

- **节点匹配**
- **边匹配**
- **拓扑匹配**

然后汇总成一个 **matching score**。  
这使系统能区分三种状态：

1. **几乎没看到目标相关信息**
2. **看到了一部分，可以用重叠结构推断剩余位置**
3. **基本看到目标，可以进入接近与验证**

也就是说，匹配分数在这里不是评估指标，而是 **控制变量**。

### 3. 三阶段探索：从盲搜到定位再到验证

#### Stage 1：Zero Matching

当匹配分数很低时，说明 scene 中几乎还没有 goal 的有效重叠。  
这时如果直接让 LLM 对整个复杂目标图做一次性推理，容易模糊。

所以作者先做一件很实用的事：

- 把目标图分解成若干 **内部相关、彼此较弱相关** 的子图；
- 再把每个子图转成可用于 frontier scoring 的描述；
- 最终选择更有希望的 frontier 做探索。

直观上，这是把“找一整个复杂关系簇”变成“先找更紧密的一小块”。

#### Stage 2：Partial Matching

当 scene graph 和 goal graph 已有部分重叠，且存在 anchor pair 时，UniGoal 开始利用结构重叠来**推断目标在哪**。

难点在于：
- scene graph 有世界坐标；
- goal graph 没有绝对坐标，只有相对关系。

作者的做法是：

1. 让 LLM 根据“左边/前方/右侧”等关系，把 goal graph 投影到一个 BEV 坐标系；
2. 利用 scene 与 goal 中的 anchor pair 做对齐；
3. 把剩余 goal 节点投影到当前场景坐标里；
4. 生成一个长期探索目标点。

这一步非常关键，因为它把“部分看见”从模糊线索，变成了**几何搜索假设**。

#### Stage 3：Perfect Matching

当匹配分数足够高且中心目标已匹配，系统不再继续大范围探索，而是进入：

- **scene graph correction**
- **goal verification**

原因很直接：  
即使匹配到了，也可能是 scene graph 构图错误、感知误检或重复物体带来的假阳性。

因此作者在目标附近构建局部子图，用：
- 新观测图像；
- 邻接节点/边关系；
- LLM/VLM 的局部更新能力；

来修正局部 scene graph，并结合：
- 图匹配得分；
- IIN 中的关键点匹配；
- 接近过程中的路径代价；

共同判定目标是否真的可信。

### 4. Blacklist：让失败匹配变成“记忆”

UniGoal 还有一个很有效的小设计：**blacklist**。

如果某次 anchor 对齐失败，或某个“看起来像目标”的候选在验证阶段失败，那么相关节点/边会被加入 blacklist，后续不再参与匹配。  
这样做改变的是搜索分布：系统不会一遍又一遍回到同一个假目标上。

如果后续 graph correction 修正了相关节点，这些黑名单项还可以被移除，避免彻底误杀。

### 核心直觉

UniGoal 的核心不是“让 LLM 更聪明”，而是**让 LLM 面对的问题更结构化**。

#### 因果链条

- **什么变了**：从“按任务定制的文本推理链”改成“统一图表示 + 图匹配驱动控制”
- **哪个瓶颈变了**：从“异构目标格式导致的隐式推理负担”变成“显式结构重叠下的阶段化决策”
- **能力发生了什么变化**：同一零样本框架可以覆盖 object / image / text 三类 goal，且对含关系上下文的目标更有优势

更具体地说：

1. **统一图表示**  
   降低了不同 goal interface 的分布差异。

2. **显式匹配分数**  
   把“是否该继续搜、是否该定位、是否该停”从 prompt 工程问题变成可控状态机。

3. **部分匹配阶段的几何对齐**  
   让系统能利用“只看见一部分”这种中间态，而不是只能在“没看到/看到了”两端切换。

4. **verification + blacklist**  
   让错误匹配不会无限重复放大。

### 策略取舍

| 设计选择 | 主要收益 | 代价/风险 | 最适合的场景 |
|---|---|---|---|
| 统一 goal/scene graph | 一套推理框架覆盖三种目标 | 上游构图错误会级联传播 | 多 goal 类型共存 |
| 匹配分数驱动阶段切换 | 搜索、定位、验证职责分明 | 依赖阈值与匹配质量 | 中长程导航 |
| Anchor pair 对齐 | 把部分重叠变成空间假设 | 对重复布局、错锚点敏感 | IIN / TextNav |
| Graph correction + verification | 降低假阳性 stop | 在线推理开销更高 | cluttered / ambiguous scene |
| Blacklist | 避免反复搜错目标 | 早期误拉黑可能漏掉正确区域 | 多个相似候选存在时 |

---

## Part III：证据与局限

### 关键证据

**信号 1：跨任务比较，证明“统一框架”不是只在单任务成立。**  
UniGoal 用单一训练免费模型同时覆盖 ObjNav、IIN、TextNav。  
其中最强信号来自 richer goal 场景：

- **HM3D IIN**：60.2 SR / 23.7 SPL，超过 Mod-IIN 的 56.1 / 23.3  
- **HM3D TextNav**：20.2 SR / 11.4 SPL，超过 GOAT 的 17.0 / 8.8

这说明它的优势主要来自**目标内部关系信息**被更好利用，而不只是“又换了个 backbone”。

**信号 2：在 ON 上只小幅提升，反而支持了作者的机制解释。**  
ON 的 goal graph 退化成单节点，stage-1 的图分解和 stage-2 的 anchor 对齐基本发挥不了作用。  
即便如此，UniGoal 仍略高于 SG-Nav（如 HM3D 54.5 vs 54.0），说明 stage-3 的 correction / verification 确实能带来稳健性，但也说明该方法最值钱的地方不在纯 object-goal。

**信号 3：消融表明收益来自“匹配驱动控制”，而非单一表示改动。**  
代表性 IIN 消融中：

- 去掉 **blacklist**：60.2/23.7 → 50.6/17.3
- 去掉 **stage 2**：60.2/23.7 → 59.0/23.2
- 去掉 **goal verification**：60.2/23.7 → 58.2/22.4

结论很清楚：  
性能提升不只是因为“用了 graph”，更因为作者把 graph matching 变成了 **策略切换信号 + 失败记忆机制**。

### 1-2 个最值得记住的指标

- **HM3D IIN**：60.2 SR / 23.7 SPL
- **HM3D TextNav**：20.2 SR / 11.4 SPL

### 局限性

- **Fails when**: 目标图从图像/文本解析错误、scene graph 漏掉关键物体关系、或环境里有大量相似重复物体时，anchor pair 可能对错，导致 stage-2 定位偏移；对 ON 这类单节点目标，方法优势会明显缩小。
- **Assumes**: 假设有 posed RGB-D、occupancy map、可在线维护 scene graph，并依赖 LLaMA-2-7B、LLaVA-v1.6-Mistral-7B、CLIP、Grounded-SAM、LightGlue 等预训练模块；虽然免训练，但在线推理延迟和系统集成复杂度并不低。
- **Not designed for**: 动态目标、室外开放世界、强实时低算力机器人平台、以及导航之后还需操作/交互的复合 embodied task。

### 可复用组件

1. **统一 goal-as-graph 接口**  
   很适合把不同 instruction / query 形式统一到同一导航或 embodied reasoning 系统里。

2. **matching-score 作为控制变量**  
   可迁移到其他“搜索 → 定位 → 验证”的 embodied pipeline。

3. **anchor-pair + BEV 对齐**  
   适合处理“只看见部分上下文”的目标定位问题。

4. **blacklist 失败记忆**  
   对所有容易陷入重复假匹配的零样本探索系统都很有参考价值。

### 总结一句

UniGoal 的贡献不在于又提出一个导航 policy，而在于把 **通用零样本导航** 重新表述为一个 **统一图表示上的显式匹配与阶段化控制问题**；它最有说服力的地方，是同一套推理骨架确实跨过了 object / image / text 三种 goal interface。

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_UniGoal_Towards_Universal_Zero_shot_Goal_oriented_Navigation.pdf]]