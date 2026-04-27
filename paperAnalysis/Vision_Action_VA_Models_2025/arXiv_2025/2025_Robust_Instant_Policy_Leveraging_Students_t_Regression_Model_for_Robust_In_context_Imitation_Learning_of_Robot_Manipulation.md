---
title: "Robust Instant Policy: Leveraging Student’s t-Regression Model for Robust In-context Imitation Learning of Robot Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/in-context-imitation-learning
  - student-t-regression
  - trajectory-aggregation
  - keypoint-tokenization
  - dataset/ManiSkill3
  - repr/3D-keypoints
  - opensource/no
core_operator: 对同一KAT上下文多次调用LLM采样候选操控轨迹，再以Student’s t回归做重尾鲁棒共识，滤除hallucination离群轨迹。
primary_logic: |
  少量人类示范 + 新场景RGB-D观测 → 将3D视觉关键点与末端执行器轨迹文本化后送入LLM并多次采样候选轨迹 → 用Student’s t回归对时间对齐后的轨迹做重尾鲁棒拟合、忽略幻觉离群点 → 输出可执行的稳健机器人轨迹
claims:
  - "在10个模拟与真实机器人任务、每任务仅10个示范的设定下，RIP平均成功率达到80%±17，高于KAT的54%±24和Diffusion Policy的50%±24 [evidence: comparison]"
  - "将Student’s t聚合替换为Gaussian聚合会使平均成功率从80%降至64%，说明重尾回归对抑制幻觉离群轨迹是关键 [evidence: ablation]"
  - "在依赖夹爪开合的任务中，基于夹爪动作保留关键时刻的下采样将RIP成功率从52%提升到74% [evidence: ablation]"
related_work_position:
  extends: "Keypoint Action Tokens (Di Palo and Johns 2024)"
  competes_with: "Keypoint Action Tokens (Di Palo and Johns 2024); Diffusion Policy (Chi et al. 2023)"
  complementary_to: "Uncertainty Quantification for In-Context Learning of LLMs (Ling et al. 2024); SelfCheckGPT (Manakul et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Robust_Instant_Policy_Leveraging_Students_t_Regression_Model_for_Robust_In_context_Imitation_Learning_of_Robot_Manipulation.pdf
category: Embodied_AI
---

# Robust Instant Policy: Leveraging Student’s t-Regression Model for Robust In-context Imitation Learning of Robot Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.15157), [Project](https://sites.google.com/view/robustinstantpolicy)
> - **Summary**: 这篇工作把 LLM 机器人“即时策略”从单次直接输出，改成多次采样后用 Student’s t 重尾回归做鲁棒聚合，从而在少样本模仿操控中显著抑制连续轨迹 hallucination。
> - **Key Performance**: 10个任务、每任务10条示范时平均成功率 **80%±17**；相对 KAT **+26 个百分点**，相对 Gaussian 聚合版 **+16 个百分点**

> [!info] **Agent Summary**
> - **task_path**: 少量人类示范 + 新场景RGB-D观测 -> 机器人末端执行器轨迹与夹爪动作
> - **bottleneck**: LLM 即时策略在连续动作空间会偶发生成偏离示范分布的离群轨迹，微小偏差就足以导致操控失败
> - **mechanism_delta**: 将单次 LLM 轨迹输出改为同上下文多次采样，并用按时间步的 Student’s t 回归提取重尾鲁棒共识轨迹
> - **evidence_signal**: 跨 10 个模拟/真实任务的比较 + Student’s t 对 Gaussian 的消融，直接证明改进来自“离群轨迹抑制”而非仅仅多采样
> - **reusable_ops**: [多次采样-鲁棒聚合, 事件感知下采样]
> - **failure_modes**: [多模态专家策略会被平均成不自然轨迹, 需要高频在线闭环控制时延迟与复杂度过高]
> - **open_questions**: [如何显式识别连续轨迹 hallucination 的来源, 如何在多最优解场景下同时保持鲁棒性与多样性]

## Part I：问题与挑战

这篇论文要解决的，不是“机器人能不能通过少量示范学会新任务”，而是更现实的一步：**在完全不微调大模型的前提下，机器人能不能稳定地执行由 LLM 即时生成的操控轨迹**。

### 真问题是什么
KAT 已经证明：把视觉和动作变成 keypoint 文本后，off-the-shelf LLM 可以像“即时策略”一样，看到几条演示就为新任务生成轨迹。  
但在机器人场景里，真正的瓶颈不是语义理解不够，而是：

- LLM 会**偶发 hallucination**
- 这种 hallucination 在文本里可能只是“答错一句话”，但在操控里会变成**离群轨迹**
- 连续控制里几厘米的偏差，就可能导致：
  - 抓不到物体
  - 碰不到把手
  - 接触方向错误
  - 错过开合夹爪时机

所以，这篇工作的核心不是重新训练更强 policy，而是给已有的 LLM instant policy 加一个**连续轨迹层面的鲁棒化器**。

### 输入 / 输出接口
论文的 ICIL 设定很明确：

- **输入**
  - 少量人类示范
  - 每条示范包含：
    - 起始 RGB-D 观测
    - 一段末端执行器动作轨迹
  - 测试时的新场景观测
- **输出**
  - 一整段机器人执行轨迹
  - 轨迹由末端执行器 3D 姿态点 + 夹爪开合状态组成

### 边界条件
这篇方法成立有几个重要前提：

- 使用 **KAT 风格** 的表示：视觉变成 3D keypoints，动作变成文本 token
- 更像**开放环轨迹生成**，不是高频闭环控制
- 任务多为**少样本日常操控**
- 演示者被默认近似遵循**单一主导策略**
- 真实部署时控制频率被明显降采样（仿真 5.5Hz，真实 1Hz）以保证稳定性

### 为什么现在值得做
因为 KAT 这类工作已经把“少量演示 + 零微调 + 直接调用 LLM”变成现实，但它距离可用还差最后一公里：**可靠性**。  
RIP 的价值就在于，它不试图替换 KAT，而是直接补上“即时策略能用但不稳”的短板。

---

## Part II：方法与洞察

这篇论文的设计哲学很清楚：

> **不要把 LLM 的一次输出当成最终答案，而要把它看成一组带噪提案；真正的策略是从这些提案里提取鲁棒共识。**

### 方法流程

#### 1. 先沿用 KAT 的表示方式
作者没有重做 perception-policy pipeline，而是直接建立在 KAT 上：

- 用 DINO 从 RGB-D 图像提取语义/几何一致的关键点
- 将视觉观测表示为 **3D visual keypoints**
- 将动作表示为末端执行器的 **3 个 3D 点 + gripper 状态**
- 然后把“示范上下文 + 新观测”一起文本化，送入 GPT-4o

这一步的意义是：**继续保留 off-the-shelf LLM 可直接充当策略的能力**。

#### 2. 不信任单次输出，而是多次采样
对同一个上下文，RIP 会向 LLM 查询 \(Q\) 次，得到多条候选轨迹。

直觉上：

- 如果模型真的“懂这个任务”，多次采样会围绕某个稳定解聚集
- 如果某次输出是 hallucination，它会偏离主簇，表现成**离群轨迹**

#### 3. 对不同长度轨迹做时间对齐
由于不同采样的轨迹长度可能不同，作者先把轨迹按 episode 起点/终点归一化到共同时间轴上。

这一步不是方法亮点，但它很关键：  
否则后面的统计聚合会把“时间错位”误认为“动作离群”。

#### 4. 用 Student’s t 回归替代 Gaussian 平均
这是 RIP 的核心。

作者不是直接对多条轨迹做简单平均，也不是用 Gaussian 回归，而是拟合一个 **Student’s t-regression** 模型来估计每个时间步上的鲁棒动作分布，然后取其均值轨迹作为最终输出。

这里的关键差异是：

- **Gaussian**：远离均值的离群点会显著拉动结果
- **Student’s t**：重尾分布对离群点更不敏感，更适合“多数样本合理、少数样本灾难性偏离”的场景

#### 5. 额外工程改动：保留夹爪事件的下采样
因为 LLM 上下文长度有限，演示轨迹需要下采样。  
作者发现均匀下采样会把“夹爪开/合”的关键时刻删掉，导致抓取类任务失败。

因此他们加入了一个简单但有效的规则：

- 起点、终点保留
- gripper 状态变化时刻保留
- 其余部分再均匀采样

这实际上是在保护**接触事件**信息。

### 核心直觉

#### what changed
从“**单次 LLM 生成 = 最终策略**”改成“**LLM 多次提案 + 重尾鲁棒共识**”。

#### which bottleneck changed
原来系统的瓶颈是：  
LLM 输出被当成**单点预测**，任何一次 hallucination 都会直接进入执行层。

RIP 把这个问题改写成：  
从一组带噪、含离群点的连续轨迹样本里，估计**主导行为模式**。

#### what capability changed
这样做带来的能力变化是：

- 不需要重新训练 LLM
- 不需要更多演示
- 只在输出端做统计鲁棒化
- 就能把“偶发严重偏航”的即时策略，变成“低数据下更可靠”的操控策略

#### why this works
因果逻辑很简单：

1. 如果 LLM 理解任务，重复采样得到的轨迹会在同一解附近聚集  
2. hallucination 轨迹会偏离这个主簇  
3. Student’s t 比 Gaussian 更能压低离群点影响  
4. 所以最终共识轨迹会更接近“稳定可执行”的那一簇，而不是被坏样本拖走

### 战略取舍

| 设计选择 | 解决的瓶颈 | 获得的能力 | 代价 / 风险 |
|---|---|---|---|
| KAT 式 keypoint 文本化 | 原始图像与连续动作不适合直接喂给 LLM | 让现成 LLM 能零微调做 instant policy | 会损失部分稠密视觉细节 |
| 同上下文多次采样 | 单次输出不稳定、无法估计不确定性 | 暴露样本间一致性与离群性 | 增加 API 调用与延迟 |
| Student’s t 回归聚合 | Gaussian/均值容易被坏样本拖偏 | 对连续轨迹 hallucination 更鲁棒 | 默认存在单一主导轨迹模式 |
| 夹爪事件感知下采样 | 均匀下采样会丢掉抓取关键帧 | 提高 grasp/release 任务可靠性 | 含启发式规则，可能依任务变化 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 跨 10 个任务的主比较信号：RIP 把“能用”变成“更稳地能用”
在 5 个仿真任务 + 5 个真实任务、每任务只给 10 条示范时：

- **RIP**: 80% ± 17
- **KAT**: 54% ± 24
- **KAT-DP**: 52% ± 24
- **DP**: 50% ± 24
- **RIPGauss**: 64% ± 17

最关键的不是绝对数值，而是这组对比说明：

- 少样本下，LLM instant policy 本身就有优势
- 但真正把性能拉开的，是**鲁棒聚合**
- 所以 RIP 的提升不是“又换了个 backbone”，而是精准命中了 hallucination 这个瓶颈

#### 2. 最强机制证据：Student’s t 明显优于 Gaussian
作者专门做了 RIPGauss 消融。  
如果只是“多采样然后平均”就够，那 Gaussian 版应该接近 RIP。事实不是这样：

- RIP: 80%
- RIPGauss: 64%

这说明贡献点不只是 sample more，而是 **heavy-tail robust aggregation**。

#### 3. 定性案例直接展示了“坏样本如何毁掉轨迹”
在 push-cube 任务中：

- KAT 的多条候选轨迹里存在明显离群样本
- Gaussian 聚合仍被这个离群样本拖偏
- 论文报告最终一步上，RIPGauss 的 X 方向比 RIP 少了 **14 cm**
- 结果是前者没碰到物体，后者成功

这个例子非常能说明问题：  
机器人失败往往不是因为“整体不会”，而是因为“被一次离群输出拖离接触几何”。

#### 4. 设计消融支持作者的因果叙事
- **Q 太小不行**：Q=2 时和 KAT 接近，说明可靠样本数不够
- **Q=5 足够**：继续增大查询数提升有限，说明主导模式已可被稳定估计
- **ν=1.5 最优**：比 Gaussian 极限更好，支持“需要更强离群抑制”
- **夹爪事件感知下采样**：抓取类任务从 52% 提升到 74%，说明接触事件的时序信息非常关键

### 局限性

- **Fails when**: 演示本身存在多个差异很大的等价最优轨迹时，RIP 可能把多种正确解平均成一条不自然甚至不可执行的轨迹；若 LLM 大多数采样都错，鲁棒聚合也无从恢复；需要高频在线闭环反应的任务中，方法难以满足时延要求。
- **Assumes**: 依赖 KAT 风格的 keypoint/action 文本化、DINO 特征提取、闭源 GPT-4o 多次并行查询，以及“演示主要围绕单一主导策略”这一假设；真实机器人测试还依赖 ALOHA/GELLO 式示教硬件与人工示范采集；执行频率被显著降低以保证稳定。
- **Not designed for**: 显式识别 hallucination 原因、在线重规划、从原始视觉端到端学习闭环控制、以及多模态专家策略建模。

### 复现与资源依赖
这篇工作虽然方法思路清晰，但工程复现并不轻：

- 用的是 **GPT-4o**，闭源 API
- 需要 **多次查询**，成本与延迟都会上升
- 需要 **DINO + 3D keypoint** 提取链路
- 真实实验依赖专门机器人平台
- 论文给了项目页视频，但**未见代码链接**，因此可复现性不算强

### 可复用组件
这篇论文最值得迁移的，不只是 RIP 本身，而是下面几个操作模板：

1. **多次采样 + 鲁棒共识**：可作为任何生成式机器人策略的后处理安全层  
2. **重尾分布替代高斯平均**：适合“少数灾难性离群样本”的连续控制输出  
3. **事件感知下采样**：对抓取、释放、接触切换等稀疏关键动作特别有效  

### 一句话结论
RIP 带来的能力跃迁，不是让机器人“更会理解任务”，而是让它在**零微调、少示范**条件下，**更少因为一次 LLM hallucination 而失败**。  
最有力的证据，就是跨模拟与真实任务的整体提升，以及对 Gaussian 聚合的直接消融。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Robust_Instant_Policy_Leveraging_Students_t_Regression_Model_for_Robust_In_context_Imitation_Learning_of_Robot_Manipulation.pdf]]