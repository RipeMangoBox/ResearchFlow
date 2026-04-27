---
title: "Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/vision-language-navigation
  - waypoint-prediction
  - multi-view-fusion
  - topological-map
  - dataset/R2R-CE
  - dataset/HM3D
  - dataset/Gibson
  - dataset/Matterport3D
  - opensource/no
core_operator: "用低视角大规模航点预训练和历史多视角注意力聚合，修正四足机器人在连续VLN中的视角高度错配。"
primary_logic: |
  人类高视角语言指令 + 机器人低机位RGBD观测 → 利用 HM3D/Gibson/Matterport3D 连通图扩展低视角 waypoint 训练，并对拓扑图 ghost node 的历史多视角特征做注意力加权聚合 → 输出更稳健的下一航点与连续导航轨迹
claims:
  - "在低视角评测下，直接将高视角训练的 ETPNav 测试到低视角会使 R2R-CE val unseen 的 SR 从 0.57 降到 0.21，说明视角高度变化本身会造成显著泛化断裂 [evidence: comparison]"
  - "在低视角消融中，仅重训练 waypoint predictor 就把 ETPNav 在 val unseen 的 SR 从 0.21 提升到 0.39，而仅重训练 navigator 只能提升到 0.32，表明主要瓶颈在航点预测而非策略头 [evidence: ablation]"
  - "在 Xiaomi Cyberdog 四个真实场景中，GVNav 在 kitchen 和 office area 的 SR 分别达到 0.40 和 0.36，均高于 ETPNav 的 0.28 和 0.24 [evidence: comparison]"
related_work_position:
  extends: "ETPNav (An et al. 2024)"
  competes_with: "ETPNav (An et al. 2024); BEVBert (An et al. 2023)"
  complementary_to: "ScaleVLN (Wang et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_Ground_level_Viewpoint_Vision_and_Language_Navigation_in_Continuous_Environments.pdf
category: Embodied_AI
---

# Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.19024)
> - **Summary**: 这篇论文把连续环境 VLN 的真实瓶颈定位为“人类高视角指令 vs 四足机器人低视角观测”的分布错配，并通过低视角大规模 waypoint 预训练与历史多视角加权聚合来提升模拟到真实部署的鲁棒性。
> - **Key Performance**: 低视角 R2R-CE val unseen 上达到 **SR 55% / SPL 45%**；真实厨房场景 **SR 40%**，高于 ETPNav 的 **28%**。

> [!info] **Agent Summary**
> - **task_path**: 人类自然语言指令 + 低机位RGBD全景重建 + 历史拓扑图 -> 下一航点/连续导航轨迹
> - **bottleneck**: 人类指令基于高视角全局感知，而机器人只有低机位且易被遮挡，导致 waypoint 候选与拓扑节点表征都建立在缺失信息上
> - **mechanism_delta**: 将系统从“高视角先验 + ghost node 历史均值特征”改为“低视角大规模 waypoint 预训练 + ghost node 注意力加权多视角融合”
> - **evidence_signal**: 低视角 ablation 中，仅重训 waypoint predictor 已把 val unseen SR 从 0.21 提到 0.39，说明瓶颈主要在候选航点生成
> - **reusable_ops**: [low-view waypoint pretraining, ghost-node attention fusion]
> - **failure_modes**: [地标从低机位始终不可见, 历史视角同样被遮挡时难以补救]
> - **open_questions**: [能否不用机械旋转相机也获得同等收益, RGBD收益与大规模航点数据收益应如何进一步拆分]

## Part I：问题与挑战

这篇工作的核心不是再做一个更强的 VLN policy，而是指出：**真实机器人部署时，观察视角高度变了，整个导航问题的输入分布也变了。**

### 1. 真正的难点是什么

传统 VLN，尤其是离散环境或模拟器中的设定，大多默认：
- 视角接近人类高度；
- 观察近似全景；
- 指令中的地标描述也天然服务于这种高视角感知。

但四足机器人不是这样。它的相机更低、可见范围更窄，更容易被桌腿、沙发、台面等遮挡。于是出现了三层错配：

1. **指令视角错配**  
   人类说“经过桌球台后左转，到棋盘桌前停下”，是基于较高、较全局的视野表达；机器人从地面看，可能根本看不到完整地标。

2. **感知模式错配**  
   模拟器里很多方法依赖全景观察；真实机器人常见是单目或 RGBD，FOV 有限。即使把模型搬到真实世界，输入接口也已经变了。

3. **上游候选航点错配**  
   在 VLN-CE 中，policy 不直接输出完整轨迹，而是依赖 waypoint predictor 给出候选航点。  
   一旦低视角让航点预测失真，后面的导航策略再强也只能在坏候选里选“相对不差”的动作。

### 2. 输入/输出接口

- **输入**：自然语言指令 + 低机位 RGBD 观测 + 历史拓扑图  
  在真实机上，作者通过旋转相机采集 12 张、每 30° 一帧的 RGBD 图像，拼成近似全景输入。
- **输出**：下一 waypoint / 拓扑目标节点，再转换为连续环境中的低层动作执行。

### 3. 为什么现在值得解决

VLN 在模拟环境里的成功率已经很高，社区正在向连续环境、真实机器人部署推进。  
这时，原先“高视角全景 + 离散图导航”的隐含假设开始失效。本文的价值就在于把这个失效点明确诊断出来：**不是单纯 sim-to-real 不好，而是“视角高度”本身就是一个被低估的 domain gap。**

### 4. 边界条件

这篇论文的设定边界也很明确：
- 室内、连续环境；
- 以拓扑图规划为主，而不是纯端到端反应式控制；
- 真实机器人为低机位四足平台；
- 成功判定是到目标点 3 米内；
- 更偏静态场景，不是动态人群或高速运动控制。

---

## Part II：方法与洞察

作者的方法可以理解为：**不大改导航器主体，而是优先修复“低视角下的候选航点生成”和“被遮挡后的节点表征”这两个上游瓶颈。**

### 方法总览

GVNav 基本沿用 ETPNav 的拓扑规划框架，但做了两件关键增强：

1. **把 waypoint predictor 重新放到低视角、大规模数据上训练**  
2. **把拓扑图里 ghost node 的历史特征从“简单平均”改成“注意力加权融合”**

这两个改动都直接作用于 policy 的输入质量，而不是只改策略头。

### 1. 低视角 waypoint predictor 扩容训练

作者认为，低机位下性能掉得最明显的地方是 waypoint prediction。  
所以他们没有只在原 R2R 数据上重训，而是借助 ScaleVLN 的思路，把公共 3D 扫描场景的连通图转移过来，构建更大的低视角航点训练集：

- 800 个 HM3D scans
- 491 个 Gibson scans
- 61 个 MP3D scans
- 总计 **212,924** 个 waypoint 训练样本
- 相比原有训练量扩大 **22.02×**

这里的因果逻辑很重要：  
waypoint predictor 学到的不是一句话到动作的映射，而是“从当前视觉几何与语义分布中，哪些方向/距离是可达且合理的”。  
如果训练分布一直是人类高度，那到低机位时，候选点分布会整体偏掉。

### 2. 多视角信息收集：给 ghost node 更好的记忆

在 ETPNav 风格的拓扑图里，节点可分为：
- 已访问节点
- 当前节点
- ghost node（尚未确认、但已被局部观察到的候选位置）

问题在于：低机位下，当前位置看到的某个 ghost node 可能被遮挡；但轨迹上更早的位置其实见过它的更清晰视角。  
如果仍然对历史视图做简单平均，清晰信息会被遮挡视图“稀释”。

GVNav 的做法是：
- 对多个视角下关联到同一 ghost node 的特征做 transformer 编码；
- 用学习到的权重，而不是均值，生成 ghost node 表征；
- 让系统优先保留“当前决策最有用”的历史视图。

换句话说，它把 ghost node 表征从：
- **“谁看过都算一点”**
变成：
- **“谁看得最清楚、最相关，就权重大”**

这正是论文所谓的 multi-view information gathering。

### 3. 系统层面的现实适配

为了让真实四足机器人也能喂给 VLN 模型近似全景输入，作者在 Xiaomi Cyberdog 上加了：
- Intel RealSense D455
- 360° 可编程旋转电机
- 每步采集 12 张 RGBD 图

所以这篇工作不是纯算法论文，也带有明显的系统适配意味：  
它承认现实机器人没有天然全景相机，于是用机械旋转去补。

### 核心直觉

**改了什么？**  
从“高视角训练的 waypoint + 当前局部视图主导的节点表征”，改成“低视角适配的 waypoint 先验 + 历史多视角选择性记忆”。

**改变了哪个瓶颈？**  
- 改变了**候选航点分布**：从人类视角偏置，转为更贴近低机位可见性/可达性；
- 改变了**信息瓶颈**：从单帧遮挡主导，转为可利用历史无遮挡视图；
- 改变了**拓扑图中的节点表示噪声**：ghost node 不再被平均池化稀释。

**能力为什么会变强？**  
因为在 VLN-CE 里，policy 的能力上限受 waypoint 候选质量强烈约束。  
如果上游候选点就错了，policy 不可能凭空“想出”正确路径。  
而多视角加权记忆，则是在低视角遮挡严重时，为 planner 提供一种“补全局部视野”的近似机制。

可概括成一句因果链：

> **低视角预训练** 改善候选点质量  
> + **历史多视角加权** 改善 ghost node 表征  
> → **拓扑规划看到的动作空间更可靠**  
> → **连续导航与真实部署更稳**

### 策略权衡

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价/折中 |
|---|---|---|---|
| 低视角大规模 waypoint 预训练 | 高低视角分布错配 | 候选航点更可执行、泛化更强 | 需要额外渲染数据与连通图资源 |
| ghost node 注意力加权融合 | 遮挡导致的局部观测缺失 | 可利用历史无遮挡视图，减少平均池化的信息稀释 | 增加记忆与计算，历史噪声也可能被放大 |
| 机械旋转相机构造全景 | 单目/FOV 不足 | 与现有全景 VLN 框架兼容 | 决策频率下降，硬件复杂度上升 |
| 复用 ETPNav 拓扑规划 | 不重写整个导航栈 | 工程上可直接继承已有 planner | 整体能力仍受 ETPNav 框架上限约束 |

---

## Part III：证据与局限

### 关键实验信号

#### 1. 比较信号：视角高度变化本身就是大坑
作者先做了最关键的一组对照：  
**高视角训练，低视角测试。**

结果显示，ETPNav 在 R2R-CE val unseen 上的 SR 从 **0.57** 直接掉到 **0.21**。  
这说明问题不是“模型略微不适配”，而是**输入分布已经发生结构性变化**。

#### 2. 消融信号：瓶颈主要在 waypoint predictor，不在 navigator
Table II 是这篇论文最有分析价值的部分。  
在低视角下：
- 仅重训 navigator：val unseen SR **0.21 → 0.32**
- 仅重训 waypoint predictor：val unseen SR **0.21 → 0.39**

这非常清楚地支持了论文主张：  
**低视角 VLN 的主瓶颈是候选航点生成，而不是后端策略本身。**

#### 3. 方法收益信号：GVNav 在模拟和真实机上都比 ETPNav 更稳
在低视角 R2R-CE 上，GVNav 相比低视角重训的 ETPNav：
- val unseen **NE 5.15 → 4.89**
- val unseen **SR 0.52 → 0.55**
- val unseen **SPL 0.43 → 0.45**

增幅不算颠覆性，但方向一致，说明多视角历史聚合提供了额外增益。

在真实 Xiaomi Cyberdog 上，GVNav 也优于 ETPNav：
- **Kitchen**: SR **0.40 vs 0.28**
- **Office Area**: SR **0.36 vs 0.24**

这说明收益不是只停留在模拟器里。

#### 4. 数据扩容信号：放大 waypoint 数据确实有效
作者还单独验证了 waypoint predictor 的训练数据扩容。  
在低视角设置下，扩容后的 predictor 能显著提高 open-space 预测比例，说明它更容易提出可走、不过分贴障碍物的航点。  
这与全文的因果链一致：**先把候选点变对，后面的导航才有可能变好。**

### 局限性

- **Fails when**: 指令依赖的关键地标从低机位始终不可见，或者所有历史视角也都被遮挡时，注意力聚合无信息可选；另外在特别拥挤、狭窄且纹理相似的室内区域，低视角地标辨识仍可能失效。
- **Assumes**: 依赖 RGBD 传感器、机械旋转形成近全景、拓扑图定位/回溯机制，以及来自 HM3D/Gibson/Matterport3D 的额外连通图与大规模低视角训练数据；真实部署还依赖笔记本 GPU（文中为 RTX 3080 Mobile 16GB）与额外硬件改装。
- **Not designed for**: 动态人群环境、室外复杂地形、高速连续控制、纯单目无全景重建的原生机器人设定，也不是端到端学习低层 locomotion 的方案。

一个额外的现实判断是：  
虽然作者在真实环境里取得了提升，但绝对成功率仍不高，很多场景 SR 还在 **0.28–0.40** 区间。  
这说明它更像是**让低机位 VLN 从“明显不可用”走向“初步可部署”**，而不是已经解决了真实机器人导航。

### 可复用组件

1. **低视角 waypoint 数据生成流程**  
   可直接迁移到其他连续环境 embodied task，用于训练更稳的候选点生成器。

2. **ghost node 的注意力式历史特征融合**  
   这是一个通用算子，适合所有“部分可见目标节点 + 历史多视角记忆”的拓扑规划系统。

3. **低 FOV 机器人上的全景重建接口**  
   虽然工程上笨重，但能让现有全景型 VLN 模型较低成本地迁移到真实机平台。

### 总结一句

这篇论文最有价值的地方，不只是提出 GVNav，而是**把低机位真实机器人上的 VLN 瓶颈重新定位为“视角高度引起的候选航点与节点表征失真”**；它的改动也因此非常聚焦，且与实验结论基本对齐。

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_Ground_level_Viewpoint_Vision_and_Language_Navigation_in_Continuous_Environments.pdf]]