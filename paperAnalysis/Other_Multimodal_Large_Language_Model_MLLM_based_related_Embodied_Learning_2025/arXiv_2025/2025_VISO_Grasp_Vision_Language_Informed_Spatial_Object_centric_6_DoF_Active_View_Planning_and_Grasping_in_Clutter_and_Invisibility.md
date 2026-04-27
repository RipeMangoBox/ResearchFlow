---
title: "VISO-Grasp: Vision-Language Informed Spatial Object-centric 6-DoF Active View Planning and Grasping in Clutter and Invisibility"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-grasping
  - task/active-perception
  - object-centric-representation
  - velocity-field-planning
  - bayesian-fusion
  - dataset/8-Real-World-Scenes
  - opensource/partial
core_operator: 以VLM维护对象级3D空间关系，并将其转化为连续速度场视角规划与多视角不确定性抓取融合。
primary_logic: |
  目标物文本提示 + 眼在手RGB-D历史观测 → VLM/开放词汇检测构建并更新对象级3D关系，按遮挡关系决定直接抓取或序列去遮挡，并用连续速度场规划目标导向NBV，再对多视角6-DoF抓取做贝叶斯不确定性融合 → 在重遮挡甚至完全不可见场景中完成目标抓取
claims:
  - "Claim 1: 在8个真实重遮挡场景上，VISO-Grasp 的平均最终成功率达到 87.5%，高于 Breyer’s 的 12.5%、Top-down 的 52.5%、Init view 的 60.0% 和 w/o GF 的 70.0% [evidence: comparison]"
  - "Claim 2: VISO-Grasp 以 3.10 的平均抓取尝试次数取得所有方法中最低的尝试成本，同时保持最高的平均整体抓取成功率 83.86% [evidence: comparison]"
  - "Claim 3: 去掉实时不确定性抓取融合后，平均最终成功率从 87.5% 降到 70.0%，说明多视角贝叶斯融合对稳定执行有实质贡献 [evidence: ablation]"
related_work_position:
  extends: "Closed-loop next-best-view planning for target-driven grasping (Breyer et al. 2022)"
  competes_with: "Closed-loop next-best-view planning for target-driven grasping (Breyer et al. 2022); Affordance-driven next-best-view planning for robotic grasping (Zhang et al. 2023)"
  complementary_to: "FoundationGrasp (Tang et al. 2025); GraspNeRF (Dai et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_VISO_Grasp_Vision_Language_Informed_Spatial_Object_centric_6_DoF_Active_View_Planning_and_Grasping_in_Clutter_and_Invisibility.pdf"
category: Embodied_AI
---

# VISO-Grasp: Vision-Language Informed Spatial Object-centric 6-DoF Active View Planning and Grasping in Clutter and Invisibility

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.12609), [Code](https://github.com/YitianShi/vMF-Contact)
> - **Summary**: 在严重遮挡乃至目标完全不可见时，VISO-Grasp 用 VLM 驱动的对象级空间推理、目标导向主动视角规划和多视角不确定性抓取融合，把“先看清、再必要时去遮挡、最后稳定抓取”做成一个闭环系统。
> - **Key Performance**: 平均最终成功率 **87.5%**；平均抓取尝试次数 **3.10**（8 个真实场景中最低）

> [!info] **Agent Summary**
> - **task_path**: 目标物文本提示 + 眼在手 RGB-D 重遮挡场景 -> 目标/遮挡物的 6-DoF 主动视角规划与抓取执行
> - **bottleneck**: 目标不可见时，系统既缺少“该往哪看”的目标级可见性信号，也缺少“该先移除谁”的遮挡因果信息；即使看到目标，单视角 grasp 仍然高方差
> - **mechanism_delta**: 将目标抓取重写为“对象级关系记忆 -> 连续 NBV 视角优化/序列去遮挡 -> 多视角不确定性抓取融合”的闭环
> - **evidence_signal**: 8 个真实场景中，完整系统 AFSR 为 87.5%，显著高于 w/o GF 的 70.0% 和固定视角基线
> - **reusable_ops**: [object-memory update, target-guided velocity field, vMF Bayesian grasp fusion]
> - **failure_modes**: [wrong occluder reasoning, dynamic-scene memory drift]
> - **open_questions**: [how to reduce VLM latency, how to handle dynamic clutter]

## Part I：问题与挑战

这篇论文要解决的不是普通 6-DoF grasp detection，而是 **target-oriented grasping under heavy occlusion**：给定一个目标物体描述，机器人需要在未知、杂乱、可能堆叠的场景中主动移动 wrist camera，必要时先移除遮挡物，最终抓到指定目标。

### 1）输入/输出接口

- **输入**：目标物文本提示 + 眼在手 RGB-D 连续观测。
- **输出**：一个闭环行为序列，而不只是单个 grasp pose：
  1. 识别当前能看到的对象；
  2. 判断目标是否可直接抓；
  3. 若不可抓，则决定该先看哪里或先移除谁；
  4. 最终执行目标或遮挡物的 6-DoF 抓取。

### 2）真正的瓶颈是什么

真正难点不是“给点云预测一个 grasp”这么简单，而是三个耦合瓶颈：

1. **可见性瓶颈**  
   目标可能被完全遮住。此时单视角抓取器根本没有目标几何，连“哪里能抓”都谈不上。

2. **视角选择瓶颈**  
   以往 NBV 方法常把问题定义成重建/渲染/信息增益优化，但“更容易重建”不等于“更容易抓到指定目标”。  
   对 target-oriented grasping 来说，关键不是看清整个场景，而是 **看清目标相关的那部分几何**。

3. **执行稳定性瓶颈**  
   即使已经从更好的视角看到目标，重遮挡场景里的单帧 grasp hypothesis 仍然可能受局部缺失、碰撞风险和视角偏差影响而波动很大。

### 3）现有方法为什么不够

- **静态视角/预定义多视角**：默认目标至少部分可见，不适合 complete invisibility。
- **重建导向的主动视角方法**：往往 target-agnostic，且依赖离散候选视角或预定义搜索空间。
- **VLM/MLLM 抓取工作**：多停留在 2D 语义或平面 4-DoF 推理，缺少可靠的 3D 空间关系建模与实时抓取闭环。

### 4）为什么现在值得做

因为两个能力现在终于可以接起来了：

- **Foundation Models / VLM** 已能提供零样本对象识别与关系推理；
- **不确定性感知的 6-DoF grasp model** 已能提供带置信度的抓取候选。

这篇论文的价值就在于：把“语义上知道谁挡住了谁”和“几何上知道相机该往哪走、抓取该何时执行”合并成一个闭环。

### 5）边界条件

论文的设定比较明确：

- 眼在手 RGB-D 相机；
- 平行夹爪；
- 场景近似静态；
- 机械臂有一定视角机动空间；
- 任务是 **目标导向抓取**，不是开放式长程操作规划。

---

## Part II：方法与洞察

### 方法骨架

VISO-Grasp 由三部分组成：

1. **AMOV3D**：多视角开放词汇 3D 对象检测与更新  
2. **TGV-Planner**：目标导向视角规划  
3. **RT-UMGF**：实时不确定性多视角抓取融合

它的核心不是某一个更强的 grasp network，而是把 **对象级关系推理、视角规划、抓取执行** 串起来。

### 1）AMOV3D：先把“场景”变成可推理的对象记忆

这部分做的事情是：

- 用 **Qwen2.5-VL** 从图像中输出对象的结构化描述，如：
  - 标签
  - 颜色
  - 纹理/图案
  - 空间关系
- 再用 **Grounding DINO + SAM2** 把这些描述落到 2D box 和 mask；
- 通过深度回投与 PCA 生成 3D oriented bounding box；
- 将不同视角下看到的对象做 **merge/add** 更新，维护一个跨时间的对象列表。

这个设计的关键点不在“检测更准”，而在于它建立了一个 **evolving object-centric scene memory**。  
这样一来，系统不再依赖单帧是否恰好看见目标，而是用历史视角不断修正“场景里有什么、谁挡住了谁”。

如果当前看不到目标，AMOV3D 还会触发：

- **VICL**：基于历史+当前信息做视觉上下文推断；
- **MoRE 风格提示**：从空间关系、材料属性、几何约束等多个 reasoning 视角投票，推断谁最可能是 occluder。

于是，系统即便“没看到目标”，也能继续推断下一步应先移除哪个物体。

### 2）TGV-Planner：把遮挡关系变成相机运动

TGV-Planner 先用规则化空间关系做高层决策：

- **Proximity**
- **Below**
- **High / Low**

据此判断：

- 目标是否能直接抓；
- 是否要先移除上层/近邻遮挡物；
- 还是先调整相机视角。

真正有意思的是它的 **Velocity-field-based NBV**。  
它不采用离散枚举相机候选位，而是直接根据“目标中心 - 遮挡物中心”的相对几何，生成一个连续速度场：

- 单遮挡物：相机沿着更可能暴露目标的一侧切向移动；
- 多遮挡物：把多个遮挡物的影响叠加；
- 当速度趋近停滞点时，意味着已经到达较优观察位。

这相当于把“往哪边绕过去才能看到目标”写成了一个连续控制律，而不是搜索问题。

### 3）RT-UMGF：不是看到一次就抓，而是把多次看到的 grasp 证据融合起来

抓取部分建立在 **vMF-Contact** 上，但作者把它改造成在线 10Hz 推理，并做了实时融合。

它的机制是：

- 每个时刻产生一批 contact grasps，包含：
  - 接触点
  - 抓取方向均值
  - 方向精度/不确定性
  - approach category
  - 抓取宽度
  - grasp quality
- 新 grasps 与历史 buffer 比较后分成两类：
  - **new**：此前没见过的 grasp cluster
  - **proximal**：与历史已有 grasp 接近，可做 cross-fusion
- 然后对位置、方向和类别分布做在线更新：
  - 位置用质量加权
  - 方向用 vMF/Bayesian 更新
  - approach bin 用累加更新

最终只有当：

- grasp quality 足够高；
- 方向精度足够高；
- 或视角规划已到停滞点且 grasp 足够可信，

系统才真正执行。

### 核心直觉

这篇论文最重要的变化，可以概括为：

**把“目标抓取”从一次性几何估计，改成“对象关系维护 -> 目标可见性优化 -> 抓取后验累积”的闭环。**

#### 变化 1：从单帧感知变成对象级关系记忆

- **what changed**：不再把每一帧独立当作一次检测，而是维护跨视角对象状态。
- **which bottleneck changed**：目标完全不可见时，像素层没有直接答案，但对象关系层仍然保留了可操作信息。
- **what capability changed**：系统能在“没看到目标”的时候继续做合理动作，比如先去除哪个遮挡物。

#### 变化 2：从重建导向 NBV 变成目标导向可见性优化

- **what changed**：优化目标从“让场景更完整”变成“让目标更可见、更可抓”。
- **which bottleneck changed**：离散视角搜索和 target-agnostic 目标函数导致大量无效观察。
- **what capability changed**：相机移动更像人类“绕着挡住目标的物体找角度”，更快暴露关键几何。

#### 变化 3：从瞬时抓取得分变成多视角抓取后验

- **what changed**：把多视角 grasp prediction 视为证据流而不是一次性打分。
- **which bottleneck changed**：单视角局部点云造成 grasp 质量和方向估计高方差。
- **what capability changed**：抓取更稳定，平均尝试次数更少。

### 为什么这个设计有效

因为它把三个原本分离的判断统一到了同一个对象中心表示里：

- **VLM** 回答“谁挡住了谁、先移谁”；
- **Velocity field** 回答“相机该往哪里动”；
- **Bayesian fusion** 回答“当前 grasp 证据是否已经够强”。

也就是说，它分别消除了：

- **信息瓶颈**：目标不可见
- **搜索瓶颈**：不知道往哪看
- **执行瓶颈**：看到以后仍抓不稳

### 战略 trade-off

| 设计选择 | 带来的能力 | 代价 / 风险 |
| --- | --- | --- |
| 多视角 open-vocabulary 对象记忆 | 零样本识别目标与 occluder，并持续修正空间关系 | 强依赖 VLM、检测与分割质量，错误会级联传导 |
| 连续 velocity-field NBV | 不必离散枚举视角，更平滑、更 target-centric | 依赖 3D box 近似质量和相机可达空间 |
| 规则化空间关系决策 | 可解释、易调试，适合快速闭环 | 阈值与几何规则可能对新场景敏感 |
| 在线 Bayesian grasp fusion | 把多视角 noisy grasp 变成更稳的执行信号 | 需要缓存历史、阈值设计，默认场景不能快速变化 |

---

## Part III：证据与局限

### 关键实验信号

#### 1）比较信号：完整闭环明显优于固定视角和传统 NBV

在 8 个真实重遮挡场景上的平均结果：

- **Ours**: AFSR **87.5%**, #AGA **3.10**, AGSR **83.86%**
- **w/o GF**: AFSR 70.0%
- **Init view**: AFSR 60.0%
- **Top-down**: AFSR 52.5%
- **Breyer’s**: AFSR 12.5%

这说明提升不只是来自“换了一个更强 grasp network”，而是来自整个闭环：
**对象关系推理 + 目标导向 NBV + 在线抓取融合** 缺一不可。

#### 2）消融信号：NBV 让目标更可见，fusion 让执行更可信

去掉 RT-UMGF 后，平均最终成功率从 **87.5%** 降到 **70.0%**。  
这表明只做到“把目标看出来”还不够，重遮挡场景里仍需要把多个视角下的 grasp 证据积累起来，才能把可见性转化成稳定抓取。

#### 3）案例信号：主动多视角不仅补几何，也补语义

作者特别提到 Scene (3) 的 Pringles 场景：  
Top-down 视角下，VLM 会把目标误识别成 “Red can”；而 VISO-Grasp 借助多视角对象更新最终做到 **5/5** 成功。  
这说明主动视角搜索并不只是补点云，还改善了高层语义推断。

### 1-2 个关键指标

- **Average Final Success Rate**: **87.5%**
- **Average Grasp Attempts**: **3.10**

### 这篇论文回答了哪三个核心问题

1. **What / Why**  
   真正瓶颈是目标不可见时缺少对象级遮挡因果信息，导致系统既不会“看”，也不会“先移谁”。现在做这件事有意义，是因为 VLM 终于能提供零样本对象关系推理，而 grasp model 又能提供不确定性感知。

2. **How**  
   作者引入的关键因果旋钮是：  
   **跨视角对象关系记忆 + 目标导向连续速度场 + 多视角贝叶斯抓取融合**。  
   这改变了两件事：  
   - 视角搜索分布从 target-agnostic 变成 target-conditioned  
   - 执行依据从单次 noisy score 变成时序后验

3. **So what**  
   能力跃迁体现在：系统第一次把“完全不可见目标”的抓取拆成可执行闭环，在 8 个实机场景里同时拿到更高成功率和更少尝试次数。最强证据是整体对比和 w/o GF 消融。

### 局限性

- **Fails when**: 场景在视角搜索和抓取融合期间快速变化；目标或遮挡物被 VLM、Grounding DINO、SAM2 错误识别；机械臂无法到达更优视角时，velocity-field 规划收益会明显下降。
- **Assumes**: 眼在手 RGB-D、平行夹爪、准静态 clutter；依赖 Qwen2.5-VL-72B、Grounding DINO、SAM2 以及基于纯仿真预训练的 vMF-Contact；空间关系规则和执行阈值需要人工设定。
- **Not designed for**: 高动态场景、强外界扰动下的高速 reactive manipulation、超出平行夹爪接触模型的复杂操作；论文也没有系统验证跨机器人/跨传感器迁移。

### 资源与复现依赖

- 作者明确承认 **VLM 推理延迟** 是现实瓶颈，尤其 VICL 的多模态上下文推理成本不低。
- 证据主要来自 **8 个自建真实场景**，不是大规模标准 benchmark，因此泛化强度仍应保守看待。
- 文中给出代码仓库，但仓库名沿用 `vMF-Contact`，从论文文本难以确认整套集成系统是否已完整开源，因此更适合视为 **partial**。

### 可复用组件

- **对象级多视角记忆**：VLM 描述 + 开放词汇检测 + mask + 深度回投 -> 可持续更新的 3D object memory。
- **目标导向 velocity-field NBV**：把“谁挡住了谁”直接变成连续相机控制信号。
- **vMF/Bayesian grasp fusion**：适合接入其他可输出 grasp uncertainty 的多视角抓取器。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_VISO_Grasp_Vision_Language_Informed_Spatial_Object_centric_6_DoF_Active_View_Planning_and_Grasping_in_Clutter_and_Invisibility.pdf]]