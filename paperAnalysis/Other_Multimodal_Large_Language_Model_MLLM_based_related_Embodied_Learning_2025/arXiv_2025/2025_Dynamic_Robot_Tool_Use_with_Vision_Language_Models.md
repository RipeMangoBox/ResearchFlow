---
title: "Physics-Conditioned Grasping for Stable Tool Use"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-tool-use
  - task/robot-manipulation
  - wrench-aware-scoring
  - rigid-body-mechanics
  - surrogate-network
  - dataset/IsaacSim
  - opensource/no
core_operator: 在VLM先确定工具、目标与接触参数后，用由刚体力学推导并由SDG-Net近似的轨迹条件扭矩/打滑/对齐代价来选择最稳定抓取。
primary_logic: |
  语言指令+场景观测 → VLM两级grounding得到工具/目标/接触点、作用法向与短时交互轨迹 → 对候选抓取预测任务诱导的扭矩放大、切向滑移与法向失配代价 → 选择最小扳手传播风险的抓取并执行稳定工具使用
claims:
  - "在固定工具、接触点和轨迹的条件下，SDG-Net相对几何抓取基线将峰值诱导腕部扭矩最多降低17.6%，并同步降低滑移与轴向偏移 [evidence: comparison]"
  - "在UR5e真实机器人四项任务上，iTuP总体成功率达到77.5%，比CoPa和去除SDG-Net的版本各高17.5个百分点 [evidence: comparison]"
  - "当保持语义grounding与轨迹规划不变、仅移除物理条件抓取评分时，扭矩驱动的旋转/滑移失败重新出现，说明收益主要来自抓取稳定性建模而非感知改进 [evidence: ablation]"
related_work_position:
  extends: "CoPa (Huang et al. 2024)"
  competes_with: "CoPa (Huang et al. 2024); GraspNet (Fang et al. 2020)"
  complementary_to: "VoxPoser (Huang et al. 2023); Physically Grounded Vision-Language Models for Robotic Manipulation (Gao et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Dynamic_Robot_Tool_Use_with_Vision_Language_Models.pdf
category: Embodied_AI
---

# Physics-Conditioned Grasping for Stable Tool Use

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv 2505.01399](https://arxiv.org/abs/2505.01399)
> - **Summary**: 这篇工作把“工具使用中的抓取”从静态几何问题改成“给定交互轨迹后最小化扳手传播风险”的动力学问题，在VLM负责找工具/接触点之后，再用物理条件抓取评分保证抓取在冲击、长杠杆和多接触场景下不打滑、不扭转。
> - **Key Performance**: 仿真中峰值腕部扭矩最多下降 **17.6%**；真实机器人四任务总体成功率 **77.5%**，相对 CoPa / w.o. SDG-Net 均提升 **17.5 个百分点**

> [!info] **Agent Summary**
> - **task_path**: 自然语言指令 + 场景观测 -> VLM定位工具/目标/接触参数 -> 候选抓取物理评分 -> 选择稳定抓取并执行 hammer/sweep/knock/reach
> - **bottleneck**: 真正瓶颈不是“看不懂该用什么工具”，而是“选出来的抓取无法承受任务轨迹诱导的扭矩放大与切向载荷”
> - **mechanism_delta**: 将抓取选择从几何/准静态评分改为基于短时交互轨迹预测 interaction wrench 的逆向规划
> - **evidence_signal**: 在固定工具、接触点和轨迹时，仅替换抓取评分就能降低峰值扭矩，并把样本从约 6.9 Nm 的失稳阈值区移开
> - **reusable_ops**: [两级语义grounding, 轨迹条件抓取代价代理网络]
> - **failure_modes**: [接触点或法向grounding错误会误导扳手估计, 柔顺接触或长时序相互作用超出刚体短时模型]
> - **open_questions**: [能否联合优化抓取与轨迹而非固定轨迹选抓取, 如何在未知摩擦/惯量与柔顺性下做鲁棒扳手估计]

## Part I：问题与挑战

这篇论文抓住的不是“VLM 会不会选错锤子”，而是一个更机械、更底层的失败源：**工具使用往往在语义上是对的，但在力学上是错的**。

### 1. 真问题是什么？
现有 VLM/LLM 机器人系统已经能较好地完成：
- 选对工具；
- 找到目标物体；
- 推断接触区域和作用方向。

但它们通常把抓取选择交给：
- 几何稳定性分数，
- force-closure，
- 或面向拾取任务训练的通用抓取网络。

这在“拿起来”时可能足够，但在“用起来”时不够。因为工具使用不是普通抓取，而是**远端接触产生的扳手传递问题**：  
作用力发生在工具头部，稳定性却取决于力如何通过抓取传到夹爪/腕部。只要杠杆臂变长，或夹爪法向和交互法向不对齐，即使工具选对、轨迹也对，抓取仍会在冲击或跟随阶段发生**旋转、滑移、失轴**。

### 2. 输入/输出接口
论文的系统接口很清楚：

- **输入**：场景观测 \(q\)、候选物体集合、自然语言指令 \(\phi\)
- **输出**：
  1. 工具抓取位姿 \(g\)
  2. 短时交互轨迹 \(\xi(t)\)

关键点在于：**抓取质量不能脱离轨迹单独评价**。  
同一个抓取，对 lifting 稳定，不代表对 hammer / reach / sweep 稳定。

### 3. 为什么现在值得解决？
因为今天的 VLM 已经把“语义 grounding”这层做得足够强，系统瓶颈自然暴露到了下一层：**机械可执行性**。  
作者的核心判断是：

> 机器人工具使用的主要失效源，已从“看不懂工具”转向“抓不住会受力的工具”。

### 4. 边界条件
这篇工作不是通用动力学控制，也不是长时序 manipulation 全栈方案。它主要针对：

- **短时交互轨迹**；
- **刚体工具与接触**；
- **已知或可由 VLM grounding 得到的工具/目标/接触参数**；
- **抓取选择问题**，而非完整策略学习。

所以它更像是在 VLM manipulation pipeline 中，补上一层此前缺失的 **wrench-aware grasp selection**。

---

## Part II：方法与洞察

作者提出 **inverse Tool-use Planning (iTuP)**。  
“inverse”的意思很重要：不是先固定抓取再想怎么运动，而是**先给定任务交互及其将产生的受力，再反过来选能扛住这些力的抓取**。

### 方法主线

整个 pipeline 可以概括成 6 步：

1. **Object-level grounding**  
   用 VLM 从自然语言和视觉观测中找出工具和目标物体。

2. **Part-level grounding**  
   进一步找接触点、交互方向、法向等参数 \(\Omega=\{c_{tool}, c_{obj}, n, d\}\)。

3. **Trajectory synthesis**  
   根据这些参数生成短时交互轨迹。

4. **Grasp sampling**  
   产生一批候选抓取。

5. **Wrench-conditioned scoring**  
   对每个抓取，不再只看几何，而是看它在该轨迹下会承受多大的：
   - 扭矩放大，
   - 切向滑移风险，
   - 法向失配。

6. **Execution**  
   选代价最低的抓取执行。

### 关键机制

作者把抓取代价拆成三部分：

- **Torque penalty**：惩罚会把远端作用力放大成腕部不稳定扭矩的抓取；
- **Slip penalty**：惩罚切向力超过摩擦容限的抓取；
- **Alignment penalty**：惩罚夹爪法向与交互法向不一致的抓取。

其物理直觉非常直接：  
当工具头部施力时，腕部不稳定性近似由 **\(\tau \approx r \times F\)** 主导。  
所以要稳定，有三种朴素但强因果的办法：

- 缩短杠杆臂 \(r\)；
- 让力方向与夹爪更对齐；
- 减小切向投影，避免滑移。

### 为什么还需要 SDG-Net？
论文没有停在解析代价上，而是训练了一个 **Stable Dynamic Grasp Network (SDG-Net)** 来近似这些结构化物理代价。

原因是现实中很难精确知道：
- 冲击瞬间的真实接触脉冲，
- 精确惯量与摩擦参数，
- 以及柔顺性带来的偏差。

因此作者采用“**解析结构 + 学习近似**”的折中：
- 物理模型定义“该惩罚什么”；
- 神经网络学习“在部分可观测和参数不确定下，怎样快速预测这个惩罚”。

这让系统能在大量候选抓取上做实时评分，而不是在线精确求解动力学。

### 核心直觉

原来系统把抓取看成“独立于动作的静态几何问题”；  
这篇论文把抓取改成“**依赖未来交互轨迹的扳手传递问题**”。

这个改动改变了什么？

- **改变的对象**：抓取排序标准  
  从“能不能拿起来”变成“能不能承受即将发生的 interaction wrench”。

- **改变的瓶颈**：信息瓶颈  
  旧方法只看到局部几何；新方法把**未来受力、杠杆臂和法向关系**也纳入评分。

- **带来的能力变化**：  
  系统会主动偏好：
  - 更靠近手柄的抓取，
  - 更短有效杠杆臂的抓取，
  - 与交互法向更一致的夹爪姿态，
  从而在 hammer / knock / reach 这类高扭矩场景里显著减少旋转和滑移。

这不是“多加一个网络所以更强”，而是一个明确的因果旋钮：

**抓取评分依据改变  
→ 抓取分布从几何稳定转向扳手稳定  
→ 高扭矩样本被移出失稳阈值区  
→ 工具使用成功率提升。**

### 战略取舍

| 设计选择 | 得到的能力 | 代价/牺牲 |
|---|---|---|
| 语义与物理严格解耦 | 能清楚证明收益来自抓取稳定性而非 VLM 变强 | grounding 错误仍会向后传播 |
| 解析物理代价 + 学习代理 | 既有物理可解释性，又能实时打分 | 仍依赖近似惯量/摩擦先验 |
| 固定短时轨迹后再选抓取 | 容易插入现有 VLM pipeline，便于模块化验证 | 不能保证抓取-轨迹联合全局最优 |
| 关注 torque/slip/alignment 三类主因 | 对冲击、长杠杆、多接触任务有效 | 对柔顺接触、复杂接触历史建模不足 |

---

## Part III：证据与局限

### 关键证据 1：控制变量仿真证明“抓取评分层”本身有效
最有说服力的实验不是全流程成功率，而是作者做了**严格隔离**：

- 固定工具身份；
- 固定接触点；
- 固定交互方向；
- 固定轨迹；

只让**抓取位姿**变化。

在这个设定下，SDG-Net 仍然相对 GQ-CNN / GraspNet 持续降低：
- 峰值腕部扭矩，
- 滑移量，
- 轴向偏差。

这说明改进不是因为“VLM 选得更聪明”，而是因为**抓取真的更能承受任务诱导扳手**。

### 关键证据 2：扭矩—失败链条被直接测到了
论文不是只报 success rate，而是给出一条因果链：

- 更高的峰值腕部扭矩  
  → 更高的滑移；
- 失败样本聚集在高扭矩区域；
- 约在 **6.9 Nm** 附近出现明显失稳阈值。

这类证据很强，因为它支持的不是“方法 A 比方法 B 好一点”，而是论文最核心的理论命题：

> 工具使用失败的主导因素之一，是任务诱导扳手跨过了抓取的摩擦/稳定边界。

### 关键证据 3：真实机器人收益集中在“扭矩放大最明显”的任务
硬件四任务总体成功率：

- **iTuP**: 77.5%
- **CoPa**: 60.0%
- **w/o SDG-Net**: 60.0%

提升最大的场景也很符合论文机制：
- **hammer / knock**：冲击大；
- **reach**：杠杆臂长；
- **sweep**：多接触累积切向载荷。

这说明收益不是平均撒在所有任务上，而是**集中出现在物理瓶颈真的存在的地方**。这恰恰说明方法命中了问题。

### 关键证据 4：杂乱场景中依然成立
在 cluttered scenes 里，iTuP 仍达到 **70.0%**，对比 CoPa 的 **47.5%**。  
作者观察到 SDG-Net 更倾向于选“靠手柄”的抓取，这与其机制完全一致：**缩短有效杠杆臂**。

---

### 局限性

- **Fails when**: 接触点/交互法向 grounding 出错时，后续扳手估计会一起偏；工具或目标存在明显柔顺性、弹性变形，或交互持续时间较长、接触拓扑不断变化时，短时刚体模型容易失真。
- **Assumes**: 依赖 VLM/分割模块正确识别工具、目标与部件；依赖类别先验近似质量、惯量、摩擦和恢复系数；依赖候选抓取生成质量；实验主要在 Isaac Sim 与 UR5e + Robotiq 2F-85 上完成，且每个真实任务试验次数有限；未见代码/模型开放，复现成本偏高。
- **Not designed for**: 抓取与轨迹的联合优化、可变形工具操作、长时序闭环操控、在线系统辨识或不确定动力学下的鲁棒控制。

### 可复用组件

1. **语义-物理解耦框架**：可直接插到已有 VLM manipulation pipeline 里。  
2. **扭矩/滑移/对齐三项结构化代价**：适合做 grasp reranking 或 failure diagnosis。  
3. **SDG-Net 这类 surrogate scorer**：适合替代难以在线精确求解的动力学打分。  
4. **“固定语义与轨迹，仅比较抓取”的因果实验范式**：很适合后续 embodied manipulation 论文做机制归因。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Dynamic_Robot_Tool_Use_with_Vision_Language_Models.pdf]]