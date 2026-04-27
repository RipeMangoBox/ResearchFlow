---
title: "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations"
venue: RSS
year: 2024
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - task/visuomotor-control
  - diffusion
  - sparse-point-cloud
  - lightweight-encoder
  - dataset/MetaWorld
  - dataset/Adroit
  - dataset/DexArt
  - dataset/DexDeform
  - repr/point-cloud
  - opensource/full
core_operator: "将单视角深度转成稀疏点云，经轻量点云编码器压缩为紧凑3D特征，再条件化到扩散策略生成多步动作。"
primary_logic: |
  单视角深度/点云 + 机器人位姿 + 少量离线示范 → 点云裁剪、FPS下采样与轻量MLP编码得到紧凑3D表示 → 条件扩散策略去噪生成短时域动作序列并执行
claims:
  - "Across 72 simulation tasks, DP3 achieves 74.4±29.9 average success versus 59.8±35.9 for image-based Diffusion Policy, a 24.2% relative improvement [evidence: comparison]"
  - "On 4 real-robot tasks with 40 demonstrations per task, DP3 reaches 85.0% average success and 0% observed safety-violation rate, while RGB/depth diffusion baselines achieve 35.0%/20.0% average success [evidence: comparison]"
  - "On 6-task ablations, point-cloud DP3 scores 78.3 average success versus 34.7/32.0/32.3 for RGB-D/depth/voxel, and the DP3 Encoder scores 78.3 versus 2.3/1.0 for PointNeXt/Point Transformer [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Diffusion Policy (Chi et al. 2023); 3D Diffuser Actor (Ke et al. 2024)"
  complementary_to: "Consistency Policy (Prasad et al. 2024)"
evidence_strength: strong
pdf_ref: "paperPDFs/Vision_Action_VA_Models_2024/RSS_2024/2024_3D_Diffusion_Policy_Generalizable_Visuomotor_Policy_Learning_via_Simple_3D_Representations.pdf"
category: Embodied_AI
---

# 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.03954), [Project](https://3d-diffusion-policy.github.io), [Code](https://github.com/YanjieZe/3D-Diffusion-Policy)
> - **Summary**: 这篇工作把单视角深度转成稀疏点云，并用一个极简点云编码器为 diffusion policy 提供几何对齐的条件，从而在少示范设置下显著提升机器人模仿学习的泛化、收敛速度与真实部署稳定性。
> - **Key Performance**: 72 个仿真任务平均成功率 74.4%，较 2D Diffusion Policy 相对提升 24.2%；4 个真实机器人任务平均成功率 85%，观测到的安全违规率为 0%。

> [!info] **Agent Summary**
> - **task_path**: 单视角深度/点云 + 机器人位姿 + 少量离线示范 -> 短时域连续动作序列
> - **bottleneck**: 少示范下，2D 视觉条件难以稳定恢复 3D 几何与空间关系，导致泛化差、收敛慢、真实机上易出现危险动作
> - **mechanism_delta**: 把 diffusion policy 的条件输入从 2D 图像特征换成经裁剪+FPS+轻量 MLP 编码的稀疏点云 3D 表征
> - **evidence_signal**: 72 个仿真任务 + 4 个真实任务的跨域比较，并辅以表征/编码器/设计细节消融
> - **reusable_ops**: [point-cloud crop+FPS, compact point-MLP conditioning]
> - **failure_modes**: [大视角变化需要手动点云变换与重设裁剪框, 超长时程任务未验证]
> - **open_questions**: [为何小点云编码器优于预训练大 backbone, 如何在开放场景下自动获得稳健的 3D 裁剪与视角不变性]

## Part I：问题与挑战

这篇论文要解决的核心问题是：**在纯离线、少示范、真实机器人操作条件下，如何学到既能泛化又能安全部署的 visuomotor policy**。

- **输入/输出接口**：输入是单视角深度相机得到的 3D 点云，以及机器人位姿；输出是短时域连续动作序列。
- **目标场景**：既包括 MetaWorld 这类低维抓取/操作，也包括 Allegro/Shadow hand 上的高维灵巧操作、articulated object 和 deformable object。
- **边界条件**：不依赖在线 RL、自动 reset、多相机系统或海量示范；真实任务每类仅 40 条演示，许多仿真任务只有 10 条示范。

真正瓶颈不在“动作生成器不够强”，而在**条件表征与控制问题的几何结构不匹配**：

1. **2D 视觉条件的数据效率太差**：用 RGB / depth image 学策略时，模型要同时从少量示范里学会深度、空间关系、视角变化和动作映射。
2. **已有 3D policy 往往不够通用**：不少方法依赖 keyframe pose extraction、prediction-and-planning 或复杂 3D 架构，更适合低维控制，不适合高频高维动作输出。
3. **真实部署还有安全问题**：视觉策略如果几何感知不稳，容易给出“离谱动作”，导致碰地、缠绕、撞击等需要人工紧急停止的行为。

**为什么现在值得解这个问题**：Diffusion Policy 已经证明扩散模型很适合建模多峰动作分布，因此当前更关键的短板转移到了“策略条件是否几何对齐”。深度相机和点云处理已经足够便宜和稳定，所以把 3D 几何直接注入 diffusion policy 变成了一个实际可落地的系统设计点。

## Part II：方法与洞察

DP3 的结构非常直接：**3D 感知模块 + 条件扩散决策模块**，两者端到端训练。

### 方法主线

1. **从深度图得到稀疏点云**
   - 单视角 84×84 深度图反投影为点云。
   - 用 bounding box crop 去掉桌面、地面等无关点。
   - 再用 FPS 下采样到 512 或 1024 个点，减少噪声并提高空间覆盖。

2. **把点云压成紧凑 3D 表征**
   - 使用一个很小的点云编码器：3 层 MLP + LayerNorm + max-pool + projection。
   - 最终只输出 **64 维** 3D feature。
   - 作者刻意**不使用颜色**，只保留 xyz，以避免外观 shortcut，提升 appearance generalization。

3. **用 diffusion policy 生成动作**
   - 将 3D feature 与机器人位姿嵌入拼接，作为条件输入给 diffusion policy。
   - 决策网络输出短时域动作序列，并在执行时滚动重规划。
   - 论文选择较短 horizon，本质上是在实时性和长规划能力之间做偏部署友好的取舍。

4. **端到端优化**
   - 感知和决策一起学，不做手工 keyframe 提取，也不先预测目标姿态再做规划。
   - 这使得方法可以覆盖高维灵巧手控制，而不局限于低维位姿控制。

### 核心直觉

这篇论文最关键的变化不是“加了 3D”，而是：

**把策略条件从外观主导的 2D 特征，换成了几何主导的紧凑 3D 特征。**

这会带来一条非常清晰的因果链：

- **what changed**：从 image/depth feature 改成 crop 后的 sparse point cloud feature。
- **which bottleneck changed**：模型不必再从少量数据里隐式恢复 3D 几何；同时 crop 去掉背景噪声，FPS 降低采样随机性，去颜色抑制外观捷径。
- **what capability changed**：空间外推更强、训练更快、跨外观/实例/轻微视角变化更稳，真实机器人上也更少出现危险动作。

更有意思的是，作者发现**简单编码器优于复杂点云 backbone**。这说明控制任务和 3D 分类/分割任务的最优归纳偏置并不一样：

- 固定相机、任务特定控制并不需要 PointNet 里的 T-Net 那种强特征变换不变性。
- BatchNorm 在少示范、多任务分布下可能反而不稳定。
- 对 diffusion policy 来说，一个稳定、低维、几何对齐的条件向量，比一个更复杂但更难优化的大表征更有价值。

### 战略取舍

| 设计选择 | 改变了什么约束 | 带来的能力 | 代价/边界 |
|---|---|---|---|
| 去颜色的稀疏点云 | 从外观优先改为几何优先 | appearance / instance 泛化更强，减少误碰撞 | 颜色语义任务不占优；依赖深度质量 |
| 轻量 MLP 点云编码器 | 降低大 3D backbone 的优化难度 | 少数据下更稳、更快 | 表达上限可能低于更大模型 |
| 点云 crop + FPS | 收紧输入分布，压掉背景噪声 | 收敛更快，成功率更高 | 需要任务相关裁剪框 |
| 短 horizon 动作扩散 | 提高重规划频率 | 更适合真实机执行与中途扰动 | 对超长时程规划支持有限 |

从消融看，**crop 是非常关键的小操作**；LayerNorm 和 sample prediction 主要改善稳定性与收敛速度；projection head 则几乎不掉精度地提升了效率。这些都说明 DP3 的贡献不是单一“大点子”，而是一个对控制分布非常贴合的系统配方。

## Part III：证据与局限

### 关键信号

- **比较信号：大规模仿真广覆盖有效**
  - 在 72 个仿真任务、7 个 domain 上，DP3 平均成功率 **74.4%**，相对 Diffusion Policy 提升 **24.2%**。
  - 这个结果重要的不是单点绝对值，而是它跨越了不同 simulator、不同手型、不同动作维度，说明收益并非只来自某个特定 benchmark。

- **比较信号：真实机器人少示范仍成立**
  - 在 4 个真实任务、每类仅 40 条示范下，DP3 平均成功率 **85%**；RGB 和 depth diffusion baseline 分别只有 **35%** 和 **20%**。
  - 尤其是 Roll-Up / Dumpling 这类变形体 + 灵巧手任务，恰好体现了 3D 几何条件对连续对位和包裹动作的重要性。

- **消融信号：不是“任何 3D 都行”**
  - 点云表征在 6 个消融任务上平均 **78.3**，明显高于 image (**40.7**)、depth (**32.0**)、RGB-D (**34.7**) 和 voxel (**32.3**)。
  - 编码器层面，DP3 Encoder 明显优于 PointNet++、PointNeXt、Point Transformer 及其预训练版本。
  - 这直接支撑论文的核心论点：**成功来自“稀疏点云 + 简单稳定编码器 + diffusion policy”的组合，而不是把 3D 模态机械塞进去。**

- **分析信号：泛化与安全的能力边界**
  - 论文分别展示了空间、外观、实例、轻微视角泛化。
  - 真实主实验中，DP3 的观测安全违规率为 **0%**，而图像/深度 baseline 为 **32.5% / 25.0%**。
  - 但要注意：这里的安全证据是经验观察，且真实评测样本数不大，不应解读为形式化安全保证。

- **附加信号：速度没有因为 3D 而牺牲**
  - 主文中 DP3 推理速度已与 2D Diffusion Policy 接近。
  - 附录的 Simple DP3 甚至把速度从 **12.7 FPS** 提到 **25.3 FPS**，只带来约 **6%** 的平均精度下降，说明真正关键的是 3D 条件设计，而不是重型 backbone。

**1-2 个最有说服力的指标**：
- 仿真：72 任务平均成功率 **74.4%**，相对基线 **+24.2%**。
- 真实机：4 任务平均成功率 **85%**，观测安全违规率 **0%**。

**局限性**
- **Fails when**: 相机视角变化较大、深度质量差、点云裁剪框设置不合适，或者任务需要显著超出短 horizon 的长期规划时；作者也明确表示未深入处理 extremely long-horizon tasks。
- **Assumes**: 单视角且已标定的深度相机、可获得机器人位姿、任务相关 crop box、少量但高质量的专家示范；真实灵巧手示范依赖人手遥操作与视觉重定向，数据采集成本不低；视角泛化实验还需要手动变换点云和调整裁剪区域。
- **Not designed for**: 颜色语义本身决定任务成败的场景、在线探索/自动重置/奖励学习范式、形式化安全认证，以及大幅相机位姿变化下的开放环境部署。

**可复用组件**
- 点云 `crop + FPS` 预处理
- 64 维轻量点云编码器
- 3D 条件化的短时域 diffusion policy
- 去颜色输入以换取外观泛化
- 对真实机友好的高频重规划设置

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2024/RSS_2024/2024_3D_Diffusion_Policy_Generalizable_Visuomotor_Policy_Learning_via_Simple_3D_Representations.pdf]]