---
title: "Pick-and-place Manipulation Across Grippers Without Retraining: A Learning-optimization Diffusion Policy Approach"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/pick-and-place
  - diffusion
  - constraint-projection
  - gripper-mapping
  - dataset/CornellGraspingDataset
  - opensource/full
core_operator: 把新夹爪的低维几何差异先映射回基准夹爪状态，再在扩散去噪末段用约束投影最小幅度修正轨迹，使其满足抓取与安全条件。
primary_logic: |
  基准夹爪示教 + 场景点云/机器人状态/稳定抓取概率图 + 新夹爪高度与开口参数
  → 对新夹爪做高度/宽度基准化映射，并在扩散采样末段对动作序列施加累计安全约束投影
  → 无需重训即可输出适配未见夹爪的安全 pick-and-place 轨迹
claims:
  - "在 6 种夹爪的实机 block 抓放任务中，完整方法平均成功率为 93.3%，高于 Diffusion Policy 的 26.7% 和 3D Diffusion Policy 的 23.3% [evidence: comparison]"
  - "去掉在线投影后，跨夹爪 seen-object 平均成功率从 93.3% 降至 33.3%，说明几何映射本身不足以保证安全执行 [evidence: ablation]"
  - "在未见物体 banana 上，方法平均成功率达到 70.0%，高于 Diffusion Policy 的 30.0% 和 3D Diffusion Policy 的 20.0% [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Diffusion Policy (Chi et al. 2023); 3D Diffusion Policy (Ze et al. 2024)"
  complementary_to: "HPT (Wang et al. 2024); Cross-embodiment robot manipulation skill transfer using latent space alignment (Wang et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Pick_and_place_Manipulation_Across_Grippers_Without_Retraining_A_Learning_optimization_Diffusion_Policy_Approach.pdf
category: Embodied_AI
---

# Pick-and-place Manipulation Across Grippers Without Retraining: A Learning-optimization Diffusion Policy Approach

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.15613) · [Code](https://github.com/yaoxt3/GADP)
> - **Summary**: 该工作把“换夹爪就要重训策略”的问题，改写成“推理时做几何映射与约束投影”的问题：只用基准夹爪示教训练一次 diffusion policy，即可零样本适配未见夹爪完成抓取放置。
> - **Key Performance**: 6 种夹爪上 seen object 平均成功率 93.3%（DP 26.7%，3D DP 23.3%）；unseen banana 平均成功率 70.0%（DP 30.0%，3D DP 20.0%）

> [!info] **Agent Summary**
> - **task_path**: 基准夹爪示教 + 点云/机器人状态/抓取概率图 + 新夹爪几何参数 -> 跨夹爪安全 pick-and-place 动作序列
> - **bottleneck**: 换夹爪会同时改变 TCP 高度、夹爪开口语义和视觉外观，导致原 diffusion policy 生成碰桌或抓空的 OOD 轨迹
> - **mechanism_delta**: 把跨夹爪适配从“重训 policy”改成“夹爪状态映射 + 去噪末段二次规划投影”的学习-优化混合推理
> - **evidence_signal**: 6 种实机夹爪比较 + 去投影/去抓取概率图 ablation 显示完整方法显著更稳
> - **reusable_ops**: [gripper-geometry remapping, denoising-time cumulative projection]
> - **failure_modes**: [rotational compensation not handled, non-two-finger/clutter-rich settings unsupported]
> - **open_questions**: [can manual gripper calibration be replaced by learned descriptors, can the projection extend to full SE(6) and richer collision constraints]

## Part I：问题与挑战

这篇论文真正要解决的，不是“如何学一个抓取策略”，而是**已经学到的抓放技能，能不能在换夹爪后直接继续用**。

### 真正瓶颈是什么
对 diffusion policy 这类 imitation learning 方法，跨夹爪失败通常来自两层错配：

1. **观测分布错配**  
   新夹爪会改变手眼视角下的外观、遮挡和点云局部几何，导致输入分布偏离训练时的基准夹爪。

2. **执行可行域错配**  
   即便策略生成的“意图”没变，同一条末端轨迹在不同夹爪上也不再等价：  
   - 长夹爪更容易碰桌；  
   - 短夹爪更容易抓空；  
   - 开口宽度语义也会变化。

所以，核心难点不是再收一批新夹爪数据，而是：**如何把基准夹爪上学到的 manipulation primitive，投到新夹爪的可执行轨迹集合里。**

### 为什么现在值得解决
真实部署里更换末端执行器很常见；如果每换一次夹爪都要重新采集示教、微调甚至重训，behavior cloning 的样本效率优势就基本被抵消了。相比依赖大规模多 embodiment 数据的方法，这篇工作更想回答：**能否只训练一次，再用推理时的结构化修正来适配硬件变化。**

### 输入/输出接口与边界
- **训练输入**：基准夹爪 \(G_0\) 下的示教序列，观测由机器人状态、场景点云、抓取概率图组成。
- **部署输入**：同类观测 + 新夹爪的几何参数（TCP 高度、最大开口）。
- **输出**：一个短时域的末端位移与夹爪命令序列，用于闭环 pick-and-place。

**边界条件**
- 只处理**两指夹爪**；
- 主要补偿**高度/TCP 平移与夹爪宽度**；
- **旋转补偿尚未覆盖**；
- 任务集中在桌面 pick-and-place，而非复杂接触操作或灵巧手操作。

## Part II：方法与洞察

作者没有走“多夹爪联合重训”的路线，而是把问题拆成三件事：  
**视觉上学不变的、状态上做对齐、采样时保安全。**

### 核心直觉

- **What changed**：从“纯学习生成轨迹”改成“学习 manipulation prior + 几何映射 + 去噪时约束投影”。
- **Which bottleneck changed**：
  - `G*prob` 把视觉输入压缩成对象中心的抓取可供性，降低夹爪外观变化带来的观测漂移；
  - gripper mapping 把新夹爪的高度/宽度重新表达成基准夹爪语义，降低状态错位；
  - denoising-time projection 直接把采样结果拉回安全/可抓取集合，解决“生成轨迹不合法”的约束瓶颈。
- **What capability changed**：策略不必为每个新夹爪重学技能，而是把“学到的抓放原语”以最小修改迁移到未见夹爪。

这套设计之所以有效，是因为作者假设跨夹爪变化主要是**低维几何失配**，而不是技能逻辑本身改变。  
因此：
- 学习模块负责“**去哪抓、怎么放**”；
- 优化模块负责“**这只新夹爪能不能安全执行这条轨迹**”。

### 关键模块

1. **稳定抓取概率图 `G*prob`**
   - 先用预训练 GG-CNN 从深度图得到抓取概率；
   - 再做阈值过滤、质心提取、圆形区域掩码，得到更稳定的 object-centric grasp cue。
   - 作用：不再让 policy 过度依赖“夹爪长什么样”，而是依赖“物体哪里可抓”。

2. **夹爪几何映射**
   - 对每个新夹爪，离线量测高度偏移和最大开口；
   - 在线把新夹爪的 TCP 高度和开口宽度映射回基准夹爪表征。
   - 作用：让 policy 看到的机器人状态语义尽量和训练时一致。

3. **去噪末段安全投影**
   - 训练阶段基本保持 diffusion policy/DP3 风格不变；
   - 推理阶段把 DDIM 末段采样改成“最小修正”的约束投影；
   - 用累计约束保证整个动作 horizon 内都尽量满足安全边界，而不是只修当前一步。
   - 作用：把原本“可能合理但不可执行”的轨迹，修成“仍像示教分布、但又能落地执行”的轨迹。

### 策略性取舍

| 设计 | 改变的瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| `G*prob` 代替直接依赖原始视觉抓取线索 | 视觉 OOD、夹爪外观扰动 | 更稳地定位抓取区域 | 依赖深度质量与 GG-CNN 先验 |
| 高度/宽度 gripper mapping | 状态语义错位 | 新夹爪零样本接入 | 需要人工标定；未覆盖旋转差异 |
| 去噪末段累计投影 | 轨迹虽像示教但不安全/不可抓 | 防碰撞、减小抓空 | 在线 QP 计算开销；约束表达仍较简化 |

## Part III：证据与局限

### 关键证据信号

- **比较信号：跨夹爪能力跳变明显**  
  在 6 种夹爪、seen block 的实机任务上，完整方法平均成功率 **93.3%**，而 Diffusion Policy / 3D Diffusion Policy 只有 **26.7% / 23.3%**。  
  这说明提升不是“小修小补”，而是确实跨过了“换夹爪就崩”的门槛。

- **泛化信号：不只是在记训练物体**  
  对 unseen banana，完整方法平均成功率 **70.0%**，明显高于 **30.0% / 20.0%**。  
  这支持作者的主张：方法学到的是可迁移的抓放原语，而不只是基准夹爪 + 单一方块的固定轨迹。

- **ablation 信号：两个模块各管一类错误**  
  - 去掉 projection，seen-object 平均成功率从 **93.3%** 掉到 **33.3%**：说明光靠学到的轨迹先验，仍无法保证不同夹爪下的执行安全。  
  - 去掉 `G*prob`，在视觉上与基准夹爪差异更大的夹爪上明显退化：说明视觉不变性不是“可有可无”的配件，而是跨夹爪成功的重要前提。

- **分析信号：机制与失败类型对得上**  
  处理后的 `G*prob` 在不同高度、物体、夹爪条件下 KL divergence 更稳定；安全分析图也显示，projection 主要修复了两类典型失败：**碰桌** 与 **抓空**。

### 局限性

- **Fails when**: 夹爪差异不再是简单的高度/开口变化，而涉及明显的姿态旋转差异、非平行两指结构、复杂三维碰撞或接触动力学时，当前映射 + z 向安全约束不足以覆盖。
- **Assumes**: 每个新夹爪都有人工标定的高度偏移与宽度缩放；依赖 Azure Kinect + RealSense 深度观测、预训练 GG-CNN、在线 QP 推理，以及 RTX 4090 级计算环境；实验主要围绕单机器人、59 条示教、每设置 5 次试验的桌面抓放场景。
- **Not designed for**: 三指/灵巧手、完整 SE(6) 姿态补偿、in-hand manipulation、密集 clutter 中的全局避障、以及需要显式力控制的接触丰富任务。

### 可复用组件

- **低维 embodiment remap**：把新硬件的几何参数先映射到基准 embodiment 语义，再交给已训练策略。
- **denoising-time minimal correction**：不重训生成模型，只在去噪末段做最小幅度约束投影。
- **稳定 affordance 压缩**：把噪声较大的 dense grasp score 压缩成稳定、对象中心的抓取区域表征。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Pick_and_place_Manipulation_Across_Grippers_Without_Retraining_A_Learning_optimization_Diffusion_Policy_Approach.pdf]]