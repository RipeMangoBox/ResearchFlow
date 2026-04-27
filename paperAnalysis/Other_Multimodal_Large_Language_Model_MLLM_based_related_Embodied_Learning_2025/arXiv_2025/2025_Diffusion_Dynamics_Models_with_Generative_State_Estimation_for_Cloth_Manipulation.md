---
title: "Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation"
venue: CoRL
year: 2025
tags:
  - Embodied_AI
  - task/cloth-manipulation
  - task/dynamics-modeling
  - diffusion
  - transformer
  - model-predictive-control
  - dataset/CLOTH3D
  - repr/mesh
  - repr/point-cloud
  - opensource/no
core_operator: 基于Transformer的条件扩散同时完成“部分点云→完整布料网格重建”和“状态历史+动作→未来网格生成”，再用MPC闭环执行折叠。
primary_logic: |
  多视角RGB-D → 融合/分割为部分点云并结合canonical template mesh → 条件扩散Transformer重建完整布料网格状态；
  当前/历史网格状态 + 机器人delta动作 → 条件扩散动力学生成未来网格序列 → 目标引导的MPPI/MPC搜索动作并闭环重估状态
claims:
  - "在T-shirt状态估计上，DPM在仿真中将CD从TRTM的5.15×10^-1降到3.22×10^-1，并在真实世界中将CD从3.18×10^-1降到2.17×10^-1 [evidence: comparison]"
  - "在oracle状态输入的长时域动力学预测中，DDM将cloth/T-shirt的MSE降到0.05/0.35×10^-3，而Transformer分别为0.61/3.22×10^-3，达到约一个数量级优势 [evidence: comparison]"
  - "系统消融表明DPM与DDM缺一不可：真实规划中DDM+DPM的成功率为cloth 9/10、T-shirt 8/10，高于DDM+Transformer的5/10、3/10以及GNN+DPM的6/10、1/10 [evidence: ablation]"
related_work_position:
  extends: "TRTM"
  competes_with: "TRTM; AdaptiGraph"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Diffusion_Dynamics_Models_with_Generative_State_Estimation_for_Cloth_Manipulation.pdf
category: Embodied_AI
---

# Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv 2503.11999](https://arxiv.org/abs/2503.11999), [Project](https://uniclothdiff.github.io/)
> - **Summary**: 论文把布料操作中的两大难点——自遮挡下的完整状态恢复、以及长时域非线性动力学预测——统一改写为条件扩散生成问题，并与MPC结合实现真实机器人折叠。
> - **Key Performance**: oracle状态输入时，DDM将cloth/T-shirt长时域MSE降至0.05/0.35×10^-3（Transformer: 0.61/3.22×10^-3）；真实折叠中cloth与T-shirt自遮挡场景成功率分别达9/10和9/10，显著高于GNN基线。

> [!info] **Agent Summary**
> - **task_path**: 多视角RGB-D/部分点云 + 历史布料网格状态/机器人动作 -> 完整布料网格与未来网格轨迹 -> MPC动作序列
> - **bottleneck**: 自遮挡导致观测严重不全，同时布料高自由度、长程耦合动力学使误差在多步预测中快速累积
> - **mechanism_delta**: 将状态估计和动力学预测都从点估计/局部消息传递改为条件扩散生成，并用mesh-patch Transformer建模全局依赖
> - **evidence_signal**: 长时域预测MSE相对最佳非扩散基线约低一个数量级，且真实世界折叠成功率在多种遮挡下稳定提升
> - **reusable_ops**: [partial-point-cloud-to-full-mesh diffusion, history-action-conditioned mesh diffusion]
> - **failure_modes**: [uncertainty-unaware planning in safety-critical settings, out-of-domain non-cloth or strongly different contact regimes]
> - **open_questions**: [how to provide calibrated uncertainty to MPC, how far the mesh-diffusion formulation transfers beyond cloth categories and embodiments]

## Part I：问题与挑战

这篇论文要解决的不是单一“抓一下布料”的局部问题，而是**部分可观测条件下的闭环布料操控**。真正卡住系统的有两个耦合瓶颈：

1. **状态估计难**：布料严重自遮挡，哪怕有多视角 RGB-D，相机看到的也只是局部点云；而规划需要的是完整 3D 网格状态。
2. **动力学建模难**：布料近似无限自由度、强非线性、长程依赖明显，局部图消息传递很容易在长时域 rollout 时积累误差。

### 输入 / 输出接口

- **输入观测**：多视角 RGB-D，经融合与分割后得到部分点云。
- **隐状态表示**：布料模板网格上的顶点 3D 位置。
- **动作表示**：与执行器无关的末端位移 delta action。
- **输出目标**：
  - DPM：从部分点云恢复完整布料网格；
  - DDM：从历史状态和动作预测未来网格；
  - MPC：据此搜索到目标折叠状态的动作序列。

### 真正的瓶颈是什么？

不是“模型容量不够”这么泛，而是：

- **信息瓶颈**：部分观测到完整状态本身是多解的；
- **结构瓶颈**：GNN 的局部消息传递不擅长建模布料长距离耦合；
- **闭环瓶颈**：一旦感知有噪声，解析模拟器或单步监督模型在长时域规划里会迅速漂移。

### 为什么现在值得做？

论文的判断是：**扩散模型 + Transformer 的表达能力，已经足够强到能把“看不见的布料状态”和“复杂变形轨迹”都当作条件生成问题来学**。再加上 500K 仿真转移数据、较逼真的深度传感器模拟，使这条路首次在真实布料折叠上变得可行。

### 边界条件

这套方法不是“无先验、全开放世界”设定，它依赖：

- 已知 **canonical template mesh**；
- 输入是**分割后的点云**，不是原始 RGB 直接端到端；
- 目标状态可获得，并且可用网格距离度量；
- 主要验证对象是 cloth / T-shirt / long-sleeve 这类布料，而非一般刚体或任意可变形物。

## Part II：方法与洞察

### 1. 统一视角：把感知和动力学都变成条件生成

论文最关键的选择，不是分别堆一个感知模型和一个动力学模型，而是把两者统一成：

- **状态估计 = 条件生成完整布料状态**
- **动力学预测 = 条件生成未来布料状态**

也就是说，它不再问“直接回归一个唯一答案”，而是问“在当前条件下，什么样的完整状态/未来状态是数据分布上最合理的”。

---

### 2. DPM：从部分点云生成完整网格

DPM 的输入是部分点云和 canonical mesh，输出是完整布料网格。

核心做法有三步：

1. **点云编码**  
   把点云分 patch，用 PointNet 提取局部几何特征，得到条件 embedding。

2. **网格 token 化**  
   在 canonical space 中对 mesh 做 patchify，相当于把布料网格离散成一组稳定 token，便于 Transformer 处理。

3. **条件扩散去噪**  
   用 Transformer 做 denoising backbone，并通过 **AdaLN + cross-attention** 把点云条件注入到网格 token 中，逐步从噪声还原出完整顶点位置。

这个设计的意义是：**用生成式先验补足被遮挡部分**，而不是强迫模型在信息不足时给出脆弱的单点回归。

---

### 3. DDM：从历史状态和动作生成未来状态

DDM 基本继承了 DPM 的 mesh diffusion 骨架，但把条件从“点云”换成了“历史状态 + 机器人动作”。

关键改动：

- 输入条件增加历史网格状态；
- 用额外的时序注意力建模多帧依赖；
- 动作用 Fourier feature 编码；
- 加入 grasp mask，显式告诉模型哪些顶点被抓取。

这样，DDM 学到的不是“下一帧平均会怎么动”，而是**在当前布料配置和操作条件下，未来可能落在哪个物理合理的状态族中**。这对长时域 rollout 尤其重要，因为布料动力学高度多模态且对局部接触很敏感。

---

### 4. 规划：把生成式动力学接到 MPC

学到 DDM 后，作者用 MPPI/MPC 做规划：

- 目标函数同时考虑终态与目标的几何距离、以及动作平滑；
- 每执行一步后，用 DPM 重新估计状态，再 replanning；
- 为了减少盲采样，加入两种启发式：
  - **概率式抓点选择**：优先采样对目标偏差大的顶点；
  - **目标引导动作采样**：根据当前状态与目标状态的位移差，初始化更合理的动作方向。

这一步很关键：它把“好 world model”真正转化为“好操控系统”，而不是停留在离线预测指标。

### 核心直觉

**这篇论文真正拧动的因果旋钮是：把布料操控中的两个核心子问题都从“确定性点估计”切换为“条件分布建模”。**

具体地：

- **从直接回归到条件扩散**  
  改变了“部分观测必须唯一决定完整状态”的错误约束，转而学习一个更宽的后验分布；结果是自遮挡下的完整网格恢复更稳。

- **从 GNN 局部传播到 Transformer 全局建模**  
  改变了动力学模型的依赖范围：不再主要依赖局部邻接消息，而能直接建模布料远距离耦合；结果是长时域误差积累明显减小。

- **从盲目采样规划到目标引导采样**  
  改变了 MPC 的搜索分布，让 rollout 更聚焦真正需要移动的区域；结果是更少采样也能找到有效折叠动作。

本质上，它不是单纯“把 diffusion 用到机器人里”，而是利用 diffusion 的分布建模能力，去缓解**遮挡导致的状态不确定性**和**复杂动力学导致的 rollout 漂移**。

### 战略取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 条件扩散做状态估计 | 部分观测到完整状态的多解问题 | 自遮挡下更稳的完整网格补全 | 训练和推理都更重 |
| Transformer mesh patch 替代 GNN | 长程依赖与局部传播限制 | 长时域预测误差更低 | 更依赖大数据和大算力 |
| canonical mesh + point cloud 条件化 | 观测噪声与形变空间过大 | 提高几何一致性与 sim-to-real 稳定性 | 需要模板网格先验 |
| target-guided MPPI | 规划采样效率低 | 更快收敛到有效抓点和动作方向 | 依赖目标状态质量与对齐精度 |
| embodiment-agnostic delta action | 执行器耦合强 | 支持并爪到灵巧手迁移 | 仍需额外的抓点/轨迹映射 |

## Part III：证据与局限

### 关键证据信号

**信号 1：感知模块确实更能“补全看不见的布料”。**  
在状态估计实验里，DPM 在 cloth 和 T-shirt 上都优于 GarmentNets、MEDOR、TRTM 和去掉 diffusion 的 Transformer。最典型的是 T-shirt 仿真场景，DPM 将 CD 从 TRTM 的 5.15×10^-1 降到 3.22×10^-1；cloth 真实场景中，CD 也从 1.72/1.85×10^-1 进一步降到 1.13×10^-1。  
**结论**：生成式后验比直接回归更适合自遮挡补全。

**信号 2：动力学模型的优势主要体现在长时域稳定性。**  
oracle 状态输入下，DDM 在 cloth/T-shirt 上的 MSE 分别为 0.05/0.35×10^-3，而 Transformer 为 0.61/3.22×10^-3，GNN 为 0.75/6.36×10^-3；即便输入换成带感知噪声的 DPM 状态，DDM 仍保持明显优势。  
**结论**：扩散式动力学建模不仅更准，而且更抗 rollout 漂移与感知噪声。

**信号 3：系统级提升不是单模块幻觉。**  
真实规划里，作者在自遮挡、外部遮挡、组合遮挡下都优于 GNN。比如 T-shirt 自遮挡场景成功率从 GNN 的 1/10 提升到 9/10。消融也显示 DPM 与 DDM 必须配套：DDM+DPM 明显优于只换感知或只换动力学。  
**结论**：感知和动力学两个生成模块共同构成了系统级能力跃迁。

### 1-2 个最值得记住的指标

- **长时域预测**：oracle 输入下，DDM 将 T-shirt MSE 从 Transformer 的 **3.22×10^-3** 降到 **0.35×10^-3**。  
- **真实操控**：T-shirt 自遮挡折叠成功率从 GNN 的 **1/10** 提升到 **9/10**。

### 局限性

- **Fails when**: 任务超出布料操控分布、接触机理更接近刚体/多物体强接触、或者系统需要显式风险控制时；当前方法没有给出校准后的不确定性，安全关键场景下 MPC 缺少置信度约束。
- **Assumes**: 已知 canonical template mesh；可获得经检测/分割/ICP 对齐后的多视角点云；依赖 500K 仿真转移数据、SAPIEN + projective dynamics 物理建模，以及 4×H100 级训练资源。
- **Not designed for**: 无模板的未知拓扑服装、原始图像端到端直接出动作、非布料的刚体接触操作。

### 复现与可扩展性提醒

- 文中提供了项目页，但**正文未明确代码/权重发布**，因此可复现性仍受限。
- 真实系统还依赖 OWLv2、SAM、ICP 等外围模块；这意味着论文不是“单模型即插即用”，而是一个完整管线。
- 若未来要扩展到更广的 deformable manipulation，最关键的新增能力不是更大模型，而是：
  1. **显式不确定性估计**；
  2. **更强的跨材质/跨拓扑泛化**；
  3. **更低成本的高保真数据获取**。

### 可复用组件

- **mesh-patch diffusion backbone**：适合任何“部分观测 → 完整可变形状态”的任务。
- **history+action-conditioned diffusion dynamics**：适合 noisy-state world model。
- **target-guided grasp/action sampling**：可直接迁移到其他 MPC 机器人操作问题。
- **embodiment-agnostic delta action 表示**：适合跨执行器部署。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Diffusion_Dynamics_Models_with_Generative_State_Estimation_for_Cloth_Manipulation.pdf]]