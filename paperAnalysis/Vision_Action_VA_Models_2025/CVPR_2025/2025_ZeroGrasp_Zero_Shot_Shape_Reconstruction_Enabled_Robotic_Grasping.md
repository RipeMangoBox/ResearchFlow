---
title: "ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping"
venue: CVPR
year: 2025
tags:
  - Embodied_AI
  - task/robotic-grasping
  - task/3d-reconstruction
  - octree-cvae
  - transformer
  - occlusion-reasoning
  - dataset/GraspNet-1Billion
  - dataset/ReOcS
  - dataset/ZeroGrasp-11B
  - repr/octree
  - repr/SDF
  - opensource/no
core_operator: 以对象级八叉树CVAE联合预测完整形状与抓取场，并用多物体关系和3D遮挡场补全被遮挡几何，再基于重建结果做接触/碰撞精修
primary_logic: |
  单视角RGB-D与实例掩码 → 反投影为对象级八叉树特征，并在octree-CVAE潜空间中加入多物体编码器与3D遮挡场进行联合重建/抓取预测 → 输出完整3D形状与经接触约束、碰撞过滤后的6D抓取姿态
claims:
  - "ZeroGrasp在GraspNet-1Billion上取得新的SOTA抓取结果，Seen/Similar/Novel AP分别为70.53/62.51/26.46，使用ZeroGrasp-11B预训练后可提升到72.43/65.45/28.49 [evidence: comparison]"
  - "ZeroGrasp在真实图像上的单视角3D重建优于Minkowski、OCNN和OctMAE，在ReOcS-hard上达到CD 6.73、F1 80.86、NC 82.95 [evidence: comparison]"
  - "3D occlusion fields、multi-object encoder以及基于重建的碰撞/接触精修均提供实质增益，其中去掉碰撞检测会使Seen AP从70.53降至49.35 [evidence: ablation]"
related_work_position:
  extends: "OctMAE (Iwase et al. 2024)"
  competes_with: "GSNet (Wang et al. 2021); EconomicGrasp (Wu et al. 2025)"
  complementary_to: "ORB-SLAM2 (Mur-Artal and Tardós 2017); ContactOpt (Grady et al. 2021)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/CVPR_2025/2025_ZeroGrasp_Zero_Shot_Shape_Reconstruction_Enabled_Robotic_Grasping.pdf
category: Embodied_AI
---

# ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.10857) · [Project](https://sh8.io/#/zerograsp)
> - **Summary**: 这篇工作把“单视角 RGB-D 下的 3D 形状补全”和“6D 抓取预测”放进同一个八叉树生成框架里，并显式建模多物体关系与遮挡，从而让抓取决策建立在更可靠的几何之上。
> - **Key Performance**: GraspNet-1Billion 上 AP(Seen/Similar/Novel)=70.53/62.51/26.46，预训练+微调后达 72.43/65.45/28.49；真实机器人成功率 75.0%，高于基线 56.25%。

> [!info] **Agent Summary**
> - **task_path**: 单视角 RGB-D + 实例掩码 / 拥挤遮挡场景 -> 物体级 3D 形状重建 + 6D 平行夹爪抓取姿态
> - **bottleneck**: 仅从局部可见表面直接回归抓取时，遮挡区域的几何、接触点和碰撞体积都不可见，导致抓取不稳且易撞到邻近物体
> - **mechanism_delta**: 将抓取预测改为“重建先行、抓取后验修正”，并在八叉树潜空间中显式加入多物体关系和3D遮挡可见性
> - **evidence_signal**: GraspNet-1Billion 全 split SOTA，且去掉碰撞检测或3D遮挡场后 AP 明显下降
> - **reusable_ops**: [object-level octree CVAE, latent multi-object transformer, reconstruction-based contact-and-collision refinement]
> - **failure_modes**: [multi-view/incremental reconstruction not supported, segmentation/depth errors can merge objects and degrade reconstruction]
> - **open_questions**: [can multi-view fusion further improve heavily occluded grasping, can reconstructed geometry be extended to placement/manipulation planning]

## Part I：问题与挑战

这篇论文针对的不是“能不能预测一个抓取姿态”，而是更具体的难点：

**在单视角、拥挤、强遮挡场景里，如何对未见物体做安全且稳定的抓取。**

### 1. 真正的瓶颈是什么？

过去很多 6D 抓取方法直接从 RGB-D 或点云回归 grasp pose。这样做的问题是：

1. **看不到的部分没有几何约束**  
   夹爪最终接触的是完整物体，而不是当前可见表面。只靠局部可见点，模型很难推断被遮挡面的法向、厚度、边界和可接触区域。

2. **碰撞规避只能隐式学**  
   在多物体贴近摆放时，抓取是否会碰到邻物，本质上需要对场景几何和空间关系有明确表示。直接回归常会出现“抓到了目标，但也撞上了旁边物体”。

3. **零样本泛化需要形状先验**  
   对未见类别/未见实例，仅记忆训练物体外观是不够的。要泛化，模型需要学到更通用的“形状补全 + 接触几何”规律。

### 2. 输入/输出接口

- **输入**：单张 RGB 图、深度图、实例掩码
- **输出**：每个目标物体的高分辨率 3D 重建（occupancy/SDF/normal）以及 dense 6D grasp pose
- **使用场景**：遮挡、堆叠、近接触的 cluttered grasping
- **硬边界**：单视角、两指平行夹爪、需要目标级分割或至少前景分割

### 3. 为什么现在值得做？

作者的判断很明确：现在有两个条件成熟了。

- **稀疏体素/八叉树重建已经足够快**  
  相比 dense voxel、NeRF 或新视角合成路线，octree 在单视角下更适合近实时输出高分辨率几何。
- **可以造出足够大的几何+抓取联合数据**  
  作者新建了 **ZeroGrasp-11B**：1M 合成 RGB-D 图像、12K Objaverse-LVIS 物体、11.3B 物理有效抓取标注。这为零样本抓取提供了大规模形状与抓取先验。

---

## Part II：方法与洞察

ZeroGrasp 的核心不是简单“多加一个 reconstruction head”，而是把**几何恢复**变成抓取预测的中间因果变量。

### 方法骨架

#### 1. 对象级八叉树输入

- 先用图像编码器提取 2D 特征
- 用实例掩码把每个物体分开
- 结合深度图把对象反投影到 3D
- 再把对象点云与图像特征转成 **octree 表示**

这样做的意义是：后续预测不再停留在稀疏点云，而是转到更适合高分辨率补全和 dense grasp 预测的层级 3D 表示。

#### 2. Octree-based CVAE：联合重建与抓取

ZeroGrasp 采用一个基于八叉树的 CVAE，同时预测：

- occupancy
- SDF
- normals
- graspness / quality / angle / width / depth 等抓取参数

这里 CVAE 的作用不是“为了 generative 而 generative”，而是为了处理**单视角补全的不确定性**。  
同一个局部可见面，背后的完整形状可能并不唯一；如果不显式建模这种不确定性，重建和抓取都容易过拟合到错误补全。

#### 3. Multi-object encoder：在潜空间做多物体关系建模

基础 prior 只看单个对象，不足以理解：

- 物体之间谁更近
- 哪些区域会互相干涉
- 哪些抓取路径虽然对目标成立，但会碰到别的物体

所以作者在潜空间加了一个 **3D transformer**，把所有对象的 latent feature 一起编码。  
它主要解决的是：**局部形状预测之外的全局排布问题**。

#### 4. 3D occlusion fields：把“谁挡住了谁”局部化

这是本文最有意思的设计之一。

问题在于：  
多物体 transformer 虽然能学空间关系，但**遮挡关系**不完全等同于“空间接近”。某个体素是否被遮挡，取决于相机视线上的可见性关系。

作者因此设计了 **3D occlusion fields**：

- 从相机向潜空间体素做简化射线判断
- 区分：
  - **self-occlusion**：被目标自身遮挡
  - **inter-object occlusion**：被邻近物体遮挡
- 再用 3D CNN 编码这些局部遮挡标记，补进 latent feature

它的关键价值是：  
**把原本需要全局视线推理的问题，转成每个局部体素可消费的“可见性特征”。**

#### 5. 基于重建的抓取精修

网络先给出初始 grasp，然后利用预测重建做后处理：

- 找左右指尖最近接触点
- 调整夹爪宽度和深度，让两侧都更贴合接触面
- 对重建结果做碰撞检测，过滤掉会撞上的 grasp

这一步实际上把“抓取是否可执行”的判断，从纯网络回归，改成了**几何后验校验**。

### 核心直觉

**ZeroGrasp 改变的是抓取决策所依赖的信息形态。**

过去很多方法的逻辑是：

> 可见局部点云/深度 → 直接回归抓取

ZeroGrasp 的逻辑变成：

> 可见局部观测 → 补全可操作几何 → 在几何上做抓取与碰撞判断

这带来的因果变化是：

1. **信息瓶颈改变了**  
   从“只能依赖当前可见表面”变成“可以利用形状先验、多物体关系和显式遮挡线索推断被遮挡区域”。

2. **约束表达方式改变了**  
   碰撞、接触、厚度这些本来很难靠网络隐式学稳的约束，被转化为可在重建几何上显式检查的条件。

3. **能力边界改变了**  
   模型不只是在 visible surface 上找 grasp，而是在**隐含完整几何**上找 grasp，因此对遮挡、贴靠摆放、未见物体更稳。

### 战略取舍

| 设计选择 | 带来的能力 | 代价/依赖 | 适用边界 |
|---|---|---|---|
| 八叉树而非 implicit/NVS | 高分辨率重建、近实时、适合 dense 3D grasp 预测 | 工程更复杂，需对象级离散结构 | 单视角对象级重建 |
| CVAE 联合建模重建与抓取 | 处理单视角补全不确定性 | 训练更复杂，需要足够大数据 | 遮挡严重、未见物体 |
| Multi-object encoder | 更好地理解邻物关系与潜在碰撞 | 增加推理成本 | cluttered scenes |
| 3D occlusion fields | 更强的被遮挡区域补全能力 | 依赖分割与相机视角几何 | 遮挡推理是主瓶颈时 |
| 基于重建的接触/碰撞精修 | 让 grasp 更物理一致 | 强依赖重建质量，增加后处理 | 平行夹爪抓取 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：3D 重建确实更强

在 **ReOcS** 和 **GraspNet-1B** 的重建评测上，ZeroGrasp 超过 Minkowski、OCNN、OctMAE。  
尤其在 **ReOcS-hard** 上仍保持优势，说明它不是只在简单无遮挡场景里有效，而是确实提升了**遮挡条件下的形状恢复**。

一个简化结论是：

> 论文不是先假设“重建好了所以抓得更好”，而是先证明“重建本身确实更好”。

#### 2. 比较信号：抓取指标达到 SOTA

在 **GraspNet-1Billion** 上：

- Ours: **70.53 / 62.51 / 26.46**（Seen / Similar / Novel）
- Ours+FT: **72.43 / 65.45 / 28.49**

这说明两件事：

- 联合重建+抓取的路线并没有拖慢主任务，反而提升了抓取质量
- 大规模合成数据预训练对 **novel objects** 泛化有直接帮助

#### 3. 因果信号：最关键的增益来自“遮挡建模 + 几何后验过滤”

消融里最有说服力的点有两个：

- 去掉 **3D occlusion fields**，抓取 AP 明显下降  
  说明“可见性建模”不是装饰模块，而是真正在补强被遮挡区域推断。
- 去掉 **collision detection**，Seen AP 从 **70.53** 掉到 **49.35**  
  说明最终性能的大头，不只是网络回归更强，而是**重建几何让后验物理过滤变得有效**。

另外，用深度图局部点云做碰撞过滤远不如用重建几何做过滤，也支持作者的中心论点：  
**局部观测不足以做可靠碰撞判断。**

#### 4. 迁移信号：真实机器人上也有收益

真实机器人实验中：

- 基线：**56.25%**
- ZeroGrasp：**75.0%**

虽然样本规模不算特别大，但它至少说明：  
这不是只在离线 benchmark 上涨分的方案，在线执行时也能转化成更高成功率。

#### 5. 运行代价仍在“可部署区间”

- 推理速度约 **212 ms**
- 约 **5 FPS**
- A100 上显存低于 **8GB**

也就是说，作者确实把“重建 + 抓取”做到了接近实时，而不是一个只能离线演示的重模型。

### 局限性

- **Fails when**: 需要增量式或多视角重建时；实例分割/深度质量差导致多个物体被合并时；重建误差较大时，接触点与碰撞过滤也会被连带放大
- **Assumes**: 单视角 RGB-D 输入；对象级实例掩码或至少前景掩码可得；两指平行夹爪；大规模合成预训练数据可用；论文报告的高效推理基于 A100 级硬件
- **Not designed for**: placement pose 预测、非平行夹爪抓取、显式多视角融合、长时序操作规划

### 资源与复现依赖

这篇工作的可扩展性很大程度上依赖于其数据和工具链：

- **ZeroGrasp-11B** 的生成需要 BlenderProc、IsaacGym、V-HACD 等合成/物理流水线
- 使用了 **SAM 2** 微调后的实例分割
- 训练依赖 A100 级 GPU
- 文中给了项目页，但**未明确提供代码仓库信息**；因此从严格意义上说，复现门槛仍不低

### 可复用组件

这篇论文最值得迁移的不只是完整系统，而是几个可拆开的操作件：

1. **3D occlusion fields**  
   可作为通用的“局部可见性编码器”，用于对象补全、场景补全、碰撞感知规划。

2. **latent multi-object reasoning**  
   在潜空间做对象间关系建模，比单对象补全更适合拥挤操作场景。

3. **reconstruction-conditioned grasp refinement**  
   先学一个粗 grasp，再用重建几何做接触与碰撞修正，这个范式可迁到其他抓取模型。

4. **shape+grasp synthetic data engine**  
   同时提供高质量几何和物理验证抓取标注的数据构建思路，对零样本操作任务很有价值。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/CVPR_2025/2025_ZeroGrasp_Zero_Shot_Shape_Reconstruction_Enabled_Robotic_Grasping.pdf]]