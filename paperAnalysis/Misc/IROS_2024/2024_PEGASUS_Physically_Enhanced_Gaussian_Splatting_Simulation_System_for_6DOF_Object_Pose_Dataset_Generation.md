---
title: "PEGASUS: Physically Enhanced Gaussian Splatting Simulation System for 6DoF Object Pose Dataset Generation"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/6d-object-pose-estimation
  - gaussian-splatting
  - physics-simulation
  - dataset/Ramen
  - dataset/PEGASET
  - opensource/full
core_operator: "将环境与物体分别重建为可刚体变换的3DGS资产，并用低模碰撞网格接入物理引擎，从而自动生成带6DoF位姿标注的高真实感训练数据。"
primary_logic: |
  实拍环境/物体图像 → 分别重建3DGS光度实体与低模几何碰撞体，并在PyBullet中模拟自然落放轨迹 → 将6DoF变换施加到3DGS对象并多视角渲染RGB/深度/分割/框/位姿，导出为BOP格式
claims:
  - "Claim 1: PEGASUS能够将独立扫描的环境与物体组合成新场景，并导出RGB、深度图、语义掩码、2D/3D框与6DoF位姿的BOP格式数据 [evidence: analysis]"
  - "Claim 2: 论文发布了与PEGASUS兼容的两套资产：含30种日本杯面的Ramen数据集，以及含21个重扫描YCB-V物体的PEGASET [evidence: analysis]"
  - "Claim 3: 使用PEGASUS生成的合成数据训练DOPE后，作者报告UR5在真实杯面抓取演示中实现了10/10连续抓取，显示出synthetic-to-real迁移能力 [evidence: case-study]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "BlenderProc2 (Denninger et al. 2023); NDDS (To et al. 2018)"
  complementary_to: "DOPE (Tremblay et al. 2018)"
evidence_strength: weak
pdf_ref: paperPDFs/Misc/IROS_2024/2024_PEGASUS_Physically_Enhanced_Gaussian_Splatting_Simulation_System_for_6DOF_Object_Pose_Dataset_Generation.pdf
category: 3D_Gaussian_Splatting
---

# PEGASUS: Physically Enhanced Gaussian Splatting Simulation System for 6DoF Object Pose Dataset Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2401.02281), [Code](https://github.com/meyerls/PEGASUS), [Project/Dataset](https://meyerls.github.io/pegasus_web)
> - **Summary**: 这篇工作把真实扫描得到的环境和物体分别做成可编辑的3D Gaussian Splatting资产，再借助物理引擎生成自然摆放，从而低成本地产生可直接训练6DoF位姿网络的高逼真BOP数据。
> - **Key Performance**: 生成60,000张训练图像约需6小时；仅用PEGASUS合成数据训练的DOPE支持UR5真实抓取演示中10/10连续抓取

> [!info] **Agent Summary**
> - **task_path**: 实拍环境/物体扫描 -> 物理一致的多视角合成场景 -> 6DoF位姿训练数据/BOP标注
> - **bottleneck**: 传统合成数据真实感不足且资产建模成本高，导致6DoF位姿估计存在明显synthetic-to-real gap
> - **mechanism_delta**: 把每个物体拆成“3DGS光度实体 + 低模碰撞网格”，把环境也做同样解耦，从而同时保留真实外观与可模拟的物理摆放
> - **evidence_signal**: DOPE在PEGASUS合成数据上训练后完成UR5真实杯面抓取演示（报告10/10连续抓取）
> - **reusable_ops**: [环境/物体分离式3DGS重建, 3DGS光度实体与低模物理网格双表示]
> - **failure_modes**: [缺乏真实阴影与重打光导致插入物体显得不自然, 纹理稀少环境会产生噪声高斯并造成遮挡伪影]
> - **open_questions**: [如何把重打光/反射/折射纳入组合式3DGS渲染, 这种数据生成方式在更复杂场景与更多类别上是否仍能稳定迁移]

## Part I：问题与挑战

这篇论文要解决的不是“如何再做一个渲染器”，而是一个更实际的问题：**如何快速、低成本地构造足够真实、又带精确6DoF标注的训练数据**，用于机器人抓取和位姿估计。

### 真正瓶颈是什么？
现有路线大致有两类：

1. **真实采集+人工标注**  
   优点是真实；缺点是昂贵、慢、难扩展，而且位姿标注本身就麻烦。

2. **纯合成数据（如NDDS、BlenderProc）**  
   优点是可无限生成；缺点是需要手工建模资产，且渲染真实感不足，容易有domain gap。

PEGASUS抓住的核心瓶颈是：

- **资产获取成本**：手工建模环境和物体太慢；
- **外观真实感**：纯CG渲染离真实零售/桌面场景仍有差距；
- **物理合理性**：如果物体只是“贴”进场景，没有接触、堆叠、落放过程，训练数据分布不自然；
- **标注可用性**：数据要能直接进入现有6DoF pose pipeline，最好兼容BOP格式。

### 为什么现在值得做？
因为**3D Gaussian Splatting (3DGS)** 提供了一个关键新条件：

- 它比NeRF更**显式**，能直接对场景元素做插入、删除、刚体变换；
- 渲染速度高，适合大规模数据生成；
- 通过真实拍摄重建，外观分布更接近真实世界。

换句话说，3DGS把问题从“重新训练一个隐式场”变成了“组合和变换可重用资产”，这让数据生成系统化成为可能。

### 输入/输出接口
- **输入**：环境图像、物体图像、相机位姿/内参、SfM稀疏点云、度量尺度校准信息
- **输出**：RGB、深度图、语义掩码、可见性掩码、2D/3D框、6DoF位姿、BOP格式元数据

### 边界条件
PEGASUS并不是通用世界模拟器，它更接近：
- 针对**已扫描刚体物体**的组合式数据生成器；
- 主要面向**桌面/平面环境**；
- 假设物体和环境可**分开扫描**；
- 更关注**外观真实 + 位姿标注可用**，而非完整光传输物理正确性。

---

## Part II：方法与洞察

PEGASUS的设计哲学很清楚：**把“看起来真实”与“摆放得合理”拆开处理**。

- “看起来真实”交给 **3DGS photometric entity**
- “摆放得合理”交给 **mesh + physics engine**

这样做避免了一个常见冲突：高保真神经/辐射场表示通常不适合直接做物理碰撞，而物理模拟用的简化网格又不够好看。PEGASUS用双表示把这两个需求解耦。

### 方法主线

#### 1. 基础环境（Base Environment）
对环境拍摄100-150张图像，使用SfM恢复相机位姿与稀疏点云，再用ArUco做**度量尺度对齐**，并把平面校正到z轴向上。

然后：
- 用3DGS重建环境的**光度表示**
- 从高斯均值提取点云
- 用 alpha shape 恢复环境网格，作为**静态碰撞体**

这一步的意义是：环境既能被真实渲染，也能进入物理仿真。

#### 2. 物体资产（Gaussian Splatting Object）
每个物体同样建立两套表示：

- **Photometric Entity**：物体的3DGS，用来渲染真实外观
- **Geometric Entity**：从点云提取并平滑后的低模mesh，用来做碰撞模拟

作者对杯面物体使用了球面/半球扫描流程，最后每个物体大约注册到约270张图像。

#### 3. 物理耦合
用 PyBullet：
- 环境mesh作为静态体
- 物体mesh作为动态刚体
- 从一定高度随机落入场景，得到自然摆放

系统记录每个时刻每个物体的：
- 平移
- 四元数姿态

再把这些6DoF变换施加到对应的3DGS对象上，生成最终可渲染场景。

#### 4. 多视角渲染与导出
相机视角从一组ground-truth pose中采样并插值，最终导出：
- RGB
- Depth
- Segmentation / visibility masks
- 2D/3D bounding boxes
- object-to-world / world-to-camera transforms

并统一保存为 **BOP格式**，可以直接喂给现有位姿估计网络。

### 核心直觉

**什么改变了？**  
作者不再把场景视为一个不可编辑的整体辐射场，而是把它拆成可独立操作的资产：环境3DGS、物体3DGS、环境mesh、物体mesh。

**哪个约束被改变了？**  
过去的瓶颈是：
- 隐式表示难编辑；
- 物理与渲染耦合在同一表示上会很难用；
- 手工建模资产成本高。

PEGASUS把约束改成：
- 用**显式3DGS**负责外观组合；
- 用**低模mesh**负责物理接触；
- 用**扫描**替代手工建模。

**能力因此发生了什么变化？**
- 从“单场景重建”变成“资产级组合生成”
- 从“视觉上像真”扩展到“摆放上也像真”
- 从“只有图像”升级到“自动获得BOP级标注”
- 从“重建论文”变成“可训练下游pose网络的数据系统”

更细一点地说，论文里一个容易被忽略但很关键的点是：**物体旋转不仅作用在高斯位置和协方差上，还作用于球谐系数**。  
这意味着物体转动后，其视角相关外观也一起正确变化，而不是只转了几何没转外观，这对真实感很重要。

### 为什么这个设计有效？
因果链条可以概括为：

**扫描得到真实外观分布**  
→ 缩小纹理/材质层面的domain gap  
→ pose网络看到的训练图像更像真实部署环境

**物理落放生成自然接触关系**  
→ 缩小物体姿态、遮挡、堆叠方式上的分布偏差  
→ 下游模型更容易适应真实抓取场景

**显式3DGS可快速组合与渲染**  
→ 允许大规模生成不同环境×物体的组合  
→ 数据覆盖面和定制效率提升

### 战略取舍

| 设计选择 | 得到的好处 | 代价/风险 |
|---|---|---|
| 环境与物体分开扫描、后组合 | 资产可复用，组合数快速增长 | 光照是“拼接”的，不是全局一致重算 |
| 3DGS做光度表示 | 新视角真实、编辑快、渲染快 | 对扫描质量敏感，纹理差场景易出噪声 |
| 低模mesh做碰撞表示 | 物理模拟效率高，便于PyBullet使用 | 碰撞边界粗糙时会引入接触误差 |
| 真实扫描替代手工CAD建模 | 降低资产制作门槛 | 仍依赖扫描流程与校准 |
| BOP格式导出 | 可直接接入现有6DoF训练流程 | 主要服务已知物体闭集设置 |

---

## Part III：证据与局限

### 关键证据

#### 1. 下游任务信号：合成到真实迁移
最重要的证据不是图像看起来逼真，而是**训练出的pose网络能否用于真实机器人**。

作者用PEGASUS生成三套数据集：
- 每套 60,000 张图像
- 对应 3 个环境
- 共 2,000 个独特场景
- 每个场景 30 个视角

然后用这些数据训练 DOPE，最终在 UR5 上做真实杯面抓取，报告：
- **连续10/10抓取成功**

这说明 PEGASUS 生成的数据至少在该任务设置下，足以支撑真实部署级的位姿感知。

#### 2. 系统吞吐信号：生成速度可接受
作者报告：
- **60,000张图像约6小时生成完成**
- 硬件为笔记本级 i9 + RTX 3080 Ti Mobile

这说明 PEGASUS不是只能做小规模展示，而是已经具备“按环境定制数据”的实用性。

#### 3. 资产规模信号：不是单对象demo
论文还配套释放了：
- **Ramen**：30种日本杯面
- **PEGASET**：21个重扫描YCB-V物体
- **9个基础环境**

这证明作者不是只展示一个漂亮案例，而是在尝试搭建一个可复用的资产生态。

### 证据该怎么解读？
这篇论文的证据更像**系统可用性验证**，而不是严格benchmark论文。

它证明了：
- 这个系统能跑通；
- 生成数据对真实机器人任务有帮助；
- 3DGS + physics 的组合是有用的。

但它**没有充分证明**：
- 比 BlenderProc / NDDS / 真实数据微调更强多少；
- 哪个组件最关键；
- 在标准BOP指标上是否稳定优于其他合成数据方案。

所以前面的 `evidence_strength: weak` 是合理的：证据是正向的，但还不够系统化。

### 局限性
- **Fails when**: 场景需要真实阴影、重打光、反射、折射或散射时；环境纹理过少导致3DGS重建噪声大时；高斯外观与物理mesh边界不一致时。
- **Assumes**: 物体与环境都能被单独扫描并重建；对象主要是刚体；需要可靠的尺度校准与姿态对齐；物理仿真依赖简化低模mesh；作者的物体采集流程使用了商用Ortery扫描系统，虽然环境可用普通相机采集，但高质量资产制作仍有设备与流程门槛。
- **Not designed for**: 非刚体/可变形物体、铰接物体、开放世界未扫描新物体、需要全局光照一致性或真实材质交互的任务。

### 可复用组件
- **3DGS光度实体 + 低模几何实体** 的双表示框架
- **面向刚体的3DGS 6DoF变换**，尤其是对球谐系数的旋转处理
- **物理仿真驱动的姿态采样器**，比纯随机摆放更自然
- **BOP格式导出层**，便于接入现有6DoF pose训练/评测工具链

### 一句话结论
PEGASUS的核心价值不在于提出了新的位姿网络，而在于把**真实扫描、显式3DGS编辑、物理摆放、多视角标注导出**串成了一条可用的数据生产线；它最像一个“高真实感6DoF数据工厂”的雏形。

![[paperPDFs/Misc/IROS_2024/2024_PEGASUS_Physically_Enhanced_Gaussian_Splatting_Simulation_System_for_6DOF_Object_Pose_Dataset_Generation.pdf]]