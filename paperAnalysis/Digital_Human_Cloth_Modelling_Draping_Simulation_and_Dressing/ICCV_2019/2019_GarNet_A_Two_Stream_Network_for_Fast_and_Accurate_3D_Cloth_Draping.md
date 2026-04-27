---
title: "GarNet: A Two-Stream Network for Fast and Accurate 3D Cloth Draping"
venue: ICCV
year: 2019
tags:
  - Others
  - task/3d-garment-draping
  - two-stream-architecture
  - mesh-convolution
  - physics-inspired-loss
  - dataset/SMPL
  - opensource/partial
core_operator: 在DQS粗对齐服装上，用人体点云流与服装网格流联合编码并回归每个服装顶点的残差位移，再用穿插/法向/弯曲约束把结果拉回物理合理区域。
primary_logic: |
  DQS粗对齐的服装模板 + 目标人体点云（+可选版型参数） → 双流网络提取人体全局/局部特征与服装点级/patch级/全局特征，并通过特征融合与物理启发损失回归每个服装顶点位移 → 输出贴体、少穿插且接近PBS结果的3D服装网格
claims:
  - "Claim 1: 在自建jeans/T-shirt/sweater测试集上，GarNet-Local相对PBS的平均顶点距离分别为0.88/0.93/0.97 cm，明显优于GarNet-Global、GarNet-Naive和DQS基线 [evidence: comparison]"
  - "Claim 2: GarNet-Local单次推理约68 ms，而文中使用的PBS需要超过7.2-19秒，达到约100×加速 [evidence: comparison]"
  - "Claim 3: 在Wang et al. 2018使用的shirt数据上，GarNet-Local的归一化距离误差为0.89%，优于GarNet-Global的1.15%和Wang et al. 2018的3.01% [evidence: comparison]"
related_work_position:
  extends: "FeaStNet (Verma et al. 2018)"
  competes_with: "Learning a Shared Shape Space for Multimodal Garment Design (Wang et al. 2018); DRAPE (Guan et al. 2012)"
  complementary_to: "Dual Quaternion Skinning (Kavan et al. 2007); DeepWrinkles (Lahner et al. 2018)"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/ICCV_2019/2019_GarNet_A_Two_Stream_Network_for_Fast_and_Accurate_3D_Cloth_Draping.pdf
category: Others
---

# GarNet: A Two-Stream Network for Fast and Accurate 3D Cloth Draping

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/1811.10983), [Project/Dataset](https://cvlab.epfl.ch/research/garment-simulation/garnet/)
> - **Summary**: 该文把3D服装垂坠建模为“在DQS粗对齐结果上预测逐顶点残差”，再用人体-服装双流特征融合和物理启发损失约束，实现在接近PBS精度下的实时服装贴体。
> - **Key Performance**: 自建数据集上平均顶点误差约0.88-0.97 cm；GarNet-Local推理约68 ms，而PBS需超过7.2-19 s。

> [!info] **Agent Summary**
> - **task_path**: 目标人体网格/点云 + DQS粗对齐服装模板（+可选版型参数） -> 服装逐顶点位移 -> 贴合人体的3D服装网格
> - **bottleneck**: 没有显式人体-服装对应关系时，很难同时建模全局姿态/体型与局部skin-cloth接触，导致褶皱不准、间隙和穿插
> - **mechanism_delta**: 将人体与服装拆成双流编码，并把局部人体特征通过最近邻池化显式注入服装顶点，再用物理启发损失约束输出
> - **evidence_signal**: 自建三类服装数据集和Wang et al. 2018数据上均优于基线，且相对PBS保持<1 cm误差并约100×加速
> - **reusable_ops**: [DQS残差化初始化, body-to-garment局部特征池化]
> - **failure_modes**: [高频细皱褶会被回归模型平滑掉, 去掉局部/patch特征或物理约束时接触区域易出现间隙与穿插]
> - **open_questions**: [如何补回高频皱褶且不牺牲稳定性, 如何扩展到动态时序与更复杂多层服装]

## Part I：问题与挑战

这篇论文解决的是一个很实际但很难的任务：**给定目标人体和一件服装模板，快速生成该服装在该人体姿态与体型上的3D垂坠结果**。目标应用是虚拟试衣、游戏、VR 这类要求接近实时响应的场景。

### 真问题是什么
Physics-Based Simulation（PBS）能给出高质量结果，但推理时间通常是秒级到十几秒级，难以用于实时系统。问题不只是“算得快”，而是要在快的同时保住三个关键属性：

1. **全局贴体正确**：衣服整体要跟着人体姿态和体型走。  
2. **局部接触合理**：肩部、腋下、腰部等区域要体现真实的 cloth-body interaction。  
3. **几何上可信**：尽量少穿插、少不自然拉扯，并保留褶皱和局部形变。

### 真正瓶颈
真正的瓶颈不是单纯的回归精度，而是：

- **人体点与服装顶点之间没有天然一一对应**；
- **仅靠全局人体特征不足以恢复局部接触关系**；
- **仅靠逐点回归容易得到“数值接近但物理不合理”的结果**，比如穿进身体里或把褶皱抹平。

### 输入/输出接口
- **输入**：
  - 目标人体 \(B\)（点云/网格）
  - 服装模板经 DQS 粗对齐后的网格 \(M\)
  - 可选：服装版型/裁片参数
- **输出**：
  - 每个服装顶点的3D位移
  - 位移加到粗对齐服装上，得到最终服装网格

### 边界条件
这篇论文关注的是**静态服装垂坠**，不是时序动态模拟。它默认：
- 有一个模板服装；
- 先用 DQS 做粗对齐；
- 用 PBS 结果作为监督信号来学一个快速近似器。

也就是说，它不是在“替代所有物理”，而是在**学习一个对 PBS 的快速逼近器**。

## Part II：方法与洞察

### 方法总览

GarNet 的核心结构是一个**双流网络**：

1. **Body Stream**
   - 输入人体点云；
   - 用 PointNet 风格结构提取人体的**点级特征**和**全局特征**；
   - 作用：编码人体姿态与体型。

2. **Garment Stream**
   - 输入 DQS 后的服装网格；
   - 一边做点级特征提取，一边用 mesh convolution 提取**patch-wise 局部邻域特征**；
   - 同时接收人体全局特征作为条件输入；
   - 作用：编码服装局部几何与整体形态。

3. **Fusion Network**
   - 将人体与服装特征拼接；
   - 用共享 MLP 对每个服装顶点输出一个3D平移量；
   - 最终得到 refined garment。

### 两个版本的差别
- **GarNet-Global**：只显式用人体全局特征。
- **GarNet-Local**：额外对每个服装顶点做最近邻人体特征池化，把局部 body cues 显式送入融合网络。

这一步很关键：它等于给每个服装顶点一个“它附近人体长什么样”的局部提示，从而更好地建模接触和贴体。

### 为什么先做 DQS 再预测残差
作者没有直接预测最终服装绝对坐标，而是预测**相对于 DQS 粗结果的逐顶点残差位移**。这相当于把问题从“从零生成一件衣服”改成“修正一个已经姿态基本对齐的衣服”。

好处是：
- 搜索空间更小；
- 姿态对齐已由几何先验完成；
- 网络可以把容量集中在局部接触、褶皱和细节修正上。

### 物理启发损失
训练时不只看顶点 L2，还加了三类约束：

- **interpenetration loss**：惩罚衣服穿进身体；
- **normal loss**：约束局部表面朝向，改善视觉外观；
- **bending loss**：约束局部弯曲/邻域几何，减少不自然尖刺和错误褶皱。

这些损失的作用不是“让网络学会物理方程”，而是**把输出限制在更像真实服装的几何流形上**。

### 核心直觉

#### 1）改了什么
从以往“低维衣服表示/纯全局条件/纯几何回归”，改成了：

- **残差化预测**：在 DQS 基础上做 refinement；
- **双流条件建模**：人体和服装分别编码，再融合；
- **局部接触显式注入**：对每个服装顶点池化局部人体特征；
- **物理启发约束**：把穿插、法向和局部弯曲纳入训练目标。

#### 2）改变了哪个瓶颈
它改变的是**信息瓶颈和约束瓶颈**：

- 信息上：从“只有全局人体条件”变成“全局 + 局部接触条件”；
- 约束上：从“只拟合坐标”变成“拟合坐标 + 满足几何/物理合理性”。

#### 3）带来了什么能力变化
- 更好地恢复局部贴体关系；
- 更少穿插；
- 比纯点级网络更能保留中频褶皱和局部形变；
- 在保持接近 PBS 结果的同时，把速度提升到实时可用。

### 为什么这个设计有效
因果上可以理解为：

- **DQS** 先解决大姿态对齐；
- **mesh conv** 让服装分支看见邻域几何，因此不再只是“每点独立回归”；
- **local body pooling** 让服装顶点知道自己附近的人体几何，缓解无对应关系问题；
- **physics-inspired loss** 把输出从“数值上接近”推向“视觉和几何上合理”。

### 策略权衡表

| 设计选择 | 解决的问题 | 收益 | 代价/风险 |
|---|---|---|---|
| DQS + 残差回归 | 直接生成难、姿态对齐难 | 降低学习难度，稳定训练 | 依赖可靠的粗对齐 |
| Garment mesh conv | 纯点级网络缺少局部结构 | 更好恢复褶皱和局部形态 | 需要固定网格邻接关系 |
| Global body features | 无人体-服装对应关系 | 提供姿态/体型条件 | 局部接触信息不足 |
| Local body pooling | 细粒度 cloth-body interaction | 减少间隙，提升贴体性 | 额外最近邻搜索与实现复杂度 |
| 物理启发损失 | 坐标回归易穿插、表面不自然 | 输出更可信 | 训练更依赖监督质量与超参 |

## Part III：证据与局限

### 关键证据信号

#### 1. 架构对比直接说明“局部 + patch 特征”是有效因子
在自建 jeans / T-shirt / sweater 数据上：

- **GarNet-Local** 的平均顶点误差分别为 **0.88 / 0.93 / 0.97 cm**
- **GarNet-Global** 略差，但接近
- **GarNet-Naive** 明显更差
- **DQS** 误差远大得多（如 jeans 为 11.43 cm）

这说明两点：
- **mesh convolution 的 patch-wise 服装特征**是必要的；
- **显式局部人体特征池化**比只用全局人体特征更有效。

#### 2. 能力跳跃不仅是精度，还有延迟
- GarNet-Local 推理约 **68 ms**
- GarNet-Global 约 **59 ms**
- PBS 约 **>7.2 s 到 >19 s**

所以它的价值不是“略快一点”，而是从离线模拟直接跨到**实时交互**可用。

#### 3. 外部数据验证了不是只会记住自建数据
在 Wang et al. 2018 的 shirt 数据上，作者把 sewing parameters 也作为输入，结果：

- **GarNet-Local：0.89%**
- **GarNet-Global：1.15%**
- **[41]：3.01%**

这说明该框架不仅能学人体-服装贴合，还能吸收**版型参数**这一额外条件。

#### 4. 消融说明损失项不是“装饰品”
- 去掉 **normal/bending**，法向误差变差，褶皱更不自然；
- 去掉 **penetration**，定量指标变化可能不大，但可视化中穿插更严重。

最强的证据不是单一数字，而是：**结构消融 + 损失消融 + 外部数据验证**三者一起支持作者主张。

### 1-2 个最关键指标
- **几何精度**：自建数据集上相对 PBS 的平均顶点误差约 **<1 cm**
- **系统速度**：推理约 **59-68 ms**，相对 PBS 约 **100×** 加速

### 局限性
- **Fails when**: 需要高频细皱褶、微小布料纹理起伏时，回归式预测会偏平滑；局部接触复杂区域若缺少局部人体特征或物理约束，容易出现间隙、伪褶皱或穿插。
- **Assumes**: 需要大量 PBS 生成的监督数据；依赖模板服装拓扑、DQS 粗对齐、人体网格/点云输入；训练数据主要来自 SMPL 身体与合成动作，分布外泛化未被充分验证；正文给出了项目/数据链接，但未明确提供完整代码与权重，复现仍依赖自行实现。
- **Not designed for**: 动态时序服装模拟、复杂多层服装交互、任意拓扑服装的零样本泛化、显式材料物理参数建模。

### 可复用组件
1. **DQS 后残差回归**：适合任何“先有粗几何、再做细修正”的3D拟合任务。  
2. **body-to-garment 局部特征池化**：适合无显式对应关系的跨表面条件建模。  
3. **物理启发几何损失**：适合把纯坐标回归拉回到更可信的几何空间。  
4. **mesh conv + point stream 双流融合**：适合同时处理规则网格局部结构与全局条件信息。

### 三个问题总结

1. **What / Why：真正瓶颈是什么，为什么现在值得做？**  
   真瓶颈是实时场景中无法承受 PBS 的高成本，同时纯数据回归又难以正确建模局部 cloth-body interaction。随着 PointNet/mesh conv 这类3D网络成熟，才有可能把“高质量垂坠”做成前向一次推理。

2. **How：作者到底拧动了哪个关键旋钮？**  
   关键旋钮是：**在 DQS 粗对齐的基础上，用双流网络把人体全局条件、人体局部接触线索和服装局部网格几何联合起来，并用物理启发损失限制输出空间**。这改变了信息瓶颈和几何约束，从而提升贴体与可信度。

3. **So what：相对前人真正跳到了哪里？**  
   能力跃迁体现在两点：  
   - 相对 PBS：把服装垂坠从秒级/十几秒级推到几十毫秒；  
   - 相对已有学习法：不再依赖低维 PCA 衣服表示，且在自建数据和 [41] 数据上都显示出更好的几何精度与更少伪影。

![[paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/ICCV_2019/2019_GarNet_A_Two_Stream_Network_for_Fast_and_Accurate_3D_Cloth_Draping.pdf]]