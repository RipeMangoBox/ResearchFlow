---
title: "RGB-SQ Grasp: Inferring Local Superquadric Primitives from Single RGB Image for Graspability-Aware Bin Picking"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/bin-picking
  - task/robotic-grasping
  - superquadric-fitting
  - global-local-fusion
  - metric-depth-estimation
  - dataset/YCB
  - dataset/GraspNet-1Billion
  - repr/superquadric
  - opensource/no
core_operator: 单张RGB先经基础模型转成实例级局部点云，再用全局-局部超二次曲面拟合与质量感知抓取采样完成无深度传感器的bin picking
primary_logic: |
  单张RGB与拥挤料箱场景 → Depth Pro与SAM2生成实例级局部点云 → 全局/局部双分支网络预测超二次曲面参数并经ICP细化 → 基于拟合质量、碰撞检测与top-down/COM优先级采样抓取位姿
claims:
  - "RGBSQGrasp在8个真实机器人bin-picking场景上取得92%的平均抓取成功率，超过PS-CNN的79.7%和MMPS的80.5% [evidence: comparison]"
  - "去掉后处理会使单物体任务中的抓取成功率从seen对象的100%降到75%，unseen对象的94.4%降到80.6%，说明ICP细化对可执行抓取至关重要 [evidence: ablation]"
  - "同时使用全局与局部特征比去掉任一编码器都能获得更低的mCD和更高的GSR，支持‘全局估尺度、局部补细节’的设计 [evidence: ablation]"
related_work_position:
  extends: "Hidden Superquadrics (Wu et al. 2023)"
  competes_with: "PS-CNN (Lin et al. 2020); MMPS (Hosseini et al. 2024)"
  complementary_to: "SCARP (Sen et al. 2023); MMRNet (Chen et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_RGBSQGrasp_Inferring_Local_Superquadric_Primitives_from_Single_RGB_Image_for_Graspability_Aware_Bin_Picking.pdf
category: Embodied_AI
---

# RGB-SQ Grasp: Inferring Local Superquadric Primitives from Single RGB Image for Graspability-Aware Bin Picking

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.02387), [Project](https://rgbsqgrasp.github.io/)
> - **Summary**: 这篇工作把单张RGB先转成实例级局部几何，再拟合为超二次曲面并用拟合质量筛选抓取区域，从而在拥挤bin-picking中摆脱对深度传感器和完整CAD模型的依赖。
> - **Key Performance**: 8个真实机器人场景平均GSR 92%；相对PS-CNN/MMPS分别提升12.3和11.5个百分点

> [!info] **Agent Summary**
> - **task_path**: 单张RGB/拥挤bin-picking场景 -> 实例级局部超二次曲面 -> 平行夹爪抓取位姿
> - **bottleneck**: 单视角遮挡下缺少稳定深度与完整点云，导致传统模板匹配或优化式SQ拟合难以恢复抓取所需几何
> - **mechanism_delta**: 用Depth Pro+SAM2把RGB转成实例级局部点云，再用全局-局部双分支SQ拟合和质量评分把“可见局部几何”转成“可执行抓取几何”
> - **evidence_signal**: 8个真实场景平均92% GSR，且移除后处理或任一编码分支都会明显降性能
> - **reusable_ops**: [cross-platform-synthetic-PCD-SQ-generation, global-local-SQ-fitting, SQ-quality-region-filtering]
> - **failure_modes**: [单个SQ难以覆盖强非凸或多部件形状, 深度估计或分割错误会直接传播到抓取采样]
> - **open_questions**: [多原语SQ分解能否进一步提升复杂物体抓取, 在更大规模真实仓储对象与不同相机上是否仍保持同样鲁棒性]

## Part I：问题与挑战

这篇论文解决的不是“从RGB恢复三维”这个泛问题本身，而是更具体的：**在拥挤、遮挡严重、相机视角有限的bin-picking场景中，如何仅凭单张RGB图像恢复足够用于抓取决策的局部几何**。

### 真正的难点
传统路径大致有三类，但都卡在 bin-picking 的约束上：

1. **解析式抓取**依赖已知物体模型或近完整点云。  
   这在料箱里通常拿不到，因为物体相互遮挡、纠缠、贴壁。

2. **深度传感器驱动的方法**把深度当作稳定几何输入。  
   但真实工业环境里，反光、透明、边缘细结构会让深度图出现洞、噪声和错位，sim-to-real 很明显。

3. **直接从RGB/点云学抓取**虽然端到端，但可解释性弱，而且对数据分布、标注和传感器质量都很敏感。

所以这里的**真实瓶颈**不是“有没有完整3D重建”，而是：

- 能否从**局部可见几何**中抽取一个足够稳定、足够低维、又直接服务于抓取的表示；
- 并且这个表示能在**单目RGB输入**和**真实机器人部署**之间保持一致性。

### 为什么现在值得做
这件事之所以现在可行，是因为两类工具成熟了：

- **单目 metric depth foundation model**：可以把RGB转成比硬件深度更稳定的几何代理；
- **超二次曲面（superquadric, SQ）表示**：用少量参数表达尺度、曲率和姿态，天然适合做抓取采样与几何推理。

论文的核心判断是：  
**抓取未必需要完整物体模型，但需要一个“抓取友好”的几何抽象。SQ 正好提供了这个中间层。**

### 输入 / 输出 / 边界条件
- **输入**：单张单目RGB图像，场景为拥挤料箱中的未知刚体物体
- **输出**：每个实例的局部SQ原语，以及可执行的平行夹爪抓取位姿
- **边界条件**：
  - 单视角、上方观测为主
  - 单个物体主要用**单个SQ**近似
  - 末端执行器是 **Robotiq 2F-85** 这类平行夹爪
  - 更偏向**刚体、日用品、近凸/可被单SQ近似**的对象

---

## Part II：方法与洞察

整套方法可以理解成四段式管线：

### 1) 跨平台合成数据：先造“局部点云 ↔ SQ参数”监督
作者先搭了一个跨模拟器数据生成流程：

- 基于 MetaGraspNetV2 的 bin-picking 场景生成思路
- 使用多个物理/仿真平台（文中图示包括 Isaac Sim、MuJoCo、PyBullet）
- 从多个近似俯视角采样，得到**部分可见点云**
- 对每个对象的部分点云配对其对应的 **SQ ground truth**
- 通过噪声、缩放、平移增强，最终形成 **36K PCD-SQ pairs**

这里的作用不是单纯“扩数据量”，而是**改变训练分布**：  
让网络在训练时就见到不同密度、稀疏度和噪声风格的点云，从而减轻 sim-to-real gap。

### 2) 全局-局部双分支 SQ 拟合网络
网络输入是预处理后的部分点云（FPS下采样到2000点），输出是SQ的参数：

- **shape head**：预测形状参数 \(\epsilon_1, \epsilon_2\)
- **scale head**：预测尺度参数 \(\alpha_1, \alpha_2, \alpha_3\)
- **rotation head**：预测旋转
- **translation head**：预测平移

关键在于它不是单一路径，而是：

- **局部分支**：DGCNN / EdgeConv，抓局部可见几何细节
- **全局分支**：PointNet风格，抓整体尺度和空间范围
- 两支都输出 1024 维特征，再融合用于参数预测
- 姿态相关输出还做了加权集成

训练目标不是直接对参数做复杂监督，而是把预测出的SQ表面采样成点云，再和GT SQ点云做 **Chamfer Distance** 对齐。

这意味着模型被训练成：  
**不是“记住物体类别”，而是从部分几何中恢复一个抓取可用的连续形状原语。**

### 3) 部署时的场景理解：RGB → 深度代理 → 实例点云 → SQ
真实部署阶段，作者不用深度传感器的深度通道，而是只用 RGB：

1. **Depth Pro**：从RGB估 metric depth
2. **SAM 2**：做分割
3. 把深度投影到分割结果上，得到**实例级 partial point cloud**
4. 对点云做预处理：
   - FPS下采样
   - RANSAC 去异常点
5. 输入训练好的 SQ fitting network
6. 用 **ICP** 做后处理细化

这个步骤的关键价值是：  
它把原本非常脆弱的“真实深度传感器读数”替换成“foundation model 生成的几何代理”，从而减少传感器噪声直接击穿抓取流程的问题。

### 4) SQ 引导的抓取采样
拿到每个实例的SQ后，系统按超二次曲面的尺度、曲率和主轴采样抓取候选，再通过三层筛选：

- **碰撞检测**：用 MoveIt! 过滤与料箱/其他物体冲突的抓取
- **SQ质量评分**：比较局部点云与拟合SQ在局部区域上的一致性，过滤欠拟合区域
- **抓取优先级**：
  - 优先 top-down 抓取
  - 若方向相同，优先接近物体 COM 的抓取

所以最终不是“只要能拟合出一个SQ就抓”，而是：  
**只在高质量拟合区域里抓，并偏向更稳的执行姿态。**

### 核心直觉

这篇论文最重要的因果改动有三层：

1. **把输入约束从“硬件深度”改成“RGB生成的几何代理”**  
   变化：RGB + foundation model 代替真实深度传感器  
   改变的瓶颈：传感器噪声与 sim-to-real 失配  
   带来的能力：面对反光/透明/边缘噪声时，场景几何更稳定

2. **把中间表示从“离散模板/完整重建”改成“连续SQ原语”**  
   变化：不依赖模板库，也不追求完整物体重建  
   改变的瓶颈：部分观察下信息不足、类别外泛化弱  
   带来的能力：只凭局部可见几何，也能恢复抓取相关的主轴、尺度和曲率

3. **把抓取决策从“拟合后直接采样”改成“质量感知采样”**  
   变化：加了ICP细化 + 区域质量评分  
   改变的瓶颈：单SQ近似误差会直接污染抓取位姿  
   带来的能力：把“粗略可拟合”变成“可靠可执行”

更直白地说，这篇工作的核心不是发明了一个更复杂的抓取网络，而是做了一个更对的表示切换：

**RGB外观 → 局部点云 → SQ几何原语 → 抓取候选**  

这个链条把高维、噪声大、难泛化的观测，压缩成了低维、可解释、直接服务抓取的几何结构。

### 战略取舍

| 设计选择 | 解决的瓶颈 | 能力收益 | 代价/取舍 |
|---|---|---|---|
| 单目 metric depth 替代硬件 depth | 深度传感器噪声与材质敏感性 | 对反光/半透明物体更稳，减少设备依赖 | 依赖外部预训练基础模型 |
| 单个连续 SQ 表示物体 | 模板库泛化差、完整重建成本高 | 表示紧凑、可解释、便于抓取采样 | 对强非凸/多部件形状欠拟合 |
| 全局+局部双分支拟合 | 部分点云下尺度与细节都不稳定 | 全局估尺度，局部补细节 | 网络结构更复杂，训练更依赖合成监督 |
| ICP + 质量评分过滤 | SQ拟合误差会直接传到抓取执行 | 明显提高可执行性与成功率 | 增加推理开销，需要阈值调节 |
| 跨模拟器数据生成 | 单一仿真风格导致 sim-to-real gap | 更稳健的部署泛化 | 仍不能完全覆盖真实材质/光照长尾 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 比较信号：真实机器人场景里，抓取成功率确实跳了
最核心结果是 8 个真实机器人 bin-picking 场景上的比较：

- **RGBSQGrasp 平均 GSR = 92%**
- **PS-CNN = 79.7%**
- **MMPS = 80.5%**

这说明它的收益不是只体现在“形状拟合更好看”，而是**真的传导到了最终执行成功率**。  
对机器人系统来说，这是最有价值的信号。

#### 2. 诊断信号：反光/透明物体是方法差异最明显的地方
论文特别强调，深度传感器在玻璃瓶等对象上容易出现边缘噪声和深度缺失。  
而 RGBSQGrasp 使用 Depth Pro 从 RGB 估 metric depth，能在这些场景下维持更稳定的场景理解。

这意味着能力提升并不只是“模型更大”，而是**把最脆弱的传感环节换掉了**。  
也就是说，它的提升有明确因果来源：**减少硬件深度依赖**。

#### 3. 消融信号：后处理和双分支不是可有可无
消融结果很说明问题：

- 去掉 **post-processing** 后，性能下降最大  
  - seen objects GSR: 100% → 75%
  - unseen objects GSR: 94.4% → 80.6%
- 去掉 **global encoder** 或 **local encoder** 都会退化  
  说明两者不是冗余，而是分别负责不同几何信息：
  - 全局分支更像是在稳住尺度/整体范围
  - 局部分支更像是在补足可见几何细节

所以这篇论文真正有效的，不是某一个单点模块，而是这条组合链路：
**foundation depth + instance segmentation + global-local SQ fitting + quality-aware grasping**

### 1-2个关键指标怎么读
- **92% 平均GSR（8个真实场景）**：代表端到端机器人系统的最终收益
- **189次抓取中的高成功率**：说明结果不是个别case，而是有一定重复执行稳定性

### 局限性

- **Fails when**:  
  - 物体形状明显不适合用单个SQ描述，比如强非凸、带孔洞/把手、明显多部件或细长不规则结构  
  - 遮挡严重到只剩极少局部可见几何时，SQ会出现姿态或尺度歧义  
  - 分割把多个相邻物体粘连，或单目深度估计在极端材质/光照下失真时，误差会直接传到抓取

- **Assumes**:  
  - 物体是刚体，且可被单个局部SQ近似  
  - 有可用的预训练 **Depth Pro** 和 **SAM2**  
  - 部署端有ICP、MoveIt!、点云处理等几何基础设施  
  - 训练阶段能构造大量 PCD-SQ 合成监督；文中训练也依赖 RTX 4090 级别GPU

- **Not designed for**:  
  - 多指灵巧手接触规划、吸盘抓取、软体/可变形物体  
  - 需要复杂拓扑理解的操作任务  
  - 需要完整物体重建或语义级操作推理的场景

### 可复用组件
这篇论文里最值得复用的模块有三类：

1. **跨模拟器 PCD-SQ 数据生成流程**  
   适合任何想做“局部几何 → 结构化原语”学习的工作。

2. **全局-局部双分支 SQ 拟合器**  
   这个模式对“部分点云下的低维形状恢复”很通用，不一定局限于抓取。

3. **SQ质量区域筛选机制**  
   本质上是把“形状拟合误差”显式转成“动作可执行性过滤”，对其他原语驱动操作也有参考价值。

### 一句话判断
这篇工作最有价值的地方，不在于把抓取完全端到端化，而在于找到了一条更稳的中间表示路线：  
**用单目RGB恢复“抓取足够用”的几何，而不是追求“重建全部真实几何”。**

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_RGBSQGrasp_Inferring_Local_Superquadric_Primitives_from_Single_RGB_Image_for_Graspability_Aware_Bin_Picking.pdf]]