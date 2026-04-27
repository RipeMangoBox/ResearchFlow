---
title: "GarNet++: Improving Fast and Accurate Static3D Cloth Draping by Curvature Loss"
venue: arXiv
year: 2020
tags:
  - Others
  - task/3d-cloth-draping
  - two-stream-network
  - graph-convolution
  - curvature-loss
  - dataset/CAESAR
  - repr/SMPL
  - opensource/partial
core_operator: 双流人体-服装网络在DQS粗配准服装上回归顶点位移，并用碰撞感知与多尺度Rayleigh商曲率损失补回PBS式褶皱细节
primary_logic: |
  目标人体网格 + DQS粗配准的模板服装 → 双流网络提取人体全局/局部特征与服装点级/patch级特征并回归每个服装顶点位移 → 结合顶点/穿模/法向/弯曲与多尺度Rayleigh商曲率监督进行训练 → 输出接近PBS且可实时运行的静态3D服装网格
claims:
  - "Claim 1: GarNet-Local在四类服装上的推理耗时为59–68ms，而PBS需>7.2–19s，约快100×，同时平均顶点误差保持在0.41–1.06cm [evidence: comparison]"
  - "Claim 2: 在Wang et al. 2018的公开服装数据上，GarNet-Local的归一化距离误差为0.43%，明显优于该方法的3.01% [evidence: comparison]"
  - "Claim 3: 多尺度Rayleigh商曲率微调能提升细节一致性；在[54]数据上平均法向误差由7.34°降至7.20°、RQ曲率损失由0.21降至0.11，但顶点距离由0.42cm升至0.45cm [evidence: ablation]"
related_work_position:
  extends: "GarNet (Gundogdu et al. 2019)"
  competes_with: "Learning a Shared Shape Space for Multimodal Garment Design (Wang et al. 2018); DRAPE (Guan et al. 2012)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/arXiv_2020/2020_GarNet++_Improving_Fast_and_Accurate_Static3D_Cloth_Draping_by_Curvature_Loss.pdf
category: Others
---

# GarNet++: Improving Fast and Accurate Static3D Cloth Draping by Curvature Loss

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2007.10867) · [EPFL CVLab](https://www.epfl.ch/labs/cvlab/)
> - **Summary**: 论文把静态3D衣物披覆拆成“DQS粗对齐 + 双流网格回归 + 物理启发损失”，并进一步用多尺度Rayleigh商曲率损失补回PBS风格的褶皱细节，从而在接近实时的速度下逼近物理仿真效果。
> - **Key Performance**: 推理约59–68ms，而PBS需>7.2–19s；在[54]公开数据上归一化距离误差0.43%，优于3.01%。

> [!info] **Agent Summary**
> - **task_path**: 目标人体网格 + DQS粗配准服装模板 -> 静态3D披覆服装网格
> - **bottleneck**: 纯顶点回归难同时建模全局姿态、局部衣身接触和高频褶皱，容易过平滑或穿模
> - **mechanism_delta**: 在GarNet双流架构上加入局部body KNN特征池化，并用多尺度Rayleigh商曲率损失替代仅靠点级监督的细节约束
> - **evidence_signal**: 四类服装上约100×快于PBS，且在公开数据上把归一化距离从3.01%降到0.43%
> - **reusable_ops**: [body-garment-two-stream-fusion, multi-scale-rayleigh-curvature-loss]
> - **failure_modes**: [极高频或波浪状细纹仍可能恢复不足, 为匹配局部曲率统计会轻微牺牲顶点距离]
> - **open_questions**: [如何扩展到动态布料与时序一致性, 如何减少对PBS配对监督和固定模板拓扑的依赖]

## Part I：问题与挑战

这篇论文解决的是**静态3D服装披覆**：给定目标人体网格，以及一件模板服装在目标姿态下经过粗略蒙皮后的形状，直接预测最终披覆后的服装网格。

### 任务接口
- **输入**：
  - 目标人体网格/点云 \(B\)
  - 模板服装网格 \(M_0\)
  - 通过 dual quaternion skinning (DQS) 粗对齐后的服装 \(M\)
- **输出**：
  - 最终服装网格 \(G^P\)，本质上是对 \(M\) 的每个顶点预测一个位移

### 真正的难点
真正的瓶颈不是“把衣服顶点挪到正确位置”这么简单，而是要同时满足三件事：

1. **全局对齐**：人体的体型与姿态决定了衣物整体下垂、拉伸、贴身程度。
2. **局部接触**：衣物哪里贴身体、哪里离身体远、哪里容易穿模，本质是局部衣身关系。
3. **高频细节**：褶皱、折痕、肩部/背部的小起伏不是单点位置误差能刻画的，而更像是**局部几何统计**问题。

### 为什么现在值得做
- **应用端压力很强**：虚拟试衣、网页端展示、交互式服装设计都需要近实时。
- **PBS太慢**：NvCloth/Marvelous Designer这类物理仿真效果好，但单次静态披覆需要秒级到十几秒。
- **早期数据驱动方法不够细**：PCA/低维子空间方法速度快，但细节丢失明显，且常需要后处理修正穿模。

### 边界条件
这篇论文有很明确的适用边界：
- 只做**静态**披覆，不建模时间序列动力学；
- 假设有**已知人体网格**和**模板服装拓扑**；
- 训练时依赖**PBS配对监督**；
- 重点是逼近物理仿真外观，而不是从RGB图像直接恢复衣服。

---

## Part II：方法与洞察

整体思路可以概括成一句话：

> 先用DQS把问题变成“小修正”，再用双流网络分别看人体和衣服，最后用物理启发损失和曲率统计把“像PBS”这件事写进训练目标里。

### 1. 先把难问题变成残差预测
作者不直接预测服装绝对坐标，而是：
- 先将模板服装用DQS粗略对齐到目标人体姿态；
- 再让网络只预测每个服装顶点的**位移残差**。

这一步的意义非常大：  
它把搜索空间从“任意3D服装形状”缩小为“围绕合理初值的局部修正”，从而更容易学，也更稳。

### 2. 双流架构：人体流 + 服装流
#### 人体流
- 用类似 PointNet 的结构处理人体点云；
- 提取两类信息：
  - **全局人体特征**：描述整体体型与姿态；
  - **点级人体特征**：给局部衣身接触提供线索。

#### 服装流
- 输入是粗对齐后的服装网格；
- 不只做点级MLP，还加入 **mesh convolution** 提取 patch-wise 局部几何；
- 同时接入人体的全局特征，让服装分支在处理中局部顶点时也知道“身体长什么样”。

#### 融合网络
- 将人体特征与服装特征拼接；
- 输出每个服装顶点的3D位移。

### 3. GarNet-Global vs GarNet-Local
作者做了两个版本：

- **GarNet-Global**：只显式使用全局人体特征；
- **GarNet-Local**：额外为每个服装顶点，从附近人体顶点中做KNN特征池化，显式注入局部衣身接触信息。

这一步解决的是一个很实际的问题：  
**仅靠全局人体embedding，网络知道“这人整体是什么姿态”，但未必知道“这个衣服点此刻离身体最近的是哪里”。**

### 4. 物理启发损失：把“合理衣服”写进目标函数
作者没有只用顶点L2，而是加入了几种很关键的监督：

- **顶点损失**：逼近PBS网格；
- **穿模惩罚**：基于最近人体顶点和法向，约束服装不要钻进身体；
- **法向损失**：让表面朝向更像PBS；
- **弯曲损失**：保持局部邻域距离关系，减少不自然拉扯。

这套损失的核心价值是：  
不是只让结果“数值接近”，而是让它在几何和接触层面也更像真正的布料。

### 5. GarNet++ 的关键增量：多尺度曲率损失
作者观察到：即便有上面的损失，结果仍可能**偏平滑**，特别是在褶皱明显的区域。

于是 GarNet++ 新增了细节监督：
- 对每个顶点取局部邻域；
- 计算邻域协方差；
- 用 **Rayleigh quotient** 近似局部最小/最大曲率统计；
- 让预测网格和PBS网格在这些局部曲率统计上接近；
- 并且在 **8/16/32 邻域**上多尺度施加。

作者也比较了基于 mean curvature normal 的方案，但认为其有两大缺点：
1. 只看一环邻域，感受野太小；
2. cotangent 权重容易数值不稳定，训练更容易发散。

### 核心直觉

**这篇论文真正调的“因果旋钮”是：把监督从一阶的点位置匹配，推进到局部二阶几何统计匹配。**

具体地说：

- **原来变什么**：主要是顶点位置、法向、局部边长；
- **现在多了什么**：局部邻域的曲率统计，以及服装点对附近身体点的显式感知；
- **瓶颈如何改变**：
  - 信息瓶颈：每个衣物点不再只靠全局body embedding猜接触关系；
  - 监督瓶颈：网络不再只被要求“坐标对”，还被要求“局部几何纹理像PBS”；
- **能力如何变化**：
  - 更少穿模；
  - 更少“衣服与身体之间的空隙”；
  - 更接近PBS的中频/高频褶皱细节。

简言之：  
**GarNet 解决“形状大体对”，GarNet++ 更进一步解决“局部褶皱也要像”。**

### 战略取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/副作用 |
|---|---|---|---|
| DQS粗配准 + 残差位移预测 | 直接生成完整服装太难 | 收敛更稳、跨姿态更容易 | 依赖模板服装与skin weights |
| 服装流中的mesh conv | 纯点级特征缺少局部几何 | 更好建模褶皱、折线 | 依赖固定网格拓扑 |
| 局部body KNN池化 | 全局人体特征不足以表达局部接触 | 贴身区域更准，body-gap更小 | 额外邻域搜索，仍是近邻近似 |
| 穿模/法向/弯曲损失 | 仅L2会生成“不物理”的表面 | 更少穿模、更自然表面 | 需要人体法向与邻域结构 |
| 多尺度RQ曲率损失 | 结果过平滑，细节缺失 | 褶皱统计更像PBS，法向更好 | 可能轻微牺牲Edist |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 速度-精度折中非常强
**信号类型：comparison**

最直接的结论是：它确实把PBS级任务做到了近实时。
- GarNet-Local / Global 推理只需 **59–68ms**
- PBS 需要 **>7.2s 到 >19s**
- 同时四类服装的平均顶点误差仍维持在 **0.41–1.06cm**

这说明它不是单纯“快很多但质量差很多”，而是实现了可用的速度-保真折中。

#### 2. patch-wise garment features 和 local body pooling 是有效的
**信号类型：ablation**

- 去掉mesh convolution得到的 **GarNet-Naive** 明显更差，说明只做点级特征不够；
- **GarNet-Local** 与 **GarNet-Global** 数值差距不算巨大，但作者指出前者在视觉上更少出现衣服与身体间的明显空隙。

所以这里的结论不是“local pooling带来数量级提升”，而是：
**它解决的是局部接触质量和视觉贴合感，而不是单一标量误差的大跳变。**

#### 3. 对外部公开数据集有明显优势
**信号类型：comparison**

在 Wang et al. 2018 的公开服装数据上：
- GarNet-Local：**0.43%** 归一化距离误差
- GarNet-Global：**0.48%**
- [54] 方法：**3.01%**

这个提升幅度很大，说明方法不仅适用于作者自建数据，也能泛化到带服装版型参数的场景。

#### 4. 曲率损失确实提升了细节，但不是“免费午餐”
**信号类型：ablation**

RQ曲率损失的作用很明确：
- 在[54]数据上，平均法向误差从 **7.34° → 7.20°**
- RQ曲率损失从 **0.21 → 0.11**
- 但平均顶点距离从 **0.42cm → 0.45cm**

这很重要，因为它说明：
**GarNet++ 优化的是“局部几何真实性”，而非单纯把每个顶点位置压到最低误差。**

也就是说，它追求的是更像PBS的褶皱统计，而不是更强的逐点拟合。

#### 5. RQ 比 mean-curvature 更稳、更通用
**信号类型：analysis**

作者的论点有实验支撑：
- mean curvature normal 方案训练中更容易出现数值不稳定；
- RQ方案可自然扩展到多尺度邻域；
- 在CAESAR身体数据上，GarNet-Local-RQ 也优于未加RQ版本：
  - T-shirt：0.53 / 8.23 / 0.044 → **0.46 / 6.56 / 0.025**
  - Sweater：0.61 / 8.11 / 0.047 → **0.56 / 6.66 / 0.031**
  - 格式分别是 `Edist / Enorm / LRQ`

这说明RQ不是只在单一数据集上“看起来更好”，而是跨身体分布也有效。

### 1-2 个最该记住的指标
- **速度**：59–68ms vs PBS 的 >7.2–19s
- **外部比较**：[54] 数据上 0.43% vs 3.01%

### 局限性
- **Fails when**: 需要恢复非常高频、波浪状、长程耦合的细纹时，曲率项仍不足以完全重建；褶皱本来就少的服装（如部分牛仔裤）上，曲率损失收益有限；当PBS对非常相近输入本身给出不一致结果时，网络很难逐点精确拟合。
- **Assumes**: 训练依赖成对PBS监督数据；假设已知目标人体网格、服装模板及其拓扑对应，并能先做DQS粗配准；复现上依赖NvCloth生成训练标签、Blender求skinning weights，以及GPU训练/推理环境。
- **Not designed for**: 动态时序布料、显式材质参数控制、从RGB直接恢复衣物、任意新服装拓扑的零样本泛化、多层服装/开拓扑复杂穿搭。

### 可复用组件
- **人体-服装双流融合**：适合任何“人体几何 + 附着物几何”的条件生成任务；
- **局部body KNN特征池化**：适合需要显式建模接触/邻近关系的网格任务；
- **碰撞感知几何损失**：适合替代昂贵后处理；
- **多尺度RQ曲率正则**：适合所有“点误差够了但表面细节太平”的3D网格回归问题。

## Local PDF reference

![[paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/arXiv_2020/2020_GarNet++_Improving_Fast_and_Accurate_Static3D_Cloth_Draping_by_Curvature_Loss.pdf]]