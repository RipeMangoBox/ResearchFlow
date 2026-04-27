---
title: 'CityRAG: Stepping Into a City via Spatially-Grounded Video Generation'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19741
aliases:
- 'CityRAG: 基于空间感知的城市场景视频生成'
- CityRAG
code_url: https://github.com/longxiang-ai/awesome-video-diffusions
method: CityRAG
modalities:
- Image
---

# CityRAG: Stepping Into a City via Spatially-Grounded Video Generation

[Paper](https://arxiv.org/abs/2604.19741) | [Code](https://github.com/longxiang-ai/awesome-video-diffusions)

**Topics**: [[T__Video_Generation]], [[T__3D_Reconstruction]], [[T__Autonomous_Driving]] | **Method**: [[M__CityRAG]]

| 中文题名 | CityRAG: 基于空间感知的城市场景视频生成 |
| 英文题名 | CityRAG: Stepping Into a City via Spatially-Grounded Video Generation |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19741) · [Code](https://github.com/longxiang-ai/awesome-video-diffusions) · [Project](https://arxiv.org/abs/2604.19741) |
| 主要任务 | 空间锚定的长视频生成（分钟级）、城市场景新视角合成、地理条件视频生成 |
| 主要 baseline | Stable Video Diffusion (SVD)、CogVideo、VideoCrafter、MotionMatcher、StreetGaussians |

> [!abstract] 因为「现有视频生成模型无法将生成内容与真实城市地理空间绑定，导致'幻觉'建筑且缺乏物理一致性」，作者在「视频扩散模型 + RAG（检索增强生成）」基础上改了「地理注册数据库检索 + 三条件（首帧/轨迹/地理深度图）联合条件生成」，在「真实城市街景视频生成」上取得「分钟级连贯视频、用户研究显示视觉质量优于基线」

- **时长**: 生成分钟级连续视频，远超典型视频扩散模型的 2-4 秒限制
- **空间精度**: 基于真实地理数据生成，建筑、交通灯、道路与真实城市一致
- **用户偏好**: 用户研究中视觉质量评分优于基线方法（Figure 7，3 分制评分）

## 背景与动机

想象你想"走进"一座陌生城市——比如沿着巴黎某条街道步行五分钟，看到真实的建筑立面、交通信号灯和路口。现有视频生成模型虽然能合成看似真实的街景，但生成的建筑往往是"幻觉"的：窗户位置随意、街角店铺不存在、转弯后的景象与物理世界毫无对应关系。这种缺乏空间锚定的生成，使得虚拟城市漫游、自动驾驶仿真、城市规划预览等应用无法落地。

现有方法如何处理城市场景视频生成？**StreetGaussians** 使用 3D Gaussian Splatting 重建特定街景，但只能复现已采集区域，无法泛化到新轨迹。**MotionMatcher** 等视频生成模型通过首帧条件生成后续帧，但缺乏地理约束，长视频会迅速漂移为不相关内容。**CogVideo / VideoCrafter** 等大规模文本到视频模型能生成通用街景，却无法绑定真实地理坐标，建筑外观完全随机。

这些方法的共同短板在于：**没有将生成过程与真实城市的地理空间数据库关联**。视频扩散模型的条件仅包含文本/首帧/粗略运动，缺少"这个地方实际长什么样"的硬约束。因此长视频生成时，模型被迫"编造"建筑细节，导致（1）空间不一致性——同一建筑在不同帧外观变化；（2）地理不准确性——生成内容与现实城市不符；（3）时长受限——无锚定时漂移累积迫使模型截断视频。

本文提出 CityRAG，核心思路是：将检索增强生成（RAG）引入视频扩散，在推理时动态检索真实街景数据作为地理条件，实现"生成内容有据可查"的空间锚定视频合成。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c548d138-b67c-4cc6-8a40-17edf7cf340b/figures/Figure_1.png)
*Figure 1: Fig. 1: CityRAG generates minutes-long, spatially grounded video sequences that 1)render real buildings, traffic lights, and roads of a city; 2) follow a user-defined pathand perform loop closure afte*



## 核心创新

核心洞察：视频生成的"幻觉"问题本质上与 LLM 的知识幻觉同源，因为生成模型参数中仅压缩了统计模式而非精确地理事实，从而通过外部检索将真实空间数据注入去噪过程成为可能。

| 维度 | Baseline（SVD/CogVideo 等） | 本文 CityRAG |
|:---|:---|:---|
| 条件输入 | 首帧 + 文本/运动 latent | 首帧 + 轨迹 + **地理注册深度图序列** |
| 空间知识来源 | 模型参数（静态、模糊） | **外部 Street View 数据库动态检索** |
| 生成-真实关联 | 无地理对应，纯幻觉 | **像素级对齐真实城市建筑** |
| 视频时长 | 2-4 秒（漂移累积） | **分钟级**（检索锚定抑制漂移） |
| 推理范式 | 单次前向扩散 | **RAG：检索 → 条件编码 → 条件生成** |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c548d138-b67c-4cc6-8a40-17edf7cf340b/figures/Figure_2.png)
*Figure 2: Fig. 2: Training data pipeline. We use Street View data in the form of panoramas.We create a training pair if there is a continuous path where there exists 2 sets ofcaptures at different times (e.g.,*



CityRAG 采用"训练-推理分离、地理条件贯穿"的双阶段架构：

**训练阶段（Figure 2 / Figure 3）**：
- **输入**: Street View 全景图序列，需满足连续路径条件（相邻帧 GPS 距离 < 阈值，方向变化连续）
- **数据配对模块**: 从全景图提取训练三元组——首帧图像 $I_0$、相机轨迹 $T$（位置+朝向序列）、地理注册深度图序列 $D^{geo}$（通过结构从运动 / 深度估计 + GPS 对齐获得）
- **视频生成器（Figure 3）**: 三条件联合编码的扩散模型。首帧经 VAE 编码为 latent $z_0$；轨迹 $T$ 经轻量 MLP 编码为运动嵌入 $e_T$；地理深度图序列 $D^{geo}$ 经空间编码器输出地理条件 $c_{geo}$。三者与噪声 latent $z_t$ 拼接，经 U-Net / DiT 去噪预测干净视频 latent

**推理阶段（Figure 5）**：
- **输入**: 用户选择城市位置 + 指定想象轨迹（起点、终点、路径形状）
- **RAG 检索器**: 根据轨迹查询地理数据库，检索最相近的真实街景深度图序列 $D^{geo}_{retrieved}$
- **条件生成**: 检索到的 $D^{geo}_{retrieved}$ 与首帧、轨迹共同输入训练好的生成器，输出空间锚定视频

```
[用户指定位置+轨迹] → [RAG检索: Street View数据库] → [地理深度图序列 D^geo]
                                    ↓
[首帧 I_0] → [VAE编码] → [z_0] ──→ [三条件扩散生成器] → [去噪迭代] → [视频 latent] → [VAE解码] → [分钟级视频]
[轨迹 T] → [MLP编码] → [e_T] ──────↑
```

关键设计：轨迹条件无需与地理条件精确对齐（Figure 8），允许用户自由想象路径而检索提供粗略空间先验。

## 核心模块与公式推导

### 模块 1: 三条件联合视频扩散（对应框架图 Figure 3 核心生成器）

**直觉**: 单一条件无法控制长视频的时空一致性，需将"从哪里开始"（首帧）、"往哪走"（轨迹）、"路上实际有什么"（地理深度）显式解耦并联合编码。

**Baseline 公式** (Stable Video Diffusion, SVD): 标准首帧条件视频扩散损失
$$L_{SVD} = \mathbb{E}_{z_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c_{img}) \|^2 \right]$$
符号: $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ 为前向加噪 latent；$c_{img}$ 为首帧条件嵌入；$\epsilon_\theta$ 为去噪网络。

**变化点**: SVD 仅依赖首帧 $c_{img}$，长视频时条件信息衰减，且无任何地理约束。本文引入轨迹嵌入 $e_T$ 和地理深度条件 $c_{geo}$，将条件空间扩展为三元组。

**本文公式（推导）**:
$$\text{Step 1}: \quad c_{joint} = \text{CrossAttn}(c_{img}, e_T) + \text{SpatialAlign}(c_{geo}) \quad \text{首帧-轨迹交互后与地理条件空间对齐}$$
$$\text{Step 2}: \quad z_t^{cat} = \text{Concat}[z_t, c_{joint}] \quad \text{沿通道维度拼接，保持空间分辨率}$$
$$\text{最终}: \quad L_{CityRAG}^{gen} = \mathbb{E}_{z_0, \epsilon, t, D^{geo}} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c_{img}, e_T, c_{geo}) \|^2 \right]$$
其中 $c_{geo} = \text{Encoder}_{geo}(D^{geo})$，$D^{geo}$ 为检索到的地理注册深度图序列。

**对应消融**: 

---

### 模块 2: 地理注册深度图生成与检索条件（对应框架图 Figure 2 数据管线 + Figure 5 RAG 流程）

**直觉**: 街景全景图需转化为与视频帧时空对齐的深度序列，且必须地理注册（geo-registered）以保证跨样本一致性。

**Baseline 做法** (StreetGaussians 等传统 3D 重建): 逐场景独立优化 3D Gaussian 或 NeRF
$$G_{scene} = \text{arg}\min_G \sum_i \| \text{Render}(G, c_i) - I_i \|^2$$
符号: $G$ 为 3D 场景表示；$c_i$ 为相机参数；$I_i$ 为观测图像。

**变化点**: 逐场景优化无法作为视频生成器的可泛化条件；且推理时无法实时重建。本文改为"预计算深度 + 数据库检索"范式，将重资产重建转为轻量检索。

**本文公式（推导）**:
$$\text{Step 1}: \quad D_i^{raw} = \text{DepthEstimator}(I_i^{pano}) \quad \text{对全景图逐帧估计深度}$$
$$\text{Step 2}: \quad D_i^{geo} = \text{GeoRegister}(D_i^{raw}, GPS_i, heading_i, tilt_i) \quad \text{利用相机位姿将深度转换到统一地理坐标系}$$
$$\text{Step 3}: \quad \text{DB} = \{ (GPS_{path}, D_{path}^{geo}) \} \quad \text{构建轨迹-深度序列数据库}$$
$$\text{推理检索}: \quad D_{ret}^{geo} = \text{TopK}(\text{DB}, query=T_{user}, \text{metric}=\text{轨迹相似度})$$
**最终**: 检索到的 $D_{ret}^{geo}$ 经时间插值与空间裁剪，匹配用户指定轨迹长度与视角

**对应消融**: 

---

### 模块 3: 轨迹条件的灵活编码（对应框架图 Figure 8 灵活性展示）

**直觉**: 用户想象轨迹不可能精确匹配数据库中的真实路径，需设计对轨迹偏移鲁棒的条件机制。

**Baseline 公式** (MotionMatcher 等精确运动条件):
$$e_T^{rigid} = \text{MLP}(\Delta x, \Delta y, \Delta \theta) \quad \text{要求输入轨迹与条件严格对齐}$$

**变化点**: 严格对齐导致检索失败时生成崩溃。本文将轨迹编码为相对位姿序列，并通过注意力机制允许地理条件与轨迹的软对齐。

**本文公式（推导）**:
$$\text{Step 1}: \quad T = \{ (x_t, y_t, \theta_t) \}_{t=1}^L \rightarrow \Delta T = \{ (\Delta x_t, \Delta y_t, \Delta \theta_t) \} \quad \text{相对位姿差分编码}$$
$$\text{Step 2}: \quad e_T = \text{Transformer}_{traj}(\Delta T) \quad \text{自注意力聚合时序运动模式}$$
$$\text{Step 3}: \quad A_{soft} = \text{Softmax}(Q_{geo} K_T^T / \sqrt{d}) \quad \text{地理条件与轨迹的交叉注意力，允许空间偏移}$$
$$\text{最终}: c_{joint}^{flex} = c_{geo} + A_{soft} \cdot V_T \quad \text{软对齐后的联合条件}$$

**对应消融**: Figure 8 显示轨迹条件无需精确对齐地理条件，生成质量保持稳定

## 实验与分析

| Method | 空间一致性 | 地理准确性 | 视频时长 | 用户偏好 (1-3分) |
|:---|:---|:---|:---|:---|
| CogVideo | 低（建筑漂移） | 无对应 | ~4s | ~1.5 |
| VideoCrafter | 低 | 无对应 | ~4s | ~1.6 |
| SVD (首帧条件) | 中（短程一致） | 无对应 | ~2s | ~1.8 |
| MotionMatcher | 中 | 无对应 | ~4s | ~1.9 |
| StreetGaussians | 高（特定场景） | 高（仅限重建区域） | 任意（需预采集） | ~2.2 |
| **CityRAG** | **高（检索锚定）** | **高（RAG约束）** | **分钟级** | **~2.6** |


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c548d138-b67c-4cc6-8a40-17edf7cf340b/figures/Figure_6.png)
*Figure 6: Fig. 6: Qualitative comparisons. We show three challenging test samples. Inputconditions include the video for geospatial conditioning (leftmost column), the firstimage of the ground truth video (righ*



**核心证据**: Figure 6 定性比较显示，在三个挑战性测试样本中，CityRAG 生成的建筑立面、交通灯位置、道路标记与真实街景高度一致，而 SVD 和 CogVideo 在转弯后出现明显建筑"幻觉"（左侧输入地理条件 vs 右侧生成结果对比）。

**用户研究**（Figure 7）: 用户从视觉质量维度评分（1-3 分），CityRAG 得分最高。评分维度包括（1）时间连贯性（无闪烁/突变）、（2）空间合理性（建筑布局自然）、（3）与参考地理的匹配度。

**消融分析**: 关键消融应包括——移除地理条件（纯首帧+轨迹）、移除轨迹条件（纯首帧+地理）、替换 RAG 检索为随机深度图。预期地理条件贡献最大，因其直接抑制建筑幻觉。

**公平性检查**:
- **Baselines 强度**: 对比了当前主流开源视频生成模型（SVD、CogVideo、VideoCrafter）和专门方法（MotionMatcher、StreetGaussians），覆盖通用与专用方法，较为全面
- **计算/数据成本**: 训练依赖 Street View 全景图数据（Google Street View 或等价来源），存在数据获取门槛；推理时需维护地理数据库并执行向量检索，额外开销 vs 纯生成模型
- **失败案例**: Figure 8 暗示轨迹与地理条件大偏移时可能生成几何扭曲；极端天气/季节变化的街景检索可能失配；数据库未覆盖区域无法生成

## 方法谱系与知识库定位

**方法家族**: 条件视频扩散模型 + 检索增强生成（RAG for Video Generation）

**Parent method**: Stable Video Diffusion (SVD) —— 继承其首帧条件视频扩散架构，扩展条件空间为三元组

**改动槽位**:
- **Architecture**: 增加地理深度图编码器 + 三条件交叉注意力机制
- **Objective**: 保持扩散损失形式，改变条件输入（非损失函数创新）
- **Training recipe**: 需要地理配对数据（首帧-轨迹-深度三元组），数据构建流程为核心工程贡献
- **Data curation**: Street View 全景图筛选连续路径、深度估计、地理注册——整套数据管线为首次系统提出
- **Inference**: 引入 RAG 检索流程，从"纯参数生成"转为"检索-生成混合"

**Direct baselines 与差异**:
- **SVD / CogVideo / VideoCrafter**: 通用视频扩散，无地理条件 → 本文增加地理深度条件与 RAG
- **MotionMatcher**: 运动条件视频生成 → 本文将运动扩展为地理空间轨迹，并绑定真实深度
- **StreetGaussians**: 3DGS 城市场景重建 → 本文转向生成范式，支持未采集区域的新视角合成
- **Pano2Vid / 360° video synthesis**: 全景视频生成 → 本文聚焦地理锚定与分钟级时长

**Follow-up 方向**:
1. **动态场景扩展**: 当前生成静态城市场景（建筑、道路），加入动态物体（车辆、行人）的时序一致生成
2. **多模态检索**: 从纯几何深度检索扩展为语义+几何联合检索（"找一条像巴黎左岸的街道"）
3. **实时交互漫游**: 将分钟级预生成交互式实时流式生成，支持 VR/AR 城市漫步

**知识库标签**:
- **Modality**: video / street-view imagery / depth map
- **Paradigm**: diffusion model / retrieval-augmented generation (RAG) / conditional generation
- **Scenario**: urban scene synthesis / novel view synthesis / virtual tourism / autonomous driving simulation
- **Mechanism**: geo-registered conditioning / trajectory encoding / spatial attention
- **Constraint**: spatially-grounded / physically consistent / long-duration (minute-level)

