---
title: "DeepVerse: 4D Autoregressive Video Generation as a World Model"
venue: arXiv
year: 2025
tags:
  - Video_Generation
  - task/video-generation
  - diffusion
  - flow-matching
  - state-token
  - dataset/VBench
  - opensource/no
core_operator: 将RGB、深度与raymap组成4D状态，并把几何历史与几何检索记忆一起作为条件进行自回归未来视频生成
primary_logic: |
  单张图像/近期4D历史 + 文本动作 + 几何记忆 → 3D VAE压缩与token级历史拼接、流匹配扩散预测未来RGB/深度/视角 → 长时一致的4D未来序列
claims:
  - "Claim 1: 采用 token-wise 历史拼接的 2B MM-DiT 在 32–128 帧的 VBench 六项指标上几乎全面优于 channel-wise 方案，但平均 GFLOPs 从 1049.4 增至 1280.9 [evidence: comparison]"
  - "Claim 2: 加入 depth 模态后，模型在 60/120 帧的 subject/background consistency 均优于去除 depth 的版本，且 FVD 曲线整体更低 [evidence: ablation]"
  - "Claim 3: 加入 spatial condition 的几何记忆检索后，模型在“离开再返回”的长时场景中更能恢复原空间布局并保持时序连贯性 [evidence: case-study]"
related_work_position:
  extends: "Aether (Zhu et al. 2025)"
  competes_with: "GameNGen (Valevski et al. 2024); WorldMem (Xiao et al. 2025)"
  complementary_to: "FramePack (Zhang and Agrawala 2025); History-guided Video Diffusion (Song et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_3D_Vision_Priors/arXiv_2025/2025_DeepVerse_4D_Autoregressive_Video_Generation_as_a_World_Model.pdf
category: Video_Generation
---

# DeepVerse: 4D Autoregressive Video Generation as a World Model

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.01103) · [Project](https://sotamak1r.github.io/deepverse/)
> - **Summary**: 这篇论文把世界模型的自回归单位从“仅RGB视频帧”升级为“RGB+深度+raymap 的4D状态”，并加入几何检索记忆，从而显式缓解长时滚动中的尺度歧义、漂移和遗忘问题。
> - **Key Performance**: depth 消融下 120 帧 subject consistency 为 0.8165 vs 0.7681；token-wise 历史拼接在 32–128 帧 VBench 六项指标上几乎全面优于 channel-wise。

> [!info] **Agent Summary**
> - **task_path**: 单张图像/近期4D历史 + 文本动作 -> 未来4D状态序列(RGB, depth, raymap)与长时视频
> - **bottleneck**: 纯视觉自回归世界模型把3D世界压成2D观测，导致尺度歧义、位姿漂移和空间记忆失配
> - **mechanism_delta**: 用 depth+raymap 显式补全隐藏几何状态，并用基于几何邻近性的外部记忆替代仅靠时间邻近的历史条件
> - **evidence_signal**: depth 模态消融稳定改善 FVD 与 60/120 帧一致性指标，且 token-wise 历史注入优于 channel-wise
> - **reusable_ops**: [4D-state-factorization, geometry-aware-memory-retrieval]
> - **failure_modes**: [real-world-domain-shift, fast-camera-rotation-or-view-jump]
> - **open_questions**: [can-geometry-be-learned-without-synthetic-labels, can-memory-retrieval-be-learned-end-to-end]

## Part I：问题与挑战

### 这篇论文真正要解决什么问题？
DeepVerse 关注的不是普通视频续写，而是**把视频生成当成世界模型（world model）**：给定当前观察和动作，模型要持续预测未来世界状态。

作者认为，现有 interactive world model 的根本缺口在于：

1. **状态表示太“扁平”**  
   多数方法直接把 RGB 观察当状态来滚动预测，但真实世界是 3D/4D 的，视频只是 2D 投影。  
   结果是：同一张图像可能对应多种深度尺度与空间解释，尤其在视角变化时会出现**尺度歧义**。

2. **长时自回归会累积几何误差**  
   一旦相机位姿、深度或物体空间关系预测错一点，后续帧会基于错误继续生成，导致**drift** 和时间不一致。

3. **“遗忘”其实是检索错了历史**  
   仅靠最近几帧做条件，对“走开再回来”“重新看到旧区域”这类场景不够。真正相关的历史片段不一定时间上最近，而可能是**空间上最接近**。

### 为什么现在值得做？
这件事在 2025 年变得可行，主要因为三件事叠加：

- 已有强大的预训练视频生成先验，可作为自回归生成底座；
- 可以通过合成数据和自动标注管线拿到**精确深度与相机位姿**；
- 3D VAE / flow matching / MM-DiT 让“几何状态 + 视频生成”能共用一套 latent 生成框架。

### 输入/输出接口与边界条件
**输入：**
- 单张起始图像，或最近一段 4D 观测历史；
- 文本动作/控制信号；
- 从历史记忆库检索出的空间条件。

**输出：**
- 未来若干步的 4D 状态序列：`RGB + depth + raymap`；
- 对应的长时视频滚动结果。

**边界条件：**
- 训练核心依赖游戏/合成数据；
- 数据预处理会过滤**过快旋转、突变视角、几乎不动**的视频片段；
- 动作接口主要是**文本化控制**，不是原生低层控制器信号。

---

## Part II：方法与洞察

### 方法骨架

#### 1. 用 4D 状态替代纯 RGB 观察
论文把每个时刻的“状态代理”定义为：
- `v_t`：RGB 观察
- `g_t`：几何信息，包括 depth 和 camera/viewpoint

具体实现里：
- **depth** 以适合 VAE 编码的方式参数化；
- **viewpoint** 用 **raymap** 表示：每个像素携带相机位置/射线方向相关几何信息。

这一步的关键不是“多加一个模态”本身，而是把原本隐含在图像背后的**空间结构**显式搬到状态里。

#### 2. 历史注入：从 channel-wise 改成 token-wise
作者比较了两种把历史信息喂给 MM-DiT 的方式：

- **Model 1: channel-wise concatenation**  
  把历史帧在通道维拼起来，再 patchify。
- **Model 2: token-wise concatenation**  
  每个时刻分别 patchify 成 token，再在 token 序列上拼接。

最终采用的是 **token-wise**。  
直觉上，channel-wise 会把“时间信息”压进单个 token 内，Transformer 更难显式区分历史与当前噪声；而 token-wise 让时间关系变成**可被注意力直接访问的离散单元**。

#### 3. 几何记忆：按空间相关性检索，而不是按时间近邻截断
DeepVerse 维护一个全局 memory pool，把历史 4D 观测都对齐到初始帧定义的全局坐标系里。  
当前时刻需要额外条件时，不是盲目取更久远的若干帧，而是根据：

- 空间位置接近程度
- 视角/方向相似度

去检索**最相关的历史状态**，再作为 spatial condition 注入模型。

这使得模型在“离开某地后又回来”的时候，能重新拿到与当前空间最匹配的旧观察，而不是只依赖最近时间窗口。

#### 4. 长时推理：滑窗 + 重缩放
训练时模型只看固定长度 clip，但推理时要滚很长。  
作者采用：
- sliding window；
- 用当前窗口首帧的深度尺度做规范化；
- 窗口完成后再 rescale 回全局坐标，并写回 memory。

这相当于在**局部稳定生成**和**全局空间一致**之间做一个工程化折中。

#### 5. 文本控制替代专用 controller 模态
相比把 joystick / controller 作为新模态引入，作者刻意选择**文本条件**：
- 复用已有 text-conditioned 生成先验；
- 更容易把不同控制器映射成统一接口；
- 方便后续迁移和微调。

代价是动作表达的精确性可能不如原生低层控制信号。

### 核心直觉

**这篇论文真正拧动的“因果旋钮”有三个：**

1. **从 RGB-only 自回归 → 4D 状态自回归**  
   - 改变了什么：把隐藏的几何结构显式纳入状态。
   - 改变了哪个瓶颈：缓解 POMDP 下“观察非马尔可夫、单图尺度不确定”的信息瓶颈。
   - 带来什么能力：视角变化时更稳，长时 rollout 更不容易漂。

2. **从时间邻近历史 → 空间相关历史检索**  
   - 改变了什么：记忆读取依据从“最近”变成“最相关”。
   - 改变了哪个瓶颈：缓解 revisit 场景中的遗忘。
   - 带来什么能力：走远再回来时，背景和布局更容易恢复一致。

3. **从 channel 混合历史 → token 级显式时间单元**  
   - 改变了什么：Transformer 看到的是“分离的历史 token”，而不是压进通道里的混合特征。
   - 改变了哪个瓶颈：降低历史融合时的表示缠结。
   - 带来什么能力：更好的时序一致性，但代价是 token 数更多、算力更高。

### 为什么这个设计有效？
一句话概括：**作者不是试图用更强的视觉模型去“猜”几何，而是直接把几何当作生成时必须维护的状态变量。**

这很关键，因为世界模型的难点不只是画得像，而是：
- 相机动了以后，世界还能不能在空间上自洽；
- 旧区域再次出现时，模型还能不能记得原来的布局；
- 长时滚动后，局部误差会不会把全局世界坐标搞乱。

DeepVerse 的 4D 表示 + 几何记忆，正好分别对应这三个点。

### 战略取舍表

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/妥协 |
|---|---|---|---|
| RGB → RGB+depth+raymap | 纯视觉状态不充分、尺度歧义 | 更稳的视角变化与空间一致性 | 需要精确几何标签与更复杂预处理 |
| channel-wise → token-wise 历史注入 | 历史信息在 token 内缠结 | 更好的长时时序建模 | token 数增加，GFLOPs 更高 |
| 仅近期历史 → 几何记忆检索 | revisit 场景遗忘 | 更强长时空间连续性 | 依赖位姿预测质量与检索启发式 |
| 专用 controller 模态 → 文本控制 | 多控制器接口不统一 | 更强泛化接口、复用预训练能力 | 控制粒度和精确性可能受限 |
| 固定 clip 推理 → 滑窗+重缩放 | 训练/推理长度不一致 | 可持续长时 rollout | 尺度传递与全局对齐更复杂 |

---

## Part III：证据与局限

### 关键实验信号

#### 1. 结构对比：token-wise 历史注入优于 channel-wise
**信号类型：comparison**

作者在相同 2B 参数规模下比较两种历史融合架构。  
结果显示，**token-wise concatenation** 在 32/64/96/128 帧的 VBench 六项指标上几乎全面更优。

这支持了一个很具体的结论：  
> 对自回归视频世界模型来说，历史信息作为“独立 token”被访问，比把时间压进通道里更利于减少长期漂移。

但它不是白来的：平均 GFLOPs 从 **1049.4** 升到 **1280.9**。

#### 2. 深度模态消融：depth 不是锦上添花，而是关键状态变量
**信号类型：ablation**

作者拿掉 depth，只保留 raymap 相机表示，做了严格对照。  
结果：
- FVD 曲线整体更差；
- 120 帧时  
  - subject consistency：**0.8165 vs 0.7681**
  - background consistency：**0.9109 vs 0.8965**

这说明仅有相机/viewpoint 信息还不够，**显式深度**才是缓解尺度歧义和环境理解不足的关键。

#### 3. 空间记忆在“离开再回来”场景中有效
**信号类型：case-study**

论文给出可视化：加入 spatial condition 后，模型在返回先前区域时更能恢复原本布局；不加时更容易漂成新场景或背景错位。

这部分证据方向是合理的，但主要仍是**案例级**，定量支撑相对弱于 depth 消融。

#### 4. 质性能力：可从单图像启动，滚出长时未来
**信号类型：case-study**

模型从单张游戏图、真实图、AI 生成图出发，都能生成连续未来。  
这说明 4D 表示确实提升了“从静态观察进入动态世界”的能力，但这部分仍以展示为主，不足以证明强泛化。

### 证据强弱怎么判断？
我会把这篇的证据强度定为 **moderate**，原因是：
- 有清晰的架构比较和 depth 消融；
- 但核心评测主要集中在 VBench/FVD，外部基准和强基线覆盖仍有限；
- 空间记忆的关键收益更多由可视化案例支撑。

### 局限性
- **Fails when**: 从合成/游戏分布切换到真实世界时泛化会下降；遇到论文预处理阶段就过滤掉的快速旋转、突发视角跳变、极端运动时，几何估计与长时滚动稳定性会变差。
- **Assumes**: 需要大规模带精确深度和相机位姿的合成数据、可靠的自动标注管线、2B 级 MM-DiT 与 3D VAE、约 23,000 A100 GPU hours；推理时还依赖位姿足够准确以支撑记忆检索。论文提供项目页，但未在文中明确说明完整代码/权重已开源。
- **Not designed for**: 闭环策略学习、奖励建模、真实物理精确仿真、无需几何监督的通用现实世界世界模型，或需要原生低层控制器精确响应的场景。

### 可复用部件
- **4D 状态分解**：`RGB + depth + raymap`
- **token-wise 历史注入**：适合自回归扩散/flow-matching 视频模型
- **geometry-aware memory retrieval**：按空间相关性而非时间邻近读取历史
- **长时滑窗重缩放**：在局部生成稳定与全局坐标一致之间做桥接
- **文本化动作接口**：便于把不同控制器映射到统一条件空间

## Local PDF reference

![[paperPDFs/Building_World_Models_from_3D_Vision_Priors/arXiv_2025/2025_DeepVerse_4D_Autoregressive_Video_Generation_as_a_World_Model.pdf]]