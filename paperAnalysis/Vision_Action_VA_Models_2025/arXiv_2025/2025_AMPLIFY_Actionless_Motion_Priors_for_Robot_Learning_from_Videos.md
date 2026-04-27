---
title: "AMPLIFY: Actionless Motion Priors for Robot Learning from Videos"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/imitation-learning
  - task/video-understanding
  - state-token
  - keypoint-tracking
  - inverse-dynamics
  - dataset/LIBERO
  - "dataset/BridgeData v2"
  - "dataset/Something-Something v2"
  - repr/keypoint-trajectory
  - opensource/no
core_operator: 将稠密2D关键点运动压缩为离散运动 token，并把策略学习拆成“无动作视频上的前向运动预测”和“机器人交互数据上的逆动力学解码”。
primary_logic: |
  动作自由视频/机器人交互数据 + 当前图像/任务描述/本体状态 → 用点跟踪抽取关键点速度并经 FSQ 离散成运动 token，前向模型从图像与语言预测未来运动 token，逆动力学再把 token 映射为动作块 → 在少动作标注、跨人机 embodiment 与零目标任务动作数据下仍可执行任务的机器人策略
claims:
  - "Claim 1: On LIBERO, BridgeData v2, and Something-Something v2, AMPLIFY predicts future keypoint trajectories more accurately than ATM, Track2Act, and a Seer+CoTracker baseline, including 0.006 vs 0.022 MSE and 0.629 vs 0.250 pixel accuracy on LIBERO [evidence: comparison]"
  - "Claim 2: In zero target-task action-data generalization on LIBERO, AMPLIFY reaches 0.52/0.80/0.69/0.41 success on Long/Object/Spatial/Goal while BC baselines are near zero, yielding a reported 27× average gain over the best BC baseline [evidence: comparison]"
  - "Claim 3: Ablations on LIBERO-Long show that local-window classification outperforms coordinate MSE for motion tokenization (ΔAUC 0.919 vs 0.883), and a 16-step inverse-dynamics horizon outperforms 4/8-step horizons for downstream success (0.75 vs 0.36/0.64) [evidence: ablation]"
related_work_position:
  extends: "ATM (Wen et al. 2024)"
  competes_with: "ATM (Wen et al. 2024); Track2Act (Bharadhwaj et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); AVDC (Ko et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_AMPLIFY_Actionless_Motion_Priors_for_Robot_Learning_from_Videos.pdf
category: Embodied_AI
---

# AMPLIFY: Actionless Motion Priors for Robot Learning from Videos

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.14198), [Project](https://amplify-robotics.github.io/)
> - **Summary**: 这篇工作把“看视频学会应该发生什么运动”与“机器人如何把该运动执行出来”分开，用离散关键点运动 token 作为中间接口，让策略能有效利用无动作标注的人类/机器人视频。
> - **Key Performance**: 在 LIBERO 上轨迹预测 MSE 0.006 vs 0.022（相对 ATM 约 3.7× 更好）；在目标任务**零动作数据**设定下，LIBERO 四个泛化子集平均成功率 60.5%，而 BC 基线接近 0。

> [!info] **Agent Summary**
> - **task_path**: 动作自由视频/机器人交互数据 + 当前 RGB/语言/本体状态 -> 未来运动 token -> 机器人动作块
> - **bottleneck**: 最丰富的数据是无动作视频，但传统策略学习需要成对动作标签；同时像素级视频预测和直接轨迹回归会把容量浪费在外观细节而非可执行运动上
> - **mechanism_delta**: 用离散关键点运动 token 统一前向动力学与逆动力学接口，使“学会该怎么动”和“学会机器人如何跟随该运动”可分别扩展
> - **evidence_signal**: 多数据集轨迹预测显著优于 ATM/Track2Act/Seer，且在 LIBERO 零目标任务动作数据下取得 60.5% 平均成功率
> - **reusable_ops**: [dense-grid-keypoint-tracking, FSQ-motion-tokenization]
> - **failure_modes**: [2D轨迹对应多解动作, 随机场景中动作效应与外生扰动难分离]
> - **open_questions**: [能否扩展到3D运动token, 能否仅靠在线探索数据训练更强逆动力学]

## Part I：问题与挑战

### 1. 这篇论文真正要解决什么问题？
机器人学习里最贵的不是模型，而是**动作标注的专家示范**。  
Behavior Cloning 直接学 `观测 -> 动作`，所以只能吃到带动作标签的数据；但现实里最丰富的数据来源恰恰是**没有动作标签的视频**，包括互联网人类视频、无标注机器人回放、非专家交互数据。

因此真正的问题不是“视频能不能做预训练”，而是：

- **如何把无动作视频转成对控制真正有用的先验？**
- **如何让这种先验在少量动作数据、跨 embodiment、人类视频、甚至零目标任务动作数据时仍然可用？**

### 2. 真正瓶颈在哪里？
作者认为瓶颈有两层：

1. **监督接口错位**  
   视频告诉你“未来会发生什么视觉/运动变化”，但策略学习需要“现在该输出什么动作”。  
   这两种数据模态没有天然对齐接口。

2. **表示层级错位**  
   直接做像素视频预测太贵，而且会逼模型建模纹理、背景、光照等对控制不关键的细节。  
   而直接在像素坐标上回归关键点轨迹，又容易：
   - 计算重
   - 回归到均值
   - 难处理多模态运动
   - 难泛化到新任务/新主体

### 3. 为什么是现在？
因为两个条件成熟了：

- **动作自由视频足够多**：互联网与现实回放远多于机器人专家示范。
- **点跟踪能力变强**：CoTracker/TAPIR 一类方法让稠密 2D 点跟踪在遮挡和长时段下更实用，能把“运动”从“外观”中抽出来。

### 4. 输入/输出接口与边界
论文假设有两类数据：

- **视频数据** `V = {(o_t, g)}`：只有观测与任务描述，可来自人类或机器人
- **交互数据** `R = {(o_t, q_t, a_t)}`：有观测、本体状态、动作，不要求都与目标任务同分布

目标是学到：
- `前向动力学`：当前图像 + 任务 → 未来运动 token
- `逆动力学`：当前图像 + 本体状态 + 运动 token → 动作块

边界也很明确：

- 主要建模 **2D 图像中的运动**
- 假设环境更偏**确定性动力学**
- 仍然需要**一部分机器人动作数据**来学 inverse model
- forward model 需要任务描述/任务条件

---

## Part II：方法与洞察

### 方法主线：把策略拆成三个模块
AMPlIFY 的结构可以概括成：

1. **先从视频里提取稠密关键点轨迹**
2. **把轨迹压缩成离散 motion tokens**
3. **分别学习前向动力学和逆动力学**

这样，原来单体的 `observation -> action`，变成了：

`observation -> future motion tokens -> action chunk`

这一步是整篇文章最关键的机制变化。

### 1. 关键点预处理：先把“运动”显式拿出来
作者对每一帧初始化一个 `20×20` 的均匀网格，共 `400` 个点，用 CoTracker 跟踪未来 `T=16` 帧。  
关键点不是手工挑任务相关点，而是**每帧重新初始化稠密均匀网格**，好处是：

- 不依赖物体分割或关键点启发式
- 相机移动时仍有覆盖
- 长时遮挡问题更小

而且他们不直接用坐标，而是转成**单步速度**，因为真正关心的是“往哪里动”。

### 2. Motion Tokenization：把连续轨迹压成离散运动词汇
这是第一层核心创新。

作者用 FSQ 把关键点速度序列编码成离散 token。  
但更重要的是它的**解码目标**不是直接回归 `(x, y)` 坐标，而是对每个点输出一个**局部窗口内的位移分类分布**。

直觉上：

- 坐标回归会把多种可能运动平均掉
- 局部窗口分类更像是在说：“这个点下一步更可能往这几个邻近方向动”

这给了模型两种好处：

- **显式局部运动偏置**
- **更好表达多模态未来**

### 3. Forward Dynamics：用任何视频学“应该怎么动”
前向模型输入：

- 当前图像
- 任务文本

输出：

- 未来一段 motion tokens

实现上是自回归 transformer，但重要的不是架构细节，而是其训练来源：  
**它完全可以只用无动作视频训练**。

也就是说，这个模块专门学：
- 对于任务 `g`
- 从当前观察 `o_t`
- 世界接下来应该呈现什么运动模式

这相当于学的是 **task-conditioned motion prior**，而不是动作策略本身。

### 4. Inverse Dynamics：用少量交互学“机器人如何跟随这种运动”
逆动力学模型输入：

- 当前图像
- proprio
- motion tokens

输出：

- 一个动作 chunk

关键点是：**它不看 goal**。  
作者故意把它做成一个**通用 reference follower**：

- forward model 负责决定“参考运动是什么”
- inverse model 只负责“机器人如何把这个参考运动执行出来”

这样 inverse model 就能吃任意交互数据，不必全是同任务专家示范。

### 核心直觉

#### 直觉一：把监督对象从“像素外观”换成“离散运动”
**改变了什么**：从预测视频像素/静态表示，改成预测关键点运动 token。  
**改变了哪个瓶颈**：削弱了外观纹理、背景、光照等无关因素造成的信息污染，也显著压缩了预测空间。  
**带来了什么能力**：forward model 更容易从异构视频里学到跨任务可迁移的动态先验。

#### 直觉二：把单体 BC 换成“前向动力学 + 逆动力学”
**改变了什么**：不再强迫一个模型同时学“任务目标”和“机器人执行”。  
**改变了哪个瓶颈**：原来必须依赖 paired action labels 的约束，只留在 inverse model；forward model 可以独立用海量无动作视频训练。  
**带来了什么能力**：少样本学习、跨人机迁移、零目标任务动作数据泛化。

#### 直觉三：把连续回归换成局部离散分类
**改变了什么**：motion tokenizer 的重建目标从坐标 MSE 改为局部窗口分类。  
**改变了哪个瓶颈**：避免“多种未来平均成不动/模糊”的回归问题。  
**带来了什么能力**：轨迹预测更锐利，也更利于下游动作解码。

#### 为什么这个设计在因果上有效？
因为它把原来纠缠在一起的两个问题拆开了：

- **what**：为了完成任务，世界应该出现怎样的运动？
- **how**：我的机器人要输出怎样的动作，才能让世界呈现这个运动？

无动作视频非常适合学 `what`，交互数据非常适合学 `how`。  
AMPLIFY 的贡献就是找到一个足够轻、足够通用、又足够动作相关的中间接口：**latent keypoint motion tokens**。

### 策略层面的 trade-off

| 设计选择 | 带来的收益 | 代价 / 边界 |
|---|---|---|
| 稠密均匀关键点网格，而非任务特定关键点 | 不依赖分割、目标点标注，跨任务通用 | 包含许多无关点，性能受 tracker 质量影响 |
| 离散 motion token，而非像素视频预测 | 预测空间更小、效率更高、更易泛化 | 有量化误差，token 语义不显式 |
| 局部窗口分类，而非坐标回归 | 更能表达局部多模态运动，减少均值化 | 默认位移落在局部窗口内 |
| 前向/逆动力学解耦 | 无动作视频和交互数据可独立扩展 | 模块间存在误差传播，不是端到端最优 |
| inverse model 不看 goal | 学成通用 reference follower，利于跨任务复用 | 如果 motion token 不够充分，会增加动作歧义 |

---

## Part III：证据与局限

### 关键实验信号

#### 1. Comparison：轨迹预测确实更准
这不是只在一个数据集上有效，而是跨三个视频数据源都更强：

- **LIBERO / BridgeData v2 / Something-Something v2**
- 对比 **ATM、Track2Act、Seer+CoTracker**

最强信号是 LIBERO 上：
- **MSE：0.006 vs 0.022**
- **Pixel Accuracy：0.629 vs 0.250**

这说明 AMPLIFY 学到的不是“看起来合理”的抽象，而是更接近真实未来运动的表示。

#### 2. Comparison：真正的能力跃迁发生在低标注与分布外场景
作者也很诚实：  
在标准 in-distribution、示范充足的设定下，传统 BC 仍然能很强，AMPLIFY 并不是单纯追求满数据场景下的绝对峰值。

它真正的优势出现在这三类场景：

- **Few-shot**：当每个任务只有 2/5/10 个 demo 时，视频预训练明显提升成功率；2 demo/任务时，相比 ATM 平均约 **1.94×**。
- **Cross-embodiment**：加入人类无动作视频训练 forward model 后，真实机器人三项任务平均成功率从 **0.42 提升到 0.58**（相对 Diffusion Policy baseline）。
- **Zero target-task action data**：这是最强证据。LIBERO Long/Object/Spatial/Goal 上分别达到 **0.52 / 0.80 / 0.69 / 0.41**，平均 **60.5%**；BC 基线几乎为 0。

如果只记一个实验，应该记这个零目标动作数据结果，因为它最直接证明：  
**motion prior 真的把“从视频学到的东西”转成了可执行策略能力。**

#### 3. Ablation：机制不是拍脑袋，关键设计有因果支撑
几组消融和方法主张对得上：

- **局部窗口分类 > 坐标 MSE**  
  tokenizer 的 `ΔAUC 0.919 vs 0.883`
- **更长 inverse horizon 更好**  
  16-step 的下游成功率 `0.75`，高于 4-step 的 `0.36` 和 8-step 的 `0.64`
- **冻结轻量视觉编码器也够用**  
  ResNet-18 与更大视觉骨干差距很小，说明收益主要来自表示与分解策略，而不只是更大 backbone

这说明它的提升不只是“参数更多”或“训练更久”，而是接口设计本身在起作用。

#### 4. 额外外溢信号：它不只是控制接口，也是一个可复用 world-model 表示
作者还把 predicted motion tokens 接到 AVDC 视频预测模型上，结果：

- **PSNR：15.93 → 16.40**
- **SSIM：0.56 → 0.59**

这说明这些 token 不只是方便 action decoder，它们也携带了对未来动态有用的结构化信息。

### 局限性

- **Fails when:** 任务需要从 2D 轨迹中恢复精确 3D 接触几何、存在严重视角歧义、或同一视觉运动可对应多种控制动作时，inverse dynamics 会遇到天然多解问题；在随机环境里，外生扰动和 agent action 也难仅凭 state-to-state 变化区分。
- **Assumes:** 依赖高质量离线点跟踪预处理（文中使用 CoTracker），且这一预处理比普通 BC 更重；需要任务描述与一定量机器人交互/动作数据来训练 inverse model；当前实验主要基于较稳定视角与偏确定性的动力学，真实机器人设置还使用多视角静态 RGB 相机。论文文本给出了项目页，但未明确代码发布。
- **Not designed for:** 完全零机器人动作数据的控制学习、显式 3D/力学推理、以及强随机环境中的通用策略学习；它也不是为纯像素级长时视频生成或在线规划而设计的系统。

### 可复用组件

- **稠密点跟踪 + 速度化预处理**：把视频转成动作相关的运动信号
- **FSQ motion tokenizer**：把连续关键点运动压成离散 token
- **goal-conditioned forward / goal-free inverse split**：可插入不同数据源与动作头
- **motion-token conditioning interface**：不仅能接 policy，也能接 video generation 模块

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_AMPLIFY_Actionless_Motion_Priors_for_Robot_Learning_from_Videos.pdf]]