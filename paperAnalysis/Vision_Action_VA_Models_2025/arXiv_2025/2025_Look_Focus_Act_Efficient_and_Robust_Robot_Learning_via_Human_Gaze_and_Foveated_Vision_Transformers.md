---
title: "Look, Focus, Act: Efficient and Robust Robot Learning via Human Gaze and Foveated Vision Transformers"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - task/bimanual-manipulation
  - flow-matching
  - foveated-tokenization
  - gaze-supervision
  - dataset/GIAVA
  - dataset/AV-ALOHA
  - opensource/full
core_operator: "用人类 gaze 监督学出“看哪里”，再以预测注视点驱动中心高分辨率、周边低分辨率的 ViT patch 切分，并用流匹配策略生成机器人动作块"
primary_logic: |
  左目相机图像 + 机器人本体状态 + 示教期同步人类 gaze → 训练 gaze 预测器或联合 gaze-动作策略，并据预测 gaze 做 foveated tokenization → 以更少视觉 token 的 ViT/Q-Former/flow-matching 解码器输出 whole-body 动作块
claims:
  - "Claim 1: 采用 gaze 引导的 foveated tokenization 可将 ViT token 数从 324 降到 20、GFLOPs 从 1905.4 降到 115.6，并带来约 7x 训练提速和 3x 推理提速 [evidence: comparison]"
  - "Claim 2: 在带 MAE 预训练的标准仿真设定下，Fov-UNet 在 6 个任务中的 3 个取得最佳结果，包括 PourTestTube 92% 和 ThreadNeedle 92% [evidence: comparison]"
  - "Claim 3: 在含 distractor 的场景中，foveated 策略整体上比均匀 token baseline 更稳健，但 Fov-Act 在 HookPackage 上会因外围目标被低分辨率化而明显退化 [evidence: analysis]"
related_work_position:
  extends: "AV-ALOHA (Chuang et al. 2024)"
  competes_with: "Gaze-Based Dual Resolution Deep Imitation Learning (Kim et al. 2021); EyeRobot (Kerr et al. 2025)"
  complementary_to: "MAE (He et al. 2022); DINOv2 (Oquab et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Look_Focus_Act_Efficient_and_Robust_Robot_Learning_via_Human_Gaze_and_Foveated_Vision_Transformers.pdf
category: Embodied_AI
---

# Look, Focus, Act: Efficient and Robust Robot Learning via Human Gaze and Foveated Vision Transformers

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.15833), [Project](https://ian-chuang.github.io/gaze-av-aloha/)
> - **Summary**: 论文把人类注视点当作机器人视觉的监督先验，用 gaze 引导的 foveated ViT 代替全图均匀编码，在更低计算开销下提升了操作策略对干扰物的鲁棒性，并在部分高精度任务上带来成功率增益。
> - **Key Performance**: ViT token 数从 324 降到 20，GFLOPs 从 1905.4 降到 115.6（约 -94%）；训练速度约提升 7x、推理约提升 3x。

> [!info] **Agent Summary**
> - **task_path**: 左目相机图像 + 机器人本体状态 + 主动视觉设定 -> 16 步机器人动作块（可选同时预测下一时刻 gaze）
> - **bottleneck**: 机器人 IL 里对整张图均匀高分辨率编码既昂贵又容易把表示预算浪费在无关背景上
> - **mechanism_delta**: 把 human gaze 变成 token 分配控制信号，让 ViT 只在注视点附近保留高分辨率 patch、周边区域粗编码
> - **evidence_signal**: 6 个仿真任务和 2 个真实任务上，20-token 的 foveated 策略通常与 324-token 的 Fine baseline 持平或更好，且 distractor 场景更稳健
> - **reusable_ops**: [image-id gaze synchronization, gaze-guided foveated tokenization]
> - **failure_modes**: [small peripheral target after gaze shift can confuse Fov-Act, real-robot gains are modest on low-data high-precision insertion]
> - **open_questions**: [which manipulation regimes benefit most from foveation, how to reuse large-scale pretrained ViTs under nonuniform tokenization]

## Part I：问题与挑战

这篇论文真正想解决的，不是“机器人能不能看”，而是“机器人是否把算力花在了真正该看的地方”。

### 1. 真正瓶颈是什么
现有机器人 imitation learning 常见做法是：把整张相机图像均匀切成 patch，交给 ViT，再和本体状态一起回归动作。问题在于：

- **计算瓶颈**：ViT 的自注意力复杂度随 token 数近似二次增长。对操作任务而言，全图细粒度编码成本很高。
- **信息瓶颈**：很多操作任务真正决定动作的，只是局部接触区、孔位、抓取点、倒液边缘等少量区域。
- **泛化瓶颈**：全图均匀处理会给背景杂物与目标区域相同的建模预算，容易学到与任务无关的视觉相关性。
- **部署瓶颈**：训练时可以拿到人的 gaze，但测试时拿不到，因此系统必须自己学会“看哪里”。

### 2. 为什么现在值得做
作者认为时机成熟有两个原因：

1. **VR 头显自带眼动追踪**，使“示教动作 + gaze + 视角控制”的同步采集变得可行。
2. **ViT 已成为机器人视觉骨干**，其高 token 成本让“非均匀分配视觉分辨率”变得特别有价值。

### 3. 输入/输出接口与边界条件
- **训练输入**：GIAVA 平台采集的左目图像、机器人本体状态、同步的人类 gaze、机器人动作示教。
- **测试输入**：只有图像和本体状态；gaze 需要由模型内部估计。
- **输出目标**：长度为 16 的动作 chunk；在端到端版本里，gaze 也被当作动作空间的一部分联合预测。
- **边界条件**：
  - 面向的是 **桌面操作/双臂操作 + 主动视觉** 场景；
  - 策略实际只用 **左目相机图像** 做决策，立体视觉主要用于人类遥操作反馈；
  - 依赖可移动相机臂和眼动采集硬件，不是“任意机器人直接即插即用”。

---

## Part II：方法与洞察

方法可以概括成一句话：**先学会“看哪里”，再把计算集中在那个地方附近。**

### 1. GIAVA：把 gaze、视角和操作动作一起采进来
作者在 AV-ALOHA 基础上扩展出 GIAVA：

- 两只机械臂负责操作；
- 第三只 7-DoF active-vision 机械臂模拟人的头/颈移动来调相机视角；
- 人类操作者用 VR 头显和控制器同时控制视角与双臂；
- Meta Quest Pro 提供眼动追踪；
- 通过给相机帧打 **image ID**，把 gaze 与图像严格同步，缓解传输延迟带来的错位；
- 对来不及标注的 gaze 帧做插值。

这一步的价值是：论文不是只在算法上“假设有 gaze”，而是把 **可采集、可对齐、可训练** 的系统闭环补齐了。

### 2. 策略骨干：ViT 编码 + Q-Former 压缩 + Flow Matching 输出动作
整体策略结构很标准但搭配合理：

- 图像先经 **ViT** 编码；
- 再用 **Q-Former** 把大量视觉 token 压缩成少量条件 token；
- 本体状态用 MLP 编码；
- 最后由一个 transformer 式的 **flow-matching action decoder** 输出动作 chunk。

这里 flow matching 的作用不是本文创新重点，重点是它提供了一个稳定的动作生成器，使作者能把实验焦点放在“视觉 token 怎么分配”上。

### 3. 两种 gaze 估计路线
因为测试时没有真人 gaze，作者比较了两条路：

#### A. Fov-UNet：两阶段
- 先用低分辨率图像通过 UNet 预测 gaze heatmap；
- 再通过 spatial softmax 得到 gaze 点；
- 用该 gaze 做 foveated tokenization；
- 同时把 gaze 追加到策略条件里。

**优点**：有显式空间归纳偏置，预测更稳。  
**缺点**：多一套 gaze 网络，训练和推理更重。

#### B. Fov-Act：端到端
- 直接把 gaze 当作动作空间一部分；
- 策略一次性联合预测未来 gaze 和机器人动作；
- 下一时刻的 foveation 由上一时刻预测的 gaze 驱动。

**优点**：结构更简洁、部署链路更短。  
**缺点**：没有 heatmap 级空间偏置，而且“根据当前帧预测未来 gaze”更容易漂。

### 4. Foveated tokenization：核心视觉改动
作者把来自图像分割工作的 foveated tokenization 改造成机器人视觉版本：

- 注视点附近：**小 patch、密集采样**
- 周边区域：**大 patch、稀疏采样**
- 先把图像平移，使预测 gaze 对齐到 foveation pattern 中心
- 超出边界部分零填充
- 最后把不同大小 patch 统一缩放后送入标准 ViT

对比基线：
- **Fine**：18×18 的均匀细网格，共 324 token
- **Coarse**：4×5 的均匀粗网格，共 20 token
- **Foveated**：同样 20 token，但把分辨率预算集中在 gaze 附近

一个很现实的问题是：**非均匀 tokenization 不能直接复用常见开源 ViT 预训练权重**。因此作者为 Fine / Coarse / Foveated 三种 patch 方案都分别做了 MAE 预训练，以保证比较公平。

### 核心直觉

**变化点**：从“全图均匀切 patch”改成“由 gaze 决定 token 密度分布”。  

**改变了什么约束**：
- 原先的约束：所有区域共享同等分辨率预算；
- 现在的约束：高分辨率预算只分给最可能与任务相关的局部区域，外围只保留粗上下文。

**为什么这会带来能力变化**：
1. **降低计算**：token 数骤减，ViT 的 pairwise attention 成本同步下降。
2. **降低干扰耦合**：背景 distractor 不再和目标区域争夺同等表示容量。
3. **保留关键精度**：高精度操作真正需要的是局部几何细节，而不是整张图都高精度。
4. **利用人类先验**：human gaze 不是普通辅助信号，而是“任务相关性”的直接监督。

换句话说，作者引入的关键 causal knob 不是“再堆一个更强模型”，而是**重新分配视觉表示预算**。

### 策略权衡表

| 方案 | 改了什么 | 主要收益 | 主要代价/风险 |
|---|---|---|---|
| Fine | 324 个均匀细 patch | 细节最全，最保守 | 计算和显存最贵，容易处理过多无关背景 |
| Coarse | 20 个均匀粗 patch | 很省算力 | 局部精度不足，细操作容易丢信息 |
| Fov-UNet | 先预测 gaze，再做 20-token foveation | 兼顾精度、鲁棒性和效率，整体最稳 | 需要额外 UNet，系统更复杂 |
| Fov-Act | 联合预测 gaze 和动作 | 端到端、效率更好 | gaze 更易漂移，外围小目标切换场景更脆弱 |

---

## Part III：证据与局限

### 1. 关键证据信号

#### 信号 A：效率不是小修小补，而是量级变化
最直接的证据来自 token 数和 ViT 计算：

- **324 token → 20 token**
- **GFLOPs: 1905.4 → 115.6**
- 训练延迟从 **833.2 ms/step** 降到 **123.8 ms/step（Fov-UNet）** 或 **108.2 ms/step（Fov-Act）**
- 推理 chunk 延迟从 **334.7 ms** 降到 **105.7 / 87.9 ms**

这说明作者不是靠“轻微剪枝”换来一点提速，而是把视觉主瓶颈直接改写了。

#### 信号 B：省算力的同时，没有明显牺牲操作能力
在标准仿真设置下，Fov-UNet 不只是“勉强不掉点”，而是在若干精细任务上更好：

- MAE 预训练下，**PourTestTube 92% vs Fine 68%**
- **ThreadNeedle 92% vs Fine 84%**
- **SlotInsertion 70% vs Fine 57%**

这支持一个关键结论：**对于高精度操作，真正重要的是局部高质量视觉，而不是全图都细。**

#### 信号 C：foveation 的真正价值在干扰场景更明显
作者专门构造了 distractor 场景。整体趋势是：

- foveated 策略在 cluttered/unseen distractor 环境下通常比均匀 token baseline 更稳；
- 真实机器人 Ball 任务加 distractor 时，**Fov-UNet 56% vs Fine 48%**；
- 仿真中的 ThreadNeedle、PegInsertion 等任务也显示出 foveated 版本更抗干扰。

这说明 gaze 不是只做“计算压缩”，还在做 **attention regularization**：让策略少看无关物体。

#### 信号 D：两阶段 generally 胜过端到端
Fov-Act 更简洁，但 Fov-UNet 大多更稳。论文解释也合理：

- Fov-UNet 用 heatmap 预测 gaze，有更强空间结构先验；
- Fov-Act 直接回归未来 gaze，在动态或视线切换任务上更容易错。

这个差异在 **HookPackage** 上最明显：目标小钩子位于外围，需要从包装盒切到周边小目标时，Fov-Act 因外围被粗编码而更难重新锁定目标。

### 2. So what：相对以往工作的能力跃迁在哪里
相对传统 gaze-for-robotics 工作，这篇论文的跃迁不只是“把 gaze 当额外输入”，而是把 gaze 变成了：

- **视觉分辨率分配器**
- **ViT token budget 控制器**
- **抗干扰的结构性先验**

所以它带来的不是单一维度提升，而是一个更有工程意义的组合收益：  
**更低算力 + 更强鲁棒性 + 某些精细任务上的更高成功率。**

### 3. 局限性

- **Fails when**: 任务要求把注意力从当前中心快速切换到外围小目标时，foveated 周边低分辨率会放大 gaze 误差，Fov-Act 在 HookPackage 上就是典型例子；真实机器人上对极高精度、低数据量插入任务（如 Toothbrush）提升有限，甚至略逊于 Fine。
- **Assumes**: 训练期需要同步的人类 gaze 标注、VR 遥操作、主动视觉机械臂与相机-眼动同步链路；非均匀 tokenization 还要求单独做 MAE 预训练，不能直接无缝复用标准 ViT 预训练权重。
- **Not designed for**: 无 gaze 监督的纯离线模仿学习、没有主动视角控制的通用机器人平台、开放环境中的大范围视觉搜索任务。

### 4. 资源与可复用组件
这篇工作的可复用价值很高，尤其在系统层面：

- **可复用组件**
  - 图像帧 ID 驱动的 **gaze-image 同步机制**
  - 可插拔的 **gaze-guided foveated tokenization**
  - 两种 gaze 学习 recipe：**两阶段** 与 **联合 gaze-action**
  - 带 distractor 的操作评测协议
- **主要依赖**
  - Meta Quest Pro 级别的眼动硬件
  - 带主动视觉臂的 AV-ALOHA/GIAVA 类平台
  - 为非均匀 patch 模式额外做视觉预训练的计算预算

整体看，这篇论文最值得记住的点是：**它把“人类看哪里”变成了机器人“把算力放哪里”的可学习先验。** 这比单纯加一个注意力模块更本质，因为它直接改写了视觉编码的资源分配方式。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Look_Focus_Act_Efficient_and_Robust_Robot_Learning_via_Human_Gaze_and_Foveated_Vision_Transformers.pdf]]