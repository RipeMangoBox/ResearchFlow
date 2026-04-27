---
title: "KineDex: Learning Tactile-Informed Visuomotor Policies via Kinesthetic Teaching for Dexterous Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/dexterous-manipulation
  - task/contact-rich-manipulation
  - diffusion
  - kinesthetic-teaching
  - force-control
  - dataset/KineDex
  - opensource/no
core_operator: "用手把手动觉示教直接采集带触觉演示，并让策略联合预测关节位置与指尖目标力，再通过虚拟位移式力控执行接触操作"
primary_logic: |
  手把手示教采集的双视角RGB/触觉/本体状态 → 人体遮挡分割与视频修补、触觉增强扩散策略学习、联合预测关节目标与指尖法向力 → 通过力控稳定执行接触密集型灵巧操作
claims:
  - "KineDex在9个接触密集灵巧操作任务上平均成功率达到74.4%，相比去掉力控的变体提升57.7个百分点 [evidence: comparison]"
  - "在Cap Twisting、Toothpaste Squeezing和Syringe Pressing三类高接触反馈任务上，引入触觉输入将平均成功率从35.0%提升到61.7% [evidence: ablation]"
  - "在5个数据采集对比任务上，KineDex示教成功率约98%，明显高于teleoperation的39%，且采集速度超过2倍 [evidence: comparison]"
related_work_position:
  extends: "DexForce (Chen et al. 2025)"
  competes_with: "DexForce (Chen et al. 2025); Open-TeleVision (Cheng et al. 2024)"
  complementary_to: "3D Diffusion Policy (Ze et al. 2024); Dexterity from Touch (Guzey et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_KineDex_Learning_Tactile_Informed_Visuomotor_Policies_via_Kinesthetic_Teaching_for_Dexterous_Manipulation.pdf
category: Embodied_AI
---

# KineDex: Learning Tactile-Informed Visuomotor Policies via Kinesthetic Teaching for Dexterous Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.01974) · [Project](https://dinomini00.github.io/KineDex/)
> - **Summary**: 这篇工作把灵巧手示范从“远程重定向”改成“手把手直接带动机器人手”，从而采到真实触觉/力信息，再训练一个同时输出位置与指尖力的多模态策略来完成接触密集型灵巧操作。
> - **Key Performance**: 9任务平均成功率 74.4%；相对无力控变体提升 57.7 个百分点，示教采集成功率约 98% vs teleoperation 的 39%

> [!info] **Agent Summary**
> - **task_path**: 双视角RGB+五指触觉阵列+本体状态 / 接触密集灵巧操作 -> 末端位姿+手部关节目标+5个指尖法向力目标
> - **bottleneck**: 高保真、机器人原生的触觉示范难以采集；同时 kinesthetic 视频里的操作者遮挡会造成训练-部署视觉分布错位
> - **mechanism_delta**: 用 hand-over-hand 示教替代 teleoperation 重定向，并把动作从“只预测位置”改成“位置+指尖力”，再用虚拟位移式力控把期望接触力落实到执行层
> - **evidence_signal**: 去掉力控后9任务平均成功率从74.4%降到16.7%，去掉inpainting则全任务为0
> - **reusable_ops**: [human-occlusion inpainting pipeline, position-plus-force action head]
> - **failure_modes**: [severe occlusion degrades inpainting quality, thumb morphology mismatch requires two-handed teaching]
> - **open_questions**: [can more biomimetic hands enable single-hand kinesthetic teaching, can richer 3D/contact representations improve hard insertion and squeezing tasks]

## Part I：问题与挑战

**What / Why：真正的瓶颈不是“策略网络不够强”，而是“示范里的接触信息在进入机器人学习链条时被丢了”。**

现有灵巧手示范采集常走两条路：

1. **teleoperation / retargeting**：把人手动作映射到机器人手。  
   问题是人手和机器人手存在运动学失配，很多接触动作最后只剩“姿态像”，没有“受力对”。
2. **视频到机器人动作**：更容易扩展，但同样缺少机器人执行时的真实触觉闭环。

对这篇论文关心的任务——如拧瓶盖、挤牙膏、按压注射器、插充电头——**“碰到物体”不够，关键是“持续施加合适的力”**。  
如果示范里没有高保真触觉/力信息，策略即使学会了手指位置，也可能只是在物体表面“轻轻碰一下”，无法稳定完成操作。

### 输入/输出接口

- **输入观察**：
  - 双视角 RGB（前视 + 腕部）
  - 五指触觉阵列
  - 本体状态（机械臂末端位姿 + 手部关节）
- **输出动作**：
  - 末端位姿
  - 灵巧手关节目标
  - 每个指尖的目标法向力

### 边界条件

这不是一个通用机器人系统报告，而是一个**单臂 + 单灵巧手 + 接触密集型 manipulation**方法：

- 依赖指尖触觉传感器
- 主要对**法向接触力**建模
- 任务以单手日常物体操作为主
- 不覆盖双手协作示教

### 为什么现在值得做

这件事之所以现在可行，靠的是三件事同时成熟：

- **更拟人化的灵巧手硬件**：允许 hand-over-hand 直接示教
- **视频修补/inpainting 模型**：能把示教时的人手遮挡从训练图像里移除
- **Diffusion Policy 一类多模态 imitation backbone**：能吃下视觉、触觉和本体状态的联合输入

---

## Part II：方法与洞察

KineDex 的核心不是单点创新，而是把**数据采集、观测分布对齐、动作语义、执行控制**四个环节连成闭环。

### 方法主线

#### 1. 用 kinesthetic teaching 直接采集“机器人原生”触觉示范

作者让操作者直接“戴着/带着”机器人手做动作：

- 四个非拇指通过背部环形带与操作者手指耦合
- 由于机器人拇指与人拇指形态不完全匹配，拇指由另一只手单独控制

这样做的收益很直接：

- **消除 retargeting 误差**
- 操作者能直接感受到接触力
- 采到的是机器人手真实接触下的触觉与运动数据

采集的数据包括：

- 前视和腕视 RGB
- 机械臂与手部本体状态
- 五指稠密触觉
- 由触觉点聚合得到的指尖 3D 力

#### 2. 用 inpainting 解决“训练看到人，部署看不到人”的视觉 OOD

kinesthetic teaching 有个明显副作用：  
前视相机会拍到操作者身体/手臂，如果直接训练，部署时场景里没这个人，视觉分布就错了。

作者的处理方式是：

- 先用 **Grounded-SAM** 分割人体区域
- 再用 **ProPainter** 对遮挡区域做视频修补

这一步不是锦上添花，而是必要条件。论文里 **w/o Inpainting 全任务 0 成功**，说明原始 kinesthetic 视频不能直接拿来训 visuomotor policy。

#### 3. 用触觉增强的 Diffusion Policy 学“位置 + 力”

策略 backbone 采用 Diffusion Policy，但把输入和输出都改了：

- **输入**：视觉 + 触觉 + 本体状态
- **输出**：不仅预测关节/末端目标，还预测**每个指尖的目标法向力**

这里的关键变化是：  
模型不再只回答“手该到哪里”，而是同时回答“接触后该施多大力”。

作者只监督法向力分量，原因也很工程化：当前手指的主动施力主要沿这个方向更稳定可控。

#### 4. 用 force-informed control 把预测力转成真实接触

这是整篇论文最关键的执行层设计。

问题在于：  
单纯位置控制时，手指到达物体表面后，若目标位置不再继续“往里压”，控制器就不会持续产生足够接触力。于是会出现：

- 抓不紧
- 拧不动
- 挤不出
- 一碰就滑

作者的解法可以理解为：

- 让策略先预测“我希望这个指尖施加多大力”
- 再把这个期望力转成一个**虚拟位移**
- 控制器据此把目标位置“推”进物体一点，从而借由物体反作用力形成稳定压力

所以 KineDex 的动作不是普通 position action，而是 **force-informed action**。

### 核心直觉

KineDex 真正改变的是三个“分布/约束”：

1. **示范分布变了**：  
   从“人手动作重定向到机器人”  
   变成“机器人手直接被人带动完成接触”。  
   结果是示范天然落在机器人可执行的接触分布上。

2. **观测分布变了**：  
   从“训练时总有人的遮挡、部署时没有”  
   变成“训练和部署都尽量是机器人视角的干净画面”。  
   结果是视觉 policy 不再学到错误的上下文依赖。

3. **动作语义变了**：  
   从“到哪个姿态”  
   变成“到哪个姿态 + 保持多大接触力”。  
   结果是策略从几何模仿升级为交互模仿，能做挤压、旋拧、按压、插入这类接触密集任务。

更直白地说：

> 这篇论文不是单纯给 policy 多加了个 tactile token，  
> 而是把 imitation learning 的学习目标从“轨迹复现”改成了“接触状态复现”。

### 为什么这个设计有效

- **kinesthetic teaching** 让示范直接在机器人本体上发生，因此保留了真实接触结果
- **tactile input** 弥补视觉看不到、或被遮挡的局部接触状态
- **force prediction + force control** 让策略输出真正能影响物理交互质量的变量
- **inpainting** 消除了训练/部署视觉失配这个致命噪声源

### 战略权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| Hand-over-hand kinesthetic teaching | retargeting 失配、无直接触感 | 高保真触觉示范、采集更快 | 需要操作者直接接触机器人，依赖硬件形态 |
| Inpainting 预处理 | 人体遮挡造成视觉 OOD | raw demo 可用于训练 | 严重遮挡时修补质量可能下降 |
| 触觉增强输入 | 视觉难以感知局部接触状态 | 旋拧/挤压/按压更稳 | 需要高分辨率触觉硬件与校准 |
| 位置+力联合动作 | 位置动作无法表达接触强度 | 可持续施压、减小打滑/接触失败 | 依赖力到位移的固定映射增益，泛化边界有限 |

---

## Part III：证据与局限

### 关键证据信号

- **信号 1｜多任务比较**  
  在 9 个接触密集任务上，KineDex 平均成功率 **74.4%**。这说明该方法不只适用于简单抓取，还能覆盖插入、旋拧、挤压、按压等更依赖接触质量的任务。

- **信号 2｜力控消融最强**  
  去掉 force control 后，平均成功率直接掉到 **16.7%**。  
  这基本证明：该方法的能力跃迁不在 backbone，而在“策略预测的力是否能被执行层真实落地”。

- **信号 3｜触觉的作用是任务选择性的**  
  对简单 pick-and-place，去掉 tactile 只会中等幅度下降；  
  但在 **Cap Twisting / Toothpaste Squeezing / Syringe Pressing** 这些高接触反馈任务上，平均下降 **26.7 个百分点**。  
  结论很清楚：触觉主要补的是视觉难以可靠恢复的接触状态，而不是所有任务都等权需要。

- **信号 4｜视觉 OOD 是硬门槛**  
  不做 inpainting 时，**全任务成功率为 0**。  
  这说明 raw kinesthetic demo 不能直接端到端喂给 visuomotor policy，观测分布对齐是必要步骤，不是工程细节。

- **信号 5｜数据采集效率比较**  
  在 5 个对比任务上，KineDex 示教成功率约 **98%**，teleoperation 只有 **39%**；同时采集速度超过 **2 倍**。  
  小规模用户研究也支持这一点：所有参与者都认为 KineDex 更适合采触觉数据和复杂任务，80% 认为更易用。

### So what：相对以往方法，能力跳跃在哪里？

这篇论文最有价值的地方，是把灵巧操作里的两个常见断点同时补上了：

1. **数据侧断点**：以前拿不到高保真的机器人原生触觉示范  
2. **控制侧断点**：以前即使学到“该用力”，控制器也未必真能施出这个力

KineDex 用 kinesthetic teaching 补第一个断点，用 force-informed control 补第二个断点。  
因此它的提升不是“更会看图了”，而是**更会维持物理接触了**。

### 局限性

- **Fails when**: 遮挡严重到 inpainting 无法稳定恢复关键区域时，训练数据质量会明显下降；同时在更依赖精细定位与接触推理的任务上，当前表示仍不足，比如 Charger Plugging、Toothpaste Squeezing 一类难任务成功率仍明显低于简单抓取。
- **Assumes**: 需要带高分辨率指尖触觉的灵巧手、双相机、机器人本体状态同步采集，以及 Grounded-SAM + ProPainter 的预处理链路；每个任务大约需要 100–150 条示范、训练 500 epochs；执行阶段还假设“目标力 → 虚拟位移”的固定增益在不同任务间可共用。
- **Not designed for**: 双手协作示教、无触觉硬件的平台、直接在含人体遮挡的原始视频上训练、以及当前形态下的单手完成拇指全控制示教。作者也明确指出，由于拇指形态失配，目前一个机器人手通常需要操作者双手共同带动。

### 复现与可扩展性备注

- 论文有项目页，但正文里**没有清晰声明代码/数据已发布**，因此复现性应保守看待。
- 真正影响复现门槛的不是 Diffusion Policy 本身，而是：
  - 触觉硬件
  - kinesthetic teaching 机械接口
  - 人体遮挡分割与视频修补流程
  - 力控增益调参与系统集成

### 可复用组件

- **人体遮挡 demo 清洗链路**：`Grounded-SAM mask -> video inpainting`
- **动作语义升级模板**：把 imitation policy 的输出从“位置”扩成“位置 + 接触力”
- **执行层接口**：用虚拟位移把期望力接到现有位置控制器上
- **任务适用面**：尤其适合挤压、旋拧、按压、插入等“接触质量比轨迹几何更关键”的任务

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_KineDex_Learning_Tactile_Informed_Visuomotor_Policies_via_Kinesthetic_Teaching_for_Dexterous_Manipulation.pdf]]