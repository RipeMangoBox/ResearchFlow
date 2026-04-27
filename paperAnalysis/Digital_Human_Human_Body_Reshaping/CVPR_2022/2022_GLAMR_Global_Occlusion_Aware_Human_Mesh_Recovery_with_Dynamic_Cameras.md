---
title: "GLAMR: Global Occlusion-Aware Human Mesh Recovery with Dynamic Cameras"
venue: CVPR
year: 2022
tags:
  - Others
  - task/human-mesh-recovery
  - task/video-understanding
  - conditional-vae
  - autoregressive-modeling
  - global-optimization
  - dataset/AMASS
  - dataset/3DPW
  - "dataset/Dynamic Human3.6M"
  - repr/SMPL
  - opensource/partial
core_operator: 先用生成式动作补全恢复长期遮挡下的局部人体运动，再用局部动作驱动的自中心全局轨迹预测作为锚点，联合优化相机与人体轨迹。
primary_logic: |
  动态相机单目视频 + MOT/ReID 与相机坐标系初始 SMPL 估计
  → Transformer-CVAE 自回归补全缺失 body pose，LSTM-CVAE 从局部动作预测自中心全局轨迹
  → 以预测轨迹为锚点联合优化相机外参与人体全局轨迹
  → 输出跨长期遮挡、全局坐标一致的 SMPL 网格序列
claims:
  - "在 Dynamic Human3.6M 上，GLAMR w/ KAMA 将 G-MPJPE / G-PVE 从最佳非 GLAMR 基线 1318.1 / 1330.3 降到 806.2 / 824.1 [evidence: comparison]"
  - "在 AMASS 动作补全基准上，所提 motion infiller 将 sampled FID 从 ConvAE 的 31.4 降到 16.7，并把 reconstructed PA-MPJPE 从 72.8 降到 36.1 [evidence: comparison]"
  - "去掉 trajectory predictor 会使 Dynamic Human3.6M 上的 G-MPJPE 从 806.2 恶化到 1750.8，说明预测轨迹锚点对全局恢复是关键 [evidence: ablation]"
related_work_position:
  extends: "KAMA (Iqbal et al. 2021)"
  competes_with: "ConvAE (Kaufmann et al. 2020); OpenSfM"
  complementary_to: "COLMAP (Schonberger and Frahm 2016); NASA (Deng et al. 2020)"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Human_Body_Reshaping/CVPR_2022/2022_GLAMR_Global_Occlusion_Aware_Human_Mesh_Recovery_with_Dynamic_Cameras.pdf
category: Others
---

# GLAMR: Global Occlusion-Aware Human Mesh Recovery with Dynamic Cameras

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2112.01524) · [Project](https://nvlabs.github.io/GLAMR)
> - **Summary**: 这篇论文面向动态相机单目视频中的人体恢复，核心做法是先用生成式模型补全长期遮挡造成的局部动作缺口，再从局部动作预测全局轨迹，并把该轨迹当作锚点去联合优化相机与人体，从而得到全局一致的 SMPL 网格序列。
> - **Key Performance**: Dynamic Human3.6M 上 GLAMR w/ KAMA 达到 G-MPJPE 806.2、G-PVE 824.1；AMASS 动作补全上 sampled FID 16.7。

> [!info] **Agent Summary**
> - **task_path**: 单目动态相机视频 -> 跨遮挡的全局坐标 SMPL mesh/pose 序列
> - **bottleneck**: 长时完全遮挡导致 body motion 缺失，同时动态相机下人体全局轨迹与相机位姿联合估计欠约束
> - **mechanism_delta**: 用“局部动作补全 + 从局部动作预测自中心全局轨迹 + 轨迹锚点联合优化”替代“可见帧直接回归 + SLAM 对齐”
> - **evidence_signal**: Dynamic Human3.6M 上 G-MPJPE 从最佳非 GLAMR 基线 1318.1 降至 806.2，且去掉 trajectory predictor 后退化到 1750.8
> - **reusable_ops**: [occlusion-conditioned autoregressive motion infilling, egocentric-trajectory anchored camera-human joint optimization]
> - **failure_modes**: [tracking或2D关键点错误级联传播, 多人强交互与细粒度形体细节无法被当前独立SMPL建模充分覆盖]
> - **open_questions**: [能否端到端联合训练减少多阶段误差传播, 能否引入场景与多人交互约束提升长时遮挡恢复]

## Part I：问题与挑战

这篇论文解决的是一个很具体但过去被低估的问题：**给定动态相机拍摄的单目视频，恢复人物在统一全局坐标系中的 3D SMPL 网格序列，即使人物会长期被遮挡、漏检，甚至完全走出视野。**

### 任务接口
- **输入**：动态相机视频；人物跟踪与 re-ID 结果；可见帧上的初始人体 mesh 估计；2D 关键点。
- **输出**：每个人在统一世界坐标中的 SMPL 序列，包括全局平移、全局朝向、body pose 和 shape。

### 真正的瓶颈
真正难点不是“某一帧 3D pose 回归不准”，而是两个耦合瓶颈：

1. **遮挡导致的观测断裂**
   - 常规 human mesh recovery 依赖可见人体或检测框。
   - 一旦出现**完全遮挡、漏检、出画**，序列中间会直接断掉。
   - 这不是小噪声问题，而是**整段 latent motion 缺失**问题。

2. **动态相机下的人体-相机联合歧义**
   - 现有很多方法只在相机坐标或 root-relative 坐标里恢复人体。
   - 对于动态相机，想得到全局轨迹，就必须同时知道**人体怎么动**和**相机怎么动**。
   - 只靠 2D 投影去同时解这两者，本质上是**欠约束**的。

### 为什么现在值得解
作者的判断很清楚：现在已经有两类基础设施可用，足以把这个问题从“不可能”推到“可做”：
- 上游已有 KAMA / SPEC 这类能给出**相机坐标系绝对人体**的估计器；
- AMASS 提供了足够强的人体动态先验，可支撑**长时动作补全**与**局部动作到全局轨迹的学习**。

### 边界条件
这篇方法并不是在任意条件下都能稳定工作，它隐含了几个前提：
- 至少要有一部分可见片段给 motion infiller 提供 context / look-ahead；
- 人体运动模式不能严重偏离 AMASS 学到的动作分布；
- 需要可靠的 MOT / re-ID / 2D keypoints 作为上游观测支撑。

---

## Part II：方法与洞察

### 方法主线

GLAMR 的整体逻辑可以理解为四段式流水线：

1. **Stage I：先拿到“有缺口的相机坐标人体序列”**
   - 用 MOT + re-ID 得到人物轨迹。
   - 用 KAMA 或 SPEC 在可见帧估计人体 pose/shape/translation。
   - 得到的是**相机坐标下、带遮挡缺口**的序列。

2. **Stage II：补 body motion，不碰 root trajectory**
   - 作者没有直接补全全局轨迹，而是只补全 body pose。
   - 原因很关键：root trajectory 在动态相机场景中混合了**人动**与**相机动**，直接学容易把两者纠缠。
   - 因此先把“局部人体运动”恢复干净，再把全局问题留到后面。

3. **Stage III：从局部动作预测全局轨迹**
   - 作者观察到：一个人的**局部肢体动作**，尤其是步态、身体朝向变化，与其**全局平移/朝向**高度相关。
   - 所以他们训练一个 trajectory predictor，从补全后的 body motion 直接预测全局 trajectory。

4. **Stage IV：联合优化人体与相机**
   - 预测轨迹本身会有漂移，也不一定完全对齐视频证据。
   - 因此再做一步全局优化，把：
     - 2D 关键点重投影，
     - 初始相机坐标轨迹的一致性，
     - 预测轨迹的 regularization，
     - 相机平滑/直立约束，
     - 多人穿插惩罚
     
     合在一起，最终同时校正人体与相机。

### 核心直觉

**这篇论文真正改变的，不是单个网络结构，而是问题的约束方式。**

#### 1) 从“缺观测”变成“有运动先验的补全”
- **What changed**：把长时遮挡帧视为动作补全问题，而不是逐帧回归失败帧。
- **Constraint changed**：遮挡区间不再是自由变量，而被 AMASS 学到的人体动态 manifold 约束。
- **Capability changed**：人物即便完全看不见、甚至走出 FoV，系统也能继续生成连续合理的局部动作。

#### 2) 从“纯几何欠约束”变成“有轨迹锚点的联合重建”
- **What changed**：不再依赖 SLAM 先给世界坐标，而是从 body motion 预测一条全局 trajectory 作为 anchor。
- **Constraint changed**：人体轨迹与相机外参的联合优化不再只靠 2D 投影，而是多了一个 learned trajectory prior。
- **Capability changed**：在动态相机且 SLAM 不可靠时，仍能恢复全局一致的人体序列。

#### 3) 为什么这种设计有效
- **局部动作能泄露全局意图**：步态频率、左右摆动、躯干转向，都会提供行进方向和速度线索。
- **自中心轨迹表示降低学习难度**：网络预测的不是“世界坐标绝对位置”，而是每一步相对 heading 的位移与朝向变化，分布更稳定。
- **在 egocentric space 优化更有传播性**：某一帧的修正可以自然影响后续所有帧，因此即使中间帧不可见，也能通过未来可见帧反向校正。

### 关键模块拆解

#### A. 生成式 Motion Infiller
- 基于 **Transformer + CVAE**。
- 只编码**可见帧**，不对缺失帧做无意义 padding。
- 测试时采用**自回归滑窗补全**：
  - 前面一段作为 context；
  - 后面一段作为 look-ahead；
  - 中间缺失段被填上。
- look-ahead 的作用很关键：它让生成的结束状态能与后文可见姿态接上，减少断裂。

#### B. Global Trajectory Predictor
- 基于 **LSTM + CVAE**。
- 输入不是原始角度，而是由 SMPL 得到的局部 3D 关节运动。
- 输出的是**egocentric trajectory**，再累计回 global trajectory。
- 作者实验证明：  
  直接回归 6-DoF 全局轨迹，或者把 LSTM 换成 Transformer，都会更差。

#### C. Global Optimization
这一步不是“再调一调”，而是把整套方法闭环的关键：
- 用 2D keypoints 保证投影与视频一致；
- 用上游估计轨迹保证和原始观测不要偏离太多；
- 用 predicted trajectory 作先验锚点防止解飘走；
- 用相机平滑/直立约束减少相机估计的不稳定。

### 策略权衡

| 设计选择 | 解决的问题 | 收益 | 代价/风险 |
|---|---|---|---|
| 只补全 body motion，不直接补全 camera-coordinate root trajectory | 避免把人体运动与相机运动混在一起学 | 局部动作补全更稳定 | 需要额外 trajectory predictor |
| 自回归滑窗 + look-ahead | 支持长时缺失，并保持前后连续 | 能补全长遮挡段 | 离线推理，窗口过长时 latent 负担增大 |
| egocentric trajectory 表示 | 降低长期预测的绝对偏移难度 | 泛化更好、优化能向后传播 | 起点位置/heading 仍需后续优化求解 |
| 轨迹锚点 + 相机联合优化 | 缓解动态相机下的欠约束 | 得到全局一致解 | 依赖 keypoints、初始化与优化稳定性 |

---

## Part III：证据与局限

### 关键证据

- **信号类型：comparison｜结论：GLAMR 真正提升的是“全局一致性”，不是只把局部 pose 修一修。**  
  在 Dynamic Human3.6M 上，GLAMR w/ KAMA 的 **G-MPJPE / G-PVE = 806.2 / 824.1**，明显优于最佳非 GLAMR 基线的 **1318.1 / 1330.3**。这说明它确实解决了动态相机下的 world-coordinate 重建问题。

- **信号类型：comparison｜结论：动作补全模块本身就比现有 motion infilling 更强。**  
  在 AMASS 上，作者的 motion infiller 把 sampled **FID 从 31.4 降到 16.7**，reconstructed PA-MPJPE 从 72.8 降到 36.1。说明它补出来的不只是“能接上”，而是更像真实人类动作。

- **信号类型：ablation｜结论：轨迹锚点是因果核心，不是附加 trick。**  
  去掉 trajectory predictor，Dynamic Human3.6M 上 G-MPJPE 会从 806.2 恶化到 1750.8；不用 egocentric trajectory 做优化也会退化到 877.3。最关键的旋钮，就是“用局部动作预测并约束全局轨迹”。

- **信号类型：comparison｜结论：全局优化对多人空间关系也有效。**  
  在 3DPW 上，全局优化把相对平移误差从 1.92 m 降到 0.66 m，相对旋转误差从 1.07 降到 0.30。这表明 joint optimization 不只是让单人更稳，也改善了多人之间的相对布局。

### 局限性

- **Fails when**: 上游 tracking / re-ID 出现 ID switch；2D 关键点严重漂移；人物长期完全不可见且前后都缺少有效 context；多人近身交互（拥抱、共舞）导致“按人独立建模”的假设失效。
- **Assumes**: 依赖 AMASS 训练出的运动先验；依赖 KAMA/SPEC、HRNet、MOT/ReID 等上游组件；使用 SMPL 表示而非细粒度几何；论文量化评测使用 GT tracks，因此真实跟踪误差并未被完整计入；相机内参采用近似并在优化中固定。
- **Not designed for**: 服装/头发等非 SMPL 细节恢复；显式 human-scene contact 建模；在线实时系统；多人交互感知的联合生成。

### 资源与复现约束
- 这是一个**五阶段串联**系统，前面阶段的误差会持续传到后面。
- 论文给出的效率是：**1 分钟视频约 5 分钟处理时间**，明显不是实时方案。
- 正文明确提到有 **Dynamic Human3.6M 生成代码** 与项目页，但完整训练/推理代码开放范围在正文里没有完全展开，因此更适合按 **partial** 开源理解。

### 可复用组件
- **可见帧条件化的自回归 motion infiller**：适合任何“长时人体遮挡补全”任务。
- **egocentric trajectory 表示**：适合长时序人体全局轨迹建模与后验优化。
- **trajectory-anchor joint optimization**：适合“人体轨迹 + 相机位姿”同时未知的欠约束视频重建问题。

## Local PDF reference

![[paperPDFs/Digital_Human_Human_Body_Reshaping/CVPR_2022/2022_GLAMR_Global_Occlusion_Aware_Human_Mesh_Recovery_with_Dynamic_Cameras.pdf]]