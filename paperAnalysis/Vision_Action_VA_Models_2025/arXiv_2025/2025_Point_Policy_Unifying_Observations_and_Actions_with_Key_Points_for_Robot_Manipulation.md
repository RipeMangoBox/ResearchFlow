---
title: "Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/imitation-learning
  - point-tracking
  - triangulation
  - transformer-policy
  - dataset/PointPolicy
  - repr/3D-keypoints
  - opensource/full
core_operator: "用双视角三角化的人手/物体3D关键点统一观察与动作空间，再由Transformer预测未来机器人关键点并通过刚体几何回推出末端控制"
primary_logic: |
  双视角人类演示视频 + 每任务单帧对象点标注 → 手部检测/DIFT/Co-Tracker提取并三角化手与物体3D关键点，并将人手关键点转为机器人关键点 → Transformer预测未来机器人关键点轨迹与夹爪状态 → 用刚体几何回推出末端位姿并闭环执行
claims:
  - "Point Policy在8个真实机器人任务的同分布评测中达到106/120成功（88.3%），而最强基线MT-π为16/120（13.3%） [evidence: comparison]"
  - "Point Policy在6个未见物体任务上达到82/110成功（74.5%），而所有基线几乎为零成功 [evidence: comparison]"
  - "双视角三角化是关键因子：去掉三角化后Point Policy在3个任务上从54/60降至0/60，而P3-PO换用三角化后可从0/60升至44/60 [evidence: ablation]"
related_work_position:
  extends: "P3-PO (Levy et al. 2024)"
  competes_with: "Motion Tracks / MT-π (Ren et al. 2025); P3-PO (Levy et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); OpenTeach (Iyer et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Point_Policy_Unifying_Observations_and_Actions_with_Key_Points_for_Robot_Manipulation.pdf
category: Embodied_AI
---

# Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.20391), [Project](https://point-policy.github.io)
> - **Summary**: 论文把人手、物体状态和机器人动作都重写为机器人基座坐标系中的3D关键点轨迹，从而无需任何机器人示教，仅靠离线人类视频就能训练出可闭环执行的机器人操作策略。
> - **Key Performance**: 8个真实任务平均成功率88.3%；未见物体实例平均成功率74.5%。

> [!info] **Agent Summary**
> - **task_path**: 双视角第三视角人类演示视频 + 每任务单帧对象点先验 -> 机器人末端6DoF位姿与夹爪控制
> - **bottleneck**: 人手与机械臂的形态差异、像素表观偏移以及不稳定的3D lifting 让“人类视频 -> 机器人动作”监督无法直接对齐
> - **mechanism_delta**: 将观测和动作统一改写为机器人基座系中的3D关键点轨迹，并用刚体几何把预测点回推为可执行机器人控制
> - **evidence_signal**: 8个真实任务的人视频-only对比实验 + 三角化深度消融（54/60到0/60）
> - **reusable_ops**: [single-frame semantic point prior propagation, two-view triangulated 3D keypoints, rigid-body pose backtracking]
> - **failure_modes**: [hand/object keypoint tracking fails under occlusion, sparse points omit collision context in cluttered or obstacle-rich scenes]
> - **open_questions**: [能否扩展到未标定的第一视角互联网视频, 如何在保持泛化的同时保留稀疏上下文与碰撞信息]

## Part I：问题与挑战

这篇工作的目标很明确：**只用离线人类演示视频，不用机器人示教、不做在线RL交互，也要学出能真实部署的操作策略。**

真正的难点不是“缺少一个更强的policy网络”，而是**缺少人和机器人之间共享的状态-动作接口**：

1. **形态差异（morphology gap）**  
   训练时看到的是人手，部署时控制的是机械臂夹爪。  
   如果直接做图像到动作的BC，模型必须同时学会：
   - 识别任务对象
   - 忽略人手/机械臂的外观差异
   - 从像素恢复3D动作几何  
   这三个问题叠在一起，数据量一小就很容易崩。

2. **3D几何监督不稳定**  
   机器人控制最终需要的是末端位姿与夹爪状态，而人类视频天然只有2D像素。  
   现成RGB-D深度如果噪声大，投到3D后动作标签会直接变脏，策略就学不到可靠控制。

3. **现有路线都要额外付出机器人数据成本**  
   - 一类方法先从人类视频学表示，再用机器人数据下游微调；
   - 另一类方法用人类视频构造奖励，再靠在线RL学策略。  
   二者都没有真正绕开机器人数据或在线交互成本。

**为什么现在值得做？**  
因为关键的感知部件已经成熟：手部关键点检测、语义对应、点跟踪都可以直接调用现成视觉模型。也就是说，过去最难的“从视频里稳定提炼任务相关几何结构”这件事，现在第一次具备工程可行性。

**输入/输出接口与边界条件：**
- **输入**：双相机第三视角人类演示视频；每个任务只需对一帧做对象点标注。
- **输出**：机器人末端6DoF位姿 + 夹爪开合，闭环执行频率6Hz。
- **边界条件**：
  - 需要双相机标定；
  - 假设任务第一帧的人手姿态已知用于初始化；
  - 末端执行器被视为刚体，可用固定模板点描述。

一句话概括 Part I：  
**瓶颈不是缺少更多demo，而是缺少一个同时对“人类观察”和“机器人动作”都成立的几何语言。**

## Part II：方法与洞察

作者的策略不是直接让网络从像素里“悟出”人手如何对应机器人，而是先把整个学习问题改写到一个统一的3D关键点空间里。

### 方法主线

#### 1. 人手关键点 → 机器人关键点
- 用 MediaPipe 从两路相机里提取人手拇指和食指关键点。
- 通过双视角三角化得到人手3D点。
- 用食指/拇指的几何关系构造机器人动作标签：
  - 两指尖中点作为末端位置；
  - 相对第一帧的刚体变换给出末端朝向；
  - 两指距离阈值决定夹爪开合。
- 再围绕末端刚体模板生成一组**机器人关键点**，作为动作空间的稀疏表示。

这里的关键是：**动作不再直接表示为抽象的6DoF数值，而是表示为“未来机器人关键点该在哪里”。**

#### 2. 单帧对象点先验 → 全轨迹对象3D点
- 用户只需在一个示范视频的第一帧标注几个任务相关对象点。
- 用 DIFT 把这些点迁移到其他示范的第一帧。
- 用 Co-Tracker 在每段视频中持续跟踪这些点。
- 再通过双视角三角化，把对象点统一到机器人基座坐标系中的3D空间。

这样做的好处是：  
**模型看见的是“对象哪里在动、机器人点应该如何接近”，而不是整张图像的纹理和背景。**

#### 3. 3D点历史 → Transformer策略 → 末端控制
- 用 BAKU 的 transformer policy，但把输入从图像改成：
  - 机器人点历史
  - 对象点历史
  - 夹爪token
- 输出未来机器人关键点轨迹和夹爪状态。
- 再利用刚体几何，从预测的机器人点回推出末端位姿。
- 通过 action chunking + temporal averaging 平滑执行，最终以6Hz闭环控制机器人。

---

### 核心直觉

**作者真正调的“因果旋钮”不是更大的policy，而是把学习问题从“像素到动作”改写成“共享3D几何空间中的未来点轨迹预测”。**

#### What changed → which bottleneck changed → what capability changed

- **从原始图像 / 2D轨迹**
  → **机器人基座系中的3D关键点**
  → 去掉了外观、背景、视角的大量干扰
  → 得到更强的空间泛化和新物体泛化

- **从人手像素监督**
  → **几何构造出的机器人关键点监督**
  → 缩小了训练时“看人手”和测试时“控机械臂”的分布落差
  → 让 human-video-only 的行为克隆第一次变得可行

- **从直接回归末端动作**
  → **先预测未来机器人点，再刚体回推**
  → 观测和动作处于同一种表示空间
  → policy更容易学到“机器人点相对对象点应该怎么动”

#### 为什么这套设计有效

因为策略网络不再需要同时解决“人手长什么样”和“机械臂怎么动”这两个问题。  
它只需要学习一件更简单、也更稳定的事：

> **给定对象点和机器人点的3D相对关系，下一步机器人点该移动到哪里。**

这使得学习重点从**表观匹配**转向**几何关系建模**。  
而双视角三角化又把这个几何关系真正锚定到3D世界坐标里，避免了深度噪声把动作标签污染掉。

### 战略取舍

| 设计选择 | 改变了什么约束/瓶颈 | 带来的能力 | 代价/假设 |
|---|---|---|---|
| 稀疏3D关键点替代原始图像 | 去掉纹理、颜色、背景依赖 | 对新位置、新物体、背景干扰更稳 | 丢失稠密场景上下文，障碍物/碰撞推理变弱 |
| 双视角三角化替代传感器深度 | 提高3D点精度，减少标签噪声 | 动作监督更稳定，3D控制更可靠 | 需要双相机与标定 |
| 预测未来机器人点替代直接动作回归 | 统一观测/动作表示 | 人到机器人迁移更自然，几何一致性更强 | 依赖固定末端刚体模板 |
| 只用人类离线视频替代机器人示教/在线RL | 去掉高成本、高风险机器人数据采集 | 更便宜、更安全、数据来源更广 | 受限于手部检测、点跟踪和人机执行差异 |

## Part III：证据与局限

### 关键实验信号

#### 1. 对比信号：human-video-only 真的能学成
在8个真实任务、190段人类演示上，Point Policy 达到 **106/120 = 88.3%** 成功率。  
对比方法里：
- RGB BC 几乎全灭；
- RGB-D BC 也几乎全灭；
- P3-PO 在原始深度设置下也是全灭；
- 最强基线 MT-π 只有 **16/120 = 13.3%**。

**结论**：真正造成失败的不是“BC不行”，而是**表示空间不对**。一旦把监督和观测都改到共享3D关键点空间，human-video-only 学策略就变得可行。

#### 2. 泛化信号：不只记住训练物体
在6个未见物体任务上，Point Policy 达到 **82/110 = 74.5%**。  
而基线几乎都接近0。

更有意义的是，很多任务训练时只见过**单个物体实例**，但测试还能迁移到新盘子、新毛巾、新瓶子、新碗。  
**结论**：点表示确实把策略关注点从“这个物体长什么样”转移到“任务相关语义点之间的空间关系”。 

#### 3. 鲁棒性信号：背景杂物影响相对有限
加背景干扰后，大部分任务只出现轻微下降：
- put bread on plate：19/20 → 18/20
- put bottle on rack：26/30 → 23/30  
但 novel broom 从 4/10 降到 2/10，说明**长工具 + 新实例 + 干扰场景**仍然是脆弱组合。

**结论**：点表示对背景纹理和 distractor 确实更稳，但在工具接触和长杆几何更复杂的任务上，稀疏表示还不够。  

#### 4. 机制信号：三角化不是小trick，而是核心因果因子
这是文中最有说服力的因果证据：

- Point Policy 在3个任务上使用三角化时是 **54/60**
- 去掉三角化后变成 **0/60**
- P3-PO 原本 **0/60**
- 换上三角化后升到 **44/60**

这说明能力提升不只是“用了关键点”，而是更具体地来自：

> **高精度、世界坐标系对齐的3D关键点。**

#### 5. 反直觉信号：加机器人demo不一定更好
在4个任务上：
- human-only：**53/60**
- robot-only：**43/60**
- human+robot：**45/60**

尤其 sweep broom、make bottle upright 这类复杂动作，少量VR遥操作数据反而更噪。  
**结论**：人和机器人完成同一任务的动作分布并不天然一致，简单拼接数据不一定有效。

---

### 局限性

- **Fails when**: 手部检测、DIFT对应或Co-Tracker跟踪在遮挡下失败时，整条监督链会断；在需要绕开障碍物、利用稠密场景上下文或复杂接触几何的任务中，稀疏点表示容易不够用。
- **Assumes**: 需要双相机标定、固定第三视角、任务首帧手姿初始化、每任务单帧对象点标注；依赖 MediaPipe、DIFT、Co-Tracker 等现成视觉模块；末端执行器满足刚体模板假设；实验主要在 Franka 单臂、6DoF末端位姿控制设置上验证。
- **Not designed for**: 未标定的互联网第一视角视频直接学习、密集障碍导航、需要完整场景语义/碰撞几何的操作、以及超出固定刚体末端模板的更复杂形态控制。

补充两点对证据解读很重要：
1. **MT-π 是作者自行复现**，因为官方实现尚未公开，因此对比结果虽然差距很大，仍应保守看待。
2. **训练算力门槛不高**（单张 RTX A4000），但系统复杂度主要在感知链和双相机标定，而不在policy训练本身。

### 可复用部件

这篇 paper 最值得复用的不是某个特定网络，而是下面这些“操作子”：

- **单帧对象点先验传播**：一帧标注，跨demo初始化语义点。
- **双视角三角化3D关键点**：把人手/对象状态锚到统一世界坐标。
- **关键点动作化**：把机器人动作表示为未来关键点轨迹，而非直接关节或位姿。
- **刚体几何回推**：从预测点恢复末端位姿，适合和别的policy backbone组合。

一句话总结 “So what”：

> Point Policy 的能力跃迁，不在于更大的模型，而在于把“人类视频”和“机器人控制”首次压缩到了同一个可学习、可执行、可泛化的3D关键点接口里。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Point_Policy_Unifying_Observations_and_Actions_with_Key_Points_for_Robot_Manipulation.pdf]]