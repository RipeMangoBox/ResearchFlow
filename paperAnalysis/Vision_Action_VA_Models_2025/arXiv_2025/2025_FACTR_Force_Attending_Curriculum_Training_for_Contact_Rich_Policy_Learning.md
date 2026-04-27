---
title: "FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/contact-rich-manipulation
  - task/imitation-learning
  - curriculum-learning
  - transformer
  - force-feedback
  - dataset/Box-Lift
  - dataset/Non-Prehensile-Pivot
  - dataset/Fruit-Pick-and-Place
  - dataset/Rolling-Dough
  - opensource/full
core_operator: "训练早期强烈退化视觉输入，迫使策略先利用外部关节力矩建立接触感知，再逐步恢复视觉细节实现稳定的力-视觉融合。"
primary_logic: |
  RGB图像 + 外部关节力矩 + 遥操作示教 → 用双边力反馈遥操作收集高质量接触式演示，并在行为克隆训练中对视觉施加逐步减弱的模糊/下采样课程 → 输出对未见物体外观与几何更稳健的动作块关节控制策略
claims:
  - "Claim 1: Across four real-world contact-rich tasks, FACTR improves average unseen-object success from 61.2% for ACT (Vision+Force) to 87.5% [evidence: comparison]"
  - "Claim 2: The bilateral force-feedback teleoperation system improves task completion rate by 64.7%, reduces completion time by 37.4%, and improves subjective ease of use by 83.3% over an un-actuated leader-follower baseline [evidence: comparison]"
  - "Claim 3: Decaying visual-corruption curricula outperform fixed-scale smoothing in ablations, indicating that progressively restoring visual detail is more effective than permanently degraded vision [evidence: ablation]"
related_work_position:
  extends: "Action Chunking Transformer (ACT"
  competes_with: "Bi-ACT (Kobayashi et al. 2024); FoAR (He et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); TacDiffusion (Wu et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_FACTR_Force_Attending_Curriculum_Training_for_Contact_Rich_Policy_Learning.pdf
category: Embodied_AI
---

# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.17432), [Project](https://jasonjzliu.com/factr/)
> - **Summary**: 论文把“力反馈”同时前移到示教采集和策略训练两端：先用低成本双边力反馈遥操作采集接触丰富演示，再用逐步去模糊视觉的课程训练，强制策略先学会“看力”而不是走视觉捷径，从而显著提升未见物体泛化。
> - **Key Performance**: 未见物体平均成功率由 61.2% 提升到 87.5%；遥操作完成率 +64.7%，完成时间 -37.4%。

> [!info] **Agent Summary**
> - **task_path**: RGB图像 + 外部关节力矩 / 接触丰富行为克隆 -> 未来 k 步关节位置动作块
> - **bottleneck**: 多模态 BC 中视觉更容易提供训练捷径，导致策略忽略长时间接近零、但在接触时最关键的力信号
> - **mechanism_delta**: 用随训练递减的视觉模糊/下采样课程压制早期视觉主导，让 force token 先变得可用，再恢复视觉细节完成联合决策
> - **evidence_signal**: 4 个真实任务上未见物体成功率 87.5% vs 61.2%，且接触时刻交叉注意力明显转向 force token
> - **reusable_ops**: [双边外部关节力矩回传, 视觉退化-恢复课程训练]
> - **failure_modes**: [关节力矩传感噪声大时细粒度力控受限, 在非接触或力信号判别性弱的任务中收益有限]
> - **open_questions**: [课程强度能否自适应调度, 能否与触觉传感或扩散策略进一步结合]

## Part I：问题与挑战

这篇论文解决的是一个很具体但长期被忽视的问题：**接触丰富操作并不是“多加一个力传感器输入”就能学会的**。

### 1. 真正的难点是什么
有两个串联瓶颈：

1. **数据采集瓶颈**  
   低成本 leader-follower 遥操作通常是单边、被动的。操作者看得到机械臂，但**感觉不到接触力**，所以在抬箱子、转动物体、擀面这种任务里，很容易丢失接触、动作突兀、示教质量差。

2. **策略学习瓶颈**  
   即便把力信号喂给策略，模型也常常**不用它**。原因不是力没用，而是：
   - 视觉在整段轨迹里都“有信息”
   - 力在很多时刻接近 0，只在接触/切换阶段突然重要
   - 结果是模型学到一条更容易的路：**依赖视觉外观，忽略力模态**

所以这篇论文的核心问题其实是：

> **如何让策略在接触式操控里真正依赖力，而不是把力当作一个被拼接但未被使用的输入通道。**

### 2. 为什么现在值得解决
因为现在两个条件已经成熟：

- 许多协作机械臂已经能直接提供**外部关节力矩**估计，如 Franka Panda、KUKA iiwa。
- 低成本伺服 leader arm 已经足够便宜，可以做成**可回传力反馈**的双边系统，而不必依赖昂贵的专业力反馈设备。

这意味着：过去“没有低成本力反馈采集链路”的障碍开始消失，接下来真正限制泛化的，就变成了**训练时如何让模型学会关注力**。

### 3. 输入/输出接口与适用边界
- **输入**：RGB 图像（前视或腕部相机）+ 外部关节力矩
- **输出**：未来一段时间的关节位置动作块
- **适用任务**：接触建立、接触维持、接触切换决定动作阶段的任务  
  例如箱体抬升、非抓取翻转、柔软水果抓取放置、擀面
- **不那么适用的场景**：纯视觉自由空间操作、几乎无接触信息的任务、需要超高分辨率触觉而非关节力矩的精细操作

---

## Part II：方法与洞察

这篇论文的设计很“系统化”：**不是只改 policy，而是同时改示教入口和训练过程。**

### 方法总览

#### A. 低成本双边力反馈遥操作
作者先搭了一个低成本 bilateral leader-follower 系统：

- follower 机械臂测得的**外部关节力矩**回传给 leader
- leader 关节是**可驱动**的，因此操作者能“感觉到”环境约束
- 只回传**外部力**，而不是把 follower 的惯性/摩擦全传回来  
  这样避免操作者感受到“机械臂自身的动态噪声”
- 再叠加：
  - 重力补偿
  - 摩擦补偿
  - 冗余自由度的 null-space 姿态调节
  - 关节限位避免
  - 双臂碰撞规避

这一步的意义不是“更炫的遥操作”，而是：**提高接触式示教的可操作性和数据质量**。

#### B. 基础策略：Vision + Force 的 ACT
策略骨干是 ACT 风格的 encoder-decoder transformer：

- 预训练 ViT 把图像编码成视觉 tokens
- MLP 把关节力矩编码成一个 force token
- action tokens 通过 cross-attention 同时读取 vision / force 表示
- 输出未来一段 joint targets

也就是说，FACTR 并不是重新发明一个全新的 policy backbone，**而是在一个已有强基线 ACT 上改变多模态学习顺序**。

#### C. FACTR：Force-Attending Curriculum Training
核心训练策略很简单，但抓住了症结：

- 在训练早期，故意把视觉输入做强退化
  - 像素空间：Gaussian blur / downsampling
  - 潜空间：对视觉 token 做 blur / pooling
- 随训练推进，逐步减弱退化，恢复清晰视觉
- 还加入 warm-up，让随机初始化的 force encoder 先得到学习空间

作者的目标不是增强视觉鲁棒性本身，而是：

> **改变“模型先学什么”的顺序：先学会靠力区分接触状态，再补回视觉细节。**

### 核心直觉

**What changed**  
训练分布被人为改写：视觉从“高保真、强主导”变成“先低保真、后高保真”。

**Which bottleneck changed**  
这直接改变了多模态训练里的信息不平衡：
- 早期视觉细节被压缩后，颜色/纹理这些捷径不再好用
- 样本间可区分的剩余信号更多来自力，尤其是接触建立、接触丢失、滚动相位等时刻
- 梯度因此更容易流向 force encoder 和跨模态注意力

**What capability changed**  
策略开始学会：
- 用力信号做**mode switching**
- 在未见物体上不过拟合外观
- 在接触被打断后进行**恢复行为**

### 为什么这个设计有效
因果链条可以概括成一句话：

> **先削弱视觉捷径 → force 成为训练早期唯一可靠的判别源 → 模型学会把接触变化映射到动作阶段 → 再恢复视觉细节后，得到既能定位也能感知接触的策略。**

这和普通数据增强不一样。普通增强常常只是让视觉更鲁棒；FACTR 的目标是**改变模态间的学习优先级**。  
论文还用一个简化的 NTK 视角说明：当 blur 足够强时，不同图像会变得近似相同，于是模型必须借助别的信号（这里是 force）来解释动作差异。

### 战略性 trade-off

| 设计选择 | 好处 | 代价/风险 | 论文结论 |
|---|---|---|---|
| 只用视觉 | 实现最简单 | 容易过拟合外观，接触切换差 | 未见物体最差 |
| 直接拼接 Vision+Force | 保留两模态 | 视觉仍可能压制 force | 比视觉-only 好，但不稳定 |
| **递减式视觉退化课程** | 先学力、后补视觉细节，兼顾泛化与精度 | 需要课程超参 | 整体最佳 |
| 固定强模糊 | 能强迫关注力 | 最终缺少视觉细节 | 通常弱于递减课程 |
| 像素空间课程 | 实现直观 | 可能改动输入分布较大 | 可行 |
| 潜空间课程 | 更贴近表示层干预 | 依赖视觉编码器表示质量 | 也可行，无绝对赢家 |

---

## Part III：证据与局限

### 关键证据：能力跃迁到底体现在哪

- **比较信号：未见物体泛化显著提升**  
  在 4 个真实接触任务上，测试物体平均成功率：
  - Vision-Only：21.3%
  - Vision+Force（无课程）：61.2%
  - **FACTR：87.5%**  
  说明关键不是“有没有 force 输入”，而是“训练时有没有逼模型去用 force”。

- **任务级信号：提升集中出现在 OOD，而不是训练物体记忆**  
  训练物体上大家通常差不多，但到了测试物体：
  - Box Lift：58.3% → **91.7%**
  - Pivot：42.0% → **76.0%**
  - Fruit Pick-and-Place：73.3% → **93.3%**
  - Rolling Dough：70.0% → **80.0%**  
  这说明 FACTR 带来的主要是**泛化能力**，不是简单提高训练集拟合。

- **机制信号：注意力在接触时刻转向 force**  
  作者可视化 decoder cross-attention 后发现，FACTR 训练出的策略在箱体接触阶段会明显提升对 force token 的关注；而无课程训练的 Vision+Force baseline 对 force 的注意力不足。  
  这和作者的机制主张是对齐的：**FACTR 让 force 真正参与决策，而不是形式上拼接输入。**

- **恢复性信号：接触丢失后更会“再来一次”**  
  在箱体抬升的 recovery 实验里，第一次成功抬起后人为把箱子打落，再看第二次是否能恢复：
  - Vision-Only：4/30
  - Vision+Force：16/30
  - **FACTR：27/30**  
  这很重要，因为它表明 force 不是只在第一次接触有用，而是在**状态切换与失败恢复**中持续有价值。

- **遥操作用户研究信号：示教入口确实更好用**  
  相比无驱动 leader-follower 基线：
  - 完成率 +64.7%
  - 完成时间 -37.4%
  - 主观易用性 +83.3%  
  这支持了论文另一半贡献：**更好的 contact-rich 演示采集链路**。

- **消融信号：关键在“递减课程”，不是某个特定操作器**  
  固定模糊不如递减模糊；pixel/latent、blur/downsample、linear/cosine/step 等组合都能工作，但没有单一参数组合绝对统治。  
  这说明真正有效的因果旋钮是：**逐步释放视觉细节**。

### 局限性

- **Fails when**: 关节力矩传感噪声较大、任务需要非常细微的力差分、或者任务大部分时间没有有效接触信号时，FACTR 的收益会明显下降；对高分辨率触觉依赖很强的精细操作也不是它的强项。

- **Assumes**: 需要 follower arm 能提供外部关节力矩；需要自制可驱动 leader arm 与 3D 打印/伺服硬件；课程超参（模糊强度、scheduler、warm-up）有任务依赖；训练仍依赖示教数据。资源上，作者报告单个 leader arm 物料成本约 \$1229.95，策略训练约 2–6 小时/RTX 4090。

- **Not designed for**: 纯视觉自由空间操作、没有力矩感知能力的机械臂、需要严格高带宽力控闭环的控制器设计、或以端执行器高精度触觉阵列为核心感知的任务。

### 可复用组件

- **外部关节力矩回传式双边遥操作**：尤其适合低成本 contact-rich 示教系统
- **视觉退化→逐步恢复的课程模板**：可迁移到其他 vision+force、vision+tactile 策略
- **force token 注意力诊断**：可作为“模型到底有没有在用力”的分析工具
- **不依赖额外接触相位标注**：相较显式接触阶段分类方法，更易落地

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_FACTR_Force_Attending_Curriculum_Training_for_Contact_Rich_Policy_Learning.pdf]]