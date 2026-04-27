---
title: "FRMD: Fast Robot Motion Diffusion with Consistency-Distilled Movement Primitives for Smooth Action Generation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - consistency-distillation
  - movement-primitives
  - dataset/Meta-World
  - dataset/ManiSkill
  - opensource/no
core_operator: 将多步扩散的原始动作生成改为在 ProDMP 轨迹参数空间中做一致性蒸馏，实现满足边界条件的单步平滑动作解码
primary_logic: |
  历史 RGB 图像与本体状态 + 单次噪声轨迹样本 → 用 MPD 教师在 ProDMP 权重空间提供 PF-ODE 一致性监督，学生网络直接预测满足当前位置/速度边界的轨迹参数 → 解码出单步生成、时序平滑的未来动作序列
claims:
  - "FRMD 在 12 个 Meta-World/ManiSkill 操作任务上取得 64.8% 的平均成功率，高于 MPD 的 64.1% 和 Diffusion Policy 的 50.1% [evidence: comparison]"
  - "FRMD 的平均推理延迟为 17.2 ms，约为 MPD 的 1/10、Diffusion Policy 的 1/7，同时保持单步推理 [evidence: comparison]"
  - "在 PlugCharger-v1 的固定初始/目标条件下，FRMD 的非平滑转折数为 21，显著低于 Diffusion Policy 的 82 [evidence: case-study]"
related_work_position:
  extends: "Movement Primitive Diffusion (Scheikl et al. 2024)"
  competes_with: "Diffusion Policy (Chi et al. 2023); Movement Primitive Diffusion (Scheikl et al. 2024)"
  complementary_to: "π0 (Black et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_FRMD_Fast_Robot_Motion_Diffusion_with_Consistency_Distilled_Movement_Primitives_for_Smooth_Action_Generation.pdf
category: Embodied_AI
---

# FRMD: Fast Robot Motion Diffusion with Consistency-Distilled Movement Primitives for Smooth Action Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.02048)
> - **Summary**: 这篇工作把机器人动作生成从“原始 waypoint 的多步扩散”改成“ProDMP 轨迹参数空间中的一致性蒸馏”，从而在保持轨迹结构性的同时，把推理压缩到单步并显著改善平滑性。
> - **Key Performance**: 平均成功率 **64.8%**；平均推理延迟 **17.2 ms**（比 MPD 快约 **10×**，比 DP 快约 **7×**）

> [!info] **Agent Summary**
> - **task_path**: RGB 图像 + 本体状态 + 历史观测 -> 未来 12 步机器人动作轨迹
> - **bottleneck**: 传统 diffusion policy 在原始动作序列上逐步去噪，既难以显式建模轨迹级时序一致性，又带来不可接受的控制时延
> - **mechanism_delta**: 用 ProDMP 权重替代原始动作作为生成目标，并把 MPD 教师的多步去噪过程蒸馏成一个单步一致性学生
> - **evidence_signal**: 12 个 Meta-World/ManiSkill 任务上取得 64.8% 平均成功率，同时把推理时间降到 17.2 ms
> - **reusable_ops**: [ProDMP轨迹参数化, 教师-学生一致性蒸馏]
> - **failure_modes**: [高难接触任务成功率仍低, 大规模预训练与更长时域场景未验证]
> - **open_questions**: [单步蒸馏能否稳定扩展到更高维动作与更长规划窗口, 作为VLA动作解码器接入更大模型后是否仍保持速度与稳定性]

## Part I：问题与挑战

这篇论文要解决的真实瓶颈，不是“机器人能不能用扩散模型生成动作”，而是**扩散动作解码器在真实控制闭环里还不够像一个可部署的动作模块**。

### 1. 真问题是什么
现有 diffusion policy 在机器人里有两个核心问题：

1. **动作不够平滑**  
   许多方法直接在原始 action waypoint 上建模，等于让模型在高频、逐点的动作空间里学习整条轨迹。这样虽然灵活，但很难显式体现轨迹级的动态约束，所以容易出现 jerk、抖动和时序不连贯。

2. **采样太慢**  
   DDPM/DDIM 一类方法需要多步迭代去噪。对图像生成这可以接受，但对机器人控制来说，动作解码是在线环节，延迟会直接限制控制频率和实时性。

### 2. 输入/输出接口
论文的任务设定是标准的视觉-运动控制：

- **输入**：RGB 图像观测 + 机器人本体状态 + 历史观测窗口  
- **输出**：未来一段动作序列  
- 具体实验里，作者采用 **过去 3 帧观测 -> 未来 12 步动作**

此外，这篇方法还显式用到当前轨迹的**初始位置与速度边界条件**，因为它要把动作序列表示成 ProDMP 轨迹参数，而不是一串松散的 waypoint。

### 3. 真正的 bottleneck 在哪
作者的判断很准确：  
**瓶颈不只是“扩散采样步数太多”，而是“生成粒度不对”。**

如果你一直在原始动作点上扩散，那么：
- 一方面分布太高维，难以学到平滑的轨迹先验；
- 另一方面加速采样也容易伤害质量，因为你在一个本身就不够结构化的空间里做近似。

所以 FRMD 的思路不是单纯“给 diffusion 加速”，而是先**把动作空间换成轨迹参数空间**，再做**一致性蒸馏**。

### 4. 为什么现在值得做
这件事现在重要，是因为机器人大模型/VLA 系统越来越把 diffusion 当成 action decoder，但真正部署时会卡在两个地方：
- 控制延迟；
- 动作平滑性与连续执行质量。

FRMD 本质上是在回答：  
**能不能保留 diffusion 的多模态表达能力，但让动作输出更接近经典机器人轨迹生成器的工程可用性？**

## Part II：方法与洞察

FRMD 的设计哲学可以概括成一句话：

**先用 movement primitives 把动作生成“轨迹化、结构化”，再用 consistency distillation 把多步扩散“压缩成单步”。**

### 方法骨架

#### 1. 用 ProDMP 表示轨迹，而不是直接预测 raw actions
教师模型沿用 MPD 的思路：  
给定观测和带噪动作序列，网络并不直接输出每个时刻的动作点，而是先预测 **ProDMP 的权重向量**，再通过 ProDMP 解码器还原出整条动作轨迹。

这一步的关键价值是：
- 把轨迹放进一个更低维、更结构化的表示空间；
- 通过当前位置/速度边界条件保证连续段衔接更自然；
- 让“平滑”不只靠数据拟合，而是部分由表示本身提供。

#### 2. 用 MPD 当 teacher，把多步去噪蒸馏给 student
FRMD 并没有从零训练一个 consistency model，而是：
- 先训练一个 MPD 教师；
- 再让学生模型学习：对于同一条 PF-ODE 轨迹上的不同噪声状态，应该直接映射到同一个干净动作结果。

也就是说，学生不再学习“下一步怎么去噪”，而是学习“**从任意噪声点一步落到最终动作**”。

#### 3. 推理时只做一次前向
这是 FRMD 最大的部署收益：

- 传统 DP / MPD：需要多步 ODE/denoising 过程；
- FRMD：一次前向，直接得到干净动作轨迹。

因此它的速度优势不是小修小补，而是**采样范式改变**。

### 核心直觉

**改变了什么？**  
从“原始动作序列上的逐步扩散”改成“ProDMP 轨迹参数上的单步一致性映射”。

**改变了哪个瓶颈？**  
它同时改变了两个约束：

1. **分布层面的瓶颈**  
   原始 waypoint 分布高维、局部、缺乏显式轨迹结构。  
   改成 ProDMP 权重后，模型面对的是一个更低维、更平滑、带边界条件的轨迹流形。

2. **计算层面的瓶颈**  
   原来必须沿去噪轨迹多步逼近最终动作。  
   一致性蒸馏后，学生被训练成直接跳到 PF-ODE 轨迹的终点表示。

**能力发生了什么变化？**
- 轨迹更平滑、更时序一致；
- 推理从多步变为单步；
- 因为仍继承 teacher 的轨迹先验，所以速度提升没有明显以成功率为代价。

**为什么这不是简单拼接两个模块？**  
因为这里不是“MP + diffusion”机械相加，而是存在明确因果链：

- 用 ProDMP 重新定义生成空间  
  ↓  
- 让 teacher 在这个更结构化的空间里形成可蒸馏的轨迹先验  
  ↓  
- 用 consistency distillation 把“逐步求解”改写成“直接投影到终点”  
  ↓  
- 获得平滑 + 快速 两种能力同时提升

### 一个值得注意的细节
作者提到，FRMD 在训练时采用的是**直接预测动作样本**，而不是图像生成里常见的噪声预测。这一改动的直觉是：  
机器人动作本身位于一个更低维、结构更强的流形上，直接回归“干净动作/轨迹”更容易收敛到有意义的动作空间。

### 策略权衡表

| 设计选择 | 改变了什么约束/分布 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 用 ProDMP 权重替代 raw actions | 把高维 waypoint 分布压缩到低维、带边界条件的轨迹参数空间 | 动作更平滑、连续段更容易衔接 | 表达能力受基函数与 primitive 形式约束，复杂高频动作可能更难覆盖 |
| 用 consistency distillation 替代多步采样 | 把“逐步求解 ODE”改为“直接映射到终点” | 单步推理、显著降延迟 | 强依赖 teacher 质量；蒸馏误差可能在难任务上放大 |
| 保留 Transformer 作为主干 | 加强时序依赖建模 | 在中高难任务上保持成功率 | 比轻量模型略慢；MLP 虽快但显著掉性能 |

## Part III：证据与局限

### 关键证据

#### 1. 能力跃迁：不是只变快，也保持了任务成功率
最关键的对比信号来自 12 个 Meta-World/ManiSkill 任务的总体结果：

- **FRMD：64.8%**
- MPD：64.1%
- Diffusion Policy：50.1%

这说明 FRMD 不只是“蒸馏成一个更快但更差的模型”，而是**在近似保留 teacher 能力的同时，略微超过了 teacher**。  
尤其在中等与高难任务上，它至少没有因为单步化而明显崩掉。

#### 2. 最大增益来自部署效率
平均推理时间：

- **FRMD：17.2 ms**
- DP：119.7 ms
- MPD：168.6 ms

这是整篇论文最有说服力的信号。因为它直接对应真实机器人控制里最痛的瓶颈：**动作生成时延**。  
如果只看成功率，FRMD 比 MPD 的提升不大；但如果把成功率和延迟一起看，FRMD 的系统价值就很清楚了。

#### 3. 平滑性证据：有正向信号，但证据形式偏局部
在 PlugCharger-v1 的可视化案例里，作者用曲率阈值统计非平滑转折：

- DP：**82**
- FRMD：**21**

这支持了“轨迹参数化 + 一致性蒸馏”确实让轨迹更稳、更少振荡。  
但要注意，这部分证据主要是**单任务案例分析**，而不是跨全部任务的系统性平滑度量。

#### 4. 消融：主干网络确实重要
作者比较了三种主干：

- 完整 Transformer：总体最好
- 轻量 Transformer：更快，但高难任务略掉点
- MLP：最快，但复杂任务明显不行

这说明 FRMD 的收益不只是“换了训练目标”，还依赖一个足够强的时序建模骨干。

### 局限性

- **Fails when**: 高难、复杂接触任务下仍然不稳，hard tasks 平均成功率只有 **29.0%**；此外，平滑性优势的定量证据主要集中在 PlugCharger-v1，对所有任务和对 MPD 的系统性平滑比较仍不足。  
- **Assumes**: 依赖离线专家示范数据；依赖先训练好的 **MPD teacher**；依赖 ProDMP 这一特定轨迹表示与当前位置/速度边界条件；实验只验证了短时域窗口设定（3 帧观测到 12 步动作）；训练与复现实验默认有 GPU 资源（文中使用单张 RTX 4090）。  
- **Not designed for**: 大规模多任务预训练、语言条件控制、超长时域层级规划、以及不适合用 ProDMP 基函数描述的任意复杂动作模式。

### 复现与外推时需要特别注意
1. **没有看到代码/项目链接**，因此开源可复现性目前偏弱。  
2. 方法虽然快，但前提是你已经有一个可用的 **MPD 教师**；这使得训练链路并不比普通 policy 更简单。  
3. 论文自己也明确承认：尚未验证在**大数据、大模型、大规模任务预训练**下的表现。

### 可复用部件
这篇论文里最值得复用的不是完整系统，而是两个操作件：

- **ProDMP 轨迹参数化动作头**：可作为任意视觉/多模态策略的结构化动作输出层  
- **teacher-student 一致性蒸馏范式**：可把已有多步 diffusion action decoder 压缩成单步解码器

如果你在做 VLA 或 diffusion-based policy，这两件东西都很有组合价值。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_FRMD_Fast_Robot_Motion_Diffusion_with_Consistency_Distilled_Movement_Primitives_for_Smooth_Action_Generation.pdf]]