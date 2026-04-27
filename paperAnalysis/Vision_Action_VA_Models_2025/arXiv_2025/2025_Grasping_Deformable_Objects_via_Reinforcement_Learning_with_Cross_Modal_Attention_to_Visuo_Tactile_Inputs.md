---
title: "Grasping Deformable Objects via Reinforcement Learning with Cross-Modal Attention to Visuo-Tactile Inputs"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-grasping
  - task/deformable-object-grasping
  - reinforcement-learning
  - cross-modal-attention
  - dataset/PyBullet
  - repr/segmentation-mask
  - repr/tactile-image
  - opensource/no
core_operator: 在视觉分割图与触觉图之间引入层间跨模态时空-通道注意力，并用RL回报端到端学习对抓取有用的联合状态表示
primary_logic: |
  视觉分割图 + 触觉图 + 机器人本体状态 → 通过CNN中的CSCA模块选择性融合全局形状/姿态与局部接触压力 → 将联合表示输入SAC策略与价值网络 → 输出夹爪连续控制以稳定抓取易碎变形物体
claims:
  - "在 basic、random move 和 more objects 三个仿真环境中，DRQ-CMA 的抓取成功率均高于 DRQ-EF、DRQ-LF 与 DRQ-SO [evidence: comparison]"
  - "将编码器从早/晚融合替换为跨模态注意力后，训练 batch reward 由先升后降/波动转为持续上升，说明表示学习更稳定 [evidence: ablation]"
  - "仅用视觉分割输入不足以学习稳定抓取策略：DRQ-SO 在训练中 batch reward 持续下降，基础环境成功率仅约 5% [evidence: comparison]"
related_work_position:
  extends: "Spatio-channel Attention Blocks for Cross-modal Crowd Counting (Zhang et al. 2022)"
  competes_with: "VisuoTactile-RL (Hansen et al. 2022); Visual-Tactile Multimodality for Following Deformable Linear Objects Using Reinforcement Learning (Pecyna et al. 2022)"
  complementary_to: "Segment Anything (Kirillov et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Grasping_Deformable_Objects_via_Reinforcement_Learning_with_Cross_Modal_Attention_to_Visuo_Tactile_Inputs.pdf
category: Embodied_AI
---

# Grasping Deformable Objects via Reinforcement Learning with Cross-Modal Attention to Visuo-Tactile Inputs

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.15595)
> - **Summary**: 论文将视觉分割图与触觉图通过跨模态时空-通道注意力做选择性融合，并用 SAC 端到端训练，使简单二指夹爪也能更稳定地抓取易掉落、易挤爆的软壳变形物体。
> - **Key Performance**: 在 20 次测试协议下，DRQ-CMA 在 basic / random move / more objects 三个环境中都取得最高抓取成功率；单视觉 DRQ-SO 在 basic 环境仅约 5% 成功率。

> [!info] **Agent Summary**
> - **task_path**: 语义分割图 + 触觉图 + 本体状态 / 软壳变形物抓取 -> 夹爪连续速度控制
> - **bottleneck**: 视觉与触觉感受野、统计性质和语义粒度不同，简单拼接很难形成对“稳抓/滑落/压爆”有判别力的 RL 状态表示
> - **mechanism_delta**: 将 encoder 从早融合/晚融合改为层间 CSCA 跨模态注意力，并直接由 critic 的 RL 信号自监督训练
> - **evidence_signal**: 训练中 batch reward 唯一持续上升，且在 3 个仿真环境的 20-trial 成功率均为最高
> - **reusable_ops**: [CSCA跨模态编码器, 基于触觉接触面积的稠密奖励]
> - **failure_modes**: [随机机械臂轨迹导致成功率明显掉点, 仅视觉输入无法感知接触动力学而几乎学不会]
> - **open_questions**: [如何缓解高参数融合编码器的轨迹过拟合, 如何从PyBullet/TACTO迁移到真实触觉硬件]

## Part I：问题与挑战

这篇论文真正要解决的，不是“抓取”这个泛问题，而是一个更窄但更难的子问题：**当机械臂在移动时，二指简单夹爪如何持续调节夹持力度，既不让软壳变形物掉落，也不把它挤破**。

### 1) 任务接口与边界
- **输入**：
  - 视觉：来自 RGB 的**语义分割 mask**，而不是原始 RGB。
  - 触觉：两指尖触觉传感器生成的触觉图像。
  - 本体：末端位姿、夹爪关节位置/速度等 proprioception。
- **输出**：
  - 一个连续动作：**夹爪关节速度/位置控制**。
- **任务边界**：
  - 论文只学**夹爪闭合控制**，不学习整条机械臂的 6-DoF 运动规划。
  - 机械臂大部分运动轨迹是环境设定好的；策略主要负责“怎么夹”。

### 2) 真正瓶颈是什么
真正瓶颈不是“有没有多模态”，而是：

- **视觉**能看到物体整体形状、姿态、相对位置，但**看不到接触细节**；
- **触觉**能看到局部压力和接触面积，但**看不到整体物体运动与形变趋势**；
- 两者**感受野不同、统计分布不同、语义粒度不同**，所以简单的 early fusion / late fusion 常把“融合”留给后面的 RL 自己去猜。

对 RL 来说，这会直接变成**状态表示不稳定**：
- 同样的视觉外观，可能对应完全不同的接触压力；
- 同样的局部触觉，可能对应不同的整体姿态与掉落风险；
- critic 学到的 value 就会更噪，policy 也更难收敛。

### 3) 为什么现在值得做
这件事现在可做，主要因为三点成熟了：
1. **PyBullet + TACTO** 让变形体与触觉图像的长期交互训练变得可扩展；
2. **DrQ 式数据增强 + SAC** 让图像输入的连续控制更稳定；
3. 视觉里的**跨模态注意力**提供了一个可迁移的融合机制，能从“拼接”升级到“条件选择”。

一句话概括 What/Why：**难点不在控制头，而在如何把“全局视觉 + 局部触觉”压缩成一个对回报真正有用的状态。**

## Part II：方法与洞察

### 方法骨架

整体框架可以概括为：

1. **视觉分支**输入语义分割图，**触觉分支**输入触觉图；
2. 两个分支在 CNN 中间层和末层经过 **CSCA（Cross-modal Spatio-Channel Attention）** 模块；
3. 得到的融合特征压成一个低维 embedding；
4. 再拼接 7 维本体状态；
5. 输入到 **SAC actor / critic**，输出夹爪连续控制。

几个关键设计点：

- **用分割 mask，不用原始 RGB**  
  作者有意去掉颜色、纹理、光照这些高扰动因素，让视觉主要表达“物体形状 + 机器人位置 + 背景语义”。

- **用跨模态注意力，不用简单拼接**  
  视觉特征和触觉特征互相提供 query / key / value，使模型能按状态选择“此时该更信视觉还是更信触觉”。

- **编码器用 RL 信号自监督训练**  
  没有单独的 grasp-state 标签，编码器直接通过 critic 的学习信号被更新。  
  这意味着它不是学“通用表示”，而是学“对回报有用的表示”。

- **奖励设计加入触觉接触面积的稠密项**  
  除了成功/失败终止奖励外，还用“较高触觉响应的面积比例”作为训练期引导，鼓励形成更稳定接触，而不是只等最终成败。

### 核心直觉

**What changed**  
从“把视觉和触觉拼起来再交给 RL”改成“让视觉和触觉在特征层互相查询，并按空间/通道自适应重加权”。

**Which bottleneck changed**  
改变的是多模态表示里的**信息瓶颈**：
- 早融合的问题是把两种分布强行塞进同一个卷积流；
- 晚融合的问题是每个模态各学各的，到最后才拼接，跨模态关系学得太晚；
- CSCA 把瓶颈前移到特征层：先建立“全局形状 ↔ 局部接触”的对应，再交给策略学习。

**What capability changed**  
一旦 state representation 更接近“实际可抓稳性”，critic 对状态价值的判断就更一致，policy 就更容易学到：
- 何时该继续闭合，
- 何时已经足够接触，
- 何时存在滑落或压破风险。

### 为什么这个设计在因果上有效
1. **分割 mask 降低视觉域噪声**  
   把原本依赖颜色/纹理的视觉分布，变成更偏结构语义的分布，减小外观变化带来的状态漂移。

2. **跨模态注意力把“该看哪里”变成条件化问题**  
   不是固定把两模态同等对待，而是根据当前接触状态和物体形状，动态决定哪些视觉区域、哪些触觉通道更重要。

3. **RL 回报让表示对控制目标而非标签对齐**  
   编码器不需要人工标注“抓稳/没抓稳”，而是通过回报自动逼近“哪些特征有助于最大化长期成功率”。

4. **稠密触觉奖励缓解稀疏回报**  
   对易碎物体抓取，仅靠最终是否到达目标太稀疏；接触面积信号提供了早期学习支撑。

### 战略权衡

| 设计选择 | 改变的瓶颈 | 带来的收益 | 代价/风险 |
|---|---|---|---|
| 语义分割替代原始 RGB | 降低外观噪声 | 更关注形状、位姿和机器人相对关系 | 依赖额外分割能力或标签 |
| CSCA 层间融合 | 解决视觉/触觉异构且感受野不一致的问题 | 学到更有控制意义的联合表示 | 参数更多，更容易过拟合轨迹分布 |
| SAC + DrQ 式训练 | 改善高维输入下的探索与稳定性 | 连续控制更稳 | 训练仍然非常耗样本 |
| 触觉接触面积稠密奖励 | 缓解只靠终局奖励过稀疏 | 更快形成稳定接触 | 可能偏向“大面积接触”，不直接等价于最优力控制 |

## Part III：证据与局限

### 关键证据

**信号 1：训练曲线比较 → 说明表示学习是否稳定**  
- 现象：只有 **DRQ-CMA** 的 batch reward 呈持续上升趋势；DRQ-EF 和 DRQ-LF 初期上升后回落或波动；DRQ-SO 持续下降。
- 结论：关键增益不只是“多了触觉”，而是**跨模态选择性融合**让编码器更容易学到对回报有区分度的状态表示。

**信号 2：三种测试环境成功率比较 → 说明泛化能力是否真实存在**  
- 测试协议：每个环境 20 次试验。
- 环境设置：
  - **basic**：与训练分布一致；
  - **random move**：机械臂轨迹变成随机 waypoint 的折线运动；
  - **more objects**：从训练 2 个物体扩展到 6 个物体，包含 4 个未见对象。
- 结论：DRQ-CMA 在三个环境都拿到最高成功率，说明它学到的不是只对 seen 场景有效的静态拼接特征，而是更接近“抓取状态”的联合表示。

**信号 3：难点诊断 → 未见轨迹比未见物体更难**  
- 论文指出：**random move 比 more objects 更难**。
- 含义：有触觉帮助时，物体外形变化尚可通过局部压力信息弥补；但机械臂运动分布变化会改变整个接触动力学，这对策略泛化更致命。
- 作者还明确提到：DRQ-CMA 在分布外环境中**最多有约 40% 的成功率下降**，提示模型有过拟合倾向。

### 能力跃迁到底在哪里
相比已有 visuotactile RL，这篇论文的能力跃迁不是“换了更复杂的夹爪”或“多了更多传感器”，而是：

- 让**简单二指夹爪**也能利用视觉全局信息和触觉局部信息；
- 把 prior work 常见的“拼接融合”升级成**状态相关的选择性融合**；
- 在**训练只见少量物体、测试含未见轨迹/物体**时，仍保持相对最优。

这说明它提升的是**表示层面对控制的可用性**，而不仅是单一 benchmark 上的分数。

### 局限性

- **Fails when**: 机械臂轨迹分布明显偏离训练时的运动模式，尤其 random move 这类 zigzag/多 waypoint 轨迹下，模型成功率显著下降；此外，仅视觉输入时由于缺失接触动力学信息，策略基本学不起来。
- **Assumes**: 可获得稳定的语义分割 mask、图像式触觉传感器输入，以及 PyBullet/TACTO 中足够可信的变形物动力学；训练还依赖超长交互量（450M episode steps）、回放缓冲和 RTX 3090Ti 级算力。
- **Not designed for**: 全臂运动规划、多指灵巧手在手操作、重抓/regrasp、真实机器人部署，或超出文中模拟参数范围的复杂材料行为。

### 资源与复现约束
- 训练完全在模拟器中完成，**没有真实机器人结果**。
- 论文未给出代码/项目链接，当前是 **opensource/no**。
- 分割 mask 的可得性是关键依赖；论文虽提到可借助 Segment Anything 之类模型，但并未把这部分纳入完整系统验证。

### 可复用组件
1. **CSCA 作为通用 visuo-tactile encoder**：可直接替换其他机器人策略里的早融合/晚融合模块。  
2. **触觉接触面积稠密奖励**：对“既不能掉也不能太用力”的接触任务很实用。  
3. **语义分割替代原始 RGB 的输入抽象**：适合降低颜色、背景、纹理变化的干扰。  
4. **融合表示 + proprioception 的 actor-critic 接口**：对其他接触型 manipulation 任务也容易迁移。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Grasping_Deformable_Objects_via_Reinforcement_Learning_with_Cross_Modal_Attention_to_Visuo_Tactile_Inputs.pdf]]