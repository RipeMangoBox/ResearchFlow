---
title: "Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - inverse-dynamics
  - future-state-prediction
  - transformer
  - dataset/DROID
  - dataset/LIBERO-LONG
  - dataset/CALVIN
  - opensource/full
core_operator: 在共享 Transformer 中先做未来视觉前瞻，再让逆动力学动作 token 单向读取该前瞻信息，从而把视觉预见与动作决策闭环地端到端联合训练。
primary_logic: |
  历史多视角图像/机器人状态 + 语言或未来状态目标 → [FRS] 预测未来视觉潜变量与图像，[INV] 在读取 foresight 的条件下预测多步动作 → 得到更数据高效、长时程更稳健的机器人操作策略
claims:
  - "Seer-Large 在 CALVIN ABC-D 上达到 Avg. Len. 4.28，超过 CLOVER 的 3.53，并刷新论文报告中的最佳结果 [evidence: comparison]"
  - "在 CALVIN 消融中，仅加入视觉前瞻可将 scratch 版 Avg. Len. 从 3.31 提升到 3.41，而联合视觉前瞻与逆动力学可进一步提升到 3.64；再加入同样的联合预训练目标后可到 3.98 [evidence: ablation]"
  - "在 4 个主要真实世界任务上，DROID 预训练使平均成功率从 60.0% 提升到 78.4%，并在多物体、自然背景、新物体和强光扰动下持续优于未预训练版本 [evidence: comparison]"
related_work_position:
  extends: "CLOVER (Bu et al. 2024)"
  competes_with: "CLOVER (Bu et al. 2024); OpenVLA (Kim et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); R3M (Nair et al. 2022)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Predictive_Inverse_Dynamics_Models_are_Scalable_Learners_for_Robotic_Manipulation.pdf
category: Embodied_AI
---

# Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2412.15109), [Project](https://nimolty.github.io/Seer/), [Code](https://github.com/OpenRobotLab/Seer/)
> - **Summary**: Seer 把“预测未来会看到什么”和“据此该怎么动”放进同一个 Transformer 里端到端共训，让大规模机器人数据同时提供视觉先验与动作先验，从而显著提升长时程操作与真实世界鲁棒性。
> - **Key Performance**: CALVIN ABC-D 上 Avg. Len. 4.28；真实世界 4 个主要任务平均 SR 78.4%（scratch 为 60.0%）

> [!info] **Agent Summary**
> - **task_path**: 多视角 RGB + 机器人状态 + 语言/未来状态目标 -> 多步机械臂/夹爪动作
> - **bottleneck**: 现有可扩展机器人学习把未来视觉约束与动作学习割裂，导致纯 BC 缺少前瞻、两阶段 PIDM 又有视觉-控制分布错配
> - **mechanism_delta**: 用 [FRS] 预测未来视觉、[INV] 单向读取该前瞻并输出动作，在共享 latent 中把 world modeling 与 inverse dynamics 端到端打通
> - **evidence_signal**: CALVIN ABC-D 上 4.28 Avg. Len. 的比较结果 + Lfore/Linv 联合优于单独前瞻的消融
> - **reusable_ops**: [foresight-readout-token, unidirectional-foresight-to-action-attention]
> - **failure_modes**: [cross-embodiment 预训练在高精度任务上可能负迁移, 缺少 wrist-view 或动作控制分布不匹配时收益明显下降]
> - **open_questions**: [PIDM 能否稳定跨机器人泛化, RGB foresight 是否是接触丰富操作的最佳未来表示]

## Part I：问题与挑战

这篇工作的核心不是“再做一个更大的机器人 policy”，而是指出：**真正的瓶颈在于动作学习缺少对未来视觉结果的内生约束**。

当前可扩展机器人操作大致有两条路线：

1. **action-centric**：像 RT-1、Octo、OpenVLA 这类方法直接在大规模机器人数据上做行为克隆。  
   问题是监督几乎全压在动作上，机器人轨迹里丰富的视觉时序信息没有被充分利用，模型容易学成“看到当前就反应”的局部策略。

2. **vision-centric / two-stage PIDM**：先学视觉表示或世界模型，再训练下游控制器。  
   问题是未来视觉预测器和逆动力学控制器通常分开训练，预测出的子目标分布与动作模型真正能消费的条件并不一致，训练时和推理时存在错配。

所以这篇 paper 要解决的真问题是：

- **如何把未来视觉预见直接变成动作决策的一部分，而不是一个外接模块**
- **如何让大规模机器人数据同时贡献视觉先验、动作先验和时序先验**
- **如何在下游数据很少时，仍然学到长时程、抗干扰的 manipulation policy**

为什么是现在？因为 DROID / Open X 这类大规模机器人轨迹已经出现，且包含图像、状态、动作三种信号；与此同时，真实世界微调数据昂贵，单靠下游 BC 已经很难继续拉高泛化性能。

**输入/输出接口**也很明确：

- **输入**：历史多视角 RGB（eye-on-hand + eye-on-base）、机器人状态、语言指令；若预训练数据缺语言，则退化为未来状态目标
- **输出**：多步 action chunk（机械臂 6D 动作 + gripper 开合）
- **边界条件**：主要针对语言条件/目标条件下的机器人操作，不是通用跨 embodiment 结论

## Part II：方法与洞察

Seer 的方法可以概括成一句话：**把“未来视觉”从外部子目标改成 policy 内部 latent，并让动作头直接读这个 latent。**

### 方法骨架

Seer 用一个共享的多模态 Transformer 编码语言、图像和机器人状态，并在每个时间步插入两个读出 token：

- **[FRS]**：负责 conditional visual foresight  
  它根据目标和历史观测，预测未来时刻的 RGB 表示与图像。作者选 RGB 作为 future target，因为它语义丰富、监督直接、且在机器人数据中容易获得。

- **[INV]**：负责 inverse dynamics prediction  
  它不是只看当前观测，而是额外读取 [FRS] 所代表的未来视觉 latent，再预测从当前到未来之间的多步动作。

最关键的设计不是“同时做两个任务”，而是：

- **[INV] 可以看 [FRS]**
- **[FRS] 不反向看 [INV]**

这个**单向注意力掩码**定义了清晰的因果方向：  
未来前瞻 → 动作决策。

这样一来，动作分支在训练时面对的就是模型自己预测的未来表征，而不是另一个独立模块产出的外部子目标，视觉和动作会在同一 latent 空间里共同适配。

此外，作者还做了几个很实用的设计来适配大规模机器人数据：

- **语言缺失**：预训练时可用未来 robot state 充当 goal，避免必须依赖完整文本标注
- **随机探索/无意义行为**：通过注意力约束降低对无效历史行为的记忆式过拟合
- **多步动作 chunk**：比单步动作更稳，对 idle action 更鲁棒

### 核心直觉

**what changed**  
从“先预测未来，再单独训练控制器”变成“在同一个模型里，把未来预测作为动作条件”。

**which bottleneck changed**  
动作学习的信息瓶颈从“只看当前观测”或“看一个外部分布的子目标”变成“看由同一模型生成、且对控制有用的未来 latent”。  
这同时改变了两件事：

1. **减少 two-stage distribution mismatch**：动作头在训练/测试看到的是同一种 foresight 表征
2. **让 foresight 变得 action-aware**：因为动作损失会反向塑造 [FRS]，模型学到的不是“视觉上像未来”，而是“对控制有用的未来”

**what capability changed**  
这直接带来更强的：

- 长时程任务完成能力
- 小样本微调效率
- 场景、物体、光照扰动下的鲁棒性

也就是说，这篇 paper 的关键贡献不是单纯把 world model 加进 policy，而是**把 future prediction 从“辅助视觉任务”变成“动作生成的因果条件”**。

### 战略取舍

| 设计选择 | 带来的能力 | 代价/风险 |
|---|---|---|
| 端到端 PIDM | 降低视觉-控制分阶段错配，增强长时程决策 | 联合优化更复杂，图像预测会增加训练负担 |
| [FRS]→[INV] 单向注意力 | 明确 future-to-action 因果流，保证 latent 融合 | 结构上更依赖 attention mask 设计是否合理 |
| 多步 action chunk | 提升时间一致性，对 idle action 更稳 | 高频纠偏可能不如逐步控制细腻 |
| 冻结 MAE/CLIP 编码器 | 小数据微调更稳定，训练参数量更可控 | 对新传感器/新视觉域的适配能力受限 |
| 用 RGB 做 foresight target | 监督直接、语义清晰 | 像素级未来不一定最适合高精度接触动力学 |

## Part III：证据与局限

### 关键证据信号

**1. 长时程 benchmark 的能力跃迁很明显**  
最强信号来自 CALVIN ABC-D：Seer-Large 达到 **Avg. Len. 4.28**，显著超过 CLOVER 的 3.53。  
这说明能力提升不是“单步动作更准”这么简单，而是**连续完成多指令链条的能力更强**。

**2. 提升并非来自“多加一个视觉辅助任务”，而是来自闭环联合训练**  
消融非常关键：

- scratch 版纯动作学习：Avg. Len. 3.31
- 只加 visual foresight：3.41
- visual foresight + inverse dynamics：3.64
- 再加同样的联合预训练：3.98

这条证据最直接地支持论文主张：  
**真正有效的不是单独预测未来图像，而是让预测未来去约束动作。**

**3. 小模型也能打过更大 VLA，说明机制比单纯参数更关键**  
在 LIBERO-LONG 上，标准版 Seer 平均成功率 **87.7%**，明显高于 MPI 和 OpenVLA。论文还特别指出，Seer 总参数约 316M，而比较对象 OpenVLA 为 7B 级别。  
这说明这篇工作的“能力跳跃”主要来自**闭环训练范式**，而不是只靠更大模型。

**4. 真实世界与抗扰动结果支持其外部有效性**  
在 4 个主要真实世界任务上，DROID 预训练把平均 SR 从 **60.0% 提到 78.4%**。  
而且在：

- 多干扰物体
- 自然背景
- 新物体
- 强光变化

这些设置下，预训练版都稳定优于 scratch 版。  
这说明学到的不只是基准技巧，还有一定程度的**语义与场景鲁棒性**。

**5. “scalable learner” 这个题目有实验支撑**  
作者还给了两个支持点：

- **数据效率**：只用 10% 下游数据时，相比从头训练，LIBERO-LONG 相对提升 187%，CALVIN 相对提升 150%
- **模型扩展性**：更大 trainable 参数规模带来更好 CALVIN 结果，且预训练版本收益持续存在

### 局限性

- **Fails when**: 跨 embodiment、控制器类型、相机配置差异较大时，尤其是高精度任务且缺少 wrist-view 数据时，预训练可能只有边际收益，甚至出现负迁移；附录里的 OXE 预训练结果就显示了这一点。
- **Assumes**: 依赖大规模、带状态与动作的机器人轨迹数据；最佳结果明显受益于 DROID 这类与下游机器人形态接近的数据。训练侧依赖冻结的 MAE/CLIP 编码器、8×4090 级别算力；真实世界复现还需要 Franka + Robotiq + RealSense 双视角配置，以及每任务约 100 条示范。
- **Not designed for**: 论文并未证明广泛的跨机器人泛化、触觉主导控制、或更大范围接触丰富任务的全面适用性；真实世界评估任务数仍然有限。

### 可复用部件

这篇 paper 最值得迁移的，不一定是完整 Seer，而是几个操作符：

- **foresight readout token**：把未来状态预测显式嵌入 policy 内部
- **[FRS]→[INV] 单向注意力**：让 future latent 成为动作生成的条件变量
- **goal fallback 机制**：语言缺失时用未来状态替代目标，适配弱标注机器人数据
- **multi-step action chunk**：在长时程操作里增强时间一致性

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Predictive_Inverse_Dynamics_Models_are_Scalable_Learners_for_Robotic_Manipulation.pdf]]