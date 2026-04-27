---
title: "Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - masked-decoding
  - action-chunking
  - dataset/LIBERO
  - dataset/SimplerEnv-Fractal
  - dataset/SimplerEnv-Bridge
  - opensource/no
core_operator: "在统一VLM Transformer内部对离散化动作chunk执行掩码式离散扩散，并用置信度驱动的二次重掩码迭代细化动作token。"
primary_logic: |
  多视角RGB图像+语言指令+可选末端位姿 → 将未来动作离散为固定长度token chunk并送入统一双向Transformer → 训练时随机mask动作token做去噪预测，推理时按置信度并行保留高置信token并对不稳定token二次重掩码 → 输出可执行动作chunk
claims:
  - "在LIBERO四个suite上，Discrete Diffusion VLA达到96.3%平均成功率，高于OpenVLA-OFT (Discrete) 的95.5%和OpenVLA的76.5% [evidence: comparison]"
  - "在LIBERO-Goal的OOD语言增强测试中，Discrete Diffusion VLA从97.4%降到96.0%，仅下降1.4%，小于OpenVLA-OFT (Discrete) 的8.0%下降和OpenVLA-OFT (Cont-Diffusion) 的2.4%下降 [evidence: comparison]"
  - "在LIBERO-Goal解码消融中，max-confidence + secondary re-masking将成功率从一次并行解码的95.6%提升到97.4% [evidence: ablation]"
related_work_position:
  extends: "OpenVLA (Kim et al. 2024)"
  competes_with: "OpenVLA-OFT (Kim et al. 2025b); π0 (Black et al. 2024)"
  complementary_to: "FiLM (Kim et al. 2025b); LAPA (Ye et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2025/New_experiments_on_VL_retention_and_new_ablations_18_pages_2025/2025_Discrete_Diffusion_VLA_Bringing_Discrete_Diffusion_to_Action_Decoding_in_Vision_Language_Action_Policies.pdf
category: Embodied_AI
---

# Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2508.20072)
> - **Summary**: 论文把VLA的动作解码从“左到右AR生成”或“外挂连续扩散头”改成统一Transformer内的离散扩散去噪，使动作token能按难易度并行细化并反复纠错，同时更好保留预训练视觉-语言先验。
> - **Key Performance**: LIBERO平均成功率96.3%；SimplerEnv-Fractal overall 64.1%，SimplerEnv-Bridge overall 54.2%。

> [!info] **Agent Summary**
> - **task_path**: 多视角RGB图像 + 语言指令 + 可选末端位姿 -> 固定长度离散动作chunk
> - **bottleneck**: AR解码的固定顺序与外挂动作头的结构割裂，使动作预测难并行、难纠错，也削弱了VLM骨干对动作建模的直接支撑
> - **mechanism_delta**: 将动作chunk离散成token并放入统一双向Transformer中，用离散扩散式mask去噪、置信度排序和二次重掩码完成并行动作解码
> - **evidence_signal**: 三个机器人基准的比较结果 + LIBERO-Goal解码消融 + LIBERO-OOD先验保留测试
> - **reusable_ops**: [固定长度动作chunk离散化, 置信度驱动二次重掩码]
> - **failure_modes**: [大视觉分布偏移下成功率仍明显下降, 去噪轮数过少时性能退化到接近一次并行解码]
> - **open_questions**: [能否扩展到可变长度与更长时域动作序列, 更大规模联合预训练下VL保真与控制精度是否仍能兼得]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再做一个更强的动作头”，而是**VLA里视觉-语言骨干与动作解码之间的信息通路是否统一**。

现有VLA大致有两条路：

1. **AR动作token解码**：在统一Transformer里按左到右顺序逐token生成动作。问题是顺序固定、早错难改、前向次数随chunk长度线性增长。
2. **外挂MLP/连续扩散头**：动作建模能力往往更强，但动作去噪发生在VLM之外，训练目标和表示路径与骨干分裂，难形成真正统一、可扩展的架构。

所以瓶颈是双重的：

- **结构瓶颈**：动作头和VLM骨干割裂，视觉/语言先验难无损传到动作层。
- **解码瓶颈**：AR的固定顺序限制了并行性和纠错能力。

论文的输入/输出接口很清楚：输入是RGB图像、语言指令、可选末端位姿；输出是未来若干步的**固定长度动作chunk**。每个时间步被量化为7个动作token，采用256-bin离散化。边界条件也很明确：只用RGB，不用depth、affordance等辅助模态；并且假设动作可以被稳定离散化成固定长度token序列。

为什么现在值得做？因为VLM backbone已经足够强，VLA开始从“任务可用”转向“统一架构是否可扩展”。如果动作仍然靠外挂head处理，就很难真正继承VLM的训练范式与扩展规律。

## Part II：方法与洞察

论文的核心动作很简单但关键：**把动作chunk当成固定长度离散序列，在同一个Transformer里做mask-based discrete diffusion。**

### 核心直觉

- **什么变了**：从“下一token预测 / 外挂连续扩散头”改成“统一双向Transformer内的随机mask动作去噪”。
- **哪个信息瓶颈变了**：动作不再受左到右因子分解约束，而变成“任意子集token补全”；同时动作位点可以直接访问完整视觉、语言、动作上下文。
- **能力如何变了**：模型可以先解容易token、后解困难token；对低置信token重新mask后再修正，因此获得并行解码、误差回滚和更强一致性。

更因果地看，关键不只是“用了diffusion”，而是训练分布发生了变化：

- `单一next-token训练` → `跨mask ratio的大量infilling子任务` → `推理时可按样本置信度自适应决定解码顺序`
- `外挂动作头` → `统一Transformer内动作位点` → `更直接复用预训练VL表示`
- `一次提交不可回头` → `secondary re-masking` → `早期错误可在后续轮次被纠正`

### 方法拆解

1. **动作离散化与chunk化**  
   连续控制量先量化成离散token，再按未来H步打包成固定长度chunk。这样动作接口和VLM的离散token接口一致，也让离散扩散的mask/denoise流程成立。

2. **统一Transformer架构**  
   作者基于OpenVLA/Prismatic-7B改造，把原先因果式动作解码改成对动作位点使用**双向注意力**。视觉、语言、动作共享同一个Transformer，只在动作位置读出logits。

3. **训练：mask动作token做去噪预测**  
   训练时随机采样mask比例，只对被mask的动作token做交叉熵恢复。好处是训练目标仍然是VLM熟悉的离散token预测，而不是额外引入一套专门的连续扩散训练栈。

4. **推理：自适应解码 + 二次重掩码**  
   推理从全mask动作chunk开始。每一轮：
   - 对当前mask位点预测分布；
   - 按最大置信度或置信度gap排序；
   - 先保留更“容易”的token；
   - 对置信度过低、或相对首次提交明显变差的token再次mask。  
   这相当于把动作解码顺序从“位置先验”改成“难度先验”。

### 战略取舍

| 设计选择 | 带来的好处 | 代价/风险 |
|---|---|---|
| 离散化 + 固定长度chunk | 与VLM token接口兼容，便于并行去噪 | 丢失部分连续控制细节，且需预设chunk长度 |
| 统一Transformer内解码 | 视觉/语言/动作信息流不割裂，更利于保留VL先验 | 需要改造原因果骨干，训练/显存更重 |
| easy-first自适应解码 | 打破固定顺序，提升精度与鲁棒性 | 仍需多轮refinement，不如一次并行最快 |
| 二次重掩码 | 允许纠正早期错误，增强跨轮一致性 | 依赖阈值和调度超参，推理逻辑更复杂 |

## Part III：证据与局限

### 关键证据

- **信号1：同token化条件下优于普通并行解码**  
  在LIBERO四个suite上达到**96.3%**平均成功率，高于OpenVLA-OFT (Discrete) 的**95.5%**。这说明收益不只是“把动作离散化”本身，而是离散扩散式的可修正解码过程带来的。

- **信号2：跨机器人设置仍成立**  
  在Google Robot的SimplerEnv-Fractal上达到**71.2% visual matching / 64.1% overall**，在WidowX的SimplerEnv-Bridge上达到**54.2% overall**，整体超过π0、π0-FAST与OpenVLA-OFT等基线，说明方法不是只在单一机器人或单一数据分布上有效。

- **信号3：更能保留预训练VL先验**  
  LIBERO-Goal OOD测试中，语言增强仅下降**1.4%**，明显小于OpenVLA-OFT (Discrete) 的**8.0%**；视觉增强下降**21.0%**，也好于连续扩散基线的**29.0%**。这直接支撑了作者的主张：把动作解码并回统一Transformer，确实更利于保留视觉-语言能力。

- **信号4：关键增益来自解码策略，而非单纯增加计算**  
  在LIBERO-Goal消融中，一次并行解码是**95.6%**，max-confidence提高到**97.0%**，再加secondary re-masking到**97.4%**。同时NFE从AR的**56**降到**12**，延迟从**136.2ms**降到**68.8ms**。说明它同时改善了精度和前向效率。

### 局限性

- **Fails when**: 视觉外观变化较大时仍会显著掉点；例如LIBERO-Goal的vision augmentation仍有**21.0%**降幅。另一个明显边界是去噪轮数过少时，方法会退化为近似一次并行解码，精度下降。
- **Assumes**: 动作可以被256-bin稳定离散化，并适合固定长度action chunk；方法依赖OpenVLA/Prismatic-7B这类预训练VLM骨干和按benchmark单独微调的演示数据；SimplerEnv-Bridge实验还额外使用FiLM增强语言grounding。
- **Not designed for**: 可变长度动作序列与EOS建模、原生连续高频控制、需要depth/触觉/力觉等额外模态的控制，以及长时规划/层级任务分解本身。

资源与复现依赖也值得明确：训练通常使用4张A800、batch size 32，速度测试在单张H800上完成。论文文本声称代码可用，但当前提供材料未包含可验证的项目或代码链接，因此按记录口径标为 `opensource/no`。

### 可复用组件

- **固定长度动作chunk离散化**：把控制问题转成离散token生成问题；
- **统一Transformer内的mask去噪解码**：适合任何希望避免外挂动作头的VLA；
- **置信度排序 + 二次重掩码**：可迁移到其他离散动作解码器中做纠错；
- **把解码顺序从位置先验改成样本难度先验**：这是比具体实现更可迁移的设计思想。

![[paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2025/New_experiments_on_VL_retention_and_new_ablations_18_pages_2025/2025_Discrete_Diffusion_VLA_Bringing_Discrete_Diffusion_to_Action_Decoding_in_Vision_Language_Action_Policies.pdf]]