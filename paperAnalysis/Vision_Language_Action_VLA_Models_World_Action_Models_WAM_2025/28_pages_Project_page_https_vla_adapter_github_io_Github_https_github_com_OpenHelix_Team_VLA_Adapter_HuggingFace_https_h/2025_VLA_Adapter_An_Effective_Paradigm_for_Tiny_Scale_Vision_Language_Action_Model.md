---
title: "VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - cross-attention
  - query-token
  - dataset/LIBERO
  - dataset/CALVIN
  - opensource/full
core_operator: "用 Bridge Attention 将各层 Raw 特征与 ActionQuery 分层注入轻量 Policy，并以可学习门控控制 Raw 分支强度，实现小模型的 VL→A 高效桥接"
primary_logic: |
  第三视角图像 + 夹爪图像 + 文本指令 + 本体状态
  → VLM 提取各层 Raw 特征与各层 ActionQuery 特征
  → Policy 在每层通过 Bridge Attention 对动作 latent 分别做 Raw 交叉注意力、ActionQuery+proprio 交叉注意力与自注意力，并学习调节 Raw 注入比例
  → 输出 H 步连续动作块
claims:
  - "在 LIBERO 上，0.5B 的 VLA-Adapter 平均成功率达到 97.3%，略高于 7B OpenVLA-OFT 的 97.1%，其中 LIBERO-Long 为 95.0% [evidence: comparison]"
  - "在 LIBERO-Long 的桥接消融中，同时使用 all-layer Raw 与 all-layer ActionQuery 的条件可达 95.0%，高于仅用 all-layer Raw 的 90.6% 与仅用 all-layer ActionQuery 的 92.6% [evidence: ablation]"
  - "VLA-Adapter 的推理吞吐达到 219.2Hz、延迟 0.0365s，快于 OpenVLA-OFT 的 71.4Hz 与 0.1120s [evidence: comparison]"
related_work_position:
  extends: "OpenVLA-OFT (Kim et al. 2025)"
  competes_with: "OpenVLA-OFT (Kim et al. 2025); π0 (Black et al. 2025b)"
  complementary_to: "CoT-VLA (Zhao et al. 2025a); WorldVLA (Cen et al. 2025)"
evidence_strength: strong
pdf_ref: "paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2025/28_pages_Project_page_https_vla_adapter_github_io_Github_https_github_com_OpenHelix_Team_VLA_Adapter_HuggingFace_https_h/2025_VLA_Adapter_An_Effective_Paradigm_for_Tiny_Scale_Vision_Language_Action_Model.pdf"
category: Embodied_AI
---

# VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2509.09372), [Project](https://vla-adapter.github.io/), [GitHub](https://github.com/OpenHelix-Team/VLA-Adapter)
> - **Summary**: 这篇工作认为 VLA 的核心瓶颈不是“骨干还不够大”，而是“视觉语言信息如何被送进动作空间”；因此它先系统分析哪类层级条件最适合控动作，再用带门控的 Bridge Attention 把多层 Raw 特征和 ActionQuery 注入轻量 Policy，让 0.5B 模型在无需大规模机器人预训练阶段的前提下仍获得很强的操控性能。
> - **Key Performance**: LIBERO 平均成功率 97.3%（0.5B，略高于 7B OpenVLA-OFT 的 97.1%）；推理吞吐 219.2Hz、延迟 0.0365s

> [!info] **Agent Summary**
> - **task_path**: 第三视角RGB + 夹爪RGB + 语言指令 + 本体状态 -> 8步连续动作块/7维控制
> - **bottleneck**: 没有机器人预训练时，单靠 VLM 末层特征难以稳定映射到动作空间，细粒度感知与动作相关信息在桥接阶段被压缩丢失
> - **mechanism_delta**: 将各层 Raw 特征与各层 ActionQuery 同时作为条件，通过分层 Bridge Attention 注入动作 latent，并用可学习门控只选择性引入 Raw 分支
> - **evidence_signal**: LIBERO/CALVIN 多基准对比 + 条件类型/层位/门控/Query 数量消融
> - **reusable_ops**: [all-layer condition routing, gated cross-attention bridge]
> - **failure_modes**: [真实世界分布偏移较大时泛化下降, ActionQuery 数量过少或过多都会伤害性能]
> - **open_questions**: [该桥接范式能否迁移到双臂/人形控制, 如何与 RL 或大规模 embodied pretraining 结合]

## Part I：问题与挑战

这篇论文要解决的真问题，不是“怎么再做一个更大的 VLA”，而是：

**在视觉-语言表征已经足够强的前提下，怎样把这些表征更有效地桥接到动作空间？**

### 1. 真正的难点是什么

现有很多 VLA 方法之所以表现强，往往隐含依赖两个前提：

1. **大骨干 VLM**
2. **大规模机器人数据预训练**

这会让 VLM 的末层表征天然更“动作化”，于是后面接一个相对简单的 Policy 也能工作。但这带来明显代价：

- 训练成本高
- VRAM 占用大
- 推理吞吐低
- 小模型在没有机器人预训练时很难直接复现这种能力

论文的判断很明确：  
**问题不一定出在感知模型不够强，而更可能出在 VL→A 的接口设计错了。**

### 2. 这个问题为什么现在值得解

因为 0.5B 级别的开源 VLM 已经具备不错的视觉-语言理解能力，真正限制机器人落地的，开始变成：

- 能不能在**小模型**上做控制
- 能不能在**没有专门机器人预训练阶段**时也跑起来
- 能不能在部署时保持**高吞吐、低延迟**

换句话说，**如果桥接方式足够好，模型规模和预训练规模的必要性就能被大幅降低。**

### 3. 输入/输出接口与边界条件

本文处理的是典型的 instruction-conditioned manipulation：

- **输入**
  - 第三视角 RGB 图像
  - 夹爪视角图像
  - 文本指令
  - 本体状态（proprioception）
- **输出**
  - 连续动作 chunk（H=8）
  - 底层是 7 维控制动作

边界条件也比较明确：

- 主要面向**单臂操作**
- 任务集中在 **LIBERO / CALVIN** 这类指令驱动操作场景
- 核心目标是**高效 bridge design**
- 不是通用世界模型、也不是在线 RL 框架

---

## Part II：方法与洞察

这篇工作的思路很清楚：

> **先回答“该把什么信息送进动作空间”，再回答“怎么送进去”。**

### 1. 先做桥接条件诊断，而不是直接堆结构

作者先系统比较四种条件：

- 单层 Raw features
- 全层 Raw features
- 单层 ActionQuery features
- 全层 ActionQuery features

得到的结论很关键：

- **Raw 特征**：中层比深层更适合动作生成  
  因为深层更偏语义压缩，中层还保留较多视觉-文本细节。
- **ActionQuery 特征**：深层比浅层更适合  
  因为 ActionQuery 是从头训练的，深层更能聚合多模态信息。
- **多层 > 单层**  
  不仅效果更好，也避免手工选“最佳层”的设计成本。
- **只用 ActionQuery 不够，只用 Raw 也不够**  
  all-layer ActionQuery 整体更强，但一些困难子任务里中层 Raw 还能补充关键信息。

这一步很重要，因为它把桥接问题从“经验调结构”变成了“信息路径选择”。

### 2. VLA-Adapter 的核心结构

在此基础上，作者提出一个轻量 **L1-based Policy**，每层都做三件事：

1. **动作 latent 对 Raw features 做 cross-attention**
2. **动作 latent 对 ActionQuery + proprio 做 cross-attention**
3. **动作 latent 自己做 self-attention**

然后把这三路结果拼接，再经过 FFN 更新动作 latent。

其中最关键的是 **Bridge Attention**：

- Raw 分支保留更细粒度的视觉-语言细节
- ActionQuery 分支承担更“动作友好”的多模态聚合作用
- proprio 与 ActionQuery 合并，给动作预测提供控制状态锚点
- 对 Raw 分支使用一个可学习门控 `tanh(g)`，避免 Raw 信息过强扰乱动作分布

### 核心直觉

**变化点**：  
从“固定取某一层/某一种条件做桥接”，变成“多层条件 + 双通路注入 + Raw 分支门控”。

**改变了什么瓶颈**：  
它把原来依赖末层语义压缩的单一桥接，改成了**分层输送信息**：

- ActionQuery 负责把多模态信息压成更动作可用的接口
- Raw 特征负责补足被高层语义压缩掉的细粒度信息
- 门控负责避免 Raw 细节过量进入动作空间造成训练不稳

**带来了什么能力变化**：  
即使 backbone 很小、也没有大规模机器人预训练阶段，Policy 仍能拿到足够的动作相关信息，因此小模型也能稳定地产生高质量动作。

### 3. 为什么这个设计是因果有效的

因果链条可以概括成：

**从“末层特征是否已经动作化”这个强假设，转向“显式提供多层可控条件”**  
→ 降低了桥接时的信息瓶颈  
→ 减少了对大规模 embodied pretraining 的依赖  
→ 小模型也能保住控制性能，而且更快

论文里有一个很强的佐证：  
当 backbone 冻结时，OpenVLA-OFT 在 LIBERO-Long 上掉到 **0.0**，而 VLA-Adapter 仍有 **86.4**。这说明它不是靠“把 backbone 再训动作化”来工作，而是桥接接口本身就更有效。

### 4. 策略取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| all-layer 条件输入 | 单层选择容易丢信息、且要手工挑层 | 更稳、更通用 | 计算量高于单层方案 |
| ActionQuery 作为主桥接接口 | 原始 VLM 表征不一定天然适配动作 | 更动作导向的多模态聚合 | Query 太多会冗余 |
| Raw 分支可学习门控 | 细粒度信息有用，但直接全灌会扰乱动作分布 | 困难子任务更稳 | 需要学习到合适注入比例 |
| L1-based Policy 而非 DiT | 追求高吞吐和快速微调 | 更快、实测更强 | 不强调动作分布建模多样性 |
| 小 backbone（0.5B） | 降低部署成本 | 低 VRAM、高效率 | 真实世界开放泛化仍受限 |

---

## Part III：证据与局限

### 1. 关键证据信号

#### Signal 1：同 backbone 下只换 bridge，性能显著上升
这是最关键的因果证据。

在 LIBERO-Long 上：

- Qwen2.5-0.5B + OFT：**85.8**
- Qwen2.5-0.5B + Ours：**95.0**

同样地，在更大 backbone 上也成立：

- LLaMA2-7B + OFT：87.5 → Ours：95.2
- OpenVLA-7B + OFT：94.5 → Ours：95.4

这说明论文的增益主要不是来自“换更大模型”，而是来自**桥接范式**本身。

#### Signal 2：桥接消融与门控消融能支撑机制解释
在 LIBERO-Long 的条件类型消融中：

- all-layer Raw only：**90.6**
- all-layer ActionQuery only：**92.6**
- Raw + ActionQuery：**95.0**

再看门控：

- Raw 用学习门控、ActionQuery 全注入最好：**95.0**
- 两者都固定全注入或都门控，都会下降

这很直接地支持了作者的机制主张：

- ActionQuery 是主干条件
- Raw 是补充条件
- 补充条件需要被“选择性”注入

#### Signal 3：增益在“无机器人预训练 / 冻结骨干”场景最明显
这是本文最有价值的实际结论之一。

在冻结 backbone 的实验里：

- OpenVLA-OFT：**0.0**
- SmolVLA：**77.0**
- VLA-Adapter：**86.4**

这说明 VLA-Adapter 解决的正是小模型/冻结模型最痛的点：  
**不是把感知网络继续训得更强，而是让已有表征更有效地进入动作空间。**

#### Signal 4：不是只涨准确率，也涨效率
效率结果很亮眼：

- **219.2Hz** 吞吐
- **0.0365s** 延迟

对比 OpenVLA-OFT：

- 71.4Hz
- 0.1120s

而且 Figure 1 还给出训练显存：

- VLA-Adapter：**24.7GB**
- OpenVLA-OFT：**62GB**

这意味着它的价值不是“牺牲效率换精度”，而是同时推进了两者。

#### Signal 5：跨基准泛化并非只在一个榜单上成立
- **LIBERO**：平均 **97.3**
- **CALVIN ABC→D**：Avg. len **4.42**
- 真实世界 4 类任务：整体优于 ACT 和 OFT-style baseline

因此这不是单一 benchmark trick，至少在模拟和小规模真实环境中都显示出一致趋势。

### 2. 1-2 个最关键指标

如果只记两个数，我会记：

- **LIBERO Avg 97.3%（0.5B）**
- **219.2Hz 推理吞吐**

前者说明“小模型也能强”，后者说明“而且真能部署”。

### 3. 局限性

- **Fails when**: 真实世界分布偏移更大、任务组合更开放、对象/场景变化更多时，模型泛化仍会下降；论文也明确承认由于没有大规模 embodied pretraining 且模型很小，真实世界泛化还需提升。
- **Assumes**: 需要任务级示范数据做 imitation learning；依赖 Prismatic-VLM 风格骨干、第三视角+夹爪视角输入、本体状态输入，以及 Policy 层数与 VLM 层数对齐；ActionQuery 数量需要调参（文中 64 最优）；另外，摘要强调可在单张消费级 GPU 上 8 小时训练，但主实验设置仍基于 **4×H100**，因此“低成本”结论更像是方法潜力，而非所有实验都已在消费级硬件完整验证。
- **Not designed for**: 双臂/人形全身控制、在线交互式学习、强化学习式后优化、超长时程规划或开放世界机器人系统整合。

### 4. 可复用组件

这篇论文里最值得复用的，不是具体参数，而是以下操作模块：

- **all-layer 条件路由**：先别默认末层最好，先把多层都接入再决定
- **ActionQuery 作为 bridge token**：在冻结或弱微调 backbone 场景特别有用
- **双分支 Bridge Attention**：把“动作友好的聚合表示”和“细粒度补充表示”分开处理
- **Raw 分支学习门控**：让补充信息是“可控注入”，不是“无脑拼接”
- **L1 chunked policy**：在追求实时控制时，比 diffusion/DiT 风格更实用

![[paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2025/28_pages_Project_page_https_vla_adapter_github_io_Github_https_github_com_OpenHelix_Team_VLA_Adapter_HuggingFace_https_h/2025_VLA_Adapter_An_Effective_Paradigm_for_Tiny_Scale_Vision_Language_Action_Model.pdf]]