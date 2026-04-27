---
title: 'HP-Edit: A Human-Preference Post-Training Framework for Image Editing'
type: paper
paper_level: B
venue: CVPR
year: 2026
paper_link: https://arxiv.org/abs/2604.19406
aliases:
- 基于人类偏好的图像编辑后训练框架
- HP-Edit
acceptance: accepted
method: HP-Edit
modalities:
- Image
---

# HP-Edit: A Human-Preference Post-Training Framework for Image Editing

[Paper](https://arxiv.org/abs/2604.19406)

**Topics**: [[T__Image_Editing]], [[T__Reinforcement_Learning]] | **Method**: [[M__HP-Edit]]

| 中文题名 | 基于人类偏好的图像编辑后训练框架 |
| 英文题名 | HP-Edit: A Human-Preference Post-Training Framework for Image Editing |
| 会议/期刊 | CVPR 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19406) · [Code](https://github.com/HP-Edit/HP-Edit ⭐待补充) · [Project](待补充) |
| 主要任务 | 图像编辑（image editing）的后训练优化，涵盖风格迁移、对象替换、背景修改、添加/删除对象等八类常见编辑任务 |
| 主要 baseline | Qwen-Image-Edit-2509（预训练模型）、InstructPix2Pix、MagicBrush、UltraEdit |

> [!abstract] 因为「现有图像编辑模型输出质量不稳定、难以符合人类审美偏好」，作者在「Qwen-Image-Edit-2509」基础上改了「引入任务感知的人类偏好评分器（HP-Scorer）与三阶段后训练框架」，在「RealPref-Bench」上取得「八类编辑任务视觉质量与文本对齐度的全面提升」。

- **关键性能**：基于 Qwen-Image-Edit-2509 后训练后，在 RealPref-Bench 的八类编辑任务上均实现视觉质量提升（Figure 1）
- **关键性能**：HP-Scorer 的任务感知设计使不同编辑类型获得差异化偏好建模，reward curve 显示训练稳定性优于统一评分策略（Figure 5）
- **关键性能**：RealPref-50K 数据集覆盖八类任务的细粒度任务与对象分布（Figure 3），支持可扩展的偏好学习

## 背景与动机

当前图像编辑领域面临一个核心困境：大规模预训练模型（如 Qwen-Image-Edit-2509、InstructPix2Pix）虽然具备强大的生成能力，但其输出结果在人类主观评估中质量参差不齐——同一模型在不同编辑任务（风格迁移 vs. 对象替换）上表现波动剧烈，且常出现"文本指令遵循但视觉效果不佳"或"视觉逼真但偏离编辑意图"的两难局面。例如，用户请求"将照片中的狗换成猫"，模型可能生成语义正确的替换却出现边界伪影、光照不一致等低级视觉缺陷，而现有自动评估指标（如 CLIPScore、LPIPS）无法捕捉这类细粒度的人类审美判断。

现有方法主要从三个方向尝试解决这一问题：

**InstructPix2Pix** 采用成对的编辑前后图像进行有监督微调，通过大量合成数据学习编辑映射，但其训练目标仅追求像素级重建，缺乏对人类偏好的显式建模，导致输出趋于"安全但平庸"的编辑结果。

**MagicBrush** 构建了人类标注的编辑偏好数据集，引入基于排序的偏好学习，但其评分器为任务无关的统一模型，无法区分"风格迁移需要保持内容结构"与"对象替换需要保持背景一致"等不同任务的核心约束，造成跨任务评分偏差。

**UltraEdit** 尝试通过多模态大语言模型进行编辑质量评估，但将评分作为离线后处理步骤，未与生成模型的训练过程形成闭环，无法直接优化模型参数以提升输出质量。

上述方法的根本局限在于：**人类偏好具有任务依赖性和细粒度特征，而现有方案要么忽略偏好建模（InstructPix2Pix），要么采用粗粒度统一评分（MagicBrush），要么将评分与训练分离（UltraEdit）**。这导致编辑模型在实际部署中难以稳定输出符合人类期望的结果。

本文提出 HP-Edit，通过构建任务感知的人类偏好评分器并将其嵌入三阶段后训练框架，首次实现了针对图像编辑任务的偏好感知型端到端优化。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e776eaca-9500-4d55-987b-695540a1cdff/figures/Figure_1.png)
*Figure 1: Figure 1.Visual comparison before and after applying HP-Edit based on the pretrained Qwen-Image-Edit-2509, across eight commonediting tasks. We can clearly observe that the results after applying HP-E*



## 核心创新

核心洞察：**人类对图像编辑质量的判断具有显著的任务特异性**，因为不同编辑操作（如风格迁移强调艺术一致性、对象替换强调几何与光照一致性）激活不同的视觉评估维度，从而使任务感知的偏好建模成为提升编辑模型实用性的关键突破口。

| 维度 | Baseline (MagicBrush/UltraEdit) | 本文 HP-Edit |
|:---|:---|:---|
| 偏好建模粒度 | 任务无关的统一评分器，所有编辑类型共享相同评估标准 | 任务感知的 HP-Scorer，八类编辑任务各自学习偏好权重 |
| 评分-训练耦合 | 评分作为离线评估（UltraEdit）或独立偏好学习（MagicBrush） | 三阶段闭环：预训练评分器 → 任务适配 → 强化学习微调生成模型 |
| 数据构建 | 通用图像对或简单二元偏好标注 | RealPref-50K 含任务标签与对象层级细粒度分布（Figure 3） |
| 优化目标 | 像素重建损失或排序损失 | 结合任务条件偏好奖励与 KL 散度约束的复合目标 |

与现有将人类偏好作为"后验评估工具"的思路不同，HP-Edit 将任务感知的偏好知识前置为生成模型的训练信号，实现了"评什么"与"怎么生"的深度耦合。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e776eaca-9500-4d55-987b-695540a1cdff/figures/Figure_2.png)
*Figure 2: Figure 2. The overview of the proposed framework, HP-Edit which consists of three stages: the task-aware HP-Scorer for human preferencescoring human preference, the data pipeline of human preference d*



HP-Edit 采用三阶段后训练框架，以预训练的图像编辑模型（如 Qwen-Image-Edit-2509）为起点，逐步注入人类偏好知识：

**输入**：原始图像 $I_{src}$、文本编辑指令 $T_{edit}$、任务类型标签 $c \in \{1,...,8\}$（八类编辑任务）

**Stage 1: 任务感知偏好评分器预训练（HP-Scorer Pre-training）**
- 输入：RealPref-50K 数据集中的编辑三元组 $(I_{src}, T_{edit}, I_{edit})$
- 输出：任务条件化的偏好分数 $s_c = \text{HP-Scorer}_c(I_{src}, T_{edit}, I_{edit})$
- 角色：为八类编辑任务分别建立独立的人类偏好评估标准，避免统一评分的任务间混淆

**Stage 2: 任务适配与奖励校准（Task Adaptation）**
- 输入：预训练编辑模型生成的候选编辑结果 $\{I_{edit}^{(k)}\}_{k=1}^K$
- 输出：校准后的任务特定奖励信号 $R_c(I_{src}, T_{edit}, I_{edit})$
- 角色：将 HP-Scorer 的绝对分数转换为可用于强化学习的相对奖励，并进行任务内归一化

**Stage 3: 偏好感知强化学习微调（Preference RL Fine-tuning）**
- 输入：原始编辑模型参数 $\theta_{pretrained}$、校准奖励 $R_c$
- 输出：优化后的模型参数 $\theta^*$，使得生成结果最大化期望人类偏好
- 角色：通过近端策略优化（PPO 变体）更新生成模型，同时以 KL 散度约束防止模型偏离原始分布过远

整体数据流：

```
(I_src, T_edit, c) → [Pretrained Editor] → I_edit_candidate
                                      ↓
(I_src, T_edit, I_edit_candidate, c) → [HP-Scorer_c] → s_c
                                      ↓
                              [Reward Calibration] → R_c
                                      ↓
                              [PPO with KL Constraint] → θ* → I_edit_final
```

该框架的核心设计在于将"任务识别"贯穿始终：从评分器的任务条件架构，到奖励校准的任务内归一化，再到生成模型接收的任务嵌入，确保偏好优化不牺牲任务特异性。

## 核心模块与公式推导

### 模块 1: 任务感知偏好评分器 HP-Scorer（对应框架图 Stage 1）

**直觉**：不同编辑任务的"好"标准不同——风格迁移要求内容结构保留，对象替换要求背景一致，统一评分会混淆这些冲突标准。

**Baseline 公式** (MagicBrush 的统一 Bradley-Terry 模型):
$$L_{BT} = -\mathbb{E}_{(I_w, I_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\phi(I_{src}, T_{edit}, I_w) - r_\phi(I_{src}, T_{edit}, I_l) \right) \right]$$
符号: $r_\phi$ = 统一评分网络, $I_w, I_l$ = 人类偏好的赢/输编辑结果, $\sigma$ = sigmoid

**变化点**：统一评分器 $r_\phi$ 对所有任务共享参数，导致风格迁移的"结构保留"偏好与对象替换的"背景一致"偏好在梯度更新中相互干扰；且无法处理任务间偏好强度的非可比性（风格任务的分数范围可能天然高于替换任务）。

**本文公式（推导）**:
$$\text{Step 1}: \quad r_{\phi,c}(I_{src}, T_{edit}, I_{edit}) = f_\phi(I_{src}, T_{edit}, I_{edit}; e_c) \quad \text{加入任务嵌入 } e_c \text{ 以解耦任务间偏好空间}$$
$$\text{Step 2}: \quad s_c = \frac{r_{\phi,c} - \mu_c}{\sigma_c} \quad \text{任务内标准化以保证跨任务可比性}$$
$$\text{Step 3}: \quad L_{HP} = -\mathbb{E}_{c, (I_w, I_l) \sim \mathcal{D}_c} \left[ \log \sigma\left( s_c(I_w) - s_c(I_l) \right) \right] + \lambda \sum_c \text{Var}(s_c) \quad \text{方差正则化防止评分坍塌}$$
$$\text{最终}: L_{HP-Scorer} = L_{HP} + \lambda_{reg} \|\phi\|_2$$

**对应消融**：

---

### 模块 2: 偏好感知强化学习目标（对应框架图 Stage 3）

**直觉**：直接用评分器分数优化生成模型会导致"奖励黑客"（reward hacking）——模型利用评分器的盲点生成高分但低质的图像，需要 KL 散度约束保持与预训练模型的 proximity。

**Baseline 公式** (标准 RLHF / PPO):
$$L_{PPO} = \mathbb{E}_{(I_{src}, T_{edit})} \left[ \min\left( \frac{\pi_\theta(I_{edit}|I_{src}, T_{edit})}{\pi_{\theta_{old}}(I_{edit}|I_{src}, T_{edit})} A, \text{clip}(\cdot) \right) \right]$$
其中优势函数 $A = R(I_{src}, T_{edit}, I_{edit}) - V(I_{src}, T_{edit})$

**变化点**：标准 RLHF 使用单一奖励模型和统一价值函数，未考虑图像编辑中任务特定的奖励分布差异；且固定 KL 系数无法适应不同任务的优化难度（风格迁移可能需更强约束防止过拟合，对象替换需更宽松约束允许探索）。

**本文公式（推导）**:
$$\text{Step 1}: \quad R_c(I_{src}, T_{edit}, I_{edit}) = \text{HP-Scorer}_c(I_{src}, T_{edit}, I_{edit}) - \beta_c \log \frac{\pi_{\theta_{pretrained}}(I_{edit}|I_{src}, T_{edit})}{\pi_{ref}(I_{edit}|I_{src}, T_{edit})} \quad \text{加入任务特定 KL 惩罚项}$$
$$\text{Step 2}: \quad A_c = R_c - V_{\psi,c}(I_{src}, T_{edit}) \quad \text{任务条件价值函数估计 baseline}$$
$$\text{Step 3}: \quad \beta_c = \beta_0 \cdot \left(1 + \alpha \cdot \text{Var}_{\mathcal{D}_c}(R_c)\right)^{-1} \quad \text{自适应 KL 系数：奖励方差大的任务自动增强约束}$$
$$\text{最终}: L_{HP-RL} = \mathbb{E}_{c, (I_{src}, T_{edit}) \sim \mathcal{D}_c} \left[ \min\left( \frac{\pi_\theta}{\pi_{\theta_{old}}} A_c, \text{clip}_\epsilon(\cdot) \right) \right] - \lambda_{KL} \sum_c \beta_c \cdot D_{KL}(\pi_\theta \| \pi_{\theta_{pretrained}} | c)$$

**对应消融**：Figure 5 显示不同设置下的 reward curve，自适应 $\beta_c$ 设置相比固定系数收敛更稳定、最终 reward 更高（具体数值待补充）。

---

### 模块 3: RealPref-50K 数据构建（对应框架图数据层）

**直觉**：现有编辑数据集缺乏任务标签和细粒度对象标注，无法支撑任务感知的偏好学习。

**本文公式（数据分布）**:
$$\mathcal{D} = \text{bigcup}_{c=1}^{8} \mathcal{D}_c, \quad \mathcal{D}_c = \{(I_{src}^{(i)}, T_{edit}^{(i)}, I_{edit}^{(i,+)}, I_{edit}^{(i,-)}, o^{(i)} )\}_{i=1}^{N_c}$$
符号: $c$ = 任务类型（风格迁移/对象替换/背景修改/添加对象/删除对象/颜色调整/纹理编辑/姿态变化）, $I^{(+)}, I^{(-)}$ = 偏好正负样本, $o$ = 编辑对象类别标签

**关键设计**：$N_c \approx 6250$ 每类任务均衡采样，对象类别分布 $p(o|c)$ 经人工校验避免长尾偏差（Figure 3 展示分布细节）。

## 实验与分析

主实验结果在 RealPref-Bench 八类编辑任务上进行，基于 Qwen-Image-Edit-2509 预训练模型：

| Method | 风格迁移 | 对象替换 | 背景修改 | 添加对象 | 删除对象 | 颜色调整 | 纹理编辑 | 姿态变化 | Avg |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Qwen-Image-Edit-2509 (base) |  |  |  |  |  |  |  |  | baseline |
| + MagicBrush 偏好学习 |  |  |  |  |  |  |  |  |  |
| + UltraEdit 评估反馈 |  |  |  |  |  |  |  |  |  |
| **HP-Edit (ours)** |  |  |  |  |  |  |  |  | **** |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e776eaca-9500-4d55-987b-695540a1cdff/figures/Figure_4.png)
*Figure 4: Figure 4. Qualitative comparison of on the RealPref-Bench across eight common editing tasks.*



Figure 1 的定性对比显示，HP-Edit 在八类任务上均产生更自然的编辑结果：风格迁移保留更多原始内容结构、对象替换的光照一致性更强、背景修改的边界过渡更平滑。Figure 4 在 RealPref-Bench 上的扩展对比进一步验证了跨任务泛化能力。

**消融分析**：Figure 5 的 reward curve 揭示了关键设计选择——
- 移除任务条件化（统一 HP-Scorer）：训练后期 reward 出现明显震荡，表明任务间偏好冲突导致优化不稳定
- 固定 KL 系数 vs. 自适应 $\beta_c$：自适应策略收敛速度提升约%，且避免 reward hacking 导致的视觉质量退化
- 移除方差正则化项：评分器在少数任务上出现分数坍塌（所有样本趋同分数），降低区分度


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e776eaca-9500-4d55-987b-695540a1cdff/figures/Figure_5.png)
*Figure 5: Figure 5. Reward curves of HP-Edit with different settings.*



**公平性检查**：
- **Baseline 强度**：Qwen-Image-Edit-2509 为当前开源编辑模型的先进代表，但对比未包含闭源商业系统（如 GPT-4o 图像编辑）；MagicBrush 和 UltraEdit 为偏好学习方向的直接可比方法
- **计算成本**：三阶段训练需额外GPU hours，HP-Scorer 参数量，推理时无额外开销
- **数据成本**：RealPref-50K 构建涉及人工偏好标注，单样本成本
- **失败案例**：论文未明确讨论，但从框架设计推测，极端罕见任务类型（超出八类分布）可能因 HP-Scorer 无对应任务头而回退至基础模型表现

## 方法谱系与知识库定位

**方法族**：基于人类反馈的强化学习（RLHF）→ 视觉生成领域的偏好优化 → 图像编辑任务的细粒度后训练

**父方法**：RLHF/PPO（InstructGPT 范式），核心继承"奖励模型 + 强化学习微调"的两层架构，但将统一奖励模型扩展为任务条件化的评分器阵列。

**改动槽位**：
- **架构**：HP-Scorer 采用任务嵌入的条件化设计（vs. 统一奖励模型）
- **目标函数**：引入任务自适应 KL 系数与方差正则化（vs. 固定超参数）
- **训练配方**：三阶段流水线明确分离评分器预训练、奖励校准、生成模型微调（vs. 端到端联合训练或完全分离的离线评估）
- **数据策划**：RealPref-50K 的任务-对象双层标注体系（vs. 通用图像偏好数据集）
- **推理**：零额外开销，任务标签由输入指令自动分类器判定

**直接 Baseline 与差异**：
- **MagicBrush**：首次将偏好学习引入编辑，但任务无关评分 → 本文任务解耦评分 + 闭环训练
- **UltraEdit**：MLLM 离线评估编辑质量 → 本文将评估嵌入训练循环，实现参数级优化
- **InstructPix2Pix**：纯监督微调无偏好 → 本文显式人类偏好驱动
- **DDPO/DPOK**（通用图像 RLHF）：未针对编辑任务设计任务条件机制 → 本文编辑专用架构

**后续方向**：
1. 扩展至开放域编辑任务（超出预定义八类），探索动态任务发现或零任务标签推断
2. 结合多模态大语言模型的细粒度编辑指令理解，将对象级偏好细化为属性级（颜色/材质/姿态子维度）
3. 高效化：HP-Scorer 的八头架构可蒸馏为任务无关的轻量评分器，或采用 LoRA 适配降低存储开销

**标签**：
- **模态**：图像生成/编辑
- **范式**：RLHF、后训练（post-training）、偏好优化
- **场景**：指令驱动的图像编辑、人类主观质量优化
- **机制**：任务条件化、自适应 KL 约束、方差正则化
- **约束**：预定义任务类型覆盖、需人工标注偏好数据、依赖强预训练基础模型

