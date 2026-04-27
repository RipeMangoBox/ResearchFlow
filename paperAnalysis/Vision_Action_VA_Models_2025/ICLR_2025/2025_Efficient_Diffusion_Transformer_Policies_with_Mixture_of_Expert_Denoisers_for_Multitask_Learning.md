---
title: "Efficient Diffusion Transformer Policies with Mixture of Expert Denoisers for Multitask Learning"
venue: ICLR
year: 2025
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - task/language-conditioned-manipulation
  - diffusion
  - mixture-of-experts
  - noise-conditioned-routing
  - dataset/CALVIN
  - dataset/LIBERO-10
  - dataset/LIBERO-90
  - opensource/full
core_operator: "按扩散噪声阶段将去噪计算路由到少量专家，并把每个噪声步的专家组合预缓存融合以降低策略推理成本"
primary_logic: |
  图像/状态历史、语言目标与带噪动作块 + 当前噪声水平 → 噪声条件自注意力编码上下文并用噪声路由选择少量去噪专家 → 迭代去噪得到未来动作序列
claims:
  - "Claim 1: 在 CALVIN ABC→D 上，预训练版 MoDE 达到 4.01 的平均 rollout length，超过文中报告的 GR-1 (3.06) 与 SuSIE (2.69) [evidence: comparison]"
  - "Claim 2: 与相近参数规模的 dense transformer 相比，带缓存的 MoDE 在 batch size 512 时将推理开销从 5772 GFLOPs 降至 361 GFLOPs，并把延迟从 104 ms 降至 64 ms [evidence: comparison]"
  - "Claim 3: 在 LIBERO-10 上，同时保留输入噪声 token 与噪声条件注意力的完整 MoDE 最优（0.92），优于去掉噪声条件注意力（0.85）和 FiLM 条件化（0.81） [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Diffusion Policy Transformer (Chi et al. 2023); GR-1 (Wu et al. 2024)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/ICLR_2025/2025_Efficient_Diffusion_Transformer_Policies_with_Mixture_of_Expert_Denoisers_for_Multitask_Learning.pdf
category: Embodied_AI
---

# Efficient Diffusion Transformer Policies with Mixture of Expert Denoisers for Multitask Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2412.12953), [Project](https://mbreuss.github.io/MoDE_Diffusion_Policy/)
> - **Summary**: 论文把扩散策略的去噪过程按噪声阶段拆成可专门化的专家子任务，并利用仅依赖噪声的路由实现专家缓存，从而在多任务机器人模仿学习中同时提升成功率和推理效率。
> - **Key Performance**: 预训练版在 CALVIN ABC→D 上达到 4.01 平均 rollout length、LIBERO-90 上达到 0.95 成功率；相对同规模 dense transformer，推理 FLOPs 最多约降 90%（361 vs 5772 GFLOPs, batch 512）

> [!info] **Agent Summary**
> - **task_path**: 语言目标 + 视觉/状态观测历史 + 带噪动作块 -> 未来动作序列
> - **bottleneck**: dense diffusion transformer 在每个 denoising step 都激活整套参数，且不同噪声阶段之间迁移有限，导致算力浪费与实时部署困难
> - **mechanism_delta**: 把每层前馈去噪器改成“按噪声水平选择少量专家”的稀疏 MoE，并将固定噪声步的专家组合预先缓存/融合
> - **evidence_signal**: 跨 CALVIN/LIBERO 的 SOTA 比较结果 + 噪声条件/路由策略的系统消融
> - **reusable_ops**: [noise-conditioned routing, expert caching]
> - **failure_modes**: [expert collapse, experts>4 时利用率下降并伴随性能回退]
> - **open_questions**: [是否能结合 noise-only routing 与 token-aware routing, 如何降低 router 初始化对训练方差的敏感性]

## Part I：问题与挑战

这篇论文针对的是**语言条件、多任务机器人模仿学习**中的一个现实瓶颈：扩散策略越来越强，但也越来越贵。

### 问题是什么
输入是：
- 当前视觉或状态观测历史
- 语言目标
- 当前带噪动作序列
- 当前噪声水平

输出是：
- 未来一段动作序列（文中通常是 10-step action chunk）

评测场景覆盖 CALVIN、LIBERO 等多任务 manipulation benchmark，既有长时程任务，也有跨环境 zero-shot 泛化。

### 真正的瓶颈是什么
作者抓到的核心点不是“模型还不够大”，而是：

**扩散去噪的不同噪声阶段，本质上是不同子任务；但传统 dense diffusion transformer 让它们共享同一套活跃参数。**

这会造成两类浪费：
1. **表示浪费**：高噪声阶段偏粗粒度恢复，低噪声阶段偏精修，它们需要的计算模式并不相同。
2. **计算浪费**：每个 denoising step 都要跑完整个 dense backbone，推理延迟随模型规模上升而变得难部署。

### 为什么现在值得解
因为扩散策略已经证明在机器人 IL 中有明显优势：能表达多模态行为、处理动作不连续、能随数据规模提升。但如果还继续依赖 dense 扩展，**会直接撞上实时控制和板载算力的上限**。

### 边界条件
这篇工作有很明确的适用边界：
- 主要是 **imitation learning**，不是 online RL
- 主要是 **语言条件操作任务**
- 依赖固定 diffusion rollout（文中统一用 10 个 denoising steps）
- 图像场景依赖 frozen CLIP 文本编码器与 FiLM-ResNet 图像编码器

---

## Part II：方法与洞察

MoDE 的关键不是“把 MoE 塞进扩散模型”，而是**把专家分工轴从 token 内容，改成 diffusion 的噪声阶段**。

### 方法主线

#### 1. 噪声条件自注意力
作者把噪声水平编码成 noise token，同时把该噪声嵌入加到所有 token 上再做 self-attention。

这带来的不是简单条件注入，而是：
- 同一组图像/语言/action token
- 在不同 denoising phase 下
- 会形成不同的注意模式

也就是说，**phase 信息进入了 token 交互层**，而不是只在末端去噪头里出现。

#### 2. 按噪声水平路由专家
每个 transformer block 的前馈层被替换成 MoE 层。与 LLM 中常见的 token-based router 不同，MoDE 的 router **只看当前噪声水平**来选 top-k experts。

这样做的因果逻辑很清楚：
- 高噪声步更像“粗恢复”
- 低噪声步更像“细修正”
- 既然去噪本身是分阶段任务，就让不同专家负责不同阶段

这比“按场景 token 路由”更贴近 diffusion policy 的内在结构。

#### 3. Expert caching
这是整篇论文最有系统价值的部分。

因为路由只依赖噪声，而推理时噪声 schedule 是预先已知的，所以：
- 每个 denoising step 会用到哪些专家，可以提前算好
- 这些专家的 MLP 可以融合成一个复合 MLP
- 推理时可以去掉动态 router 的额外开销

因此 MoDE 不只是“理论上更稀疏”，而是把稀疏性变成了**真正可落地的推理加速**。

#### 4. 预训练与迁移
作者还用 Open-X-Embodiment 子集做预训练，并在下游微调时**冻结 router**。

这背后的意图是：
- 先在大规模多机器人数据上学稳定的“噪声阶段 -> 专家职责”映射
- 再让其余组件适应具体任务域

这是一个很实用的工程选择，因为它减少了下游再训练时路由漂移的问题。

### 核心直觉

**what changed**：  
把“所有 denoising step 共用同一个 dense 去噪器”改成“不同噪声区间调用不同专家去噪器”。

**which bottleneck changed**：  
模型容量从“对所有 phase 平均共享”变成“对各个 phase 稀疏分配”；同时因为路由只依赖噪声，动态路由被转化成了可预计算的静态执行路径。

**what capability changed**：  
模型既能更好地专门化不同去噪阶段，又能显著减少活跃参数和推理 FLOPs，因此获得了“更准 + 更省”的联合收益。

一句话说，MoDE 不是在问“不同任务需不需要不同专家”，而是在问：  
**既然 diffusion 的不同噪声阶段本来就是不同任务，为什么还要让它们共享同一条 dense 计算路径？**

### 战略权衡

| 设计选择 | 改变了什么 | 收益 | 代价/风险 |
|---|---|---|---|
| noise-only routing | 路由信号从 token 内容改为噪声阶段 | 专家职责更贴合 denoising phase；支持 caching | 可能忽略由场景语义驱动的细粒度路由需求 |
| noise-conditioned attention + noise token | phase 信息进入 attention 交互层 | 提升去噪阶段感知能力 | 需要稳定的噪声注入设计 |
| sparse top-k experts | 仅激活少量专家 | 扩总容量但控 active compute | 会带来 router/expert collapse 风险 |
| freeze-router finetuning | 下游只调主体参数，不改专家分工 | 迁移更稳定 | 下游域若需重分配专家，灵活性受限 |

---

## Part III：证据与局限

### 关键证据信号

#### 信号 1：跨 134 个任务，性能不是局部提升
最强证据来自 CALVIN 和 LIBERO 多个设置的对比。

- **CALVIN ABC→D**：预训练版 MoDE 达到 **4.01** 平均 rollout length，超过 GR-1 的 3.06 和 SuSIE 的 2.69
- **CALVIN ABCD→D**：从零训练的 MoDE 已有 **4.30**，预训练版进一步到 **4.39**
- **LIBERO-10 / LIBERO-90**：MoDE 为 **0.92 / 0.91**，预训练版达到 **0.94 / 0.95**

作者还汇总给出：相比第二好的 diffusion policy 变体，MoDE 在四个 benchmark 上平均提升约 **57.5%**。  
这说明它不是只在某个数据集上“碰巧有效”。

#### 信号 2：效率提升是真实推理收益，不是纸面 FLOPs
和相近参数规模的 dense transformer 相比：
- batch size 512 时，MoDE with cache 为 **361 GFLOPs**
- dense baseline 为 **5772 GFLOPs**
- 推理时间从 **104 ms** 降到 **64 ms**

在 CALVIN 单动作预测分析中，预训练版 MoDE 约 **1.53 GFLOPs/action**，而 GR-1 是 **27.5 GFLOPs/action**，但延迟接近（12.2 vs 12.6 ms）。

这类结果很关键，因为它证明 expert caching 把噪声路由的结构优势真正兑现成了系统收益。

#### 信号 3：因果旋钮是“噪声感知路由/表示”，不是单纯堆参数
在 LIBERO-10 的消融里：
- 完整 MoDE：**0.92**
- 去掉 input noise token：**0.90**
- 去掉 noise-conditioned attention：**0.85**
- 改成 FiLM noise conditioning：**0.81**

说明提升不是“MoE 参数更多所以更强”，而是**噪声条件化方式本身是有效的因果变量**。

同时还有两个很重要的结构性观察：
- **noise-only routing** 略优于 token-only routing（归一化平均 0.851 vs 0.845），且多了缓存能力
- **4 experts** 是甜点；扩到 6/8 个 experts 后，利用率不升反降，性能也会回退

#### 信号 4：专家确实学会了按噪声阶段分工
作者可视化了不同 layer、不同 noise level 的 expert usage，观察到在 **σ8 左右存在明显的专家切换**。  
这说明 router 不是随机地把 token 打散，而是真的学到了“高噪声去谁、低噪声去谁”的 phase specialization。

### 局限性

- **Fails when**: 路由训练不稳定或专家塌缩时，MoDE 的方差会上升；当 experts 数增加到 6/8 时，模型往往无法充分利用新增专家，性能反而下降；若任务更依赖场景语义而不是噪声阶段，noise-only routing 可能不够细。
- **Assumes**: 需要固定已知的 diffusion noise schedule、较大规模 demonstration 数据、冻结的语言/视觉编码器，以及不低的训练资源；预训练版用了 196k 轨迹、6×A6000 GPU 训练约 3 天，下游微调也用到 4 GPU。
- **Not designed for**: 在线强化学习、一步式超低延迟控制、以及需要主要依据 token 内容做高度动态路由的设置。

### 可复用组件

这篇论文里最值得迁移到别的 diffusion transformer / robot policy 的组件有：

- **noise-conditioned routing**：把 denoising phase 当作专家分工轴
- **expert caching / fusion**：把固定噪声步的动态 MoE 变成静态可优化执行图
- **noise-conditioned attention**：让 phase 信息影响整个 token mixing
- **freeze-router finetuning**：先固定专家职责，再做任务迁移

### So what
这篇论文的能力跃迁点，不只是“MoE 让模型更大”，而是它把**去噪阶段是多子任务**这个观察，变成了一个可执行的架构旋钮。  
因此它相对 prior diffusion transformer 的优势是：**既提升多任务机器人成功率，又把推理路径变得足够高效，朝真实部署更近了一步。**

![[paperPDFs/Vision_Action_VA_Models_2025/ICLR_2025/2025_Efficient_Diffusion_Transformer_Policies_with_Mixture_of_Expert_Denoisers_for_Multitask_Learning.pdf]]