---
title: "Do Visual Imaginations Improve Vision-and-Language Navigation Agents?"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/vision-language-navigation
  - diffusion
  - early-fusion
  - cross-modal-alignment
  - dataset/R2R
  - dataset/REVERIE
  - opensource/partial
core_operator: "把指令中的地标子短语离线生成为视觉“想象图”，作为额外模态早融合进 VLN 代理，并用文本-想象对齐损失强化视觉 grounding"
primary_logic: |
  自然语言导航指令 + 当前/历史全景观测
  → 子指令切分与名词短语过滤，使用文生图模型生成地标想象图
  → ViT+MLP 编码后与文本表示早融合，并在训练中显式拉近“想象图—对应名词短语”表示
  → 提升未见环境中的地标识别、语义消歧与导航动作/停下预测
claims:
  - "在 R2R val-unseen 上，HAMT-Imagine 将 SR/SPL 从 66.24/61.51 提升到 67.26/62.02，DUET-Imagine 将 SR 从 71.52 提升到 72.12，说明 visual imagination 对两类 VLN 架构都有稳定但温和的增益 [evidence: comparison]"
  - "在 REVERIE 上，DUET-Imagine 将 SR 从 46.98 提升到 48.28，RGS 从 32.15 提升到 32.97，表明该方法也适用于更粗粒度、目标导向的指令导航 [evidence: comparison]"
  - "消融表明顺序多子目标 imaginations 优于只给终点图像，文本-想象对齐损失优于无辅助损失，而且错误 imaginations 会把性能拉回基线附近，支持“对齐的视觉锚点”是核心因果因素 [evidence: ablation]"
related_work_position:
  extends: "ADAPT (Lin et al. 2022)"
  competes_with: "ADAPT (Lin et al. 2022); LAD (Li et al. 2023)"
  complementary_to: "ScaleVLN (Wang et al. 2023); MARVAL (Kamath et al. 2023)"
evidence_strength: strong
pdf_ref: "paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_Do_Visual_Imaginations_Improve_Vision_and_Language_Navigation_Agents.pdf"
category: Embodied_AI
---

# Do Visual Imaginations Improve Vision-and-Language Navigation Agents?

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.16394), [Project](https://www.akhilperincherry.com/VLN-Imagine-website/)
> - **Summary**: 这篇论文把导航指令中的地标描述先转成“想象图”，再作为额外视觉提示输入 VLN 代理，以降低未见环境中的语言地标 grounding 难度。
> - **Key Performance**: R2R val-unseen 上 HAMT/DUET 分别提升 +1.02/+0.60 SR；REVERIE 上 DUET 提升 +1.30 SR、+0.82 RGS。

> [!info] **Agent Summary**
> - **task_path**: 自然语言导航指令 + 当前/历史全景观测 + 离线生成的地标想象图 -> 导航动作分布/停止决策
> - **bottleneck**: 指令中的地标名词短语难以在未见环境中稳定对齐到真实视觉观测
> - **mechanism_delta**: 用 SDXL 将子指令地标生成为图像 token，并与文本早融合进入 VLN cross-modal encoder，再用文本-想象对齐损失强化 grounding
> - **evidence_signal**: R2R 与 REVERIE 上对 HAMT/DUET 的一致增益，且“正确想象 > 空想象 > 错误想象”的消融支持因果解释
> - **reusable_ops**: [sub-instruction segmentation + noun-phrase filtering, text-to-image landmark cue + text-side early fusion]
> - **failure_modes**: [imaginations 与指令错配时会误导导航, 环境专属命名或个性化 referent 无法被可靠想象]
> - **open_questions**: [额外生成成本在真实机器人部署中是否划算, imaginations 能否与更大规模 VLN 预训练或 world model 形成更强互补]

## Part I：问题与挑战

这篇论文研究的不是“代理会不会走路”，而是 **代理能不能把语言里提到的地标稳定地看懂**。

在 Vision-and-Language Navigation (VLN) 里，输入通常是：
- 一段自然语言导航指令；
- 当前与历史的全景视觉观测。

输出是：
- 下一步该朝哪个可导航方向移动，或何时 `stop`。

### 真正的问题是什么？

很多导航指令并不是纯动作序列，而是“动作 + 地标条件”的组合，例如：
- “走到台球桌左转进入厨房”
- “走到带金色毯子的卧室停下”

困难点不在于“左转/直走”这类低级动作，而在于：
1. **指令中的名词短语必须和真实环境里的视觉对象对上号**；
2. 这种对齐要发生在 **未见环境** 中；
3. 下游 VLN 训练数据覆盖的对象种类有限，导致稀有地标、属性组合、细粒度差异很难学稳。

作者的判断是：现有 VLN 大多依赖隐式 cross-modal alignment，让模型自己从语言 token 里召回视觉概念；但这件事在未见场景里并不可靠，尤其当对象稀有、描述带属性、或多个候选房间很相似时。

### 为什么现在值得做？

因为文生图模型已经强到足以把一句地标描述变成一个可用的“视觉原型”：
- 它不需要在线逐步推理；
- 可以离线预生成；
- 还能利用大规模图文知识，补足 VLN 训练集对长尾对象的缺失。

所以这篇论文的核心问题是：

> **如果把语言里隐含的地标先显式“画出来”，VLN 代理会不会更容易把指令和观测对齐？**

### 边界条件

这项方法有明确边界：
- 主要针对 **室内 Matterport3D 风格场景**；
- imaginations 是 **离线生成** 的，不是与当前环境几何严格对齐的视图；
- 只对能抽取出有效名词短语的子指令有效；
- 目标是增强 **grounding 与消歧**，不是重写整个导航范式。

## Part II：方法与洞察

### 方法骨架

整套方法分两部分：**生成 imaginations** 和 **把 imaginations 接进现有 VLN 代理**。

#### 1. 从指令中抽取可想象的地标

作者先把一条导航指令切成多个子指令（用 FG-R2R 分段），然后：
- 用 spaCy 找名词短语；
- 用人工 blacklist 过滤掉不适合作图的词，如方向词、计数词、模糊代词等。

这样做的目的，是只保留那些真正提供视觉锚点的片段，比如：
- “go past the couch” 保留；
- “go straight then left” 过滤。

#### 2. 用 SDXL 生成地标“想象图”

对每个保留的子指令，使用 SDXL 生成一张图像，并用提示词把风格往室内房产场景上引导。

结果是：
- 在 R2R 上合成出约 41k 张 imaginations；
- 平均每条指令约 2.96 张 imagination。

这些图不是未来视图预测，也不是环境重建，而是 **“语言地标的视觉原型”**。

#### 3. 把想象图作为新模态接入代理

每张 imagination 先经过：
- 预训练 ViT-B/16 编码；
- 再过一个 3 层 MLP；
- 加上 imagination modality type embedding。

随后，作者把这些 imagination embeddings **拼接到文本 embeddings 上**，再一起送入原有 VLN 代理的 cross-modal encoder。

这一步很关键：  
作者不是把 imaginations 当成环境观测，而是把它们当成 **视觉化的指令补充**，因此选择接在 **文本侧** 做早融合。

他们验证于两个基座：
- **HAMT**
- **DUET**

两者都只做最小侵入式改造。

#### 4. 加一个文本-想象对齐辅助目标

训练时，作者显式拉近：
- imagination embedding
- 对应子指令中名词短语的文本表示

直觉上，这相当于强行告诉模型：

> “这张图就是这段话里这个地标的视觉对应物。”

#### 5. 三阶段微调，避免灾难性遗忘

为了避免把原始 VLN 能力训坏，作者采用三阶段微调：
1. 先只训新加的 imagination encoder；
2. 再低学习率联训；
3. 最后统一学习率微调。

这是典型的“先接上新模态，再慢慢放开旧骨干”的稳定做法。

### 核心直觉

- **改了什么**：从“让模型仅靠语言 token 自己去想象地标”变成“直接给模型一个地标的视觉原型”。
- **改变了哪个瓶颈**：把一部分困难的 text-to-vision grounding，改写成更直接的 image-to-image 语义匹配问题。
- **带来了什么能力**：代理更容易在未见环境里识别稀有对象、属性组合，并在相似房间/相似物体之间做消歧。

更具体地说，这个设计起作用的因果链条是：

1. **文生图模型提供开放词汇视觉先验**  
   VLN 训练集没怎么见过的对象或属性组合，扩散模型也可能“画得出来”。

2. **想象图把抽象 noun phrase 变成可比对的视觉锚点**  
   代理不必只在语言空间里做 grounding，而能拿一个“视觉模板”去和观测比。

3. **对齐辅助损失把这个锚点牢牢绑回原指令**  
   防止 imagination 只是一个漂浮的额外 token，而不真正服务于对应地标。

4. **早融合让 imaginations 参与整个 cross-modal 推理过程**  
   它更像“视觉化指令”，而不是另一组环境观测。

一个很重要的因果信号是：  
**错误 imaginations 会伤害性能**，说明收益不是单纯来自加参数或正则化，而是来自 **对齐且有信息的视觉提示**。

### 策略权衡

| 设计选择 | 带来的好处 | 代价/风险 | 论文里的证据 |
|---|---|---|---|
| 顺序地为多个子指令生成 imaginations | 给整条路径提供多次 landmark cue，而不只是终点提示 | 图像数量更多，预处理成本更高 | full imaginations 优于 goal-only |
| 文本侧早融合 | 把 imaginations 当“视觉化指令”，更符合其角色 | 序列更长，cross-modal encoder 负担更大 | early fusion 略优于 LAD 风格 late fusion |
| 文本-想象对齐损失 | 稳定 imagination 与 noun phrase 的绑定关系 | 依赖分词/名词短语抽取质量 | 优于 no-loss 版本 |
| 冻结通用 ViT + 轻量 MLP | 泛化较稳，减少过拟合 | 表征未必最贴导航域 | off-the-shelf ViT 与 task-tuned ViT 接近 |
| 离线 SDXL 生成 | 不需在线大模型推理，可提前准备 | 生成开销真实存在，且图像不与环境几何对齐 | 单图生成约 3.2s/H100；但为 upfront cost |

## Part III：证据与局限

### 关键实验信号

#### 1. 比较信号：对不同 VLN 架构都有一致增益
在 R2R val-unseen 上：
- HAMT：SR 从 66.24 提到 67.26，SPL 从 61.51 提到 62.02；
- DUET：SR 从 71.52 提到 72.12。

这说明该方法不是只对某个单一结构有效，而更像一个 **可插拔的 grounding 增强器**。

#### 2. 跨设定信号：粗粒度目标式指令也受益
在 REVERIE 上，DUET-Imagine：
- SR 提升 +1.30；
- RGS 提升 +0.82。

这很关键，因为 REVERIE 比 R2R 更依赖目标识别和探索，说明 imaginations 不只是帮助“沿路走”，也帮助“最后认准目标”。

#### 3. 因果消融信号：对齐的 imaginations 才有价值
最有说服力的消融不是“有无模块”，而是：
- **correct imaginations > null imaginations > wrong imaginations**

其中：
- 去掉测试时 imaginations，性能仍略高于原基线，说明训练阶段确有正则化收益；
- 但换成错误 imaginations，收益消失甚至变差，说明真正有用的是 **语义对齐的视觉锚点**。

#### 4. 设计空间信号：顺序 imaginations 和对齐损失都有效
- 顺序多子目标 imaginations 优于只给最终目标图像；
- cosine 对齐损失优于 no-loss；
- visual imaginations 也优于只把子指令文本再平均一遍作为替代表示。

这表明收益不是“多喂一点 token”那么简单，而是 **视觉化表达本身** 带来了额外信息。

#### 5. 保真度/分析信号：生成图大多确实画对了对象
作者用开放词汇检测器检查 imaginations 与子指令 noun phrases 的一致性，发现：
- 98%+ 的子指令至少能检测到一个对应名词；
- 约 95% 的子指令能检测到全部名词。

再配合 attention 可视化与轨迹案例，论文展示了 imaginations 如何在语言 token 与环境观测之间充当“中间桥”。

### 这篇论文真正的“so what”

能力跃迁不是 SOTA 级大跳，而是更精确地说：

> **它把 VLN 的薄弱环节从“隐式语言 grounding”改成“显式视觉锚点辅助 grounding”，因此在相同 backbone 上获得稳定、可解释、可迁移的小幅提升。**

所以它的价值主要体现在：
- **机制上可解释**；
- **可以插到不同代理上**；
- **与扩大训练数据这类路线正交**。

但也要承认：
- 它没有超过依赖大规模额外环境数据的最强路线；
- 增益幅度总体是 modest 的，说明它更像是补 grounding 短板，而不是完全改变规划能力。

### 局限性

- **Fails when**: imaginations 与指令错配、目标依赖环境专属命名或个体化 referent、以及部署场景对时延/算力极敏感时；论文已显示错误 imaginations 会把性能拉回基线附近。
- **Assumes**: 能可靠切分子指令并抽取名词短语；有可用的文生图模型（SDXL）和额外算力；生成单张 1024×1024 imagination 平均约需 3.2 秒/H100，单个代理微调约 1.5 天/V100；同时依赖已有 VLN 强基座（HAMT/DUET）作为起点。
- **Not designed for**: 与当前环境严格对齐的视角预测、在线逐步 world-model 规划、长期个体化地标绑定/命名记忆，以及现实机器人中低功耗端侧部署优化。

### 可复用组件

这篇论文里最值得复用的不是完整系统，而是几类操作符：

- **子指令切分 + 名词短语过滤**：把长指令拆成可操作的地标单元。
- **离线地标想象图生成**：把语言描述转成视觉原型库。
- **文本侧早融合**：把 imaginations 当作“视觉化指令”，而不是额外观测。
- **文本-想象对齐辅助目标**：把新模态与原指令强绑定，减少漂移。
- **三阶段微调**：在现有 embodied agent 上加新模态时很实用。

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_Do_Visual_Imaginations_Improve_Vision_and_Language_Navigation_Agents.pdf]]