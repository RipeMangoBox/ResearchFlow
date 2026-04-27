---
title: "Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/vision-language-navigation
  - data-augmentation
  - diffusion
  - prompting
  - dataset/R2R
  - dataset/REVERIE
  - dataset/R4R
  - dataset/R2R-CE
  - opensource/full
core_operator: 用VLM/LLM重写观测描述与导航指令，并以扩散式T2I生成新全景，再通过“混合后聚焦”训练把少量合成样本转化为VLN泛化增益
primary_logic: |
  已标注VLN轨迹中的全景观测与人工指令 → VLM生成场景描述、LLM补充可能对象并重写描述与指令、扩散式T2I生成新全景并离散成多视角 → 得到对齐的未见观测-指令对并用于两阶段训练
claims:
  - "Claim 1: On R2R Val Unseen with CLIP ViT-L/14, RAM improves DUET from 73.22 to 76.29 SR and from 64.49 to 66.39 SPL [evidence: comparison]"
  - "Claim 2: On REVERIE Test Unseen, RAM reports 57.44 SR and 36.05 RGS, exceeding the reported ScaleVLN results of 56.13 SR and 32.53 RGS [evidence: comparison]"
  - "Claim 3: On the R2R ablation, the full pipeline with generated panoramas, object-enriched description rewriting, and rewritten instructions achieves the best Val Unseen SR of 70.29, above observation-only 67.52 and instruction-only 67.43 [evidence: ablation]"
related_work_position:
  extends: "PanoGen (Li and Bansal 2023)"
  competes_with: "ScaleVLN (Wang et al. 2023); PanoGen (Li and Bansal 2023)"
  complementary_to: "DUET (Chen et al. 2022); HAMT (Chen et al. 2021)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_Unseen_from_Seen_Rewriting_Observation_Instruction_Using_Foundation_Models_for_Augmenting_Vision_Language_Navigation.pdf
category: Embodied_AI
---

# Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.18065), [DOI/TNNLS](https://doi.org/10.1109/TNNLS.2025.3624691), [Code](https://github.com/SaDil13/VLN-RAM)
> - **Summary**: 论文提出 RAM，把已有 VLN 人工标注样本离线“改写”为新的观测-指令对：先重写场景描述并生成新全景，再依据新旧观测差异重写指令，最终在不依赖额外模拟器或网页清洗的前提下提升未见环境泛化。
> - **Key Performance**: R2R Val Unseen 上达到 **SR 76.29 / SPL 66.39**；R2R-CE Val Unseen 上 **SR 44.15**（DUET 为 37.25）。

> [!info] **Agent Summary**
> - **task_path**: 已标注全景轨迹+导航指令 / VLN数据增强 -> 重写观测-指令对 -> 导航动作预测
> - **bottleneck**: 未见环境泛化受限于稀缺且低多样性的观测-指令对，现有模拟器/网页增强又分别受场景边界或噪声清洗成本限制
> - **mechanism_delta**: 把数据增强从“采样新轨迹”改成“重写旧观测与旧指令”，并用两阶段训练在吸收多样性的同时抑制合成噪声
> - **evidence_signal**: 4个VLN基准的 unseen split 提升，且消融证明观测重写、指令重写、混合后聚焦训练都各自贡献性能
> - **reusable_ops**: [对象增强观测重写, 观察对比式指令重写]
> - **failure_modes**: [T2I重复物体伪影, 跨时间步几何一致性不足]
> - **open_questions**: [如何显式加入3D/时序一致性约束?, 如何减少对闭源LLM与T2I的依赖?]

## Part I：问题与挑战

这篇论文针对的是 **Vision-Language Navigation (VLN)** 的一个老问题：**训练数据太少，且“少”的不是纯数量，而是“多样且对齐”的 observation-instruction 对太少**。

### 1. 真正瓶颈是什么
VLN 训练需要的是三者对齐：
- 视觉观测序列
- 自然语言导航指令
- 对应动作/轨迹

已有方法的问题不在于模型完全学不会，而在于：
1. **训练环境太窄**：大多数据来自有限的 Matterport3D 等模拟器；
2. **未见环境分布差异大**：新房屋里的对象共现、空间布局、描述方式都可能变；
3. **增强数据常常不够“对齐”**：
   - simulator-based 方法能扩轨迹，但仍被已有环境束缚；
   - web-based 方法能扩视觉分布，但噪声大、清洗贵；
   - Speaker 或模板式指令生成常常空泛、重复、甚至与视觉不完全对齐。

所以，作者瞄准的不是“再堆更多数据”，而是：

> **如何低成本地产生“像未见环境、但仍与路径语义对齐”的新 observation-instruction 对。**

### 2. 输入/输出接口
- **输入**：VLN 中的全景观测序列 + 人工标注导航指令
- **输出**：导航动作序列/停止位置
- **RAM 的位置**：它不是在线推理代理，而是一个 **离线数据增强器**

这点很关键：作者没有让 GPT 在测试时每步做决策，而是把 foundation model 的知识离线蒸馏进训练数据里，降低调用频率与成本。

### 3. 边界条件
RAM 的增强边界比较明确：
- 它 **保留原始轨迹骨架**，主要改的是每一步看到什么、语言怎么说；
- 它 **不生成新的导航图拓扑**；
- 它既支持离散 VLN，也能迁移到连续环境，但核心仍是“基于已有标注轨迹重写”。

### 4. 为什么现在做
因为现在已经有了足够成熟的 foundation models，分别提供：
- VLM：看图写描述
- LLM：基于世界知识补对象、改写语言
- T2I：把新描述变成新视觉观测

这三者组合起来，才让“**不依赖额外模拟器、也不靠网页采集**”的增强真正变得可行。

---

## Part II：方法与洞察

RAM 的核心不是直接生成整条新导航任务，而是把原有高质量人工样本做成“**语义保持、表面分布改变**”的新样本。

### 方法主线

#### 1. 对象增强的观测重写
作者先对原始全景观测做场景描述，再让 LLM 往里加入“这个场景里合理可能出现”的对象，并改写描述表达。

流程可概括为：
1. 用 VLM 为每个时间步的全景观测生成 scene description；
2. 用 LLM 根据对象共现知识，显式补充新对象，并改写描述；
3. 用 T2I（论文中是 MultiDiffusion）直接生成 **全景图**；
4. 再把全景图离散成 VLN 所需的多视角图像。

这里有两个关键点：
- **不是 view-by-view 生成**，而是先生成 panorama，再切视角，所以视角内部一致性更好；
- LLM 不只是“换措辞”，而是显式引入新对象和新的显著性焦点，改变训练时的视觉分布。

#### 2. 观察对比式指令重写
新观测出来后，原始指令就不再对齐了。所以作者再做一次语言重写，但不是从零生成，而是基于“新旧观测差异”来改。

流程是：
1. 从原始指令中抽出顺序 landmarks；
2. 用视觉-文本匹配把原指令中的 landmark 对齐到原轨迹中的关键观测；
3. 再为新观测提取对应描述；
4. 让 LLM 根据“旧 landmark vs 新观测内容”的差异，重写指令中的对象/场景提法，并把动作表达换成同义说法。

这一步的意义很强：
- 它保留了原始人工指令的结构与可读性；
- 同时把语言引用对象切换到新图像里真正出现的内容；
- 比 Speaker 直接生成更不容易空泛或跑偏。

#### 3. 混合后聚焦训练
合成数据有价值，但也会带噪声。尤其 T2I 容易出现重复物体、局部不自然、几何不严谨。

作者的应对不是简单全量混训，而是两阶段：
- **Stage 1: Mixing**  
  原始数据 + 重写数据一起训练，先吃到分布多样性；
- **Stage 2: Focusing**  
  回到纯原始数据继续训练，把模型拉回高质量真实监督上。

同时，作者还在 Stage 1 对合成全景做 **random observation cropping**，通过随机裁大块并重组，进一步：
- 扩大视觉分布；
- 缓解 T2I 的重复物体问题。

### 核心直觉

这篇论文最重要的“因果旋钮”是：

> **把 VLN 数据增强的自由度，从“采样更多路径”转成“重写已有路径上的视觉-语言表面分布”。**

更具体地说：

- **What changed**：从额外模拟器/网页采集，改成 foundation models 驱动的 observation-instruction rewriting。
- **Which bottleneck changed**：
  - 放松了“环境来源必须是真实扫描/模拟器”的约束；
  - 放松了“语言只能跟着原图走”的约束；
  - 但保留了原始高质量轨迹骨架与指令结构。
- **What capability changed**：
  - 训练时见到更多对象共现、更多视觉布局、更多语言表达；
  - 因而对 unseen house 的泛化更强。

为什么这套设计有效，核心因果链是：

1. **LLM 的世界知识** 提供“这个场景里还可能有什么”的先验；
2. **T2I 的视觉合成** 把这种先验变成新的视觉分布；
3. **观察对比式指令重写** 保证语言提到的是新图里真的有的东西；
4. **混合后聚焦训练** 则避免模型被合成噪声带偏。

换句话说，作者没有直接解决“真实新环境从哪来”，而是解决了一个更实际的问题：

> **如何用原始标注样本，制造出足够像“未见环境”的训练信号。**

### 战略取舍

| 设计选择 | 带来的能力 | 代价/风险 |
|---|---|---|
| 直接重写观测并生成全景，而非从模拟器采新轨迹 | 跳出固定场景库，扩对象与布局分布 | 物理真实性与几何一致性不如真实扫描 |
| 基于 observation contrast 重写指令，而非 Speaker 直接生成 | 语言与新观测更对齐，指令更信息化 | 依赖 VLM 描述质量与 landmark grounding 准确性 |
| 先混合后聚焦，而非一步混训 | 吃到合成多样性同时抑制噪声 | 训练流程更复杂，需要调阶段与比例 |
| 先生成 panorama 再切 view | 多视角内部一致性更好、T2I 调用更省 | 仍未显式保证跨时间步一致性 |
| 离线调用 foundation models，而非测试时在线推理 | 成本更低、部署更稳定 | 增强质量受第三方模型能力上限约束 |

---

## Part III：证据与局限

### 关键证据

#### 1. 对比信号：unseen split 上稳定变强
最有说服力的不是 seen split，而是 unseen split：

- **R2R Val Unseen**：在 CLIP ViT-L/14 下，RAM 把 DUET 从 **73.22 SR / 64.49 SPL** 提升到 **76.29 SR / 66.39 SPL**。  
  这说明它确实改善了未见环境泛化，而不只是记忆训练集。
- **R4R Val Unseen**：从 **50.35 SR / 46.14 SPL** 提升到 **55.28 SR / 49.59 SPL**。  
  长指令、长轨迹场景下也有效，说明它不只是对短路径有帮助。
- **R2R-CE Val Unseen**：连续环境上从 **37.25 SR** 提升到 **44.15 SR**。  
  这说明增强不是只对离散图上的过拟合技巧。

#### 2. 对比信号：不仅导航变好，语言-视觉对齐也变好
在 **REVERIE Test Unseen** 上，RAM 达到：
- **57.44 SR**
- **36.05 RGS**

这很重要，因为 REVERIE 同时考察导航与目标 grounding。结果说明 RAM 的增益不只是“会走”，也包括“语言提到的目标更对得上图像”。

#### 3. 消融信号：三段式链路都在起作用
消融结果很干净：

- **只做观测重写** 有提升；
- **只做指令重写** 也有提升；
- **观测+指令一起重写** 最好；
- 加入 **object-enriched description rewriting** 比直接用原描述生成图像更强；
- **mixing-then-focusing + random crop** 比简单一次性混训更强。

这说明论文不是单一 trick，而是：
1. 合成视觉分布；
2. 重新对齐语言；
3. 用合适训练策略吸收它们  
三者共同决定了最终效果。

#### 4. 数据效率信号：不是靠海量合成硬堆出来
作者在补充材料里给了一个很有价值的信号：
- RAM 在 R2R/R4R 上的增强量约 **14,025**；
- ScaleVLN 的 R2R 增强量是 **4,941,710**。

而且在同等 3x 数据规模的子集对比下，RAM 仍优于 ScaleVLN 子集版本。  
这支持作者的核心判断：**关键不是“更多”增强，而是“更对齐”的增强。**

### 局限性

- **Fails when**: T2I 生成出现重复物体、局部不自然或几何错乱时，增强样本会带来噪声；对需要严格跨步视觉重叠或强3D一致性的场景，RAM 只有语义级一致性，没有显式时空约束。
- **Assumes**: 需要已有高质量人工标注轨迹-指令对作为种子；依赖 Tag2Text、CLIP、MultiDiffusion，以及闭源 ChatGPT API；补充材料显示数据生成用到约 8×RTX 3090，T2I 生成约 30 小时，API 成本约 20 美元，因此“代码开源”不等于“完全可无差别复现”。
- **Not designed for**: 生成新的导航拓扑/新路径监督；在线测试时的 LLM 推理代理；需要物理一致性和安全保证的真实机器人部署。

### 可复用组件

这篇论文有几个很值得迁移的模块：

1. **对象增强式场景描述重写**  
   适合任何“已有图像/轨迹，但想扩场景语义多样性”的任务。
2. **观察对比式指令重写**  
   很适合需要保持语言-视觉对齐的数据生成场景，不限于 VLN。
3. **混合后聚焦训练**  
   对所有“真实数据 + 合成数据”共训的问题都有参考价值。
4. **Panorama-to-view 生成范式**  
   对多视角任务比逐视角独立生成更自然。

一句话总结 “So what”：

> RAM 的能力跃迁不在于造了一个更强导航器，而在于提出了一种更高效的数据制造方式：用 foundation models 把“看过的样本”改写成“像没看过的样本”，并且尽量保住视觉-语言对齐。

## Local PDF reference

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_Unseen_from_Seen_Rewriting_Observation_Instruction_Using_Foundation_Models_for_Augmenting_Vision_Language_Navigation.pdf]]