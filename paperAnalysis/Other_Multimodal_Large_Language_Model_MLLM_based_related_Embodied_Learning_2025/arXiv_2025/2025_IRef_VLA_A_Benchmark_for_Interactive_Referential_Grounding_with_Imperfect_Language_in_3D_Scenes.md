---
title: "IRef-VLA: A Benchmark for Interactive Referential Grounding with Imperfect Language in 3D Scenes"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/referential-grounding
  - task/interactive-grounding
  - scene-graph
  - graph-search
  - heuristic-scoring
  - dataset/IRef-VLA
  - opensource/full
core_operator: 通过在大规模3D室内场景中同时加入场景图、自由空间标注与“不存在/说错了”的指代表达，把评测从单次找物体扩展为存在性判断与替代建议。
primary_logic: |
  3D场景 + 可能有误的指代表达 → 构建对象/区域/自由空间/场景图与不完美语言样本 → 评测目标定位、对象存在性判断与替代建议质量 → 揭示3D交互式指代 grounding 的鲁棒性边界
claims:
  - "IRef-VLA汇集了7,635个场景、超过11.5K个区域、286K个对象、477个类别、7.6M条空间关系和4.7M条指代表达，并把场景图、可通行自由空间和不完美引用纳入同一基准 [evidence: analysis]"
  - "3D-VisTA在IRef-VLA-Full上训练后，Nr3D零样本准确率达到44.9%，高于论文表中SceneVerse zero-shot text的43.1%，说明该数据生成流程对迁移到人类指代表达更有效 [evidence: comparison]"
  - "在不完美引用的存在性判断子任务上，graph-search基线达到94.4% F1和98.9% TN，显著高于MVT+binary classifier的78.3% F1，表明结构化场景图搜索是更强的拒答基线 [evidence: comparison]"
related_work_position:
  extends: "ReferIt3D (Achlioptas et al. 2020)"
  competes_with: "SceneVerse (Jia et al. 2024); ReferIt3D (Achlioptas et al. 2020)"
  complementary_to: "ConceptGraphs (Gu et al. 2023); TEACh (Padmakumar et al. 2021)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_IRef_VLA_A_Benchmark_for_Interactive_Referential_Grounding_with_Imperfect_Language_in_3D_Scenes.pdf"
category: Survey_Benchmark
---

# IRef-VLA: A Benchmark for Interactive Referential Grounding with Imperfect Language in 3D Scenes

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.17406), [GitHub](https://github.com/HaochenZ11/IRef-VLA)
> - **Summary**: 该工作提出 IRef-VLA，把 3D 指代 grounding 从“默认目标一定存在的单次检索”扩展为“目标存在性判断 + 替代建议”的交互式基准，并用大规模真实室内扫描场景支撑这一评测。
> - **Key Performance**: graph-search 在对象存在性判断上达到 **94.4% F1**；3D-VisTA 在 IRef-VLA-Full 训练后对 **Nr3D 零样本 44.9%**。

> [!info] **Agent Summary**
> - **task_path**: 3D室内场景点云/场景图 + 可能有误的指代表达 -> 目标对象ID 或 not-found + 替代对象
> - **bottleneck**: 现有3D grounding基准默认“引用正确且目标存在”，无法评测机器人在真实错误语言下的拒答与澄清能力
> - **mechanism_delta**: 作者把评测协议改成“存在性判别 + 替代检索”，并用场景图、自由空间和不完美语言样本系统覆盖更真实的失败分布
> - **evidence_signal**: 预训练基线迁移到 IRef-VLA 明显掉点，而 graph-search 在存在性判别上可达 94.4% F1，说明该基准确实测到了旧协议没覆盖的能力
> - **reusable_ops**: [scene-graph relation generation, imperfect-reference augmentation]
> - **failure_modes**: [view-dependent或allocentric描述, 替代建议依赖人类偏好的场景]
> - **open_questions**: [如何用人类偏好更可靠地评测替代建议, 如何把单轮grounding扩展到多轮导航对话]

## Part I：问题与挑战

这篇论文的核心不是再提一个新 grounding 模型，而是指出：**当前 3D 指代 grounding 的评测目标本身就过于理想化**。

### 真正的问题是什么？
现有基准大多默认：
1. 用户说的话是对的；
2. 场景里一定有对应目标；
3. 系统只需单次返回一个对象。

但真实机器人交互里，用户常会说错、说得不完整，或者使用与场景不匹配的描述。比如“桌子上的遥控器”，实际遥控器可能在沙发上。若 benchmark 不覆盖这种情况，模型即使在传统 grounding 上得分不错，也不代表它能在现实中稳定工作。

### 真正的瓶颈是什么？
**瓶颈是评测分布与真实交互分布不一致。**  
过去 benchmark 主要测“在封闭世界里能不能从一句正确描述中找出目标”；IRef-VLA 想测的是“当语言不完美时，模型能否先判断目标不存在，再给出合理替代”。这更接近人机交互里的澄清前置步骤。

### 输入/输出接口
- **输入**：3D 室内场景（点云、对象、区域、自由空间、场景图）+ 一句指代表达
- **输出**：
  - 若目标存在：返回对应对象
  - 若目标不存在：显式输出 not found，并给出替代候选

### 边界条件
这项工作仍然是 **grounding benchmark**，不是完整导航系统：
- 不评测路径规划或动作控制
- 主要是单轮 grounding，不是多轮对话
- 场景是静态扫描，不覆盖动态环境变化

## Part II：方法与洞察

### 这篇论文到底做了什么？
作者构建了一个新的 3D benchmark：**IRef-VLA**。它的设计重点不是“多一个数据集”，而是**把鲁棒交互 grounding 所需的结构和失败情形都显式做进 benchmark**。

### 数据与评测设计

#### 1. 大规模 3D 场景汇聚
数据来自多源室内扫描与合成场景，包括：
- ScanNet
- Matterport3D
- HM3D
- 3RScan
- ARKitScenes
- Unity 场景

总规模：
- 7,635 个场景
- >11.5K 区域
- 286K 对象
- 477 类对象
- 7.6M 空间关系
- 4.7M 指代表达

#### 2. 结构化场景表示
每个场景包含：
- 场景点云
- 对象类别、边界框、颜色
- 房间/区域标注
- 可通行自由空间
- 按区域划分的 scene graph

这里的关键价值是：**benchmark 不只给“句子—目标”，还给了支持推理的结构层**。这让后续方法可以做部分匹配、替代搜索和更细粒度的误差分析。

#### 3. 关系驱动的语言生成
作者用启发式模板从 scene graph 生成指代表达，强调三点：
- **view-independent**：不依赖观察视角
- **unambiguous**：应只对应一个目标
- **minimal**：尽量用最少描述消歧

关系类型包括 above / below / near / on / in / between / closest / farthest 等。  
此外还加入同义词，让语句形式更多样。

#### 4. “不完美引用”扩展
这是本文最关键的设计。作者不是只生成正确表达，还会对已有语句做扰动，制造**看起来合理、但场景中不存在的引用**。  
于是 benchmark 不再只问：
- “你能不能找到它？”

还会问：
- “这个东西其实不存在时，你能不能识别出来？”
- “你能不能给一个最接近用户意图的替代项？”

#### 5. 评分协议
扩展任务被拆成两个子问题：
1. **对象存在性判断**：TP/TN/FP/FN、F1
2. **替代建议质量**：按对象类别、属性、空间关系等方面的匹配度打启发式分数，其中类别和关系权重更高

这意味着 benchmark 测的不只是 localization accuracy，而是：
**拒答能力 + 恢复能力**。

### 核心直觉

过去的 3D grounding benchmark 默认一个封闭前提：**每句话都该有唯一真值对象**。  
IRef-VLA 改动的不是模型参数，而是这个前提本身。

**what changed**  
从“正确语言下的单次定位”改为“可能错误语言下的定位/拒答/替代”。

**which bottleneck changed**  
从只测“对象区分能力”，变成测“开放世界下的语义一致性判断 + 交互恢复能力”。

**what capability changed**  
模型现在必须学会：
- 识别不存在的 referent
- 利用部分关系结构找近似意图
- 为后续多轮澄清提供起点

**why this works**  
因为 scene graph 让“部分匹配”成为可操作对象：  
当完整引用不存在时，系统仍可根据类、属性、关系的子集去搜索最接近的候选。这比纯粹 end-to-end 的“必须猜一个对象”更适合评测交互鲁棒性。

### 战略取舍

| 设计选择 | 带来的能力 | 代价/风险 |
|---|---|---|
| 聚合多源 3D 扫描 | 扩大场景与类别分布，减少单一数据源偏差 | 类别映射与标注规范需要对齐，带来噪声 |
| 模板化关系语言生成 | 可大规模、可控地生成无歧义语句 | 自然度仍弱于真实人类表达 |
| 加入 imperfect references | 显式测拒答与替代能力 | 合成错误未必完全等价于真实用户错误 |
| scene-graph + heuristic scoring | 能系统评估部分匹配与替代质量 | 分数未必符合人类主观偏好 |

### 基线设计的意义
作者给了两类 baseline：
- **监督式 grounding 模型**：MVT, 3D-VisTA
- **graph-search 基线**：LLM 先把句子解析成子图，再在 scene graph 中搜索；若无精确匹配，则做部分匹配并让 LLM 选最接近的替代语句

这个 graph-search 不是说“未来一定这么做”，而是给 benchmark 提供了一个很有价值的参考：
- 当结构信息可靠时，上界大概到哪
- 纯 grounding 模型在“拒答/恢复”这件事上差距有多大

## Part III：证据与局限

### 关键证据信号

#### 信号 1：IRef-VLA 不是“更大的旧数据”，而是更难的新分布
论文复现实验显示，原本在 Sr3D 上训练的模型迁移到 IRef-VLA 时表现明显较差：
- MVT：IRef-VLA-ScanNet / Full 仅 29.0% / 17.2%
- 3D-VisTA：39.2% / 24.8%

**结论**：IRef-VLA 的难点不只是规模更大，而是语言分布、场景分布和结构需求都更不同，能暴露原 benchmark 没测到的能力缺口。

#### 信号 2：更好的合成语言流程确实提升迁移性
3D-VisTA 在 IRef-VLA-Full 上训练后：
- Nr3D 零样本：44.9%
- 高于论文表中的 SceneVerse zero-shot text：43.1%

**结论**：即使仍是合成语言，只要生成规则更贴近“最小但足以消歧”的人类指代逻辑，迁移到人类表达的效果会更好。

#### 信号 3：结构化搜索显著强于“给模型多加一个存在性头”
在 imperfect references 子任务上：
- MVT + binary classifier：78.3% F1
- Graph-search：94.4% F1，98.9% TN

**结论**：当任务目标变成“先判断有没有，再决定怎么回复”，scene graph 这类显式结构比把问题硬塞进普通 grounding 模型更有效。

#### 信号 4：替代建议仍然远未解决
graph-search 的：
- LLM parsing accuracy：94.0%
- average alternative similarity：61%

这说明替代建议可以做，但目前更多是“结构上像”，还不是“真正符合用户意图”。

### 1-2 个最值得记住的数
- **94.4% F1**：graph-search 在“目标是否存在”判断上的强基线
- **44.9% Nr3D zero-shot**：IRef-VLA 训练的 3D-VisTA 对人类表达仍有一定迁移能力

### 局限性
- **Fails when**: 输入包含明显的视角相关、相对朝向、主观感受词描述时（如 “我左边那个”“看起来更舒服的椅子”），该基准的模板语言与现有评分很难充分覆盖。
- **Assumes**: 依赖高质量 3D 分割、对象语义标签、空间关系启发式计算和自由空间标注；graph-search baseline 还依赖闭源 API 模型 gpt-4o-mini 做语句解析。
- **Not designed for**: 端到端导航控制、多轮人机对话、动态场景更新、以及真正基于人类偏好的替代建议选择。

### 复现与扩展时要注意的资源依赖
- 数据与代码是公开的，这是加分项
- 但 baseline 中的 LLM parser 依赖外部闭源服务
- 多源数据集的语义标签统一、本体映射和 Unity 标签清洗带来额外工程成本

### 可复用组件
- scene graph 生成与关系枚举流水线
- minimal / unambiguous 指代表达生成规则
- imperfect reference 数据增强方式
- “存在性判断 + 替代建议” 的评测协议
- 基于部分图匹配的替代候选检索思路

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_IRef_VLA_A_Benchmark_for_Interactive_Referential_Grounding_with_Imperfect_Language_in_3D_Scenes.pdf]]