---
title: "Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - rule-based-evaluation
  - hybrid-annotation
  - multi-turn-evaluation
  - dataset/Multi-IF
  - dataset/IFEval
  - opensource/full
core_operator: 将 IFEval 扩展为 8 语种、3 轮对话的可验证指令跟随基准，并用脚本化 strict/loose 评分与 IFR/ECR 诊断暴露跨轮遗忘和多语短板
primary_logic: |
  评测多轮多语指令跟随能力 → 基于 IFEval 生成三轮对话并做冲突过滤、翻译审校与敏感内容清洗 → 用指令级/对话级、strict/loose 四类规则分数评估并辅以 IFR/ECR 诊断 → 揭示模型的跨轮保持、错误恢复与跨语言稳健性边界
claims:
  - "Multi-IF 包含 8 种语言、4,501 个三轮对话，并以 strict/loose × 指令级/对话级四类分数综合衡量多轮指令跟随 [evidence: analysis]"
  - "在 14 个被测 LLM 中，所有模型都会随轮次增加而降分；例如 o1-preview 的跨语言平均准确率从 turn-1 的 0.877 降至 turn-3 的 0.707 [evidence: analysis]"
  - "非拉丁文字语言（Hindi、Russian、Chinese）以及 length_constraints、combination 等指令类别具有更高错误率，说明当前模型的多语与复合约束跟随能力仍薄弱 [evidence: analysis]"
related_work_position:
  extends: "IFEval (Zhou et al. 2023)"
  competes_with: "FollowBench (Jiang et al. 2023); InfoBench (Qin et al. 2024)"
  complementary_to: "MT-Bench (Zheng et al. 2023); G-Eval (Liu et al. 2023b)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Long_Horizon_Episodic_Memory/arXiv_2024/2024_Multi_IF_A_Benchmark_for_Multi_turn_Instruction_Following.pdf
category: Survey_Benchmark
---

# Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.15553) · [Dataset](https://huggingface.co/datasets/facebook/Multi-IF) · [Code](https://github.com/facebookresearch/Multi-IF)
> - **Summary**: 该文把 IFEval 扩展成一个 8 语种、3 轮对话的可验证指令跟随基准，用规则脚本 + LLM/人工混合标注系统性测出 LLM 在跨轮遗忘、错误恢复和多语言稳健性上的真实短板。
> - **Key Performance**: turn-3 跨语种平均准确率最高为 o1-preview 与 Llama 3.1 405B 并列 0.707；o1-preview 从 turn-1 的 0.877 下降到 turn-3 的 0.707。

> [!info] **Agent Summary**
> - **task_path**: 多轮多语用户指令序列 -> 模型逐轮响应 -> 指令/对话级遵循分数
> - **bottleneck**: 现有指令跟随评测主要停留在单轮英文，无法观测模型是否能跨轮持续保持已执行约束，也无法稳定诊断多语脚本差异
> - **mechanism_delta**: 把 IFEval 的可验证单轮提示扩展成三轮八语种对话，并用 strict/loose + instruction/conversation 双粒度评分，再用 IFR/ECR 拆解“遗忘”与“纠错”
> - **evidence_signal**: 14 个模型全部出现随轮数增加而系统性降分，且非拉丁文字语言错误率整体更高
> - **reusable_ops**: [单轮转多轮扩展, LLM+人工双阶段审校]
> - **failure_modes**: [跨轮遗忘先前约束, 非拉丁文字语言下长度与结构约束更易失败]
> - **open_questions**: [三轮评测能否外推到更长对话, 翻译与语言特定规则是否引入额外评测偏差]

## Part I：问题与挑战

这篇文章要解决的不是“模型不会回答”，而是**我们缺少一个能客观测出模型在真实对话里是否持续遵守指令的评测框架**。

### 真问题是什么
现有 instruction-following benchmark 大多有三个缺口：

1. **单轮偏置**：只看一轮回答是否满足约束，测不到后续回合中模型会不会忘掉之前的要求。
2. **单语偏置**：主要是英文，无法反映真实产品面向全球用户时的多语表现。
3. **主观性难评**：像“更幽默”“更自然”这种要求难以稳定自动判分。

Multi-IF 的切入点很明确：只保留**可验证指令**，例如字数、段落数、大小写、关键词、固定开头结尾、JSON 格式等，让评测可以脚本化、可重复。

### 输入 / 输出接口
- **输入**：8 种语言的三轮对话提示，每轮都可能加入新的可验证约束。
- **输出**：模型逐轮回答。
- **评测输出**：四类分数  
  - instruction-level strict  
  - conversation-level strict  
  - instruction-level loose  
  - conversation-level loose  

这里最关键的是 **conversation-level**：它要求模型在当前回合不仅满足新指令，还要继续满足此前各轮的旧指令，所以能直接测“持续遵循”。

### 为什么现在要做
现实中的 LLM 应用已经默认是：
- 多轮对话；
- 全球多语；
- 长上下文持续约束；
- 一旦漏掉格式/安全/流程要求，就会影响产品可用性。

因此，真正的瓶颈不是再做一个更大的单轮排行榜，而是把**跨轮保持能力**和**跨语言鲁棒性**变成可观测量。

### 边界条件
这个 benchmark 也有明确边界：
- 只覆盖**可脚本验证**的指令；
- 对话长度固定为 **3 轮**；
- 数据是**合成 prompt**，不是自然真实用户日志；
- 某些只适用于西文的约束（如全大写）不会硬翻到所有语言。

---

## Part II：方法与洞察

### 评测构建流程

#### 1. 从 IFEval 扩展到多轮
作者以 IFEval 的英文单轮 prompt 为起点：
- turn 1 直接用原始 prompt；
- turn 2 / turn 3 从 IFEval 的 30 类指令里随机采样类型；
- 再让 Llama 3.1 405B 把“指令类型”改写成自然语言用户请求，并显式结合之前各轮上下文。

这样得到的不是简单拼接规则，而是更像真实对话中“用户追加要求”的形式。

#### 2. 冲突指令清理
多轮合成后，最容易出问题的是前后矛盾，例如：
- 第一轮要求短回答；
- 第二轮要求至少 800 词。

作者采用两阶段清理：
- 先用 Llama 3.1 405B 自动扫描冲突；
- 再由人工审核。

这一步的作用是把 benchmark 的难度集中在“持续遵循”，而不是制造不可完成任务。

#### 3. 多语翻译
英文 golden set 再被翻译到另外 7 种语言：
- French
- Russian
- Hindi
- Italian
- Portuguese
- Spanish
- Chinese

流程也是两段式：
- LLM 初译；
- 专业人工审校与改写。

文中提到平均约 **15%** 的翻译被重写，说明作者并没有把“LLM 翻译结果”直接当金标。

#### 4. 敏感内容过滤
本地化过程中可能引入政治、宗教、人际敏感内容，因此又做了：
- LLM 初筛；
- 人工复核删除。

#### 5. 评分与诊断
最终评分不是单一准确率，而是四类分数的平均值。  
此外，作者额外定义了两个诊断量：

- **IFR**：前一轮明明做对了，后一轮却忘了的比例
- **ECR**：前一轮没做对，后一轮又补回来的比例

这使 benchmark 不只给排名，还能区分性能下降到底来自：
- 累积约束太多；
- 还是模型真的“忘了”；
- 以及模型能不能“自我修正”。

### 核心直觉

真正的变化，不是“数据多了两轮”，而是把评测对象从：

**单次回答是否满足局部约束**

变成了：

**随着对话推进，模型是否还能持续保留并执行一整串累计约束**

这带来的能力增益是可诊断的：

- **分布改变**：单轮英文 → 多轮多语、约束累积
- **测量改变**：局部 compliance → 累积 compliance
- **能力揭示改变**：只能看格式跟随 → 能看跨轮记忆、遗忘、自纠错、多语鲁棒性

换句话说，Multi-IF 把原本被单轮 benchmark 掩盖的失败模式“显影”了：
1. 旧指令会不会被新指令挤掉；
2. 多语脚本是否让格式/长度约束更难执行；
3. 强推理模型是否更能补救之前的错误。

### 设计取舍

| 设计选择 | 解决的问题 | 代价 / 局限 |
|---|---|---|
| 只保留可验证指令 | 评分客观、自动化、可复现 | 覆盖不了“幽默”“礼貌”等主观遵循 |
| 固定 3 轮对话 | 便于控制变量、做大规模横向比较 | 不能直接代表 10+ 轮真实长对话 |
| LLM 生成 + 人工审校 | 能规模化扩展多轮多语数据 | 会引入生成器/翻译器偏差，且人工成本高 |
| strict + loose 双评分 | 兼顾严格判定与模板前缀容错 | loose 仍是启发式，不同语言上可能有偏差 |
| IFR / ECR 机制诊断 | 把“降分”拆成遗忘与修复 | 只能做相关性解释，不能单独证明因果 |

---

## Part III：证据与局限

### 关键实验信号

#### 1. [多模型比较] 多轮退化是普遍现象，不是个别模型问题
作者评测了 14 个模型，结果显示**所有模型**都会随着轮数增加而掉分。  
最直接的信号是：
- o1-preview：**0.877 → 0.707**
- turn-3 最强模型也只有 **0.707**

这说明当前顶级模型在“持续遵循”上仍远未饱和。

#### 2. [诊断分析] 性能下降不只是误差累积，更包含明显“遗忘”
IFR 分析显示：
- 高性能模型的遗忘率更低；
- Llama 3.1 从 8B 扩到 405B 时，遗忘更少；
- Gemini 系列遗忘更明显。

这支持一个核心结论：  
**multi-turn 指令跟随的瓶颈之一是旧约束在新回合里被冲掉，而不只是单轮误差自然相乘。**

#### 3. [诊断分析] 强推理模型更容易纠正历史错误
ECR 分析中：
- o1-preview / o1-mini 的错误恢复率最高；
- 文中称其能恢复大约 **25%** 先前未遵循的指令。

这说明某些模型不仅“少忘”，还更可能在后续回合把之前没做好的约束补回来。  
不过要注意：这更像**相关性证据**，说明“强推理/隐藏 CoT”与恢复能力相关，而不是已被严格因果证明。

#### 4. [跨语错误分析] 多语差距真实存在，且集中暴露在非拉丁文字和复合约束
作者发现：
- 英语基本总是最好；
- Hindi、Russian、Chinese 错误率普遍更高；
- `length_constraints`、`combination` 是最难类别；
- `startend` 在 Russian / Chinese 上尤其不稳。

因此，Multi-IF 最有价值的地方之一，是把“模型多语能力不错”的泛化印象，细化成：
- 哪些语言掉队；
- 哪种约束最容易失效；
- 问题到底出在记忆、格式控制还是脚本适配。

### 局限性
- **Fails when**: 需要评估主观风格遵循、事实正确性、长达 10+ 轮的真实对话、工具调用/Agent 规划等场景时，Multi-IF 的三轮可验证约束设置不足以覆盖。
- **Assumes**: 指令必须能被规则脚本客观判定；数据构建依赖 Llama 3.1 405B、专业人工翻译/审校与语言特定的分句计数规则；这些假设会影响跨语言公平性与完全复现成本。
- **Not designed for**: 安全性、帮助度、知识正确性、偏好对齐质量、开放式对话体验等非“可验证指令遵循”能力。

### 复现与资源依赖
虽然数据与代码已开源，复现实验仍有几个实际依赖：
- 构建流程大量使用 **Llama 3.1 405B**；
- 多语质量依赖**专业人工审校**；
- 被测模型多为商业 API，且 token 设置不完全一致（如 o1 使用更高 max tokens）。

所以它的**benchmark 使用门槛低**，但**benchmark 构建门槛并不低**。

### 可复用组件
这篇工作最值得复用的不是某个分数，而是一套 benchmark engineering 模板：

1. **单轮 benchmark → 多轮 benchmark** 的扩展流程  
2. **LLM 初筛 + 人工复核** 的冲突/翻译/敏感过滤管线  
3. **strict/loose × instruction/conversation** 的双粒度评分框架  
4. **IFR / ECR** 这类把排名拆成“遗忘”和“纠错”的机制诊断指标  

如果你要做更长对话、更多语言、或 agent setting 的 instruction-following eval，这四个模块都可以直接迁移。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Long_Horizon_Episodic_Memory/arXiv_2024/2024_Multi_IF_A_Benchmark_for_Multi_turn_Instruction_Following.pdf]]