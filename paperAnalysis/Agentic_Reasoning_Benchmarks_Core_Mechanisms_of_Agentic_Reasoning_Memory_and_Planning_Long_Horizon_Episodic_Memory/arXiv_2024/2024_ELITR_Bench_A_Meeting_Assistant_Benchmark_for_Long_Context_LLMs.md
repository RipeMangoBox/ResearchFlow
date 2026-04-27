---
title: "ELITR-Bench: A Meeting Assistant Benchmark for Long-Context LLMs"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/question-answering
  - task/llm-evaluation
  - llm-as-a-judge
  - noise-injection
  - rubric-based-evaluation
  - dataset/ELITR-Bench
  - opensource/full
core_operator: "基于真实会议ASR转写构造手工问答/会话样本与可控噪声版本，并用经人工校验的GPT-4量表评分系统评测长上下文LLM的会议助手能力"
primary_logic: |
  会议助手评测目标 → 在ELITR会议转写上手工构造QA/Conv问题与答案并注入多级ASR噪声 → 用GPT-4量表评分并以专家/众包人工校验 → 揭示长上下文LLM在真实会议问答、多轮对话与噪声鲁棒性上的能力边界
claims:
  - "Claim 1: 在ELITR-Bench的三种主设置中，GPT-4/4o平均分均高于8.3，而LLaMA-3.1-8B与Phi-3-small已明显超过多数LLaMA-2系长上下文模型 [evidence: comparison]"
  - "Claim 2: 对同一批QA问题，从single-turn切到multi-turn时，LLaMA-2系模型分数显著下降，而GPT-4/4o、LLaMA-3.1-8B和Phi-3-small基本保持稳定 [evidence: comparison]"
  - "Claim 3: GPT-4评委与专家人工和众包人工评分分别达到0.82和0.78的Pearson相关，但其10分制评分实际上主要聚成约3个质量层级 [evidence: analysis]"
related_work_position:
  extends: "ELITR Minuting Corpus (Nedoluzhko et al. 2022)"
  competes_with: "LongBench (Bai et al. 2023); L-Eval (An et al. 2023)"
  complementary_to: "Retrieval Meets Long Context LLMs (Xu et al. 2024); Prometheus (Kim et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Long_Horizon_Episodic_Memory/arXiv_2024/2024_ELITR_Bench_A_Meeting_Assistant_Benchmark_for_Long_Context_LLMs.pdf"
category: Survey_Benchmark
---

# ELITR-Bench: A Meeting Assistant Benchmark for Long-Context LLMs

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.20262), [GitHub](https://github.com/utter-project/ELITR-Bench)
> - **Summary**: 这篇工作把真实会议 ASR 转写扩展成面向“会议助手”场景的长上下文基准，通过单轮/多轮问答、可控转写噪声和经人工校验的 GPT-4 打分，测出模型在真实部署分布下的能力边界。
> - **Key Performance**: GPT-4/4o 在三种主设置上的平均分约为 8.30–8.48；GPT-4 评委与专家/众包人工评分的 Pearson 相关分别为 0.82 / 0.78。

> [!info] **Agent Summary**
> - **task_path**: 长会议 ASR 转写文本 + 单轮/多轮问题 -> 会议问答回答质量评分与长上下文能力画像
> - **bottleneck**: 现有长上下文评测多是干净、泛化文本，缺少真实会议口语噪声与多轮依赖，难以测到 meeting assistant 的实际可用性
> - **mechanism_delta**: 在 ELITR 会议语料上新增 271 个手工 QA、构造 Conv 版与多级噪声转写，并以 GPT-4 量表评分结合人工校验形成统一评测协议
> - **evidence_signal**: 12 个长上下文模型对比 + 噪声鲁棒性曲线 + GPT-4 与人工评分高相关验证
> - **reusable_ops**: [manual long-context qa annotation, controllable asr-noise injection, rubric-based llm-as-a-judge validation]
> - **failure_modes**: [gpt-4 judge cannot reliably separate more than about three quality levels, llama-2-based long-context models get distracted in multi-turn mode]
> - **open_questions**: [how to extend the benchmark to passage-level rag evaluation, how much anonymization changes who-question difficulty]

## Part I：问题与挑战

这篇论文解决的不是“长上下文模型能不能读长文档”这种泛问题，而是更具体的：

**长上下文 LLM 能否在真实会议助手场景中，从嘈杂、口语化、带 ASR 错误的长会议转写里稳定回答问题？**

### 为什么现有基准不够
现有 long-context benchmark 大多有两个问题：

1. **数据分布太干净**：常见是 Wikipedia、书面文档、合成任务，不像真实会议转写那样有口语停顿、插话、ASR 错误、匿名化实体。
2. **任务过于泛化**：很多任务更像长文档检索或抽取，不能代表“没参加会议的人问会议发生了什么”这种真实助手场景。

论文认为，会议助手场景有三个更真实的难点：

- **长上下文**：单个会议平均约 11k–13k token，最长约 17.6k。
- **文本噪声**：转写来自 ASR，仍有错误和口语现象。
- **对话依赖**：在 Conv 设置里，后续问题会依赖前面对话中的指代与省略。

### 输入/输出接口
- **输入**：会议转写文本 + 用户问题；在 multi-turn/Conv 设置下还包含之前的问答历史。
- **输出**：会议相关自然语言答案。
- **边界条件**：
  - 仅覆盖 **英语会议**
  - 会议主题偏 **计算机科学/NLP**
  - 输入是 **文本转写**，不是原始音频或多模态会议流
  - 文本经过 **去标识化**，如 `[PERSON3]`

### 真正瓶颈
真正的瓶颈不是“窗口够不够长”，而是：

> **模型能否在长、脏、口语化、带实体匿名和多轮依赖的上下文里，稳定保持信息定位、会话状态和答案精确性。**

这也是为什么现在值得做：模型上下文窗口已经扩到 32k、128k 甚至更高，如果评测仍停留在干净长文档，就无法区分“纸面长上下文能力”和“可部署会议助手能力”。

---

## Part II：方法与洞察

论文的核心不是新模型，而是**把 long-context evaluation 从“泛长文”改成“真实会议助手诊断”**。

### Benchmark 设计
作者在 ELITR Minuting Corpus 上构造了 **ELITR-Bench**：

- **数据来源**：ELITR-English 的 dev/test2，共 **18 场会议**（10 dev + 8 test）
- **新增标注**：**271 个手工编写问题及标准答案**
- **问题类型**：Who / What / When / How many
- **答案位置标注**：Beginning / Middle / End / Several passages
- **两种任务设置**：
  - **ELITR-Bench-QA**：每个问题可独立回答
  - **ELITR-Bench-Conv**：问题按顺序提问，部分问题依赖前文问答

### 可控噪声设计
为了单独测“噪声鲁棒性”，作者又构造了多级 noisy transcript：

- 基于 50 万+ 对齐 ASR 数据生成 **86,148 条替换规则**
- 目标 WER：20%、40%、60%、80%、100%
- 用这些规则把同一会议转写改写成不同噪声版本

这一步很关键，因为它把两个常被混在一起的变量拆开了：
- **上下文长**
- **文本脏**

这样 benchmark 不只是排分，还能测模型对实际部署噪声的敏感性。

### 评分协议
评测采用统一 prompt，把会议转写放进上下文，再测试：

- **single-turn**：每个问题独立开新对话
- **multi-turn**：同一会议的所有问题连续提问

答案评分主要由 **GPT-4 judge** 完成，采用 **1–10 分 rubric-based evaluation**；同时再用：
- Prometheus
- 专家人工（Gold Human）
- 众包人工（Silver Human）

做评测器校验。

### 核心直觉

真正的创新不在“又做了一个 benchmark”，而在于它改变了**测量瓶颈**：

1. **从干净长文档 → 真实会议转写**  
   改变了输入分布。  
   结果是，测到的不再只是“长文定位能力”，而是“抗噪读取 + 口语理解 + 实体恢复”。

2. **从单问单答 → 会话依赖问答**  
   改变了约束。  
   结果是，能暴露模型是否会被前几轮问答干扰，是否真的能维护对话状态。

3. **从静态平均分 → 分维度诊断**  
   通过问题类型、答案位置、噪声等级和评委对齐分析，benchmark 从“排行榜”变成“能力边界图”。

换句话说，这篇论文的因果链是：

> **更真实的输入分布 + 更细的任务拆分 + 可控噪声 + 校准过的自动评分**  
> → **测量更接近真实部署中的失败模式**  
> → **能看见旧 benchmark 难以暴露的能力差异**

### 策略性权衡

| 设计选择 | 解决的测量盲点 | 带来的能力诊断 | 代价/风险 |
|---|---|---|---|
| 真实会议转写替代通用长文档 | 泛 benchmark 与部署分布不匹配 | 更能测 meeting assistant 可用性 | 领域较窄，主题偏技术会议 |
| QA + Conv 双设置 | 只测检索，不测会话状态 | 可区分单轮能力与多轮抗干扰能力 | Conv 中真正依赖上下文的问题数不多 |
| 多级 WER 噪声注入 | 难以系统评测鲁棒性 | 能画出性能-噪声曲线 | 合成噪声不完全等于真实会议 ASR |
| GPT-4 量表评测 + 人工校验 | 全人工评测昂贵且慢 | 大规模自动评分可行 | 闭源依赖，细粒度分辨率有限 |

---

## Part III：证据与局限

### 关键证据

**信号 1：代际进步是清晰可见的。**  
12 个长上下文模型比较显示，GPT-4/4o 在三种主设置中始终领先；新一代开源模型 **LLaMA-3.1-8B** 和 **Phi-3-small** 明显超过多数 LLaMA-2 系长上下文模型。  
这说明：**上下文窗口增大之后，模型代际升级确实带来了真实会议 QA 能力提升，而不只是“能塞更长输入”。**

**信号 2：multi-turn 会暴露“假长上下文”。**  
在同一批 QA 问题上，LLaMA-2 系模型从 single-turn 切到 multi-turn 时明显掉分；但 GPT-4/4o、LLaMA-3.1-8B 和 Phi-3-small 基本稳住。  
这说明：**很多模型的问题不是读不下长文本，而是会被先前对话历史干扰，无法稳定维护状态。**

**信号 3：clean transcript 上接近，不代表真实部署上接近。**  
在干净转写上，LLaMA-3.1-8B / Phi-3-small 与 GPT-4o 的差距已不大；但随着噪声上升，开源模型和 GPT-4o 的差距显著拉大。GPT-4o 在高噪声下平均分仍高于 6。  
这说明：**噪声鲁棒性仍是 frontier proprietary model 和当前开源模型的重要分水岭。**

**信号 4：GPT-4 judge 可以用，但不要过度相信它的“10 分制精度”。**  
GPT-4 judge 与专家人工、众包人工评分分别达到 **0.82 / 0.78** 的 Pearson 相关，说明它适合做大规模自动评测；但更细看分布后，作者发现 GPT-4 的分数大致只稳定区分出 **约 3 个质量层级**。  
这说明：**LLM-as-a-judge 更适合排序和粗粒度比较，不适合过细的分数解释。**

**信号 5：没有强烈的全局“lost in the middle”。**  
按答案位置分析后，作者没有在大多数模型上观察到强烈的 middle position 劣化。  
这说明：**在这个真实会议场景中，位置效应未必像某些合成设置里那样主导表现。**

### 1-2 个最关键指标
- **模型侧**：GPT-4/4o 在三种主设置的平均分约 **8.30–8.48**；LLaMA-3.1-8B 约 **7.76–7.79**。
- **评测器侧**：GPT-4 judge 与 Gold / Silver Human 的 Pearson 相关为 **0.82 / 0.78**。

### 局限性
- **Fails when**: 转到非英语、非技术领域会议，或需要音频线索、说话人声学特征、视觉材料、跨会议长期记忆的场景时，这个 benchmark 的代表性会明显下降；此外，合成噪声未必覆盖真实会议 ASR 的全部错误模式。
- **Assumes**: 有手工编写的问答和参考答案；主评测依赖闭源 GPT-4 API；噪声模拟基于 RED-ACE 的 ASR 错误分布；开放模型推理默认具备较高算力条件（文中使用单张 A100 80GB），并且评测/校验存在额外成本（如配置搜索约 \$150、众包约 £400）。
- **Not designed for**: 端到端语音会议助手、多模态会议理解、会议摘要/行动项生成质量评测、实时系统延迟/成本评估、超大规模通用长上下文统一排行榜。

### 可复用组件
- **数据层**：ELITR-Bench 的问题、答案、元数据已开放。
- **评测层**：模型生成结果与 GPT-4 judge 分数一并开放，便于复现实验。
- **协议层**：QA / Conv 双设置、答案位置标注、rubric-based judge prompt 都可迁移到其他长上下文 benchmark。
- **鲁棒性层**：可控 ASR 噪声注入流程可直接复用到其他 transcript QA 评测。

**一句话总结 So what：**  
这篇工作最重要的价值，不是再做一个长上下文排行榜，而是证明了：**当评测分布换成真实会议转写后，模型差距会从“谁更会读长文”重新洗牌成“谁真正能当会议助手”。**

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Long_Horizon_Episodic_Memory/arXiv_2024/2024_ELITR_Bench_A_Meeting_Assistant_Benchmark_for_Long_Context_LLMs.pdf]]