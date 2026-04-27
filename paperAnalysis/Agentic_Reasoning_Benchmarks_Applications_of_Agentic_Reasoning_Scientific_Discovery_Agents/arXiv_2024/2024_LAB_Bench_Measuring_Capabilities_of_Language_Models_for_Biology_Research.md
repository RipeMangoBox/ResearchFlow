---
title: "LAB-Bench: Measuring Capabilities of Language Models for Biology Research"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - multiple-choice
  - reject-option
  - human-baseline
  - dataset/LAB-Bench
  - opensource/partial
core_operator: 围绕生物科研中的文献检索、图表解读、数据库访问、序列操作与协议排障，构造带拒答选项和人类基线的多子任务评测基准。
primary_logic: |
  实用生物研究能力评测目标 → 手工+程序化构造 8 类 2457 道题，并加入“信息不足”拒答选项与公开/私有拆分 → 用 accuracy/precision/coverage、人类专家对照和开放回答抽查评测模型 → 揭示无工具 LLM 在检索依赖、长序列操作和复杂克隆任务上的能力边界
claims:
  - "在 8 个顶层类别的 accuracy 上，人类专家均高于所有被测模型；仅 Claude 3.5 Sonnet 在 TableQA precision 上以 0.90 略超人类的 0.87 [evidence: comparison]"
  - "加入拒答选项后，LitQA2、SuppQA 与 DbQA 等检索依赖任务出现显著 coverage 下滑，说明无工具模型在需要外部查找时经常选择不答 [evidence: analysis]"
  - "开放回答复核中，Claude 3.5 Sonnet 和 GPT-4o 在 CloningScenarios 仅 0.20 accuracy、在 FigQA 仅 0.30 accuracy，表明多项选择成绩部分来自干扰项排除与猜测 [evidence: analysis]"
related_work_position:
  extends: "LitQA (Lála et al. 2023)"
  competes_with: "GPQA (Rein et al. 2023); BioLLMBench (Sarwal et al. 2023)"
  complementary_to: "PaperQA (Lála et al. 2023); GeneGPT (Jin et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Scientific_Discovery_Agents/arXiv_2024/2024_LAB_Bench_Measuring_Capabilities_of_Language_Models_for_Biology_Research.pdf
category: Survey_Benchmark
---

# LAB-Bench: Measuring Capabilities of Language Models for Biology Research

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2407.10362), [Hugging Face Dataset](https://huggingface.co/datasets/futurehouse/lab-bench)
> - **Summary**: 这篇工作把“生物学考试题”升级为“生物科研实务题”，用文献、补充材料、图表、数据库、序列与克隆场景来测量 LLM 作为科研助手的真实可用性。
> - **Key Performance**: Claude 3.5 Sonnet 在 TableQA 上 precision 0.90，高于人类 0.87；但开放回答版 CloningScenarios 上 Claude 3.5 Sonnet 与 GPT-4o 均仅 0.20 accuracy。

> [!info] **Agent Summary**
> - **task_path**: 生物科研工件（论文/补充材料/图表/数据库/序列/协议） -> 多项选择答案或拒答
> - **bottleneck**: 现有科学 benchmark 主要测教材式知识回忆，难以暴露真实科研流程中的检索、图表理解、长序列操作与 protocol troubleshooting 短板
> - **mechanism_delta**: 用 8 类 practical biology tasks + 拒答选项 + 人类专家基线 + 开放回答抽查，替代单纯课本式科学问答评测
> - **evidence_signal**: 人类在绝大多数类别显著强于无工具模型，且开放回答显著拉低 CloningScenarios/FigQA 成绩
> - **reusable_ops**: [reject-option scoring, open-response spot-check]
> - **failure_modes**: [multiple-choice distractors inflate scores, human-hard tasks have noisy human baselines]
> - **open_questions**: [how much tool-augmented agents improve on LAB-Bench, how to scale reliable open-response grading]

## Part I：问题与挑战

这篇论文要解决的不是“再造一个生物问答集”，而是一个**评测错位**问题：  
现有科学 benchmark 往往测的是模型能否回答课本式、摘要级、或互联网常见的科学问题；但真正的科研助手需要处理的是：

- 从论文正文或补充材料里找到一次性出现的信息
- 看懂技术图和数据表，而不是只读 caption
- 去多个生物数据库里查证具体事实
- 对 DNA/RNA/蛋白序列做精确操作
- 在实验 protocol 出错时做 troubleshooting
- 在复杂 cloning workflow 中完成多步推理

### 真正的瓶颈是什么？

**真正瓶颈不是“模型有没有生物知识”，而是“模型能否在真实科研工件上完成任务，并在不知道时正确拒答”。**

这和传统 benchmark 的差别很大。传统 benchmark 更像“会不会考试”；LAB-Bench 更像“能不能在实验室或文献工作流里帮上忙”。

### 为什么现在要解决？

因为前沿 LLM 已经被广泛讨论为“AI scientist”或科研助手，但如果评测仍停留在教材题层面，就会出现两个问题：

1. **高分不等于有用**：模型可能只是在做 test-taking，而不是在做 research task。
2. **优化方向错误**：社区可能继续强化记忆与考试技巧，而不是检索、工具使用、序列操作和跨模态科研理解。

### 输入 / 输出接口

LAB-Bench 的输入不是单一文本，而是多种科研工件：

- 论文标题 + DOI
- 补充材料线索
- figure / table 图像
- 数据库查询目标
- DNA / RNA / protein 序列
- protocol 文本
- cloning 场景中的多个质粒/片段/酶/流程

输出则是：

- 多项选择答案
- 或明确选择“信息不足/无法回答”

### 边界条件

这篇论文的主结果有几个很重要的边界：

- **模型评测默认无工具**：不允许联网、数据库访问、检索插件
- **人类评测允许用工具**：互联网和 DNA 软件可用
- **任务是 biology-specific**：不追求覆盖全部科学
- **主要是静态单轮 QA**：并不直接评测 agent 的长期规划或真实实验执行

所以它测的是：**无工具基础模型在生物科研微任务上的裸能力边界**。

---

## Part II：方法与洞察

作者把“生物科研实务能力”拆成 8 个可标准化评测的大类，共 **2,457** 道题：

| 类别 | 主要测什么 | 关键难点 |
|---|---|---|
| LitQA2 | 论文全文检索与推理 | 问题通常不能靠摘要回答 |
| SuppQA | 补充材料检索与理解 | 信息藏在 supplementary text / tables |
| FigQA | 科研图像理解 | 无 caption，仅靠图像内容推理 |
| TableQA | 科研表格理解 | 既要读表，也要做简单运算/比较 |
| DbQA | 生物数据库访问能力 | 涉及多数据库，非单 API 可解 |
| ProtocolQA | protocol troubleshooting | 要从错误结果反推需要修改的步骤 |
| SeqQA | 序列理解与操作 | 长字符串精确操作困难 |
| Cloning Scenarios | 真实分子克隆多步推理 | human-hard，多质粒/酶/流程联动 |

题目构建上，作者采用了**程序化生成 + 专家手工编写**的混合路线：

- **程序化**：适合 SeqQA、DbQA 这类结构化任务
- **手工专家生成**：适合文献、图表、protocol、cloning 这类高语义密度任务

此外，数据集还设计了：

- **80% public / 20% private split**：监控未来 contamination
- **accuracy / precision / coverage 三指标**
- **human expert baseline**
- **小规模 open-response 复核**

### 核心直觉

LAB-Bench 的核心不是“题更多了”，而是**测量瓶颈被改了**。

#### 1. 从“知识分布”换到“科研工件分布”
- **改变了什么**：题目不再主要来自教材事实，而来自论文正文、supplement、数据库、图表、序列和实验流程。
- **改变了哪个瓶颈**：把评测分布从“训练语料里常见的标准知识”切到“低频、长尾、工具依赖、工件依赖”的真实科研分布。
- **带来了什么能力诊断**：能分辨模型是在背知识，还是能做 research-grounded task。

#### 2. 从“强制作答”换到“允许拒答”
- **改变了什么**：每题都有“信息不足/不回答”选项。
- **改变了哪个瓶颈**：把“会不会答”与“敢不敢乱答”分开。
- **带来了什么能力诊断**：可以同时看 accuracy、precision、coverage，测到不确定性表达与选择性回答能力。

#### 3. 从“只看 MCQ 分数”换到“MCQ + 开放回答校验”
- **改变了什么**：作者在部分任务上做了 open-response 复核。
- **改变了哪个瓶颈**：抑制模型靠排除干扰项、考试技巧拿分。
- **带来了什么能力诊断**：能看出模型是真会推理，还是只是“会猜题”。

### 为什么这套设计有效？

因为它在因果上对准了科研助手最关键的几个误差来源：

- **检索缺失**：LitQA2 / SuppQA / DbQA 强迫模型面对“没有外部访问时到底会不会”
- **跨模态缺失**：FigQA / TableQA 把文本流畅性和技术图表理解拆开
- **符号操作缺失**：SeqQA / CloningScenarios 暴露长序列与多步克隆推理的精度问题
- **校准缺失**：拒答机制揭示模型是否知道自己不知道

### 战略权衡

| 设计选择 | 获得的诊断能力 | 代价 / 妥协 |
|---|---|---|
| 多项选择 + 拒答 | 可同时测 accuracy / precision / coverage | 强依赖 distractor 质量，易高估真实能力 |
| 手工 + 程序化混合出题 | 兼顾规模与专业性 | 标注昂贵，题型一致性难完全保证 |
| public/private split | 能监控 contamination | 社区无法完全复现全量测试集 |
| 人类专家基线 | 给出科研可用性的现实参照 | 覆盖不全，human-hard 任务招募困难 |
| 开放回答抽查 | 检查 MCQ 分数是否虚高 | 评分成本高，难以大规模化 |

---

## Part III：证据与局限

### 关键证据

**信号 1：人类仍显著领先，尤其是在真实检索和复杂实验任务上。**  
这是这篇论文最核心的结论。人类专家在几乎所有顶层类别上都明显优于模型，尤其是：

- LitQA2
- SuppQA
- DbQA
- Cloning Scenarios

这说明即便前沿模型已经很强，它们距离“可靠生物科研助手”仍有明显差距。

**信号 2：拒答机制揭示了“高 precision 不代表真能用”。**  
最典型的是 SuppQA 这类需要查 supplementary 的任务。  
例如 Claude 3.5 Sonnet 的 SuppQA **precision 可到 0.75**，但 **coverage 只有 0.04，accuracy 仅 0.02**。  
这说明模型并不是“会做”，而是“只在极少数自信题上作答”。对科研助手来说，这种能力离可用还很远。

**信号 3：表格比技术图更容易，说明视觉科研理解并不均衡。**  
Claude 3.5 Sonnet 在：

- **TableQA precision = 0.90**，高于人类 **0.87**
- 但 **FigQA precision = 0.54**，仍远低于人类 **0.82**

这表明“看懂表格图片”并不等于“看懂科研 figure”。技术图往往需要跨元素、多跳视觉推理，这仍是明显短板。

**信号 4：序列任务不是一概不会，而是“局部匹配还行，长序列精确操作很差”。**  
作者发现模型在某些 primer 选择类任务上能接近甚至超过人类，但在以下类型明显失效：

- 从长模板中定位子序列
- 计算 PCR 扩增片段长度
- 推断 restriction digest 片段数和长度

也就是：**局部模式匹配可以，长程精确操作不行。**

**信号 5：开放回答复核显示 MCQ 成绩被高估。**  
在开放回答设置下：

- CloningScenarios：Claude 3.5 Sonnet / GPT-4o 都只有 **0.20 accuracy**
- FigQA：两者都只有 **0.30 accuracy**

这个结果直接支持作者的判断：一些 MCQ 正确答案其实是靠“排除不合理选项 + 猜测”得到的，而不是通过扎实的科研推理得到的。

### 1-2 个最值得记住的数字

- **TableQA**：Claude 3.5 Sonnet precision **0.90**，人类 **0.87**
- **CloningScenarios 开放回答**：Claude 3.5 Sonnet / GPT-4o 均仅 **0.20 accuracy**

前者说明某些结构化视觉任务已接近甚至超过人类；后者说明真正高难、长链条科研任务还远未解决。

### 局限性

- **Fails when**: 需要评测真实 agent 执行、网页导航、湿实验闭环、长时程规划或开放式科研写作时，LAB-Bench 的静态 MCQ 形式会失真；在 CloningScenarios 这类 human-hard 任务上，若干扰项设计不够强，模型分数会被虚高。
- **Assumes**: 依赖高成本生物专家出题与复核；主实验假设模型通过商业 API 且无外部工具；人类基线允许使用互联网与 DNA 软件，因此人机比较不是完全等条件；公开数据仅约 80%，完整复现实验需私有集。
- **Not designed for**: 覆盖全部生物学子领域、给出最佳提示词下的模型能力上限、或直接评估带工具 scientific agents 的最终性能。

### 复现与资源依赖

这篇论文的可扩展性受几个现实条件影响很大：

- **专家标注成本高**：人工生成高质量科研题很贵
- **闭源模型依赖**：主结果包含商业 API 模型
- **工具设置不对称**：人类可用工具，模型无工具
- **全量数据不公开**：为防污染保留私有测试集

因此，它更像一个**高价值但高维护成本**的 benchmark 基础设施。

### 可复用组件

即使不做 biology benchmark，这篇论文也提供了几个很可迁移的评测操作：

- **拒答选项 + accuracy/precision/coverage 三指标**
- **public/private split 做 contamination 监控**
- **专家手工 + 程序化混合构题**
- **open-response 抽查，用来识别 MCQ 虚高**
- **先让前沿模型生成草题，再由专家修订，用来探索模型“能力空白区”**

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Scientific_Discovery_Agents/arXiv_2024/2024_LAB_Bench_Measuring_Capabilities_of_Language_Models_for_Biology_Research.pdf]]