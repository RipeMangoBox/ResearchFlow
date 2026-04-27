---
title: "Large Language Models are Zero-Shot Reasoners"
venue: NeurIPS
year: 2022
tags:
  - Others
  - task/multi-step-reasoning
  - task/question-answering
  - chain-of-thought-prompting
  - two-stage-prompting
  - prompt-engineering
  - dataset/MultiArith
  - dataset/GSM8K
  - dataset/StrategyQA
  - opensource/partial
core_operator: 用统一触发语“Let's think step by step”先诱导零样本中间推理，再用第二次提示从推理文本中抽取最终答案
primary_logic: |
  自然语言问题 + 通用推理触发语 → 先生成显式链式推理 → 再用答案抽取提示将推理收束为数字/选项/Yes-No标准答案
claims:
  - "On text-davinci-002, Zero-shot-CoT raises MultiArith accuracy from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% over standard zero-shot prompting [evidence: comparison]"
  - "Using the same fixed trigger on PaLM 540B raises MultiArith accuracy from 25.5% to 66.1% and GSM8K from 12.5% to 43.0%, showing the effect transfers across model families [evidence: comparison]"
  - "On MultiArith, instructive reasoning triggers outperform misleading or irrelevant prompts, and the trigger Let's think step by step. gives the best reported accuracy (78.7%) among 16 tested templates [evidence: ablation]"
related_work_position:
  extends: "Few-shot-CoT (Wei et al. 2022)"
  competes_with: "GPT-3 few-shot prompting (Brown et al. 2020); Few-shot-CoT (Wei et al. 2022)"
  complementary_to: "Self-Consistency (Wang et al. 2022); Instruction Tuning (Ouyang et al. 2022)"
evidence_strength: strong
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/NeurIPS_2022/2022_Large_Language_Models_are_Zero_Shot_Reasoners.pdf"
category: Others
---

# Large Language Models are Zero-Shot Reasoners

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2205.11916) · [Code](https://github.com/kojima-takeshi188/zero_shot_cot)
> - **Summary**: 论文证明，大语言模型的多步推理能力并不一定依赖 few-shot 示例；只要用统一短语触发“先推理再作答”，就能把原本隐藏的零样本推理能力显式调出来。
> - **Key Performance**: text-davinci-002 上 MultiArith 17.7%→78.7%，GSM8K 10.4%→40.7%

> [!info] **Agent Summary**
> - **task_path**: 自然语言推理题（算术/符号/逻辑/常识） -> 标准化答案（数字/选项/Yes-No）
> - **bottleneck**: 标准 zero-shot 提示迫使模型直接报最终答案，把潜在多步推理压缩掉，并把推理过程与答案格式约束耦合在一起
> - **mechanism_delta**: 用统一 reasoning trigger 先展开显式 CoT，再用第二次、按答案格式轻量定制的 prompt 抽取最终答案
> - **evidence_signal**: 12 个数据集的系统比较 + 模型规模分析，显示困难推理任务上 Zero-shot-CoT 相对 standard zero-shot 有大幅跃升
> - **reusable_ops**: [通用推理触发语, 推理-答案两阶段提示]
> - **failure_modes**: [小模型无法稳定展开推理, 推理链后续漂移导致答案被覆盖或输出多个候选]
> - **open_questions**: [为什么少数短语特别有效, 生成的CoT是否忠实对应模型内部真实推理]

## Part I：问题与挑战

这篇论文真正要回答的不是“如何用更多示例教会模型推理”，而是：**大模型是否本来就具备零样本多步推理能力，只是标准提示没有把它激活出来？**

- **核心难题**：在算术、符号、日期推理、状态跟踪这类 system-2 任务上，标准 zero-shot 提示通常要求模型直接输出答案，结果往往表现很差，而且随模型规模扩大也不明显改善。
- **真正瓶颈**：此前 Few-shot-CoT 的成功，让社区默认“复杂推理必须靠手工构造 step-by-step exemplars”。作者指出，这里面混杂了两个因素：  
  1) exemplar 让模型进入“解释式推理”模式；  
  2) exemplar 还顺带教会了答案格式。  
  因此，以前的强结果不一定说明“必须 few-shot”，也可能只是说明“必须显式展开推理”。
- **输入/输出接口**：输入是自然语言问题；输出是 benchmark 所要求的数字、选项或 Yes/No。论文严格限制在**零样本、无微调、无外部工具、同一个 reasoning trigger 跨任务复用**。
- **为什么现在值得做**：因为 PaLM/GPT-3 级模型已经足够大，few-shot CoT 的结果暗示了潜在推理能力确实存在；如果零样本就能调出这类能力，就能显著降低样本工程成本，并给 reasoning 研究提供更干净的 baseline。

## Part II：方法与洞察

方法本身非常简单，但切得很准：**不是给模型更多任务知识，而是改变它开始生成时的“解题姿势”。**

### 方法概览

1. **第一阶段：Reasoning Extraction**  
   将问题包装成  
   `Q: [问题] A: Let's think step by step.`  
   让模型先生成一段中间推理链。

2. **第二阶段：Answer Extraction**  
   把“原问题 + 第一阶段推理文本 + 因此答案是...”拼起来，再调用同一个模型输出最终答案。  
   这一步本质上把“自由推理文本”收束为 benchmark 需要的格式。

3. **Answer Cleansing**  
   最后用简单规则提取第一个合法数字/选项/Yes-No，避免自由文本影响评测。

一个重要细节是：**真正跨任务固定不变的是 reasoning trigger**；第二阶段的答案抽取提示仍然会按答案类型做轻量定制，比如数字题和多选题的尾部提示不同。

### 核心直觉

这篇论文拧动的关键因果旋钮是：

- **What changed**：把模型的下一个 token 目标，从“直接预测最终答案”改成“先预测解释性 token / 中间状态 token”。
- **哪个瓶颈变了**：原来的 zero-shot prompt 把推理与答案格式压缩在一次短输出里，信息瓶颈很强；现在先允许模型把潜在计算过程展开，再单独做答案格式收束。
- **能力如何变化**：对足够大的 LLM 来说，这相当于把原本隐含在参数里的多步计算模式外化出来，于是困难任务不再只是“猜最终答案”，而能走“分解步骤 → 汇总答案”的路径。

为什么这个设计有效？因为 few-shot exemplars 很可能同时承担了“进入推理风格”和“提示输出格式”两种作用。作者把这两部分拆开后发现：

- **推理风格** 可以被一句通用 trigger 激活；
- **输出格式** 只需要一个很轻量的第二阶段提示即可；
- 因而，不需要每个任务都手工写完整的 CoT exemplars。

| 设计选择 | 改变了什么 | 收益 | 代价/风险 |
|---|---|---|---|
| 单一句子 trigger | 从“直接答题模式”切到“显式推理模式” | 跨任务通用，零样本可用 | 对 wording 敏感 |
| 两阶段 prompting | 将推理生成与答案格式解耦 | 降低格式错误，减少 exemplar 依赖 | 需要两次调用，延迟和成本更高 |
| 不用 few-shot exemplars | 去掉任务特定样本工程 | 部署简单，迁移方便 | 上限仍低于精心设计的 Few-shot-CoT |
| 贪心解码 + 规则解析 | 结果确定、便于复现 | 比较稳定 | 无法靠采样纠错，错误推理会被固化 |

## Part III：证据与局限

### 关键证据

**1. 多数据集比较信号：能力跳跃首先体现在困难推理任务上。**  
最强证据是相对 standard zero-shot 的大幅提升，而不是相对精心调过的 few-shot CoT。

- MultiArith：**17.7% → 78.7%**
- GSM8K：**10.4% → 40.7%**
- Coin Flip：**12.8% → 91.4%**
- Date Understanding：**49.3% → 67.5%**

这说明收益不仅限于算术，还覆盖符号和部分逻辑任务。

**2. 跨模型家族信号：不是某一个 API 的偶然现象。**  
在 PaLM 540B 上，同一 trigger 也有效，例如 GSM8K **12.5% → 43.0%**。这支持论文的主张：它更像是在释放“大模型已经学到的能力”，而不是某个特定模型的 prompt hack。

**3. 模型规模信号：该方法主要对“大模型已具备潜在推理能力”这一前提成立。**  
小模型几乎无法稳定受益，而随着模型变大，Zero-shot-CoT 的收益明显增强。  
所以这个方法不是“给所有模型加一个 magic phrase 都会变强”，而是“给足够大的模型一个展开潜在推理的接口”。

**4. 机制支持信号：prompt 内容确实重要，而且是“鼓励推理”这件事在起作用。**  
16 个模板的鲁棒性实验显示，鼓励思考的提示普遍优于误导或无关提示，其中 “Let’s think step by step.” 最好。这说明效果并非随便加一句文本都能得到。

**5. 与 prior work 的位置关系很清楚。**
- **优于 standard zero-shot**：这是论文最强结论。
- **优于 standard few-shot**：即使不给 exemplars，也明显强过普通 few-shot 直接答题。
- **弱于精心构造的 Few-shot-CoT**：例如 MultiArith 上 78.7% vs 93.0%。  
  所以它的意义是**建立超强 zero-shot baseline**，而不是宣称已超过 task-specific CoT 的天花板。

### 局限性
- **Fails when**: 需要从多个合理候选里稳定收敛到唯一答案的常识题；模型较小或推理能力不足时；长推理链后半段发生漂移，导致已经得到正确中间结论后又继续生成并改坏最终答案时。
- **Assumes**: 可访问足够大的预训练 LLM；需要手工指定按答案格式划分的第二阶段提示和规则解析器；论文最强结果依赖 OpenAI/PaLM 等闭源 API，且训练数据细节不透明，这会影响严格复现与机制解释。
- **Not designed for**: 保证 CoT 文本一定忠实反映内部推理；替代精心设计 Few-shot-CoT 的最优性能；处理需要外部工具、验证器或搜索过程的高精度数学求解。

### 可复用组件

- **通用 reasoning trigger**：`Let's think step by step`
- **推理先行、答案后抽取** 的两阶段 prompting
- **轻量 answer cleansing**：把开放式文本收束成 benchmark 可评分格式
- **可正交叠加的模块化接口**：后续可与 Self-Consistency 等解码策略组合

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/NeurIPS_2022/2022_Large_Language_Models_are_Zero_Shot_Reasoners.pdf]]