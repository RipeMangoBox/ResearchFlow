---
title: "Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios"
venue: ACL
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - llm-as-judge
  - tree-structured-planning
  - key-value-matching
  - dataset/UltraTool
  - opensource/full
core_operator: 用“真实查询驱动 + 工具无关自然语言规划 + 缺失/干扰工具集”的六维评测协议，分解诊断 LLM 在复杂场景中的完整工具利用能力。
primary_logic: |
  真实复杂用户查询 → 先生成与工具集解耦的树状自然语言计划 → 通过缺失工具集/干扰工具集分别评测规划、工具创建与工具使用六个维度 → 输出模型在真实工具代理场景中的能力边界
claims:
  - "UltraTool 构建了一个覆盖 22 个领域、2,032 个工具、5,824 个样本的双语基准，并同时评测规划、工具创建、工具使用共六个维度 [evidence: analysis]"
  - "在 UltraTool 上，GPT-4 取得最高总体成绩，中文/英文分别为 76.04% 和 74.58%，明显领先于所有开源模型 [evidence: comparison]"
  - "用于规划与工具创建的 GPT-4 多维 LLM-as-Judge 与人工评分高度一致，整体 Pearson 相关大约达到 0.85（规划）与 0.87（创建）[evidence: analysis]"
related_work_position:
  extends: "MetaTool Benchmark (Huang et al. 2023)"
  competes_with: "ToolBench (Qin et al. 2023b); API-Bank (Li et al. 2023)"
  complementary_to: "Toolformer (Schick et al. 2023); CREATOR (Qian et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Multi_Turn_Tool_Use/ACL_2024/2024_Planning_Creation_Usage_Benchmarking_LLMs_for_Comprehensive_Tool_Utilization_in_Real_World_Complex_Scenarios.pdf
category: Survey_Benchmark
---

# Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2401.17167), [GitHub](https://github.com/JoeYing1019/UltraTool)
> - **Summary**: 该文提出 UltraTool，用真实复杂查询把 LLM 的工具能力拆成“先规划、再创工具、再用工具”的六维评测，避免以预定义工具集反向造题带来的失真。
> - **Key Performance**: GPT-4 总分 76.04%（中文）/74.58%（英文）；GPT-4 评审与人工评分在规划/工具创建上的总体 Pearson 相关约为 0.85/0.87。

> [!info] **Agent Summary**
> - **task_path**: 真实复杂用户查询 + 候选/缺失/干扰工具集 -> 树状规划、缺工具判断、创工具、选工具、填参数 -> 六维评测分数
> - **bottleneck**: 现有基准把查询构造绑定到预定义工具集，导致只能测“匹配已有 API”，测不到规划先行与缺工具时的创建能力
> - **mechanism_delta**: 先独立评估工具无关的自然语言树状规划，再通过缺失工具集和干扰工具集把 awareness / creation / selection / usage 分离测量
> - **evidence_signal**: 14 个 LLM 的中英双语六维对比 + GPT-4 评审与人工评分高相关
> - **reusable_ops**: [真实查询驱动造题, 工具无关规划先评测]
> - **failure_modes**: [JSON 格式错误导致结果不可解析, 指令不遵循与幻觉导致步骤错位和漏答]
> - **open_questions**: [若换成可执行真实 API 排名是否稳定, LLM-as-Judge 在开放式工具创建上的偏差有多大]

## Part I：问题与挑战

这篇工作的核心不是再做一个“会不会调 API”的榜单，而是指出：**真实世界的工具代理能力，至少包含规划、发现工具缺口、创建工具、决定是否用工具、选对工具、填对参数** 六个环节。现有 benchmark 大多只测最后一环，甚至很多数据还是“先有工具，再按工具反向生成查询”，这会带来两个根本问题：

1. **评测对象被预定义工具集绑死**  
   如果 query 本身就是围绕已有工具造出来的，模型只要学会“把问题映射回已知 API”即可，测不到真正的任务分解与工具缺口识别。

2. **真实复杂性被低估**  
   现实任务通常跨域、多步、带约束，还可能需要新工具。仅测单次调用或固定工具选择，无法反映真实 agent 工作流。

3. **能力纠缠，无法定位瓶颈**  
   如果模型最终失败，你不知道是：
   - 没有先规划好，
   - 没发现现有工具不够，
   - 不会设计新工具，
   - 还是选错工具、参数填错。

### 这篇 paper 要解决的真瓶颈

**真瓶颈是“测量瓶颈”而不是单纯“模型瓶颈”**：以前的评测协议本身就把复杂工具使用压扁成了固定工具上的调用匹配问题，所以无法诊断真实 tool agent 的弱点。

### 输入 / 输出接口

UltraTool 把评测接口拆得很清楚：

- **规划**：`Query Q -> 树状自然语言计划 P`
- **工具创建**：`Plan P + 不完整工具集 T̂ -> 缺工具判断 + 新工具定义`
- **工具使用**：`Plan P + 完整/干扰工具集 -> 是否用工具 + 选哪个工具 + 参数怎么填`

这使它测的不是最终答案，而是**中间决策链条**。

### 边界条件

- 数据来自**真实复杂查询**，不是纯合成 query。
- 工具是**tool skeleton**，不是真正可执行 API。
- 支持中英文双语，原始收集语言是中文，再翻译到英文。
- 数据规模：22 个领域、5,824 个样本、2,032 个工具；平均每样本 12.27 步，且 39.61% 含 nested tool call。

---

## Part II：方法与洞察

### 1. 评测集怎么构出来

UltraTool 的构建流程有三段：

1. **Query Collection**  
   让不同领域专家先写真实复杂需求，再用 GPT-4 做 generalization / complication 扩展，最后人工审查。
   - 这一步的关键不是扩数据量，而是保证 query 分布更接近真实用户需求。
   - 与“工具先行”的 benchmark 相反，它是**query-first**。

2. **Solution Annotation**  
   用 GPT-4 先生成**不依赖预定义工具集**的树状自然语言计划，再补工具创建、计划 refinement、tool call annotation。
   - 计划里有父节点与叶子节点
   - 叶子节点分为：
     - tool-free step
     - tool-usage step

3. **Manual Refinement**  
   6 位专家做细致修订并双重检查，修正冗余步骤、逻辑缺口、参数来源、工具选择等问题。

### 2. 它到底评什么

UltraTool 将完整工具利用流程拆成六维：

1. **Planning**
2. **Tool Creation Awareness**
3. **Tool Creation**
4. **Tool Usage Awareness**
5. **Tool Selection**
6. **Tool Usage**

其中最有价值的两点是：

- **显式评估自然语言规划**
- **显式评估缺工具时的工具创建**

这两项正是此前 benchmark 普遍缺失的。

### 3. 评测协议的关键设计

- **规划评测**：用多维 LLM-as-Judge，从准确性、完整性、可执行性、语法、结构合理性、效率六个维度打分。
- **工具创建评测**：同样用多维 LLM-as-Judge，关注格式合规、准确性、内容合理性、可执行性、丰富度。
- **工具感知/选择**：用 key-value accuracy。
- **参数生成**：用 key-value Levenshtein distance，避免对表述方式过度脆弱。

此外，他们还人为构造：

- **不完整工具集 T̂**：测试模型能不能发现“工具不够”
- **含干扰工具的工具集 T̄**：测试模型会不会选错工具

这让原本缠在一起的 agent 能力，被拆成了可观测的局部决策。

### 核心直觉

**作者真正改变的旋钮是：从“工具驱动造题”切到“查询驱动 + 计划先行 + 工具可缺失”的评测视角。**

具体来说：

- **What changed**  
  从“给定工具，生成对应 query，再测调用”  
  变成“先给真实 query，先做工具无关规划，再分别测缺工具感知、创建和使用”。

- **Which bottleneck changed**  
  过去的限制是：query 和 plan 空间被已有工具集提前裁剪。  
  现在把计划先抽出来，等于把**任务分解能力**从**API 库覆盖度**里解耦出来。

- **What capability changed**  
  Benchmark 不再只测“会不会调用现有工具”，而是能测：
  - 会不会先分解任务
  - 知不知道现有工具不够
  - 能不能设计合理新工具
  - 在干扰工具下会不会选错
  - 参数能否从上下文和前序输出中正确接地

### 为什么这套设计有效

因为它把模糊的 “tool agent competence” 转化成了几类可控反事实：

- **把工具拿掉**：看模型能否识别缺口并补工具
- **给工具加干扰项**：看模型能否稳健选择
- **先不谈工具，只谈计划**：看模型是否真的理解任务结构

也就是说，它改变的是**测量分辨率**，不是简单增加难题数量。

### 战略性权衡

| 设计选择 | 带来的收益 | 代价 / 风险 |
|---|---|---|
| 真实专家 query，而不是工具反推 query | 更接近真实需求分布 | 标注成本高，专家分布可能引入偏置 |
| 先评自然语言规划，再评工具 | 把规划能力从工具覆盖中解耦 | 规划评分依赖 judge 模型 |
| 用 tool skeleton 而非真实 API | 可以覆盖“需要新工具”的情形 | 不能直接代表真实执行成功率 |
| 缺失工具集 + 干扰工具集 | 能拆分 awareness / creation / selection | 难度受构造策略影响 |
| GPT-4 作为多维评审 | 可扩展、可评开放文本 | 闭源依赖，评审偏置难完全排除 |

---

## Part III：证据与局限

### 关键证据看什么

#### 信号 1：闭源模型仍明显领先，尤其在创建与使用环节
- **信号类型**：comparison  
- **结论**：GPT-4 在中英双语上都最高，分别达到 **76.04% / 74.58%**。
- **含义**：当前 tool-agent 能力最难的部分，不是“写出像样的计划文字”，而是更结构化的创建/选择/填参。

一个很有意思的细节是：很多开源模型在**planning 的语法性**上不差，但在**完整性、可执行性、工具创建**上明显掉队。说明它们常常“说得像计划”，但不是真的把任务拆对了。

#### 信号 2：模型变大有用，但增益主要体现在结构化工具能力
- **信号类型**：analysis  
- **结论**：开源模型总体上参数越大表现越好；Qwen-72B 是最强开源，Mistral-7B 在 7B 档位很有性价比。
- **含义**：规模扩展对 tool utilization 有帮助，但还没抹平闭源差距，尤其是 JSON 格式、工具创建、参数绑定这些更“结构化”的输出任务。

#### 信号 3：LLM-as-Judge 不是乱打分，和人工评分高度一致
- **信号类型**：analysis  
- **结论**：规划与工具创建评测中，GPT-4 打分与人工评分总体 Pearson 相关约 **0.85** 与 **0.87**。
- **含义**：对这种开放文本、树状计划、工具定义任务，LLM-as-Judge 至少在这套协议下是可用的，不是纯噪声。

#### 信号 4：JSON 格式能力是隐藏瓶颈
- **信号类型**：analysis  
- **结论**：JSON format correct rate 与整体表现正相关。
- **含义**：很多模型不是“不知道该做什么”，而是“知道一点但输出结构不稳定”，最终使结果不可解析、不可执行。

### 1-2 个最关键指标

- **总体最好成绩**：GPT-4 达到 76.04%（中文）/ 74.58%（英文）
- **评审可靠性**：GPT-4 judge 与人工总体相关约 0.85（规划）/ 0.87（工具创建）

### 局限性

- **Fails when**: 你想把这个 benchmark 的结果直接解释为“真实 API 执行成功率”时会失效，因为这里的工具主要是功能 skeleton，不是可执行环境。  
- **Assumes**: 依赖 GPT-4 参与数据扩写、计划标注、工具创建、翻译和部分评测；还依赖专家人工 refinement 与双重校验，复现成本不低。  
- **Not designed for**: 长时多轮交互式 agent 执行、在线环境反馈、工具调用成本/时延优化、安全攻击面评估，这些都不是它的核心目标。  

### 复用价值

这篇工作最值得复用的不是某个分数，而是几个设计模式：

1. **query-first benchmark construction**  
   先从真实需求出发，而不是从工具库出发。

2. **plan-first evaluation**  
   先把计划抽出来单独测，避免把“不会规划”和“工具不够”混在一起。

3. **counterfactual toolset design**  
   通过缺失工具和干扰工具，分别测 awareness、creation、selection。

4. **结构化输出作为独立能力诊断**  
   JSON 合规不是工程细节，而是 tool agent 能否落地的关键能力。

**一句话总结**：UltraTool 的能力跳跃不在于“更难”，而在于它第一次比较系统地把真实工具代理的完整决策链条拆开来测，因此能更准确地告诉你：模型到底是不会想、不会补、不会选，还是不会填。  

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Multi_Turn_Tool_Use/ACL_2024/2024_Planning_Creation_Usage_Benchmarking_LLMs_for_Comprehensive_Tool_Utilization_in_Real_World_Complex_Scenarios.pdf]]