---
title: "GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning"
venue: arXiv
year: 2024
tags:
  - Others
  - task/agent-safety
  - task/policy-enforcement
  - code-generation
  - retrieval-augmented
  - chain-of-thought
  - dataset/EICU-AC
  - dataset/Mind2Web-SC
  - opensource/no
core_operator: 将自然语言安全请求先分解为任务计划，再生成并执行守护代码，以确定性地拦截目标智能体的违规动作
primary_logic: |
  安全请求+目标智能体规格+输入/输出日志 → 检索相似历史示例并生成任务计划 → 基于工具箱生成并执行守护代码 → 输出允许/拒绝标签及违规原因
claims:
  - "在 EICU-AC 上，GuardAgent 在四种核心 LLM 下达到 98.4%–99.1% 的标签预测准确率，并把解释准确率提升到 95.7%–97.5%，整体优于 Model-Guarding-Agent 与硬编码规则基线 [evidence: comparison]"
  - "在 Mind2Web-SC 上，GuardAgent 达到 83.5%–93.0% 的标签预测准确率，并在所有设置下保持 100% 的最终响应准确率，说明其能在不降低目标智能体任务性能的前提下执行安全控制 [evidence: comparison]"
  - "在 EICU-AC 上移除相关工具箱函数与记忆示例后，GuardAgent 的标签预测准确率从 98.7% 降至 90.8%，且代码可执行率下降，说明检索示例与受限工具接口对稳定守护至关重要 [evidence: ablation]"
related_work_position:
  extends: "NeMo Guardrails (Rebedea et al. 2023)"
  competes_with: "LlamaGuard (Inan et al. 2023); RigorLLM (Yuan et al. 2024b)"
  complementary_to: "ReAct (Yao et al. 2023); Reflexion (Shinn et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Medical_and_Clinical_Agents/arXiv_2024/2024_GuardAgent_Safeguard_LLM_Agents_by_a_Guard_Agent_via_Knowledge_Enabled_Reasoning.pdf
category: Others
---

# GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.09187), [Project](https://guardagent.github.io/)
> - **Summary**: 这篇工作把“让模型读安全规则后直接判决”改成“让一个 guard agent 把规则翻译成可执行代码并审查目标 agent 的动作日志”，从而以非侵入方式保护医疗和网页智能体。
> - **Key Performance**: EICU-AC 上 LPA 最高 99.1%、EA 最高 97.5%；Mind2Web-SC 上 LPA 最高 93.0%、EA 最高 94.0%，且 FRA 始终为 100%。

> [!info] **Agent Summary**
> - **task_path**: 安全请求 + target agent 输入/输出日志 -> allow/deny 标签 + 违规原因
> - **bottleneck**: 自然语言 guardrail 很难把文本政策精确对齐到 agent 的结构化动作、数据库列和网页操作上
> - **mechanism_delta**: 把“直接语言判决”改成“检索示例→生成任务计划→生成并执行守护代码”的两阶段守护流程
> - **evidence_signal**: 双基准对比中同时提升 guardrail accuracy 与 explanation accuracy，且对 target agent 的 FRA 无损
> - **reusable_ops**: [similarity-based demo retrieval, toolbox-constrained code generation]
> - **failure_modes**: [broad-rule-to-query linking failure, lower executability without toolbox/memory]
> - **open_questions**: [automatic toolbox construction, stronger reasoning or multi-agent decomposition]

## Part I：问题与挑战

### 1. 真问题是什么
这篇论文抓住的不是“LLM 会不会输出有害文本”，而是更具体也更难的瓶颈：

**当 LLM 变成 agent 后，它输出的不再只是文本，而是动作、代码、API 调用、数据库访问、网页点击。**  
此时安全约束也从“文本 harmfulness 分类”变成了“这个动作是否违反角色权限/平台规则/用户政策”。

这带来两个旧方法解决不了的问题：

1. **传统 LLM guardrail 不够用**  
   LlamaGuard / 文本 moderation 更适合判断“这段文本是否暴力、色情、仇恨”，不擅长判断：
   - 一个护士是否能访问 diagnosis 表；
   - 一个没有驾照的用户能否租车；
   - 某个网页动作是否违反年龄/地域/会员规则。

2. **把规则硬编码进 target agent 也不理想**  
   如果直接把安全规则塞进目标 agent 的 system prompt，目标 agent 需要同时做两件事：
   - 完成原任务；
   - 自己审查自己是否违规。  
   这种“侵入式”设计会干扰原任务性能，论文里确实测到 FRA 明显下降。

### 2. 真正瓶颈在哪里
作者认为真正瓶颈是：

> **如何把自然语言写成的安全请求，可靠地映射到 agent 的结构化输入/输出日志和动作轨迹上，并得到可执行、可审计、低歧义的判决。**

难点不在“看懂一句规则”，而在“把规则绑定到正确的字段、数据库、列名、用户属性和动作上”。

论文中的两个典型场景正好说明这一点：

- **医疗 agent：访问控制**
  - 输入：角色 + 医疗问题
  - 输出：是否允许访问某些数据库/列
- **网页 agent：安全策略**
  - 输入：网页任务 + 用户画像
  - 输出：是否允许执行某个网页动作

### 3. 为什么现在要解决
因为 LLM agent 已经开始进入高风险场景：
- 医疗
- 企业/组织数据访问
- 自动网页操作
- 更广义的工具调用系统

这些系统一旦出错，不是“说错一句话”，而是**真的执行了不该执行的动作**。  
所以现在需要的不是更强的文本审查器，而是**动作级、日志级、策略级的守护器**。

### 4. 输入/输出接口与边界条件
GuardAgent 的接口非常清楚：

- **输入**
  - 安全请求 `Ir`
  - 目标 agent 规格说明 `Is`
  - 目标 agent 的输入 `Ii`
  - 目标 agent 的输出日志 `Io`
- **输出**
  - 标签 `Ol`：0=允许，1=拒绝
  - 解释 `Od`：具体违规原因

它的边界也很明确：

- 必须能看到 target agent 的输入和日志；
- 规则需要能落到某种可执行检查上；
- 它是**并行守护**，不是修改 target agent 内部权重或训练过程。

### 5. 论文为此构建了两个专用基准
为了把问题落地，作者还新建了两个 benchmark：

- **EICU-AC**
  - 面向 EHRAgent
  - 评测医疗场景的 access control
  - 316 个样本，3 种角色，226 个 unique 问题
- **Mind2Web-SC**
  - 面向 SeeAct
  - 评测网页安全策略
  - 200 个样本，6 类规则

这两个 benchmark 的价值在于：它们把“agent safety”从泛泛讨论变成了**可比较、可诊断的动作级评测问题**。

---

## Part II：方法与洞察

### 1. 方法主线
GuardAgent 的核心不是再训练一个 safety classifier，而是让另一个 agent 来“守护”目标 agent。流程分两步：

1. **任务规划（Task Planning）**
   - 读取安全请求、agent 规格、target agent 输入/输出日志
   - 从 memory 里检索相似 demonstration
   - 生成一个分步骤的 action plan

2. **守护代码生成与执行（Guardrail Code Generation & Execution）**
   - 根据 action plan 和可调用函数列表生成代码
   - 在外部执行引擎中执行
   - 根据执行结果返回 allow/deny 与原因

这是一个很重要的设计转移：

- 以前：`规则文本 -> 语言判断`
- 现在：`规则文本 -> 计划 -> 代码 -> 执行结果`

### 2. 关键组件

#### 2.1 记忆增强的任务规划
作者没有让 LLM 零样本直接理解所有安全请求，而是引入 memory module，存放过去案例：

- target input/output
- 任务计划
- 守护代码

检索方式很简单：基于当前 `Ii + Io` 与历史案例的 **Levenshtein distance** 找相似示例。  
然后用这些 demonstration 做 in-context learning。

论文还规定了一个四步 planning 模板，本质是在强制 LLM 做局部化推理：

1. 总结 guard request 的关键词  
2. 从 guard request 中筛出和当前输入相关的信息  
3. 从 target agent 日志中提取与规则相关的动作/字段  
4. 指导后续代码生成去做比对和执行

这一步的作用是：  
**先把“该看哪条规则、该看日志的哪个部分”说清楚，再进入代码生成。**

#### 2.2 受工具箱约束的代码生成
第二步不是自由生成任意程序，而是给 LLM 一组**可调用函数**和接口说明，例如论文里用到的：

- `CheckAccess`
- `CheckRules`

然后要求模型只用这些函数生成 guardrail code。

这解决了两个老问题：

- 降低幻觉函数/幻觉 API；
- 把自然语言规则变成显式的结构化比较。

一旦代码执行成功，输出就是确定性的，不再依赖 LLM 最终那句自然语言判词。

#### 2.3 外部执行与调试
GuardAgent 还接了执行引擎（实验里默认是 Python）。  
如果代码执行报错，还可以调用一个 debugging 机制修复错误。

论文说这步在主实验里很少触发，说明前面的“受限工具+示例检索”已经显著提高了可执行率。

#### 2.4 非侵入式部署
GuardAgent 不改 target agent 的 prompt、不改训练、不改内部逻辑，只做旁路守护：

- target agent 正常提议动作；
- GuardAgent 审查该动作是否满足安全请求；
- 若违规，则 deny。

这就是作者强调的 **non-invasive**。

### 核心直觉

### 3.1 what changed
从“让 LLM 直接读规则并给结论”，改成“让 LLM 先做规则-日志对齐，再把判定逻辑写成代码执行”。

### 3.2 which bottleneck changed
改变的是**安全规则落地时的信息瓶颈**：

- 原来瓶颈：  
  规则、字段、动作都在自然语言空间里，容易被 LLM 模糊匹配、善意补全、错误泛化。
- 现在瓶颈被改写为：  
  规则被绑定到显式变量、函数调用、集合比较、字段映射上。

尤其在这类任务里，错误往往不是“不会推理”，而是：
- 把不允许的库“善意地补进可访问列表”；
- 混淆同名字段来自哪个数据库；
- 没把用户属性与规则条件绑定起来。

代码执行把这些问题从“语言解释问题”变成“程序匹配问题”。

### 3.3 what capability changed
因此带来的能力变化是：

- 从 **文本级 moderation** 升到 **动作级 policy enforcement**
- 从 **单模型主观判词** 升到 **可执行、可审计、可解释的守护**
- 从 **每个 agent 各写一套硬编码规则** 升到 **可扩展到新 agent/新规则的旁路守护**

### 3.4 为什么这个设计有效
因果上，GuardAgent 有效不是因为“又多用了一个 LLM”，而是因为它改了判定机制：

1. **记忆检索**  
   减少从新规则到新任务的冷启动翻译难度。

2. **先规划后编码**  
   把大问题拆成“找相关规则 / 找相关动作 / 做比对”。

3. **工具箱约束**  
   把自由生成空间压缩到可执行且已知的接口上。

4. **代码执行而不是语言判决**  
   最终结果由执行逻辑决定，降低 LLM 的“自作主张”。

论文里的 case study 很典型：  
prompt-only baseline 会把护士不该访问的 diagnosis 数据库“体贴地”加入可访问列表；而 GuardAgent 因为执行的是显式检查代码，不会做这种“善意篡改”。

### 4. 战略取舍

| 设计选择 | 带来的能力增益 | 代价/风险 |
|---|---|---|
| 非侵入式 guard agent | 不破坏 target agent，本身可插拔，FRA 可保持 100% | 需要完整日志，增加一次额外推理/执行时延 |
| 任务计划 + 代码执行 | 判定更确定、可审计、减少语义歧义 | 依赖执行引擎，规则必须能程序化表达 |
| 检索式 demonstrations | 少样本即可迁移到新规则/新 agent | 检索质量差时性能会掉 |
| 工具箱约束生成 | 降低幻觉函数，提升可执行率 | 新领域需要人工扩展 toolbox |
| 调试回路 | 增强鲁棒性 | 仍无法完全消除逻辑错误和语法错误 |

---

## Part III：证据与局限

### 1. 关键实验信号

#### 信号 A：文本 guardrail 不能直接迁移到 agent 守护
最强的负面对照是 LlamaGuard。  
在两个 benchmark 上它的 LPA 都接近随机：

- EICU-AC：50.3
- Mind2Web-SC：51.0

这说明传统“文本有害性” guardrail 并不能自然迁移到动作级 agent safety。

#### 信号 B：GuardAgent 相比 prompt-only guard 更稳定，尤其体现在解释准确率
在 **EICU-AC** 上，GuardAgent 几乎把问题做成了“高精度访问控制”：

- GPT-4 版本：**LPA 98.7 / EA 97.5**
- Llama3.3-70B 版本：**LPA 99.1 / EA 96.9**

相比之下，Model-Guarding-Agent 虽然有时 LPA 也不低，但 **EA 明显差很多**。  
这很关键：说明 prompt-only guard 可能“判断对了”，但**没真正找对违规原因**，而 GuardAgent 在精确定位违规数据库/列/规则上更可靠。

#### 信号 C：在网页 agent 上也能跨域成立
在 **Mind2Web-SC** 上，任务更开放、规则更杂，但 GuardAgent 仍然优于基线：

- Llama3.3-70B：**LPA 93.0 / EA 94.0**
- GPT-4：**LPA 90.0 / EA 80.0**

这里的意义是：方法不是只对“数据库权限匹配”有效，也能处理网页动作的规则型约束。

#### 信号 D：非侵入式是实打实的能力增益
论文最重要的系统级收益之一不是 LPA，而是 **FRA 始终 100%**。

也就是说，GuardAgent 不会伤害 target agent 的原任务能力。  
反过来，硬编码规则的侵入式基线会明显拖垮原任务：

- 例如 EHRAgent + GPT-4 的硬编码规则版本，**FRA 只有 3.2%**

这说明 capability jump 不只是“多判对几题”，而是：

> **第一次把安全守护从 target agent 内部职责中剥离出来，同时不牺牲原任务性能。**

#### 信号 E：ablation 支持“记忆 + 工具箱”确实是关键因果旋钮
在 EICU-AC 上拿掉相关 toolbox 和 memory 后：

- LPA：**98.7 -> 90.8**
- 可执行率也下降

这说明方法有效的关键不是“任何 LLM 都能自己想明白”，而是：
- demonstration 检索在帮助规则对齐；
- toolbox 约束在帮助代码落地。

### 2. 能力跃迁到底体现在哪
和 prior work 相比，这篇 paper 真正的跃迁不只是分数，而是把守护机制从：

- **content moderation**
- **硬编码规则**
- **自然语言 judge**

推进到了：

- **policy-grounded**
- **action/log-aware**
- **code-executable**
- **non-invasive**

所以它的“so what”很明确：  
如果你的系统已经是 tool-using LLM agent，GuardAgent 这类架构比普通 LLM guardrail 更贴近真实部署问题。

### 3. 局限性

- **Fails when**: 安全规则覆盖范围很宽、与当前查询/动作的关联需要额外语义桥接时，GuardAgent 仍会漏连规则；论文中 Rule 5（电影/音乐/视频的地域限制）就是代表性失败点。另一个失败边界是缺少匹配 demonstrations 或相关函数时，代码可执行率和判定精度会下降。
- **Assumes**: 能访问 target agent 的完整输入、输出日志和 agent specification；安全请求能被翻译成某种可执行检查；系统具备外部执行引擎；部署前有少量人工 demonstrations 和函数库；核心 LLM 需要有较强推理能力。
- **Not designed for**: 低时延实时控制、连续控制类 embodied 场景、无法从日志直接观测的隐性风险、完全开放世界且没有工具支持的复杂政策组合。

### 4. 资源与复现依赖
有几个现实约束值得明确写出：

- **开源状态不明确**：文中给了 project page，但正文未明确给出代码仓库，因此复现透明度仍有限。
- **benchmark 构建有人力成本**：EICU-AC 的角色权限由临床专家和 ChatGPT 共同制定。
- **运行开销增加**：GPT-4 版 GuardAgent 在 EICU-AC / Mind2Web-SC 上平均约 45.4s / 37.3s，每次守护比 prompt-only baseline 更慢。
- **toolbox 仍是手工设计**：这限制了新领域零成本迁移。

### 5. 可复用组件
这篇工作最值得迁移的不是某个具体 benchmark，而是以下操作模块：

- **旁路式 guard wrapper**：不改 target agent，只对其动作做 allow/deny
- **四步任务规划模板**：先找规则，再找日志，再做比较
- **相似案例记忆检索**：用历史 guard 示例降低新任务冷启动难度
- **受限函数驱动的代码生成**：把自由文本判决变成可执行策略检查
- **执行后解释输出**：不仅给 deny，还给具体违规原因

---

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Medical_and_Clinical_Agents/arXiv_2024/2024_GuardAgent_Safeguard_LLM_Agents_by_a_Guard_Agent_via_Knowledge_Enabled_Reasoning.pdf]]