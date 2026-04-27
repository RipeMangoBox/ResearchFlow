---
title: "Playing repeated games with Large Language Models"
venue: arXiv
year: 2023
tags:
  - Evaluation
  - task/MLLM-evaluation
  - repeated-games
  - social-chain-of-thought
  - dataset/2×2-games
  - opensource/full
core_operator: 将有限轮2×2重复博弈文本化为提示链评测协议，并用“先预测对手再决策”的SCoT干预LLM的合作与协调行为
primary_logic: |
  社会互动能力评测目标 → 把收益矩阵、轮次历史与问句转成文本提示，让LLM与LLM/手工策略/人类重复对弈 → 比较得分、合作率、协调率与行为轨迹 → 揭示LLM在自利博弈强、在惯例形成与协调博弈弱的能力边界
claims:
  - "在六类有限重复2×2博弈的总体比较中，GPT-4取得五个被测LLM中最高的总体归一化得分0.854，并在囚徒困境家族上表现尤强 [evidence: comparison]"
  - "在有限轮囚徒困境中，GPT-4在对手仅首轮背叛、后续持续合作时仍持续背叛；即便显式告知该模式，行为也基本不变 [evidence: case-study]"
  - "社会链式思维提示（SCoT）提升了GPT-4与人类在Battle of the Sexes中的协调成功率与人类感知到的‘像人程度’，同时未显著改变人类在囚徒困境中的平均得分 [evidence: comparison]"
related_work_position:
  extends: "Towards the scalable evaluation of cooperativeness in language models (Chan et al. 2023)"
  competes_with: "Large language models as simulated economic agents: What can we learn from Homo silicus? (Horton 2023); Using large language models to simulate multiple humans and replicate human subject studies (Aher et al. 2023)"
  complementary_to: "Chain-of-thought prompting (Wei et al. 2022); Boosting theory-of-mind performance in large language models via prompting (Moghaddam & Honey 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Embodied_Agents/arXiv_2023/2023_Playing_repeated_games_with_Large_Language_Models.pdf
category: Evaluation
---

# Playing repeated games with Large Language Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2305.16867), [Code/Data](https://github.com/eliaka/repeatedgames)
> - **Summary**: 论文把行为博弈论中的有限重复2×2博弈做成文本交互评测床，系统测量LLM在合作、报复与协调上的社会行为，发现其更擅长自利型博弈而不擅长形成协调惯例；SCoT可部分修复这一问题。
> - **Key Performance**: GPT-4在六类博弈上的总体归一化得分最高为0.854；在人机Battle of the Sexes中，SCoT显著提升参与者得分（β=0.74）与协调率（β=0.33）。

> [!info] **Agent Summary**
> - **task_path**: 文本化收益规则+交互历史 -> 当前轮二元动作选择/社会行为诊断
> - **bottleneck**: 模型能识别收益结构和对手模式，但难把“预测到对手行为”转换成“按惯例协调行动”
> - **mechanism_delta**: 将一次性问答替换为有限轮重复博弈评测，并加入“先预测对手下一步、再选择自己动作”的SCoT提示
> - **evidence_signal**: Battle of the Sexes中，SCoT同时提升人机协调率与参与者得分，而囚徒困境中的分数不显著下降
> - **reusable_ops**: [payoff-matrix-to-text, prompt-chained-history, opponent-prediction-before-action]
> - **failure_modes**: [permanent-retaliation-after-single-defection, inability-to-follow-turn-taking-conventions]
> - **open_questions**: [whether-scot-scales-to-larger-or-indefinite-games, whether-training-can-internalize-conventions-without-prompting]

## Part I：问题与挑战

这篇论文真正关心的，不是“LLM会不会解一道博弈题”，而是：**当LLM要和同一个对手连续互动时，它表现出的到底是合作、报复、协调，还是僵化自利？**

### 现有评测缺了什么
以往很多LLM社会能力研究，要么是单轮问答，要么是单次经济学游戏。这样的设置能测到：
- 是否读懂规则；
- 是否会给出看起来合理的答案。

但**测不到**更关键的动态社会能力：
- 遇到一次背叛后会不会原谅；
- 能不能形成轮替、礼让之类的惯例；
- 是否能把“理解对手”真正转化为“调整自己的动作”。

### 这篇论文的输入/输出接口
论文把每个2×2博弈写成文本规则，并把历史轮次拼接进提示中：

- **输入**：收益矩阵的文字描述 + 前几轮双方选择与得分 + 当前轮问题  
- **输出**：当前轮二选一动作

边界条件很明确：
- 双人、双动作；
- 全信息；
- 固定同一对手；
- 有限轮（10轮）；
- 主要是英文文本提示、temperature=0、单token动作输出。

### 真正瓶颈是什么
真正瓶颈不是“算不出哪格收益高”，而是**历史依赖的社会策略形成**：

1. 看到收益结构；
2. 读出对手行为模式；
3. 把这个模式变成自己的下一步动作；
4. 在自利与共同收益之间权衡。

论文的核心发现正好卡在第2步到第3步之间：**GPT-4常常能看懂对手，但不会按看懂的结果去协调。**

### 为什么现在值得解决
因为LLM已经不是一次性回答器，而是在客服、助理、协作代理中反复和人互动。  
只看单轮正确率，无法回答“它是不是个好搭档”。这篇论文的贡献就在于把评测重心从**静态答题**推进到**动态社会行为**。

---

## Part II：方法与洞察

这篇论文本质上做了两件事：

1. **造一个能暴露社会策略的评测仪器**；
2. **用最小提示干预去测试行为是否可塑。**

### 评测协议怎么搭的

#### 1. 大规模重复博弈扫描
作者先让5个LLM在六类2×2博弈家族中两两对战（含自博弈），共10轮重复互动，覆盖：
- Win-win
- Prisoner’s Dilemma family
- Unfair
- Cyclic
- Biased
- Second-best

关键不是只看某一局输赢，而是看**归一化得分**与行为轨迹，从而比较不同模型在不同社会结构下的偏好。

#### 2. 用“简单对手”隔离失败模式
为了不只看总分，作者又在两个经典游戏里加入手工策略：

- **囚徒困境**：总合作、总背叛、首轮背叛后持续合作  
  用来测“是否原谅”
- **Battle of the Sexes**：总选A、总选B、轮替策略  
  用来测“是否会形成轮替惯例”

这一步很关键，因为它能把“模型分低”拆成更具体的行为原因。

#### 3. 用人类实验验证外部效应
作者再让195名参与者与GPT-4或SCoT版GPT-4对战：
- 看协调/合作是否改善；
- 看人类是否更愿意把它当成人类对手。

这使论文不只是离线评测，还带有人机交互外部效度。

### 核心直觉

#### 改了什么
- 从**单轮任务**改为**带历史的重复博弈**
- 从**直接选动作**改为**先预测对手、再选动作（SCoT）**

#### 哪个瓶颈被改变了
- 前者改变的是**测量瓶颈**：把原先测不出的报复、宽恕、轮替、惯例形成显性化。
- 后者改变的是**信息瓶颈**：把模型对对手的隐式判断强制外显，减少“看懂了但没把它接到动作选择上”的断层。

#### 带来了什么能力变化
- 评测层面：作者能区分“不会预测对手”与“会预测但不跟随”。
- 干预层面：SCoT让GPT-4在协调博弈里开始更像一个能配合的搭档，而不只是坚持自己偏好。

#### 为什么这个设计有效
因果上，这不是简单“多问一句话”：

- 在BoS里，基础GPT-4会重复选自己偏好选项；
- 但单独让它预测时，它又能逐步识别出对手轮替模式；
- 说明问题不在感知，而在**决策接口**：预测没有进入行动。

SCoT的作用，就是把“对手下一步会做什么”变成决策前的显式中间变量，让协调策略成为可执行选项，而不只是隐含在模型内部的模式匹配结果。

同理，在PD里加入“对手可能会犯错”的说明，会把一次背叛从“敌意证据”改写为“噪声证据”，从而减弱永久报复倾向。

### 战略权衡

| 设计选择 | 改变了什么瓶颈 | 得到的能力/诊断 | 代价 |
|---|---|---|---|
| 文本化10轮重复博弈 | 把静态理解变成动态策略测试 | 能暴露报复、宽恕、协调惯例 | 游戏空间仍很小，生态较简化 |
| 加入手工简单策略 | 精确定位行为缺陷 | 能分离“不会原谅”与“不会轮替” | 对手不够真实、策略过于规则化 |
| SCoT先预测再行动 | 把隐式对手模型接入动作选择 | 协调率提升，行为更像合作伙伴 | 提示敏感、增加token与推理开销 |
| 全信息且已知终局 | 方便严格控制比较 | 结论更干净、可解释性更强 | 不代表未知终局/长期关系中的策略 |

---

## Part III：证据与局限

### 关键证据

#### 证据1：跨六类博弈的总体比较
最强的总体信号是：**GPT-4在五个被测LLM里总体最好（0.854）**。  
但更重要的是结构性差异：
- 在**Win-win**和**PD family**里表现强；
- 在需要让渡自身偏好、形成协调惯例的博弈里更弱。

这说明现阶段LLM的“强”，很大程度上是建立在**自利、防御、报复型策略与收益结构一致**时。

#### 证据2：囚徒困境里的“永久报复”
当对手只在首轮背叛、之后一直合作时，GPT-4仍持续背叛。  
而且即使明确告诉它“对手只会背叛一次”，它依然基本不回到合作。

这把它在PD类游戏中的高分解释清楚了：**不是它更擅长建立互信，而是它更不宽恕、也更稳定地自利。**

#### 证据3：Battle of the Sexes里的“看懂但不做”
论文最有洞察力的结果在这里：

- 让GPT-4作为玩家时，它很难跟随轮替策略；
- 但让它先做预测时，它又能从第3轮或第5轮开始预测出对手在轮替。

所以缺陷不是“看不出规律”，而是**预测—行动脱节**。  
这比简单说“LLM不会协调”更细，也更有可修复性。

#### 证据4：SCoT在人机互动中有实际收益
在人类实验里，SCoT带来两个重要结果：
- 在**BoS**中提升参与者平均得分与协调率；
- 人类更容易觉得自己对面是“另一个人”。

而在**PD**中，人类平均得分没有显著变化，但联合合作率有所提升。  
这说明SCoT的收益主要体现在**协调性增强**，而不是无差别提高所有任务分数。

### 1-2个关键指标
- **总体博弈表现**：GPT-4总体归一化得分 **0.854**，为五个模型中最高。
- **人机协调收益**：在Battle of the Sexes中，SCoT使参与者平均得分提升 **β=0.74**，协调率提升 **β=0.33**。

### 局限性
- **Fails when**: 需要连续动作、隐藏信息、多人博弈、未知/无限时域、长期声誉积累时，本文协议未必还能准确刻画LLM社会行为；尤其“轮替惯例”是否能扩展到更复杂协作场景仍未知。
- **Assumes**: 10轮、全信息、双人双动作、英文文本提示、temperature=0、单token回答；最强结果依赖GPT-4与Claude等闭源API，且人类实验基于Prolific英文参与者。
- **Not designed for**: 训练新模型、证明LLM拥有真正的Theory of Mind、评估现实谈判/多方组织协作/长期自治代理系统。

### 复现与扩展时要特别注意
虽然代码和数据公开，但论文的关键结论仍部分建立在**闭源模型API**之上；模型版本漂移会影响可重复性。  
此外，这里的“SCoT有效”更像是**推理脚手架有效**，不等于模型已经内化了稳定社会规范。

### 可复用组件
- **payoff-matrix-to-text**：把收益矩阵改写成自然语言规则，适合快速搭建社会行为评测。
- **prompt-chained-history**：将多轮历史显式拼接进提示，测试历史依赖策略。
- **opponent-prediction-before-action**：先预测对手、再行动的两阶段提示，可作为协调型任务的通用模板。
- **robustness grid**：选项重命名、收益单位替换、cover story替换、收益矩阵渐变，适合验证行为模式是否只是提示伪影。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Embodied_Agents/arXiv_2023/2023_Playing_repeated_games_with_Large_Language_Models.pdf]]