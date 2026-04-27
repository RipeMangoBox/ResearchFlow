---
title: "BEARCUBS: A benchmark for computer-using web agents"
venue: COLM
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - live-web-evaluation
  - trajectory-validation
  - workaround-filtering
  - dataset/BEARCUBS
  - opensource/full
core_operator: 以实时网页短答题、人类验证轨迹和抗文本绕过筛题，测量计算机使用型 Web Agent 的真实找信息能力
primary_logic: |
  评测真实网页信息检索能力 → 构造必须访问 live web 且尽量不能被文本捷径绕过的短答案题目，并附人类验证轨迹 → 以人工执行和统一答案判定比较人类与多类代理 → 揭示多模态交互、来源可信度、规划与执行效率的能力边界
claims:
  - "BEARCUBS 上人类准确率为 84.7%，显著高于最佳计算机使用代理 ChatGPT Agent 的 65.8%，表明当前前沿 web agent 在真实网页找信息任务上仍未达到人类水平 [evidence: comparison]"
  - "简单搜索增强与闭卷 LLM 几乎无法解答 BEARCUBS（最佳零样本基线 DeepSeek R1 为 8.1%，搜索增强模型最高仅 5.4%），说明题目基本不被参数记忆或搜索摘要覆盖 [evidence: comparison]"
  - "多模态子集显著放大代理能力差距：ChatGPT Agent 为 54.5%，而 Operator 为 12.7%，Anthropic Computer Use 与 Proxy 均为 9.1%，暴露了精细控制和复杂交互瓶颈 [evidence: analysis]"
related_work_position:
  extends: "AssistantBench (Yoran et al. 2024)"
  competes_with: "AssistantBench (Yoran et al. 2024); WebArena (Zhou et al. 2024)"
  complementary_to: "OSWorld (Xie et al. 2024); WebSuite (Li & Waldo, 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_BEARCUBS_A_benchmark_for_computer_using_web_agents.pdf"
category: Survey_Benchmark
---

# BEARCUBS: A benchmark for computer-using web agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.07919), [Project / Leaderboard](https://bear-cubs.github.io/)
> - **Summary**: 该工作提出一个面向实时网页环境的 111 题基准，用“难以被文本捷径绕过”的短答案信息检索任务，真实评测 computer-using web agent 的搜索、浏览、点击与多模态交互能力。
> - **Key Performance**: 人类 84.7% vs ChatGPT Agent 65.8%；多模态子集 ChatGPT Agent 54.5% vs Operator 12.7%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言信息检索问题 / 实时网页与多模态交互 -> 唯一短答案 + 轨迹级能力诊断
> - **bottleneck**: 现有 web-agent 评测多依赖模拟环境或可被文本检索绕过，测不到真实 live web 上的交互、找源与执行能力
> - **mechanism_delta**: 将评测分布切到 live web，并用人类验证轨迹、Google-adversarial 题目设计和 Deep Research 事后过滤来压缩文本捷径空间
> - **evidence_signal**: 111 题 benchmark 上，人类 84.7% 明显高于最强 agent 65.8%，且多模态题将模型差距进一步放大
> - **reusable_ops**: [人类验证轨迹, 文本绕过过滤]
> - **failure_modes**: [精细鼠标/键盘控制不稳, 复杂数据过滤与长轨迹规划易陷入循环]
> - **open_questions**: [如何把来源可信度纳入标准评分, 如何长期维护 live-web 基准避免污染]

## Part I：问题与挑战

**What / Why：真正的瓶颈不是“会不会回答”，而是“评测是否真的迫使 agent 去做真实网页操作”。**

BEARCUBS瞄准的是一类新系统：能看屏幕像素、操控虚拟键盘和鼠标的 computer-using web agents。问题在于，已有评测常有三类失真：

1. **环境失真**：很多基准运行在模拟/合成网页中，不能反映真实网页的动态性、脆弱性和不可预期性。  
2. **能力失真**：不少任务最终仍可被 HTML、搜索摘要或文本推理“绕过”，并没有真正测到视频、3D、游戏、复杂 UI 操作。  
3. **难度失真**：部分旧基准已接近饱和，难以继续拉开前沿系统差距。

所以，这篇论文解决的核心不是再造一个更大的数据集，而是**修正评测分布**：把测试从“可控网页上的代理行为”转成“真实网络上的信息查找与交互”。

### 任务接口与边界

- **输入**：一条自然语言信息检索问题。
- **代理行为**：在 live web 上搜索、浏览、点击、观看、导航、必要时进行多模态交互。
- **输出**：一个**短、唯一、易判分**的事实答案。
- **边界条件**：
  - 题目必须能在公开网页上完成；
  - 不依赖付费墙、登录、人工协助；
  - 主要评测**信息查找**，不是开放式写作或长期事务执行；
  - 多数题目指定或隐含可靠来源，强调“找到正确来源”而不只是“猜对答案”。

### 这件事为什么是“现在”必须做

因为 web agent 已从“只读网页”走向“可实际用电脑”，如果评测还停留在静态 HTML 或模拟网页，就无法回答最关键的问题：  
**这些系统到底能不能在现实网页中稳定完成有摩擦、有噪声、有多模态依赖的任务？**

## Part II：方法与洞察

**How：作者引入的关键因果旋钮，是“让题目必须经过真实网页交互才能解开”。**

### 评测设计框架

BEARCUBS 的构造思路很明确：

- 共 **111** 道题，约一半文本型、另一半多模态型；
- 覆盖 **108 个顶级 URL**，避免系统只记住少数网站套路；
- 每题都带有：
  - gold answer
  - 人类验证过的可行浏览轨迹
  - 访问过的网站列表

题目筛选遵循四条硬约束：

1. 问题信息足够但不过量；
2. 答案必须短、唯一、好判定；
3. 对 Google Search 具有对抗性；
4. 答案必须公开可访问。

尤其关键的是第 3 点：  
作者不只要求“搜索摘要里找不到”，还专门用 **Deep Research 做事后过滤**，删除那些虽然看起来是多模态题、但实际上可以被纯文本路径绕过的问题。

### 核心直觉

过去很多 benchmark 的隐性漏洞是：**你以为你在测交互，实际上你在测搜索或参数记忆。**

BEARCUBS 改变了三件事：

1. **分布变了**：从模拟网页转向 live web，加入现实世界的页面变动、访问摩擦、UI 细节和页面不稳定性。  
2. **约束变了**：通过抗搜索设计和 workaround 过滤，减少“只靠文本就能答对”的捷径。  
3. **观测信号变了**：不只看最终是否答对，还保留人类验证轨迹，能分析 agent 到底卡在找源、交互、还是规划。

因此它获得的不是单纯“难一点”的测试，而是**更接近真实失败机理的诊断能力**：  
- 是没找到正确页面？  
- 找到了但不会点/拖/筛选？  
- 还是明明看到来源，却转而引用了不可靠二手信息？

### 为什么这个设计有效

因果上，BEARCUBS 之所以能测出真实能力，是因为它把成功路径压缩成了：

**正确找源 + 正确交互 + 正确读取 + 正确作答**

只要其中任何一个环节缺失，题目就更难被“语言模型猜测”掩盖。  
同时，**短答案**降低了判分噪声，**人类验证轨迹**保证了任务是可解的，而不是“作者自己也未必做得出来”的伪难题。

### 战略取舍

| 设计选择 | 带来的能力诊断 | 代价 / 风险 |
|---|---|---|
| live web 而非模拟网页 | 更真实地测到网页脆弱性与交互摩擦 | 页面会变，基准会老化、污染 |
| 短且唯一的答案 | 判分简单清晰，减少评测歧义 | 任务表达能力有限，不覆盖开放式答案 |
| 人类验证轨迹 | 能定位 agent 卡在何处 | 标注和维护成本高 |
| 过滤文本 workarounds | 真正测到多模态/交互能力 | 题目构造难，数据规模难做大 |
| 人工运行商业 agent | 能覆盖无 API 的闭源系统 | 评测慢，复现成本高，受时间上限影响 |

## Part III：证据与局限

**So what：这个 benchmark 真正把前沿 web agent 的能力差距“测出来了”，而且测出来的不是纸面差距，而是失败模式差距。**

### 关键证据

1. **可解，但远未被攻克**  
   人类准确率 **84.7%**，说明题目并非不可做；最佳 agent ChatGPT Agent 为 **65.8%**，说明真实网页找信息仍有明显人机差距。

2. **它确实不是“搜一下就行”的 benchmark**  
   闭卷 LLM 和简单搜索增强几乎全军覆没，说明题目没有被参数记忆或 snippet 检索轻松覆盖。  
   这点对 benchmark 很重要：如果简单搜索就能做，computer use 的评测意义就会塌掉。

3. **多模态交互是当前最大能力鸿沟**  
   人类对不少多模态题反而觉得更直观，但 agent 普遍更差。  
   最强 ChatGPT Agent 在多模态子集也只有 **54.5%**，其余 computer-use agents 多在 10% 左右，说明瓶颈主要在：
   - 精细控制
   - 复杂 UI 操作
   - 长轨迹规划
   - 对多模态信息的主动利用

4. **“答对”不等于“找对来源”**  
   Deep Research 虽排第二，但不少正确答案依赖次级来源或未充分落地来源。  
   这说明未来评测不能只看 correctness，还要看 **source credibility**。

### 论文给出的最强诊断结论

一句话总结：  
**当前前沿 web agent 已经比旧系统强很多，但离“可靠地在真实网页上替人找信息”还有明显差距。**

最直接的跃迁信号不是“比基线高几分”，而是：
- ChatGPT Agent 相比 Operator 有明显跨代提升；
- 但一旦任务要求精细交互、复杂筛选或长时间稳定执行，系统仍会变慢、卡住、循环、放弃，或转向不可靠来源。

### 局限性

- **Fails when**:  
  - 任务没有唯一短答案，或需要开放式长回答时；  
  - 目标网页频繁变化、内容下线、答案被公开泄露后；  
  - 需要登录、重度 CAPTCHA、付费墙或长期账户状态时。

- **Assumes**:  
  - 需要持续的数据维护来替换失效/污染题目；  
  - 依赖人类编写并验证轨迹，人工成本高；  
  - 评测很多时候依赖闭源商业系统，且部分系统无 API，只能手工运行；  
  - 时间上限会影响结果（论文中大多数 agent 有 15 分钟上限，ChatGPT Agent 例外到 45 分钟）。

- **Not designed for**:  
  - 系统化多语言/跨文化能力比较；  
  - 安全性、可信对齐或人机协作评测；  
  - 需要长期计划、事务执行或开放式任务产出的通用电脑使用评测。

### 可复用组件

- **抗文本绕过的数据构造流程**：先设计，再用强文本代理做事后过滤。  
- **短答案 live-web QA 格式**：很适合做低歧义 leaderboard。  
- **人类验证轨迹**：适合做 agent failure analysis。  
- **自动答案评估器**：论文给出自动 evaluator，二分类准确率接近 98.7%，可复用于类似 benchmark。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_BEARCUBS_A_benchmark_for_computer_using_web_agents.pdf]]