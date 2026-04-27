---
title: "TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets"
venue: "ICLR Workshop"
year: 2025
tags:
  - Others
  - task/financial-market-simulation
  - bdi-framework
  - dynamic-social-network
  - llm-agents
  - dataset/Xueqiu
  - dataset/Guba
  - dataset/CSMAR
  - opensource/no
core_operator: "用 BDI 结构化 LLM 投资者代理，并通过时间衰减交易相似图与热度推荐把社交传播耦合进订单驱动市场仿真。"
primary_logic: |
  真实用户画像/交易记录 + 股票/新闻/公告数据 → BDI代理形成并更新信念、主动检索信息、在动态社交网络中传播观点并提交订单 → 订单撮合与社交反馈共同更新市场状态，输出价格序列、成交量与泡沫/恐慌等宏观涌现现象
claims:
  - "TwinMarket 在四类金融 stylized facts 上整体比 ABM-HPM 和 ABM-BH 更接近真实市场统计，例如收益峰度 5.24 优于 4.47/4.99，波动聚集指标 0.89 优于 0.82/0.72 [evidence: comparison]"
  - "去掉 BDI 或去掉代理异质性都会显著削弱市场真实性与拟合度，真实指数相关性分别从 0.77 降到 0.34 和 -0.61 [evidence: ablation]"
  - "在谣言注入实验中，sell/buy 比从 0.495 升至 0.997，并伴随信念下修与价格下跌，说明信息传播可触发系统性 turbulence [evidence: case-study]"
related_work_position:
  extends: "EconAgent (Li et al. 2024)"
  competes_with: "ASFM (Gao et al. 2024); EconAgent (Li et al. 2024)"
  complementary_to: "FinCon (Yu et al. 2024); TrendSim (Zhang et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/World_Models_for_Science/ICLR_Workshop_2025/2025_TwinMarket_A_Scalable_Behavioral_and_Social_Simulation_for_Financial_Markets.pdf"
category: Others
---

# TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.01506), [Project](https://freedomintelligence.github.io/TwinMarket)
> - **Summary**: 论文把 BDI 认知结构、动态社交传播和订单驱动交易系统耦合到 LLM 投资者代理中，用来研究“个体偏差如何经由社交互动放大为市场级泡沫、恐慌与波动”。
> - **Key Performance**: 与真实指数序列的相关性达 0.77、RMSE=0.02；在 4 类金融 stylized facts 上整体优于 ABM-HPM 与 ABM-BH 基线。

> [!info] **Agent Summary**
> - **task_path**: 用户画像+市场/社交流 -> 交易/发帖 -> 价格、成交量与市场涌现
> - **bottleneck**: 静态规则代理无法同时建模认知偏差、局部信息传播和价格反馈闭环
> - **mechanism_delta**: 用 BDI 代理 + 时间衰减交易相似图 + 订单撮合，把个体认知状态直接耦合到社会传播与价格形成
> - **evidence_signal**: 社交交互消融使 Corr 0.77→-0.50，RMSE 0.02→0.18
> - **reusable_ops**: [BDI代理状态更新, 时间衰减社交图与热度推荐]
> - **failure_modes**: [连续撮合或高频市场外推弱, 单一中国A股语境泛化不足]
> - **open_questions**: [开源模型能否复现实验, 真实关注图替换交易相似图会否改变结论]

## Part I：问题与挑战

TwinMarket针对的不是“让代理会选股”这么窄的问题，而是一个更难的系统问题：**怎样让大量异质投资者在真实信息流和社交互动下，稳定地产生像真实市场那样的宏观统计规律与社会涌现现象**。

### 真正的瓶颈
1. **规则式 ABM 太刚性**  
   传统金融 ABM 往往把代理写成少数固定策略，能复现部分统计规律，但很难表达过度自信、从众、情绪波动、信息误读等行为金融里的关键非理性因素。

2. **已有 LLM 经济代理缺少“社会传播—价格形成”闭环**  
   只让代理读新闻做决策，不足以生成真实市场里的放大机制。真实波动常来自：信息先在社交网络里扩散，再触发同步买卖，最后反映到价格和成交量。

3. **微观可解释性与宏观可扩展性难兼得**  
   代理过于自由，行为难分析；代理过于规则化，又失去真实性。本文的核心动机是：保留 LLM 的行为丰富度，同时给代理一个可追踪、可干预的认知结构。

### 输入/输出接口
- **输入**：真实用户画像与交易痕迹、股票/行业数据、新闻、公司公告、社交帖子
- **中间状态**：代理的 belief / desire / intention，以及局部社交图中的可见信息
- **输出**：买/卖/持有、发帖/转发、belief 分数、价格序列、成交量，以及泡沫、衰退、极化、谣言冲击等宏观现象

### 边界条件
这篇论文的有效范围比较明确：
- 场景主要是**中国 A 股**，核心围绕 **SSE 50** 及其 10 个聚合行业指数
- 主验证实验多为 **100 个 GPT-4o 代理**、约 5 个月仿真；扩展实验到 **1000 代理**
- 市场是**简化的订单驱动系统**，作者明确提到当前采用**单日 call auction**与**zero-sum environment**
- 社交关系不是来自真实关注图，而是由**交易行为相似性**近似构造

### 为什么现在值得做
因为 LLM 已经具备三个关键能力：
- 能读新闻、公告和帖子
- 能维持 persona 并表现出一定偏差
- 能在文本交互中形成局部社会影响

而金融市场又有成熟的 stylized facts 作为验证标尺，因此它是检验“LLM 是否真的能支撑社会科学仿真”的理想试验场。

## Part II：方法与洞察

### 方法主线
TwinMarket把系统拆成微观代理和宏观环境两层。

1. **微观层：BDI 驱动的投资者代理**
   - **Belief**：代理结合 persona、市场信息、新闻和社交帖子理解环境；每日结束后再根据反馈自评并更新 belief。
   - **Desire**：代理主动生成查询，检索相关股票和资讯，而不是只被动接收输入。
   - **Intention**：代理把当前信念落实为行动，包括交易决策和社交行为。

2. **宏观层：动态社交网络 + 订单驱动市场**
   - 用**时间衰减的交易强度相似性**构造社交图，近似“谁更可能影响谁”
   - 用**热度+时效**排序给每个代理推荐局部帖子，限制感知野
   - 代理订单进入撮合系统，成交结果更新价格、成交量，再反过来影响下一轮 belief

3. **数据 grounding**
   - Xueqiu：初始化用户画像与偏差
   - Guba：辅助推荐系统
   - CSMAR / Sina / 10jqka / CNINFO：提供股票、新闻与公告流

### 核心直觉
**这篇论文真正改的，不是“让 LLM 去做交易”，而是把市场仿真从静态规则映射改成了一个“局部可见、可更新信念、可传播”的认知系统。**

因果链如下：

- **什么变了**  
  静态、同质、弱交互的代理  
  → 变成 **有 BDI 内隐状态、有异质画像、有局部社交暴露** 的代理群体

- **哪种瓶颈变了**  
  - 决策不再只由当前价格触发，而是由**持续更新的 belief** 驱动  
  - 信息暴露不再近似 iid，而是被**相似人群局部过滤**  
  - 价格不再是外生背景，而是由代理订单**内生形成并反向塑造下一轮信念**

- **能力为何变化**  
  金融涌现现象往往依赖两种耦合：  
  1）**跨时间依赖**：belief 会延续、修正、积累  
  2）**跨个体依赖**：社交传播会同步化、极化、放大  
  TwinMarket恰好把这两个缺失项补上，所以更容易自然地产生 fat tails、volatility clustering、opinion leaders、rumor cascades 和 self-fulfilling prophecy。

### 战略权衡
| 设计选择 | 改变了什么 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| BDI 认知框架 | 从一次性反应改为可更新信念状态 | 更容易生成自我实现、损失厌恶、情绪持续性 | 依赖 prompt 设计与 LLM 稳定性 |
| 异质用户画像 | 从同质代理改为多风格、多偏差代理 | 更接近真实收益分布与财富分化 | 画像估计依赖真实交易样本，存在采样偏差 |
| 动态社交图 + 热度推荐 | 从全局同质信息流改为局部受限传播 | 能出现 opinion leader、极化、谣言扩散 | 图构造阈值和时间衰减会影响传播形态 |
| 订单驱动撮合 | 从“想法模拟”变成“行动影响价格” | 能把微观决策闭环到宏观价格序列 | 当前市场机制仍明显简化 |
| 扩展到 1000 代理 | 从 toy simulation 走向规模化群体互动 | 提升价格拟合和群体统计稳定性 | API/算力成本上升，复现难度增加 |

### 与前作相比的机制增量
- 相比 **EconAgent**：TwinMarket把经济代理从相对孤立的决策单元推进到**带社交传播的市场系统**
- 相比 **ASFM / 传统 ABM**：TwinMarket用 LLM+BDI 替代固定规则，更强调**非理性偏差与局部互动**
- 相比一般 social simulation：它不是只看帖子传播，而是把传播结果真正落实到**交易与价格形成**

## Part III：证据与局限

### 关键证据
- **[comparison] 宏观真实性**  
  论文用 4 类金融 stylized facts 做验证：fat tails、leverage effect、volume-return relationship、volatility clustering。TwinMarket 在这些统计规律上整体都比 ABM-HPM 和 ABM-BH 更接近真实市场。  
  最直观的指标是：**模拟指数与真实指数 Corr=0.77，RMSE=0.02**。

- **[ablation] 关键因果旋钮有效**  
  最有说服力的不是“做出了现象”，而是做了消融：  
  - 去掉 **BDI**：相关性从 0.77 降到 0.34  
  - 去掉 **异质性**：相关性降到 -0.61  
  - 去掉 **社交交互**：RMSE 从 0.02 升到 0.18，相关性从 0.77 降到 -0.50  
  这说明提升不是单靠 LLM 文本能力，而是来自认知状态、异质群体和传播闭环的组合。

- **[case-study] 谣言注入揭示系统级放大机制**  
  当负面 rumor 被定向送给高中心性用户后，belief 开始系统性下修，sell/buy ratio 从 **0.495** 升到 **0.997**，并伴随价格下跌与群体极化。  
  这直接支持了作者的主张：**局部信息不对称 + 社交扩散** 能被放大成市场 turbulence。

- **[scaling] 规模本身也是性能旋钮**  
  在 1000 代理实验里，随着每日活跃交易代理比例提高，成交量上升且 RMSE/MAE 单调下降，说明框架不是只能在小规模玩具环境里工作。

### 局限性
- **Fails when**: 需要连续撮合、高频交易微结构、衍生品杠杆、做市商制度、跨市场传染时，当前结论未必成立；单一中国 A 股设定限制了外推性。
- **Assumes**: 交易相似性可以近似社交联系；有限真实用户样本足以初始化可靠 persona；闭源 LLM 能稳定维持角色与偏差；新闻/公告等外生信息流可持续提供。
- **Not designed for**: 实盘预测、可执行交易策略生成、严格的政策定量评估，或监管级微结构仿真。

### 资源与复现依赖
- 主实验依赖 **GPT-4o**，稳定性实验还用到 **Gemini-1.5-Flash**
- 长时程、多代理仿真意味着 API 成本和模型版本漂移都会影响复现
- 论文提供了 **project page**，但正文里**没有明确代码开源说明**
- 部分真实交易数据具敏感性，即便做了匿名化，也提高了完整复现门槛

### 可复用组件
- **BDI 代理骨架**：适合迁移到其他“认知—行动—反馈”型社会仿真
- **时间衰减行为相似图**：不依赖显式社交关系，也能近似传播路径
- **热度+时效的感知野机制**：适合建模“谁看见什么”
- **订单驱动市场闭环**：为把文本代理意见变成可量化宏观指标提供接口

### 一句话结论
TwinMarket 的核心价值不在于“做了一个金融聊天机器人市场”，而在于它把**认知偏差、局部社交传播、订单反馈**连成了一个可干预的仿真闭环，因此比传统 ABM 更自然地解释了：**为什么个体层面的非理性，会在群体层面长成泡沫、恐慌与波动聚集。**

![[paperPDFs/World_Models_for_Science/ICLR_Workshop_2025/2025_TwinMarket_A_Scalable_Behavioral_and_Social_Simulation_for_Financial_Markets.pdf]]