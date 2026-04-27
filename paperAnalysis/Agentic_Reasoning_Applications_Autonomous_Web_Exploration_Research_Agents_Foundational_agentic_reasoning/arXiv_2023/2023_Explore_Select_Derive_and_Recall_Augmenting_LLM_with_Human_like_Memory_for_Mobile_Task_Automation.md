---
title: "Explore, Select, Derive, and Recall: Augmenting LLM with Human-like Memory for Mobile Task Automation"
venue: arXiv
year: 2023
tags:
  - Others
  - task/mobile-task-automation
  - hierarchical-memory
  - in-context-learning
  - human-in-the-loop
  - dataset/MobileGPT
  - opensource/partial
core_operator: 通过任务-子任务-动作三级 App 记忆，把 LLM 的移动操作从“每次从零推理”改为“检索已学子任务并自适应重放”
primary_logic: |
  用户指令 + 当前 App 界面/离线探索缓存 → Explore 枚举当前页可执行子任务，Select 选择并补全参数，Derive 生成原子 UI 动作并写入层级记忆 → Recall 按高层任务匹配并通过属性匹配/少样本提示自适应重放已学子任务
claims:
  - "在 18 个 App、185 个任务的综合评测上，MobileGPT 首次执行任务的成功率为 82.7%，分别比 AutoDroid 和 AppAgent 高 8.0 与 15.3 个百分点 [evidence: comparison]"
  - "在参数变化的重复任务上，MobileGPT 的 warm-start 成功率达到 98.75%，相对同 prompt 的 derive-only 基线将延迟和 LLM 查询成本分别降低 62.5% 与 68.8% [evidence: ablation]"
  - "其基于子任务的页面分类在 269 个界面上仅产生 3 个假阳性、0 个假阴性，优于已有文本和视觉页面分类方法 [evidence: comparison]"
related_work_position:
  extends: "AutoDroid (Wen et al. 2023)"
  competes_with: "AutoDroid (Wen et al. 2023); AppAgent (Yang et al. 2023)"
  complementary_to: "SUGILITE (Li et al. 2017); Pix2Struct (Lee et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2023/2023_Explore_Select_Derive_and_Recall_Augmenting_LLM_with_Human_like_Memory_for_Mobile_Task_Automation.pdf
category: Others
---

# Explore, Select, Derive, and Recall: Augmenting LLM with Human-like Memory for Mobile Task Automation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2312.03003), [Project](https://mobile-gpt.github.io/)
> - **Summary**: 这篇论文提出 MobileGPT，把移动任务自动化拆成“探索-选择-推导-回忆”四步，并把成功执行压缩成可参数化的子任务记忆，从而让 LLM 不必每次都从零操作手机界面。
> - **Key Performance**: 18 个 App、185 个任务上首次执行成功率 82.7%；参数变化的重复任务上 warm-start 成功率 98.75%，相对基线延迟/成本下降 62.5%/68.8%

> [!info] **Agent Summary**
> - **task_path**: 自然语言/语音任务指令 + 当前 Android 界面 + App 记忆 -> 参数化子任务序列 -> 原子 UI 动作执行/任务完成
> - **bottleneck**: 现有 LLM 手机代理每次都在原子动作层从零推理，缺少可复用的中层记忆单元，导致重复任务不稳定、昂贵且难以纠错
> - **mechanism_delta**: 用 Explore-Select-Derive 把动作搜索先收缩为“当前页可执行子任务”，再把成功执行写成任务-子任务-动作三级记忆，Recall 时通过属性匹配和少样本提示自适应重放
> - **evidence_signal**: 跨 18 个 App 的对比实验 + 同 prompt 消融，显示首次执行更准，重复执行接近确定性且显著更省时省钱
> - **reusable_ops**: [界面HTML化压缩表示, 任务-子任务-动作层级记忆]
> - **failure_modes**: [无可靠文本可访问树的App难以解析, 重复或变化过大的UI属性会让动作适配失效并需LLM/用户修复]
> - **open_questions**: [如何扩展到跨App长程任务规划, 如何在隐私敏感场景下做端侧低延迟部署]

## Part I：问题与挑战

这篇论文的系统名是 **MobileGPT**。它想解决的不是“让 LLM 会点按钮”这么简单，而是更实际的问题：**怎样让手机任务自动化既能第一次做对，又能第二次更稳、更快、更便宜地重做类似任务**。

### 1. 真正的问题是什么
现有 LLM 手机代理（如 AutoDroid、AppAgent）通常直接从：
- 用户指令
- 当前屏幕
- 原子动作集合（click / input / scroll ...）

一步步推出下一步操作。

这种方式的问题不在于 LLM 完全看不懂界面，而在于它**每次都在最底层动作空间里重新搜索**。结果是：

1. **可靠性差**：同一任务今天做对，明天可能做错。
2. **重复劳动**：相似任务共享大量公共步骤，但系统不会复用。
3. **成本高**：每一步都要查询大模型，延迟和费用都上升。
4. **纠错难沉淀**：即使用户修好了一个错误，系统也不一定能把这次修复转化为以后可复用的知识。

### 2. 真正的瓶颈在哪里
论文抓得很准：**瓶颈不是缺少更强的原子动作生成器，而是缺少位于“用户意图”和“原子 UI 动作”之间的中层记忆表示**。

作者认为，人类做手机任务时不会记住每次点击的绝对位置，而是会记住：
- 这个页面能做哪些“子任务”
- 当前该选哪个子任务
- 这个子任务通常怎么完成
- 下次遇到相似目标时如何改参数并复用

所以核心瓶颈是：**如何把 App 交互经验组织成可复用、可参数化、可适配界面变化的“子任务记忆”**。

### 3. 为什么现在值得做
因为 LLM 已经具备了两个关键前提：
- 能从文本化 UI 表示中理解页面功能；
- 能基于上下文做参数填充、动作推理和少样本适配。

一旦这些能力成立，移动任务里高度重复的结构就可以被“记忆化”，于是冷启动学习的成本能被后续大量 warm-start 摊薄。这正是现实手机使用场景里最有价值的点。

### 4. 输入/输出接口与边界
**输入**：
- 用户自然语言/语音指令
- 当前 Android 界面（通过 Accessibility 抽取）
- 已构建的 App memory

**输出**：
- 子任务选择与参数
- 原子 UI 动作序列
- 最终任务完成/回答当前界面内容

**边界条件**：
- 主要面向 **Android 单 App 任务**
- 依赖可访问性树里的文本和层次结构
- 初次学习仍可能失败，需要自纠或人工修复
- 对支付、发送消息等不可逆操作，需要更谨慎的人机确认

---

## Part II：方法与洞察

### 1. 系统主线：Explore → Select → Derive → Recall

#### Explore：先理解“这个页面能做什么”
系统先离线探索 App 页面：
- 用随机探索器和用户轨迹监控访问尽可能多的页面；
- 将页面转换成简化 HTML 表示；
- 让 LLM 为当前页列出可执行子任务，格式是 function-call 风格：
  - 子任务名
  - 描述
  - 参数
  - 对应关键 UI 元素

这一步的本质是：**把页面从“像什么”改写成“能做什么”**。

#### Select：在当前页可执行子任务中选一个最合适的
给定：
- 用户指令
- 当前页面表示
- 当前页可用子任务列表

LLM 选择下一步应执行的子任务，并补全参数。  
如果参数不完整，系统会追问用户。  
如果该子任务已经学过，就直接走记忆，不再从零推导动作。

#### Derive：只在选中的子任务内部推导原子动作
不是让 LLM 面对整个任务直接输出 click/input，而是先限定目标子任务，再在这个局部目标下生成原子动作。  
这样动作搜索空间被明显缩小。

#### Recall：把“做过一次”变成“以后可重放”
任务完成后，系统会把：
- 高层任务
- 子任务序列
- 每个子任务对应的动作模板

写入记忆。下次遇到相同或相近任务，系统优先走 Recall，只做：
- 任务匹配
- 参数填充
- 动作适配

而不是重新经历完整的 Explore-Select-Derive。

### 2. 关键实现件

#### 2.1 界面表示：Accessibility → 简化 HTML
论文先把 Android layout 转成简化 HTML：
- 保留层次结构
- 保留文本/描述/可交互属性
- 删除无语义或无交互价值的节点
- 给每个元素分配唯一 index

这样做的作用有两个：
1. 比纯截图更容易让 LLM理解文本密集型页面；
2. 比原始 layout 更短，平均减少 84.6% token。

#### 2.2 三级层级记忆：任务 / 子任务 / 动作
这是论文最核心的结构。

- **Node（页面节点）**：不是按外观，而是按“子任务集合”定义页面
- **Edge（子任务边）**：表示某个子任务如何由动作序列完成
- **Task（任务）**：表示在哪些页面执行哪些子任务

这意味着记忆单位不是“整任务录像”，而是**可跨任务共享的子任务模块**。

#### 2.3 按功能而非外观进行页面分类
页面匹配不是看截图是否像，而是看：
- 当前屏幕是否具备执行某些子任务所需的关键 UI 元素
- 必要时再用子任务相似度做二次校验

这使得“长得不完全一样但功能相同”的页面能共享记忆。

#### 2.4 动作适配：属性模板化 + 少样本兜底
动作在存储时不会死记 `ui_index=5`，而是泛化为：
- 元素属性（id/text/description）
- 参数占位符（如 `[contact_name]`）

回忆时再把参数填回去，并在当前屏幕中寻找匹配 UI。

如果属性匹配失败：
- UI 属性缺失
- 多个 UI 属性相同
- App 更新导致属性变化

则调用 LLM 做 **in-context adaptation**，把过去正确的例子作为 few-shot 样例提示，让 LLM“按以前正确做法重推一次”。

#### 2.5 双重纠错：自纠 + 人在回路
系统承认 LLM 会错，所以提供两层修复机制：

1. **Self-correction**
   - 无效 UI index
   - 不可点击元素
   - 卡在循环滚动/无页面变化  
   系统会把这些错误转成下一轮 prompt 的反馈。

2. **HITL repair**
   用户可以在 Explore / Select / Derive 三层直接修：
   - 补/删子任务
   - 改选错的子任务或参数
   - 直接演示正确动作序列

关键点在于：**修复不会只停留在这一次，而会写回记忆**。

### 核心直觉

这篇论文真正引入的“因果旋钮”是：

> **把 LLM 的决策对象，从“直接在全局原子动作空间里找下一步”，改成“先在当前页功能空间里找子任务，再在子任务内部生成或重放动作模板”。**

这带来三层变化：

1. **搜索分布变了**  
   原来每一步都在 click/input/scroll 的大动作空间里盲搜；  
   现在先收缩到“当前页哪些功能可用”，动作推理只发生在被选中的子任务内。

2. **信息瓶颈变了**  
   原来历史成功执行几乎无法复用；  
   现在历史被压缩成“参数化子任务模板 + 功能页面图”，相似任务能共享记忆。

3. **能力边界变了**  
   原来系统最多是一次性 agent；  
   现在它更像一个会“学 App”的代理：第一次学，第二次回忆，第三次迁移到相似任务。

换句话说，作者不是单纯让 LLM 更聪明，而是**给 LLM 加了一个更符合人类任务学习方式的中间表示层**。这层表示同时降低了首次执行时的推理难度，也把后续重复任务的成本降下来了。

### 设计取舍

| 设计选择 | 直接收益 | 代价/风险 |
|---|---|---|
| 简化 HTML 而非纯截图 | 保留层次与文本语义，冷启动更准 | 依赖可访问性树，遇到图像主导界面受限 |
| 离线 Explore 预采页面功能 | 在线执行时少问 LLM，记忆起点更好 | 需要一次性探索时间与费用 |
| 任务-子任务-动作三级记忆 | 可跨任务复用，warm-start 大幅加速 | 需要维护页面分类与记忆一致性 |
| 属性模板化动作 | 能适配参数变化与页面内容变化 | UI 属性冲突或更新会破坏匹配 |
| few-shot 适配兜底 | 弥补规则匹配失败，减少重复错误 | 仍依赖闭源 LLM 的稳定性 |
| HITL 修复写回记忆 | 把偶发错误转化成长期收益 | 用户需要介入，不可逆操作难恢复 |

---

## Part III：证据与局限

### 1. 关键证据信号

#### 信号 A：跨系统对比说明它不只“记得住”，也“第一次更容易做对”
在 18 个 App、185 个任务的综合评测中，MobileGPT 首次执行成功率 **82.7%**，高于：
- AutoDroid：74.7%
- AppAgent：67.4%

这说明它的收益并不只来自回忆阶段；即便是冷启动，**更完整的界面表示 + 子任务分解** 也已经提升了决策质量。

#### 信号 B：真正的能力跳跃发生在 warm-start
最关键结果不是冷启动分数，而是“学过一次以后还能否稳定迁移到参数变化的同类任务”。

在自建 MobileGPT 数据集上：
- warm-start 成功率：**98.75%**
- 相比同 prompt 的 derive-only 基线：
  - 延迟下降 **62.5%**
  - 成本下降 **68.8%**

这直接支持论文主张：**层级记忆不是锦上添花，而是把 LLM 手机代理从一次性推理器，变成可复用的任务执行器。**

#### 信号 C：回忆不是死记硬背，而是带适配能力的重放
论文还给了两个支撑点：
- 页面分类：269 个页面上仅 **3 个假阳性、0 个假阴性**
- in-context action adaptation：当属性匹配失败时，53 个需要 few-shot 重推的动作全部成功适配

这说明 Recall 阶段不是简单 replay，而是“**参数化模板 + 当前屏幕约束**”下的动态重放。

#### 信号 D：HITL 修复的价值在于把错误变成资产
23 人用户研究显示：
- 除 1 人外，参与者都更偏好 MobileGPT
- repair 机制 SUS 为 **64.6**，属于“可接受但仍有改进空间”
- 有修复机制时，用户对自动化系统的可接受准确率门槛从 **96% 降到 84%**

这意味着修复机制虽然不完美，但显著改善了真实使用可接受性。

### 2. 论文最有价值的结论
这篇论文最重要的结论可以概括为一句话：

> 在移动 UI 自动化里，真正值得缓存的不是 LLM 的最终回答，而是“按功能组织、可参数化复用的子任务经验”。

这比单纯缓存 prompt-response 更强，因为它：
- 对参数变化有适应性
- 能跨任务共享局部能力
- 能把人类修复变成长期记忆

### 3. 局限性

- Fails when: App 没有可靠的文本可访问树（如部分 Flutter / Web App / 地图 / 相机等图像主导界面）；界面上存在多个属性几乎相同的控件且上下文不足；执行发送消息、删除联系人等不可逆动作后，一旦出错很难自动回滚。
- Assumes: 依赖 Android Accessibility Service 提取层次化 UI；依赖 GPT-4/GPT-3.5 在线 API 与外置 Python 服务器；需要离线探索或用户使用轨迹来预热记忆；高质量 warm-start 很大程度上默认允许系统先被人类修好再写入记忆。
- Not designed for: 原生跨 App 长流程任务规划；完全端侧、隐私封闭式部署；无需用户确认的支付/授权等安全关键任务。

### 4. 资源与复现边界
需要特别指出的复现假设有三点：

1. **闭源模型依赖明显**  
   论文实现基于 GPT-4-turbo / GPT-3.5-turbo，性能和成本都受 API 行为影响。

2. **需要一次性预探索成本**  
   离线随机探索每个 App 约 10–15 分钟，总计成本约 \$10.78，能覆盖评测中 89.65% 所需页面。  
   这不算高，但意味着系统不是零准备成本。

3. **端上负担低，但前提是重计算在服务端**  
   端上资源开销不大，但这是建立在大量推理外包给服务器和在线模型之上的。

### 5. 可复用组件
这篇论文里最值得迁移到其他 agent 系统的部件有：

- **Accessibility → 简化 HTML 表示**：保留结构、压缩 token
- **功能页面图**：按“能做什么”而非“长什么样”组织状态
- **任务-子任务-动作三级记忆**：适合重复任务场景
- **属性模板化动作适配**：把 UI index 变成语义属性 + 参数占位符
- **规则自纠 + HITL 修复闭环**：把失败转化成未来成功样本

---

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2023/2023_Explore_Select_Derive_and_Recall_Augmenting_LLM_with_Human_like_Memory_for_Mobile_Task_Automation.pdf]]