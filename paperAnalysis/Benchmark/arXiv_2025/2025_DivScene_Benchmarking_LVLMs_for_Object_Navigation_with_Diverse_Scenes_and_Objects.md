---
title: "DivScene: Benchmarking LVLMs for Object Navigation with Diverse Scenes and Objects"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/object-navigation
  - task/MLLM-evaluation
  - chain-of-thought
  - imitation-learning
  - bfs-planning
  - dataset/DIVSCENE
  - dataset/DIVSCENEep
  - opensource/full
core_operator: "用 GPT-4 + Holodeck 扩展开放词汇场景分布，并以 BFS 最短路径与 CoT 动作模板构建对象导航评测集和强基线。"
primary_logic: |
  开放词汇导航评测需求 → 用 GPT-4 + Holodeck 生成 81 类多样场景并以 BFS 采样最短路径回合 → 用 SR/SPL/SEL 统一评测 LVLM，同时用带 CoT 的行为克隆训练 NATVLM → 揭示现有模型能力边界并给出低成本提升路径
claims:
  - "DIVSCENE/DIVSCENEep 将开放词汇对象导航扩展到 4,614 个房屋、81 种场景类型和 5,707 类目标对象，覆盖度显著超过 ProcTHOR、iTHOR 与 HM3D-OVON 一类现有基准 [evidence: analysis]"
  - "在 DIVSCENEep 上，现成 LVLM 对开放词汇对象导航仍明显不足：整体最强闭源基线 GPT-4o 的成功率为 37.66%，多数开源模型不超过 20.99% [evidence: comparison]"
  - "基于 BFS 最短路径对 Idefics 2 做带 CoT 的行为克隆可将整体成功率提升到 56.17%，而去掉解释轨迹后仅为 28.09%，说明解释式监督是关键增益来源 [evidence: ablation]"
related_work_position:
  extends: "HM3D-OVON (Yokoyama et al. 2024)"
  competes_with: "ProcTHOR (Deitke et al. 2022); HM3D-OVON (Yokoyama et al. 2024)"
  complementary_to: "VLFM (Yokoyama et al. 2024a); SG-Nav (Yin et al. 2024)"
evidence_strength: strong
pdf_ref: "paperPDFs/Benchmark/arXiv_2025/2025_DivScene_Benchmarking_LVLMs_for_Object_Navigation_with_Diverse_Scenes_and_Objects.pdf"
category: Survey_Benchmark
---

# DivScene: Benchmarking LVLMs for Object Navigation with Diverse Scenes and Objects

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.02730) · [Code](https://github.com/zhaowei-wang-nlp/DivScene)
> - **Summary**: 论文构建了一个大规模开放词汇对象导航基准 DIVSCENE，并证明只用 BFS 最短路径生成的低成本示范、配合 CoT 动作解释，就能把 LVLM 的导航能力显著拉高。
> - **Key Performance**: NATVLM 在 DIVSCENEep 上整体 **SR=56.17%**，高于 GPT-4o 的 **37.66%**；零样本迁移到 iTHOR / ProcTHOR 时 **SR=72.79% / 53.12%**。

> [!info] **Agent Summary**
> - **task_path**: 目标对象类别 + 第一视角 RGB / 位姿 / 短历史 -> 离散下一步导航动作
> - **bottleneck**: 现有 ObjectNav 基准的场景和对象词汇太窄，导致 LVLM 的开放词汇导航能力既测不准也学不出来
> - **mechanism_delta**: 把“导航”转成大规模多样场景上的“带 CoT 解释的下一步动作模仿学习”，用 BFS 最短路径替代昂贵人工示范
> - **evidence_signal**: NATVLM 整体 SR 56.17%，相对 GPT-4o 提升 18.51 个点；去掉 explanation traces 后降到 28.09%
> - **reusable_ops**: [LLM生成场景描述并调用Holodeck建屋, BFS最短路径蒸馏为动作监督]
> - **failure_modes**: [长时程探索时在局部区域徘徊, 受限上下文下难以维持长期记忆与全局搜索]
> - **open_questions**: [如何把短视野模仿策略升级为主动探索策略, 合成场景上的收益能否稳定迁移到真实机器人]

## Part I：问题与挑战

这篇论文真正要解决的，不只是“让 LVLM 会走路”，而是更具体的：

**如何在开放词汇、长尾对象、复杂多场景条件下，真实评估并提升 LVLM 的对象导航能力。**

### 1. 旧基准为什么不够用
作者指出，现有对象导航基准大多有两个结构性问题：

1. **场景分布太窄**  
   例如只覆盖少量家庭场景、少量房间类型。

2. **目标词汇太小**  
   常见设置只含十几到几十类目标物体，离“open-vocabulary”很远。

这会带来一个直接后果：  
模型即使在旧基准上表现尚可，也不代表它真的具备了**开放词汇对象理解 + 空间决策 + 导航执行**能力。

### 2. 真正瓶颈是什么
我认为这篇文章抓到的核心瓶颈是：

> **瓶颈不只在模型，而在训练/评测分布本身太弱。**

如果场景只是在少数家庭房间里找少数常见物体，那么模型更多是在利用：
- 房间共现先验
- 简单目标词匹配
- 短程局部搜索

而不是在学习真正的开放词汇导航。

### 3. 任务接口与边界条件
这篇文章把任务定义得很清楚：

- **输入**：
  - 目标对象类别名
  - 当前第一视角 RGB 图像
  - agent 的位置与朝向
  - 最近若干步历史状态
- **输出**：
  - 下一步离散动作：`MoveAhead / RotateLeft / RotateRight / Done`

环境边界条件：

- 平台：**AI2THOR**
- 步长：**0.25m**
- 旋转：**90°**
- 成功判定：目标出现在视野中，且距离 **< 1.5m**
- 最长轨迹：**200 步**

### 4. 为什么是现在做
现在做这件事成立，有两个现实原因：

- **LVLM 已经足够强**，值得认真测试其 embodied 能力，而不只是 VQA/文档理解。
- **Holodeck + Objaverse + GPT-4** 让大规模自动生成多样 3D 场景变得可行，终于能构造更像“开放世界”的评测分布。

---

## Part II：方法与洞察

这篇论文实际上有两层贡献：

1. **DIVSCENE / DIVSCENEep：一个新的开放词汇对象导航基准**
2. **NATVLM：一个用该基准训练出来的强基线**

### 1. 基准怎么构建
作者先从 MIT Scenes 补充整理出 **81 种 scene type**，分属 5 大类。  
然后流程是：

- 用 GPT-4 给场景类型补充属性  
  例如 lighting、room style、objects、users 等
- 生成自然语言 house description
- 把 description 输入 **Holodeck**
- 在 AI2THOR 上自动生成房屋

最终得到：

- **4,614 houses**
- **81 scene types**
- 全部房屋中 **22,696 object types**
- 采样出 **22,962 episodes**
- episode 中包含 **5,707 类目标对象**

这一步改变的是**数据分布覆盖面**。

### 2. episode 怎么采样
作者把房屋离散成网格图后：

- 随机采样初始位置
- 随机采样目标对象
- 用 **BFS** 搜索最短路径
- 再把路径转成动作序列
- 执行动作并记录每一步观测

一个很实用的小设计是：  
BFS 在路径搜索时对“旋转变化”加入额外代价，使得到的 expert path 更容易被语言模型模仿，因为它不只是最短，还更“动作友好”。

### 3. NATVLM 怎么训练
NATVLM 基于 **Idefics 2 (8B)** 微调，但关键不是单纯输出动作，而是输出：

> **带 CoT 解释的下一步动作**

输入里除了当前图像和状态，还包括：

- 最近 **8 步**的位置与动作
- 最近 **4 张**视觉历史图像

输出响应模板分三步：

1. 比较当前位置与目标/推荐点的相对关系
2. 看图判断前方是否有障碍、是否需要转向
3. 给出最终动作

而且这些 explanation traces 不是 LLM 自动生成，而是**人工设计模板 + 坐标填充**。  
这使监督更稳定，也更贴近“空间决策规则”。

### 核心直觉

这篇论文最重要的因果改动可以概括为：

> **把“稀疏成功/失败的导航问题”，改写成“在更宽分布上、每一步都可监督的空间决策问题”。**

具体是三层变化：

1. **场景/对象分布变宽了**  
   从封闭词表、小量家庭场景，变成 81 类场景和数千目标对象。  
   → 改变了评测瓶颈  
   → 终于能测出 LVLM 在开放词汇导航上的真实短板。

2. **监督从稀疏回报变成最短路动作示范**  
   用 BFS 最短路径做 imitation learning。  
   → 改变了学习瓶颈  
   → 不再依赖昂贵人工演示，也避免 RL 奖励设计复杂。

3. **动作标签变成带解释的动作标签**  
   输出不只是 “RotateLeft”，而是“为什么此时该转”。  
   → 改变了信息瓶颈  
   → 模型更容易把位置关系、障碍判断、终止条件绑定到动作上。

换句话说，作者并不是让 LVLM 学“全局规划器”，而是先让它学会一个更可靠的**局部决策器**。

### 为什么这个设计有效
因为对象导航里，很多失败不是“看不见目标”，而是：

- 不知道**此刻**该不该前进
- 不知道前方障碍是否要求转向
- 不知道什么时候该 `Done`

CoT 解释把这些隐含判断显式化，模型就不容易只学 prompt 表面形式。

### 战略取舍

| 设计选择 | 改变了什么 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| GPT-4 + Holodeck 生成 81 类场景 | 扩大评测分布 | 更接近开放词汇导航 | 合成场景真实性受生成质量约束 |
| BFS 最短路径替代人工示范 | 提供廉价、确定性的 expert data | 可大规模行为克隆 | 主要学到“最短路模仿”，不是主动探索 |
| CoT 动作模板监督 | 显式注入空间因果 | 动作预测显著提升 | 推理更慢，且位置差计算会出错 |
| 只保留 8 步/4 图像历史 | 控制上下文长度和算力 | 短程导航足够有效 | 长时程记忆不足 |

---

## Part III：证据与局限

### 1. 关键证据：这个基准确实测出了能力缺口
最强的诊断信号不是 NATVLM 本身，而是：

- **多数 blind LLM 几乎接近随机**
- 加 caption 的 LLM 也只小幅提升
- 说明“把图像压成文字描述”不足以支撑导航决策
- 即使是 **GPT-4o**，在 DIVSCENEep 上整体也只有 **37.66% SR**

这说明：  
**现成 LVLM 的开放词汇对象导航能力并没有随着通用多模态能力自然涌现出来。**

### 2. 关键证据：BFS + CoT 不是装饰，而是主要增益来源
最有说服力的实验是消融：

- **NATVLM**：整体 **SR 56.17 / SPL 46.15**
- **去掉 explanation traces**：整体 **SR 28.09 / SPL 23.56**

这个掉幅非常大，说明提升并不只是“微调了 Idefics 2”，而是**解释式监督改变了动作学习质量**。

另一个细节也很有意思：

- 如果给模型额外提供 gold positional difference，性能还能继续提升  
  → 说明模型仍会在位置差计算上犯错  
  → 当前瓶颈部分转移到了**空间数值对齐能力**

### 3. 关键证据：不是纯粹记住训练集
作者还做了两个值得看的泛化信号：

- **few-shot**：只用 20% 训练数据时，测试 SR 已到 **38.89%**，基本追平 GPT-4o 的 **38.27%**
- **zero-shot transfer**：
  - iTHOR：**72.79% SR**
  - ProcTHOR：**53.12% SR**

这说明模型学到的并非完全是场景记忆，而确实包含一部分可迁移的导航决策规律。

### 局限性
- **Fails when**: 轨迹很长、需要大范围主动探索、目标长时间不进入视野时，NATVLM 容易在局部区域来回徘徊，缺乏持续探索能力。
- **Assumes**: 训练依赖 AI2THOR 的离散动作空间、GPT-4 + Holodeck 生成场景、BFS 最短路径监督，以及 Idefics 2 骨干；作者报告训练需 8×A100，部分闭源基线还依赖 OpenAI API。
- **Not designed for**: 连续控制、真实机器人低层执行、以及与使用深度/全景/场景图等额外模态的复杂导航系统做严格公平对比。

### 可复用组件
这篇论文最值得复用的不是某个单点模型，而是三类“操作子”：

1. **开放词汇场景生成流程**  
   场景类型 → 属性采样 → LLM 描述 → Holodeck 建屋

2. **最短路径蒸馏流程**  
   网格离散化 → BFS → 动作序列化 → 观测采集

3. **解释式动作监督模板**  
   位置比较 + 障碍判断 + 终止判断 的 CoT 响应模板  
   外加 `MoveAhead` 下采样、冲突样本过滤等数据清洗策略

**一句话总结 So what：**  
这篇工作最重要的价值，是把“LVLM 是否真的会开放词汇导航”从一个模糊问题，变成了一个可大规模测、可低成本训、可清晰诊断的基准问题。

![[paperPDFs/Benchmark/arXiv_2025/2025_DivScene_Benchmarking_LVLMs_for_Object_Navigation_with_Diverse_Scenes_and_Objects.pdf]]