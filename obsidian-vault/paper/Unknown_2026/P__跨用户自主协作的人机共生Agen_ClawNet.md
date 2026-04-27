---
title: 'ClawNet: Human-Symbiotic Agent Network for Cross-User Autonomous Cooperation'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19211
aliases:
- 跨用户自主协作的人机共生Agent网络
- ClawNet
method: ClawNet
---

# ClawNet: Human-Symbiotic Agent Network for Cross-User Autonomous Cooperation

[Paper](https://arxiv.org/abs/2604.19211)

**Topics**: [[T__Agent]] | **Method**: [[M__ClawNet]]

| 中文题名 | 跨用户自主协作的人机共生Agent网络 |
| 英文题名 | ClawNet: Human-Symbiotic Agent Network for Cross-User Autonomous Cooperation |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19211) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 跨用户自主协作、人机共生智能体网络、治理感知的多智能体协调 |
| 主要 baseline | OpenClaw, AutoGen, MetaGPT, CrewAI |

> [!abstract] 因为「现有Agent框架局限于单用户边界，无法自主跨越用户权限边界实现协作」，作者在「OpenClaw」基础上改了「引入人机共生架构与治理感知协作机制」，在「跨用户协作场景」上取得「实现自主跨边界任务分解与执行」

- **关键性能**: 相比OpenClaw，ClawNet实现跨用户任务完成率
- **关键性能**: 治理冲突检测准确率
- **关键性能**: 人机协作满意度评分

## 背景与动机

当前大型语言模型（LLM）驱动的Agent系统已能完成复杂任务规划与执行，但面临一个根本性瓶颈：所有协作被锁定在单一用户边界内。例如，当用户A需要与用户B协调完成一份联合报告时，现有系统无法自主跨越账户、权限和组织边界发起协作，必须依赖人工手动转发、授权和确认。

现有方法如何处理这一问题？**AutoGen** 通过对话编程实现多Agent协作，但所有Agent共享同一用户上下文，无跨用户概念；**MetaGPT** 引入软件公司角色分工模拟，但角色属于同一虚拟组织，不涉及真实用户边界跨越；**CrewAI** 提供任务委派流程，仍假设所有Agent在同一操作域内运行；**OpenClaw** 首次提出开放Agent网络，支持Agent发现与调用，但缺乏治理机制，跨用户协作需显式人工授权每一步。

这些方法的共同短板在于：**将用户边界视为不可逾越的硬约束**，导致跨用户协作退化为"人工桥接"模式——每次跨边界交互都需人工确认，丧失自主性。更深层的矛盾是：完全自主跨越边界侵犯用户主权，完全人工控制又丧失效率。这一 tension 催生了核心研究问题：**如何设计既尊重用户治理主权、又能实现高效自主协作的Agent网络？**

本文提出 ClawNet，以"人机共生"为核心范式，通过双层架构分离用户主权域与协作执行域，使Agent能在预设治理规则下自主完成跨用户任务分解、协商与执行。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5f406f9a-4ce7-4c26-a1db-916e4341c77d/figures/Figure_1.png)
*Figure 1 (motivation): Paradigm comparison between OpenClaw and ClawNet.*



## 核心创新

核心洞察：**人机共生不是简单的人在回路（human-in-the-loop），而是将人类治理意图编码为可计算的约束契约，使Agent在约束空间内获得完全自主性**，因为传统"人工审批每一步"模式将人类置于执行路径上造成瓶颈，而将治理规则前置为边界条件后，Agent可在规则包络内自主决策，从而使大规模跨用户自主协作成为可能。

| 维度 | Baseline (OpenClaw) | 本文 (ClawNet) |
|:---|:---|:---|
| 跨用户协作模式 | 显式人工授权每一步 | 治理规则预授权，Agent自主执行 |
| 人机关系 | 人类作为审批节点（阻塞式） | 人类作为治理契约制定者（约束式） |
| 架构组织 | 单层开放网络，无用户边界概念 | 双层架构：主权域（Human Core）+ 协作域（Agent Swarm） |
| 冲突解决 | 无内置机制，依赖外部仲裁 | 治理感知冲突检测与自动协商 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5f406f9a-4ce7-4c26-a1db-916e4341c77d/figures/Figure_3.png)
*Figure 3 (architecture): High-level architecture of ClawNet.*



ClawNet 采用**双层耦合架构**，数据流如下：

**输入层**：自然语言任务请求（如"协调张三和李四完成Q3财务合并报告"）→ 解析为结构化协作意图，识别涉及用户集合与任务类型。

**Human Core（人类主权域）**：每个参与用户拥有独立的治理内核，存储三项核心资产：① 用户画像（偏好、技能、可用性）；② 治理契约（可自动化操作的权限白名单、需人工确认的黑名单、资源访问上限）；③ 信任网络（对其他用户的显式信任评级）。该层**只输出规则，不介入执行**。

**Agent Swarm（协作执行域）**：由跨用户共享的Agent集群组成，包含三类角色：① **Orchestrator**（协调器）——接收任务，基于治理契约进行可执行性验证与任务分解；② **Negotiator**（协商器）——当子任务涉及多用户资源冲突时，在约束空间内自动协商替代方案；③ **Executor**（执行器）——在单用户授权边界内调用工具链完成原子操作。Agent间通过**治理感知消息总线**通信，每条消息携带发送方与接收方的治理上下文标签。

**输出层**：任务完成结果 + 执行轨迹审计日志（供Human Core追溯与契约更新）。

```
[Task Request] 
    ↓
[Human Core: User A] ←→ [Governance Contract] ←→ [Human Core: User B]
    ↓ (rule export)              ↓ (rule export)
[Agent Swarm] ──→ Orchestrator → Task Decomposition
                    ↓
              Negotiator ←→ Governance-aware Conflict Resolution
                    ↓
              Executor(s) → Tool Calls → [Output + Audit Log]
```

## 核心模块与公式推导

### 模块 1: 治理契约形式化（对应框架图 Human Core层）

**直觉**: 将自然语言治理意图转化为可计算的约束边界，使Agent能本地判断操作合法性而无需远程问询。

**Baseline 公式** (OpenClaw 显式授权): 每一步操作 $a_t$ 需实时获取用户确认信号 $c_t \in \{0,1\}$，有效动作空间为
$$\mathcal{A}_{\text{base}}(s_t) = \{a_t \text{mid} c_t = 1\}$$
符号: $s_t$ = 当前状态, $a_t$ = 候选动作, $c_t$ = 人工确认信号（每步阻塞等待）。

**变化点**: OpenClaw 的 $c_t$ 造成 $O(T)$ 次人工交互复杂度；本文将确认前置为预授权约束集合，将交互降至 $O(1)$（契约制定阶段）。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{G}_u = \langle \mathcal{W}_u, \mathcal{B}_u, \mathcal{Q}_u \rangle \quad \text{将用户}u\text{的治理意图编码为三元组}$$
其中 $\mathcal{W}_u$ = 白名单操作集合（完全自主），$\mathcal{B}_u$ = 黑名单操作集合（禁止），$\mathcal{Q}_u$ = 配额函数（资源使用上限）。

$$\text{Step 2}: \text{Valid}(a_t; \mathcal{G}_u) = \mathbb{1}[a_t \in \mathcal{W}_u] \cdot \mathbb{1}[a_t \notin \mathcal{B}_u] \cdot \mathbb{1}[Q(a_t) \leq \mathcal{Q}_u] \quad \text{本地合法性校验}$$

$$\text{最终}: \mathcal{A}_{\text{ClawNet}}(s_t; \mathcal{G}_u) = \{a_t \text{mid} \text{Valid}(a_t; \mathcal{G}_u) = 1\}$$

**对应消融**: 

---

### 模块 2: 治理感知任务分解（对应框架图 Orchestrator）

**直觉**: 协调器必须在分解阶段即识别跨用户依赖，避免将不可自治的子任务分配给单一Agent。

**Baseline 公式** (标准Hierarchical Planning): 给定任务 $T$，递归分解为子任务集合
$$T = \text{bigcup}_{i} t_i, \quad \text{s.t.} \quad \forall i, \text{Agent}(t_i) = \text{arg}\max_{a} \mathbb{P}(\text{success}|a, t_i)$$
符号: $t_i$ = 原子子任务, Agent($\cdot$) = 任务分配函数。

**变化点**: 标准分解忽略用户归属约束，可能产生需要跨用户权限的"非法"子任务；本文引入治理兼容性约束到分配目标中。

**本文公式（推导）**:
$$\text{Step 1}: \text{User}(t_i) = \{u \text{mid} \text{Resource}(t_i) \cap \text{Domain}(u) \neq \emptyset\} \quad \text{识别子任务涉及的用户集合}$$

$$\text{Step 2}: \text{Auto}(t_i; \{\mathcal{G}_u\}) = \text{bigwedge}_{u \in \text{User}(t_i)} \left( \exists a \in \mathcal{A}(s; \mathcal{G}_u): a \text{ realizes } t_i \right) \quad \text{判断全自治可行性}$$

$$\text{Step 3}: \text{Decompose}(T) = \{(t_i, \text{mode}_i)\} \text{ where } \text{mode}_i = \begin{cases} \text{AUTO} & \text{if } \text{Auto}(t_i)=1 \\ \text{HYBRID} & \text{if } \exists u: \text{Auto}_u(t_i)=1 \wedge \exists u': \text{Auto}_{u'}(t_i)=0 \\ \text{BLOCKED} & \text{otherwise} \end{cases}$$

$$\text{最终}: \min_{\{(t_i, \text{mode}_i)\}} \sum_i \mathbb{1}[\text{mode}_i = \text{BLOCKED}] + \lambda \cdot \mathbb{1}[\text{mode}_i = \text{HYBRID}] \quad \text{优化目标：最小化阻塞与混合模式数量}$$

**对应消融**: 

---

### 模块 3: 约束空间协商（对应框架图 Negotiator）

**直觉**: 当多用户治理规则冲突时（如A的白名单与B的黑名单交集非空），需在约束交集外寻找帕累托改进方案。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{G}_{\text{joint}} = \text{bigotimes}_{u \in U} \mathcal{G}_u \quad \text{联合治理空间为各用户约束的笛卡尔积投影}$$

$$\text{Step 2}: \text{Conflict}(\{t_i\}) = \{t_i \text{mid} \mathcal{A}(s; \mathcal{G}_{\text{joint}}) = \emptyset\} \quad \text{识别冲突子任务（无可行动作）}$$

$$\text{Step 3}: \text{Negotiate}(t_i) = \text{arg}\max_{t_i' \in \text{Alt}(t_i)} \sum_{u} \text{Utility}_u(t_i') \cdot \mathbb{1}[\mathcal{A}(s; \mathcal{G}_u, t_i') \neq \emptyset] \quad \text{在替代方案中寻找满足全部约束的最优解}$$

$$\text{最终}: t_i^* = \begin{cases} t_i & \text{if } t_i \notin \text{Conflict} \\ \text{Negotiate}(t_i) & \text{if } \text{Alt}(t_i) \cap \text{Feasible} \neq \emptyset \\ \text{ESCALATE} & \text{otherwise} \end{cases}$$

**对应消融**: 

## 实验与分析

主实验结果：

| Method | 跨用户任务完成率 | 人工交互次数 | 平均协商轮数 | 治理违规率 |
|:---|:---|:---|:---|:---|
| AutoGen |  |  | N/A |  |
| MetaGPT |  |  | N/A |  |
| CrewAI |  |  | N/A |  |
| OpenClaw |  | $O(T)$ |  |  |
| **ClawNet** | **** | **$O(1)$** | **** | **** |


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5f406f9a-4ce7-4c26-a1db-916e4341c77d/figures/Table_1.png)
*Table 1 (comparison): Paradigm comparison across representative agent frameworks.*



**核心发现分析**: ClawNet 的核心优势应体现在**人工交互次数从 $O(T)$ 降至 $O(1)$**（契约制定阶段一次性投入），同时通过治理感知分解将**治理违规率压降至理论零值**（所有执行动作均经预授权约束校验）。若跨用户任务完成率与OpenClaw持平或更优，则证明"预授权不牺牲效能"的关键假设成立。

**消融实验**（推断设计，若可用）：
- 移除 Negotiator 模块：冲突子任务直接 ESCALATE，预期混合模式比例上升、端到端完成率下降
- 将 Human Core 降级为 OpenClaw 式实时审批：交互次数陡增，任务延迟上升
- 简化治理契约为二元允许/禁止（去掉配额 $\mathcal{Q}_u$）：预期资源超限违规率上升

**公平性检查**: 
- **Baseline强度**: OpenClaw 是最接近的直接可比方法（同作者团队前作），但 AutoGen/MetaGPT/CrewAI 并非为跨用户场景设计，存在"降维打击"质疑；理想应补充跨用户专用 baseline（如未开源的同类工作）。
- **计算/数据成本**: Human Core 的契约制定需用户一次性投入，存在冷启动成本；Agent Swarm 的分布式协商增加通信开销，需报告实际延迟数据（待补充）。
- **失败案例**: ESCALATE 机制将不可调和冲突返还人工，若 ESCALATE 率过高则核心主张受损；需报告不同冲突密度下的 ESCALATE 率曲线（待补充）。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5f406f9a-4ce7-4c26-a1db-916e4341c77d/figures/Figure_2.png)
*Figure 2 (example): Governance-aware cross-border collaboration scenario.*

 展示治理感知跨境协作场景的具体执行轨迹，可辅助定性验证机制有效性。

## 方法谱系与知识库定位

**方法家族**: Multi-Agent System (MAS) → LLM-based Agent Network → 开放/跨组织Agent协作

**父方法**: **OpenClaw**（Zhang et al., 同团队前作）。ClawNet 继承其"开放Agent网络"核心概念，但关键 slot 改变：① **架构**从单层网络改为双层人机共生架构；② **目标函数**从最大化Agent连接数改为最大化自主协作度同时约束治理风险；③ **训练/部署**引入契约预授权机制替代实时审批。

**直接 Baseline 差异**: 
- **vs AutoGen**: AutoGen 聚焦对话编程范式，无用户边界概念；ClawNet 显式建模用户主权域与跨域治理。
- **vs MetaGPT**: MetaGPT 模拟软件公司角色流，角色为虚拟设定；ClawNet 的用户-Agent绑定是真实身份映射。
- **vs OpenClaw**: OpenClaw 开放但"无治理"；ClawNet 以治理契约实现"有约束的开放"。

**后续方向**: 
1. **动态契约学习**: 当前治理契约静态预设，未来可基于执行反馈自动优化契约条款（强化学习+用户偏好学习）。
2. **跨组织联邦治理**: 扩展至企业级B2B场景，引入区块链或可信执行环境保障契约不可篡改。
3. **多模态治理感知**: 当前文本契约扩展至视觉、代码等模态的细粒度权限控制。

**知识库标签**: 
- **modality**: 文本/结构化规则
- **paradigm**: 人机共生 (human-symbiotic), 预授权治理 (pre-authorized governance)
- **scenario**: 跨用户协作, 跨境/跨组织任务协调
- **mechanism**: 约束空间规划, 治理感知消息传递, 自动协商
- **constraint**: 用户主权保留, 零治理违规, 最小化人工阻塞

