---
title: "CityBench: Evaluating the Capabilities of Large Language Models for Urban Tasks"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - simulator-in-the-loop
  - multimodal-evaluation
  - llm-assisted-annotation
  - dataset/CityBench
  - opensource/full
core_operator: "以多源城市数据和交互式城市模拟器统一构造 8 个城市任务，对 LLM/VLM 的城市理解、推理与决策能力做系统诊断"
primary_logic: |
  城市世界模型评测目标 → 融合地理/视觉/人类活动数据并构建可交互 CitySimu → 在理解与决策两类 8 个任务上统一评测 30 个 LLM/VLM → 揭示其语义能力、数值能力与地理偏置边界
claims:
  - "Claim 1: CityBench 构建了一个覆盖 13 个城市、8 个城市任务、20K+ 图像和 30K+ 问题的统一评测平台，并对 30 个知名 LLM/VLM 做了系统测试 [evidence: analysis]"
  - "Claim 2: GPT-4o 在街景地理定位上达到 City Acc 0.862、Loc Acc 0.797，但在人口密度预测上 RMSE 2.32 仍劣于专用基线 RemoteCLIP 的 1.966，说明通用模型并未在专业城市数值任务上统一胜出 [evidence: analysis]"
  - "Claim 3: 在交通信号控制与跨城市定位/移动预测中，模型表现显著受任务类型与城市分布影响，最佳模型队列长度仍高于 Max-Pressure，且 Cape Town、Nairobi 等城市结果更弱，暴露出数值推理短板与地理偏置 [evidence: analysis]"
related_work_position:
  extends: "V-IRL (Yang et al. 2024)"
  competes_with: "V-IRL (Yang et al. 2024); AgentBench (Liu et al. 2023)"
  complementary_to: "CityGPT (Feng et al. 2024); LLMLight (Lai et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2024/2024_CityBench_Evaluating_the_Capabilities_of_Large_Language_Model_as_World_Model.pdf
category: Survey_Benchmark
---

# CityBench: Evaluating the Capabilities of Large Language Models for Urban Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.13945), [Code](https://github.com/tsinghua-fib-lab/CityBench)
> - **Summary**: 这篇工作用 CityData + CitySimu 把“LLM 是否懂城市”从零散静态问答升级为跨城市、可交互、多模态的系统评测，并证明现有模型更擅长语义理解，明显不擅长专业数值与控制任务。
> - **Key Performance**: GPT-4o 在街景地理定位上达到 City Acc 0.862 / Loc Acc 0.797；但交通信号控制中最优模型队列长度 52.459 仍明显落后于 Max-Pressure 的 36.898。

> [!info] **Agent Summary**
> - **task_path**: 城市多源观测（文本/街景/卫星/模拟状态） -> 城市理解与决策答案/动作 -> 统一评分
> - **bottleneck**: 缺少同时覆盖多模态、动态交互、跨城市分布差异的统一城市评测平台
> - **mechanism_delta**: 用 CityData+CitySimu 把静态城市知识测试扩展成 8 个带环境反馈的理解/决策任务
> - **evidence_signal**: 30 个模型在 13 个城市上的结果一致显示“语义任务相对强、数值/控制任务明显弱”，并伴随显著地理偏置
> - **reusable_ops**: [多源城市数据融合, simulator-in-the-loop评测]
> - **failure_modes**: [格式错误, 无效动作, 拒答/幻觉]
> - **open_questions**: [如何降低长尾城市的地理偏置, 如何让通用LLM稳定处理城市数值与控制任务]

## Part I：问题与挑战

这篇论文的核心不是再做一个城市数据集，而是回答一个更实用的问题：

**如果把 LLM/VLM 当作“城市世界模型”候选者，它们到底在哪些城市任务上可靠，在哪些地方会失效？**

### 真正的难点是什么

已有工作大多只测其中一小块能力：

- 只看**静态地理知识**，例如城市、坐标、道路关系；
- 只看**单一模态**，例如纯文本 GeoQA 或单独的遥感任务；
- 只看**单任务小范围**，很难区分模型是真的“懂城市”，还是只记住了某类模板或某几个热门城市；
- 很少覆盖**动态交互**，也就测不出导航、移动预测、交通控制这类真正接近城市运行的问题。

所以真正瓶颈不是“模型参数里有没有城市知识”，而是：

**模型能否在异构、多模态、局部可观测、带状态转移的城市环境中，把先验知识转成可执行判断和动作。**

### 为什么现在需要解决

因为 LLM/VLM 已经开始被尝试用于：

- 城市视觉理解
- 移动行为预测
- 导航
- 交通信号控制

如果没有系统 benchmark，大家很容易把局部成功误判成“已经具备城市世界建模能力”。CityBench 的价值就在于，它把这个判断从 anecdotal demo 变成了统一、可重复的诊断。

### 输入 / 输出接口与边界

**输入：**

- 文本问题
- 街景图像
- 卫星图像
- 模拟器状态：附近道路/POI、全景图、轨迹历史、路口排队长度、车速等

**输出：**

- 分类或文本答案
- 坐标/人口密度等数值预测
- 下一步位置或道路选择
- 导航动作
- 交通信号相位决策

**边界条件：**

- 13 个全球城市
- 8 个任务，分为“感知理解”和“规划决策”两大类
- 使用公开地图、街景、卫星、人类活动数据以及合成 OD 数据
- 结论针对的是**benchmark 下的能力边界**，不是现实部署安全性证明

---

## Part II：方法与洞察

### 评测框架怎么搭起来

CityBench 由三层组成：

1. **CityData**
   - 整合 OSM 地理数据、Google/Baidu 街景、Esri 卫星图、人类活动数据（Foursquare check-in、合成 OD）
   - 额外做了地图重建，使原始 OSM 能支持 lane topology、building-lane connection 等模拟需求

2. **CitySimu**
   - 提供基础环境 API
   - 支持三类模拟：
     - 个体移动模拟
     - 城市视觉环境模拟
     - 微观交通模拟
   - 把“只会答题”变成“要在环境里感知-决策-执行”

3. **CityBench**
   - 8 个任务：
     - 感知/理解：街景定位、人口密度预测、基础设施识别、GeoQA
     - 决策：移动预测、户外视觉语言导航、城市探索、交通信号控制
   - 题目生成采用模板 + LLM/VLM 辅助
   - 质量控制采用 LLM 过滤/重写 + agent 执行验证 + 作者人工抽检

### 核心直觉

**它改变的不是模型，而是“测量瓶颈”。**

过去很多城市评测，本质上测的是：

- 模型是否记住了城市事实
- 模型是否能对某个单一输入做静态回答

CityBench 改成了：

- 在**多模态输入**下测
- 在**动态状态转移**下测
- 在**可执行动作接口**下测
- 在**多城市分布差异**下测

于是被测分布从“互联网静态知识回忆”切换成了“带环境反馈的城市任务分布”。

这带来两个直接诊断收益：

1. **能区分语义能力和数值/控制能力**
   - 模型可能会看图、会做语义推断
   - 但不一定会做精确地理估计、数值预测、交通控制

2. **能把隐性问题显式暴露出来**
   - 例如格式错误、无效动作、拒答、幻觉
   - 这些在普通 QA 里可能只是“答得不好”，但在 agent setting 里会直接导致任务失败

### 为什么这个设计有效

因果上看，CityBench 有效是因为它把模型从“只输出看起来合理的话”推到了“必须输出可被环境执行的动作”。

一旦动作真的进入模拟器：

- 纯语言流畅性不再够用；
- 只靠记忆热门城市事实也不够；
- 模型必须把先验知识、当前观测、任务目标三者结合起来。

这也是为什么论文能稳定观察到：
- 语义理解类任务相对更强；
- 数值、专业知识、控制类任务显著更弱；
- 长尾城市上更容易暴露偏置。

### 战略取舍

| 设计选择 | 解决的测量盲点 | 带来的收益 | 代价 / 风险 |
|---|---|---|---|
| 多源城市数据融合 | 单一模态看不出城市理解全貌 | 同时评估文本、街景、卫星、行为数据 | 依赖外部数据质量与覆盖度 |
| simulator-in-the-loop | 静态 QA 无法测规划与控制 | 能测导航、探索、交通控制等闭环任务 | 模拟真实性限制外推性 |
| 13 城市全球覆盖 | 单城市 benchmark 易被“记忆”污染 | 能暴露地理偏置与跨城市泛化差异 | 城市间数据不均衡更难校准 |
| 模板 + LLM/VLM 构题 + 质检 | 纯人工构题难扩展 | 快速构造大规模多样题目 | 仍可能残留模板偏差和质检盲点 |

---

## Part III：证据与局限

### 关键证据信号

**信号 1｜横向比较：现有 LLM/VLM 更像语义助手，不像可靠的城市数值/控制器。**

- 街景定位上，GPT-4o 达到很强表现：City Acc 0.862、Loc Acc 0.797。
- 但在人口密度预测中，GPT-4o 的 RMSE 2.32 仍差于专用遥感基线 RemoteCLIP 的 1.966。
- 交通信号控制中，最佳模型 InternVL2-40B 的 queue length 为 52.459，明显差于 Max-Pressure 的 36.898。

**结论**：通用模型在“看懂/说清”层面已有竞争力，但在“精确估计/动态控制”层面仍不可靠。

---

**信号 2｜绝对分数：很多城市任务距离可用仍很远。**

- GeoQA 最优也只有 0.398，离满分 1.0 很远。
- Mobility Prediction 的 top-1 最高仅 0.159，说明即使最强模型也没有表现出稳定的人类移动规律建模能力。

**结论**：把 LLM 当城市世界模型，目前更多是“部分能力可用”，远不是“任务级成熟”。

---

**信号 3｜跨任务排名不稳定：没有通吃模型。**

- GPT-4o 并没有在所有视觉任务上都最好。
- Llama3-70B 在 mobility prediction 和 urban exploration 上更强。
- 开源模型在部分任务上并不输闭源模型，但没有任何一个系列稳定领先所有任务。

**结论**：城市任务异质性很高，单一总榜意义有限；更需要按能力维度诊断，而不是只看平均分。

---

**信号 4｜跨城市差异明显：存在地理偏置。**

- 图像定位和移动预测在不同城市上波动很大。
- 论文指出 Cape Town、Nairobi 等城市更难，而 New York、London、Paris 等公开网络存在更强的城市通常更容易。
- 作者还用 Google/Wikipedia 条目规模做了初步旁证。

**结论**：模型对城市的“认识”部分依赖训练语料的城市曝光度，而不是稳定的普适城市理解。

---

**信号 5｜误差分析：很多失败不是“不会想”，而是“不会按接口行动”。**

常见失败包括：

- misformatted output
- invalid action
- refusal
- hallucination
- logic error

**结论**：如果目标是把模型接进真实城市系统，接口可靠性和执行一致性可能跟知识本身同样关键。**

### 局限性

- **Fails when**: 需要真实在线城市反馈、实时更新地图/街景、连续细粒度控制、或超出 13 个城市覆盖分布的场景时，CitySimu 的结论不能直接等价为真实部署表现。
- **Assumes**: OSM、Google/Baidu Maps API、Esri 卫星图、Foursquare-checkin、合成 OD 数据具有足够质量；题目生成中的模板与 LLM/VLM 辅助标注足够可靠；模型能遵守输出格式；复现实验还依赖云 API、本地 GPU、vLLM/VLMEvalKit 等基础设施。
- **Not designed for**: 给出真实城市系统的安全认证、覆盖所有城市任务、评估训练后城市专用微调策略的最优性，或替代真实世界 A/B 测试。

### 可复用组件

这篇工作的可复用价值其实很高：

- **CityData**：城市多源数据整合管线
- **mosstool**：面向模拟的地图重建工具
- **CitySimu APIs**：移动、导航、交通控制环境接口
- **题目生成与质检流程**：模板生成 + LLM 重写/过滤 + 执行验证

### 一句话总结

CityBench 的真正贡献不是证明“LLM 已经是城市世界模型”，而是更清楚地证明：

**现有模型只在城市语义理解上接近可用，在专业数值、动态控制和长尾城市泛化上仍有明显短板。**

## Local PDF reference

![[paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2024/2024_CityBench_Evaluating_the_Capabilities_of_Large_Language_Model_as_World_Model.pdf]]