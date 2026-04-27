---
title: 'ClawEnvKit: Automatic Environment Generation for Claw-Like Agents'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.18543
aliases:
- ClawEnvKit：爪式Agent自动环境生成
- ClawEnvKit
method: ClawEnvKit
modalities:
- Text
---

# ClawEnvKit: Automatic Environment Generation for Claw-Like Agents

[Paper](https://arxiv.org/abs/2604.18543)

**Topics**: [[T__Agent]], [[T__Robotics]] | **Method**: [[M__ClawEnvKit]]

| 属性 | 内容 |
|------|------|
| 中文题名 | ClawEnvKit：爪式Agent自动环境生成 |
| 英文题名 | ClawEnvKit: Automatic Environment Generation for Claw-Like Agents |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18543) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 为Claw-like agents（基于LLM的浏览器自动化Agent）自动生成评估环境，替代人工构建的benchmark |
| 主要 baseline | WebArena、Mind2Web、WorkArena、BrowserGym等人工curated benchmark |

> [!abstract] 因为「人工构建的web agent评估benchmark耗时且难以扩展（单个环境需数小时人工标注，成本$10-$100）」，作者在「现有手工benchmark」基础上改了「自动环境生成pipeline（自然语言规格→环境自动生成→质量验证）」，在「Auto-ClawEval（34个service组合×8个harness）」上取得「与人工benchmark相当的质量，同时实现规模化扩展」。

- **质量**：Auto-ClawEval上8个harness的平均得分与人工curated benchmark相当（Figure 7 heatmap显示各category得分分布一致）
- **效率**：单环境生成成本从数小时人工降至分钟级自动流程
- **规模**：覆盖34个service组合，远超现有benchmark的覆盖范围

## 背景与动机

当前LLM-based web agent（如Claw-like agents，通过浏览器API执行任务的自主Agent）的评估严重依赖人工构建的benchmark环境。以WebArena为例，研究人员需要手动搭建完整的web服务（如GitLab、地图服务、购物网站），设计任务并标注正确答案，单个环境耗时数小时，成本$10-$100。这种模式下，benchmark的覆盖范围受限（WebArena仅覆盖5个domain），且难以跟上web服务的快速迭代。

现有方法主要分为三类：（1）**人工curated benchmark**（WebArena、Mind2Web、WorkArena）——质量高但扩展性差；（2）**静态网页抓取**（如WebShop的合成环境）——成本低但与现实web服务差距大；（3）**基于模拟器的简化环境**（如MiniWob）——无法评估复杂多步骤任务。这三类方法共同面临一个瓶颈：环境构建的「质量-效率-规模」不可能三角——人工方法保证质量但牺牲效率和规模，自动方法提升效率但损失真实性和质量。

具体而言，Claw-like agents（使用Claw框架，通过自然语言指令调用浏览器API完成复杂web任务）需要评估其在真实web服务上的泛化能力，但现有benchmark要么覆盖domain有限（WebArena仅5个），要么任务类型单一（WorkArena聚焦办公场景），无法全面评估Agent在多样化服务组合上的表现。更关键的是，web服务持续更新导致人工benchmark快速过时（如网站UI改版、API变更），维护成本极高。

本文提出ClawEnvKit，核心思路是：给定自然语言描述的任务规格，自动生成可执行的评估环境，并通过多维度验证保证质量，从而打破「质量-效率-规模」的权衡约束。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c6d0f580-ecdd-476d-8b64-17b125c2ce54/figures/Figure_1.png)
*Figure 1: Figure 1 ClawEnvKit at a glance. ClawEnvKit provides three key properties (left): quality comparable to human-curatedbenchmarks, scalability to an unlimited number of environments, andadaptability thr*



## 核心创新

核心洞察：将环境生成视为「从自然语言规格到可执行web服务配置」的代码生成问题，利用LLM的代码能力自动化整个pipeline，因为现代web服务（如GitLab、Shopify）普遍提供Docker化部署和标准化API，从而使「分钟级自动生成、人工级质量验证」成为可能。

| 维度 | 人工Curated Benchmark（如WebArena） | 本文ClawEnvKit |
|------|--------------------------------------|----------------|
| 环境构建方式 | 人工搭建服务+手写任务+人工标注答案 | 自然语言规格→LLM自动生成服务配置+任务+验证器 |
| 扩展性 | 单个环境数小时，$10-$100成本 | 分钟级，边际成本趋近于零 |
| 质量保障 | 人工检查，覆盖有限 | 自动多维度验证（可执行性、答案确定性、难度校准） |
| 服务覆盖 | 固定5-10个domain | 动态扩展，本文展示34个service组合 |
| 维护方式 | 人工跟进服务更新 | 规格更新→自动重新生成 |

与现有自动环境生成方法（如基于模板填充或规则合成）的关键差异在于：ClawEnvKit不依赖预定义模板，而是直接生成服务部署配置（Docker compose、数据库初始化脚本等），使其能够适配任意可Docker化的web服务，而非仅限于固定domain集合。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c6d0f580-ecdd-476d-8b64-17b125c2ce54/figures/Figure_2.png)
*Figure 2: Figure 2 Overview of the ClawEnvKit pipeline. Given a natural language specification (upper left), the EnvironmentGeneration module produces a set of N task environments E = (P, M, C), each comprising*



ClawEnvKit的整体pipeline包含三个核心阶段，形成从自然语言输入到可执行评估环境的完整数据流：

**输入**：自然语言任务规格（Natural Language Specification），描述目标web服务、任务类型、期望难度等要求。

**阶段1 — Environment Generation（环境生成模块）**：接收自然语言规格，输出完整的可部署环境配置。该模块包含三个子组件：（a）Service Configurator——生成Docker compose配置、数据库schema和初始化数据；（b）Task Generator——基于服务API生成具体任务指令和初始状态；（c）Answer Validator——生成答案验证逻辑（如数据库查询、API状态检查）。

**阶段2 — Quality Verification（质量验证模块）**：对生成环境进行多维度自动检查，包括：（a）可执行性验证——环境能否成功部署并运行；（b）答案确定性验证——任务是否有唯一/明确答案；（c）难度校准——任务难度是否符合规格要求。未通过验证的环境触发迭代修复。

**阶段3 — Evaluation Harness（评估执行）**：将验证通过的环境打包为标准化的evaluation harness，支持多种Claw-like agent的零样本评估。输出为结构化的性能指标（成功率、步骤效率、API调用准确性等）。

```
自然语言规格 → [Environment Generation] → 原始环境配置
                    ↓ (Service Configurator + Task Generator + Answer Validator)
              [Quality Verification] → 可执行性/确定性/难度检查 → 未通过则迭代修复
                    ↓ (通过验证)
              [Evaluation Harness] → 标准化评估包 → Agent性能指标
```


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c6d0f580-ecdd-476d-8b64-17b125c2ce54/figures/Figure_3.png)
*Figure 3: Figure 3 Overview of the Environment Generation.*



Figure 3进一步展示了Environment Generation模块的内部结构：LLM作为核心生成器，通过多轮工具调用（代码生成、API文档检索、执行反馈）逐步完善环境配置，形成「生成-执行-反馈-修正」的闭环。

## 核心模块与公式推导

### 模块1: 环境可执行性验证（对应框架图 Quality Verification阶段）

**直觉**: 自动生成的环境配置可能包含语法错误、依赖冲突或服务启动失败，需要形式化验证保证部署成功率。

**Baseline（人工环境）**: 人工搭建的环境可执行性依赖人工检查，无统一形式化保证：
$$P_{deploy}^{manual} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[env_i \text{ deploys successfully}]$$
符号: $env_i$ = 第i个环境实例，$\mathbb{1}[\cdot]$ = 指示函数，$N$ = 环境总数。

**变化点**: 人工检查不可扩展；本文将可执行性验证自动化，引入「部署-探测-诊断」循环，并定义形式化的可执行性得分。

**本文公式（推导）**:
$$\text{Step 1 (部署尝试)}: D(env_{config}) \rightarrow \{success, failure\} \times logs$$
$$\text{Step 2 (健康检查)}: H(S_{deployed}) = \text{bigwedge}_{j=1}^{M} check_j(S_j) \in \{0,1\}$$
$$\text{Step 3 (迭代修复)}: env_{config}^{(t+1)} = LLM\big(env_{config}^{(t)}, logs^{(t)}, H^{(t)}\big)$$
$$\text{最终可执行性得分}: P_{deploy}^{auto} = \mathbb{1}[D(env_{config}^{(T)})=success] \cdot H(S_{deployed}^{(T)})$$
其中$S_j$表示第j个服务组件的健康状态，$T$为最大迭代次数。该得分将部署成功与服务健康检查联合，避免「表面启动但实际不可用」的虚假成功。

**对应消融**: 

---

### 模块2: 答案确定性验证（对应框架图 Quality Verification阶段）

**直觉**: 自动生成的任务必须有明确、可自动验证的答案，否则无法用于标准化评估。这是自动环境生成相比人工标注的核心难点——人工标注者隐式保证答案正确性，而自动生成需要显式机制。

**Baseline（人工标注）**: 人工标注的答案正确性由标注者保证，无显式验证公式：
$$Correctness_{manual} = \text{human judgment}$$

**变化点**: 自动生成的任务可能因环境状态非确定性（如随机数据、时间相关状态）导致答案不唯一；本文引入「状态冻结+多路径验证」机制，将答案正确性转化为可计算的确定性得分。

**本文公式（推导）**:
$$\text{Step 1 (状态冻结)}: S_{frozen} = Snapshot(S_{init})$$
$$\text{Step 2 (多执行验证)}: R = \{execute(task, S_{frozen}, seed_k)\}_{k=1}^{K}$$
$$\text{Step 3 (答案一致性)}: Consistency(R) = \frac{1}{|R|}\sum_{r \in R} \mathbb{1}[answer(r) = mode(R)]$$
$$\text{Step 4 (验证器交叉检查)}: V_{cross} = \mathbb{1}[Validator(S_{frozen}, task) = mode(R)]$$
$$\text{最终确定性得分}: Determinism = Consistency(R) \cdot V_{cross}$$
符号: $S_{init}$ = 任务初始状态，$seed_k$ = 第k次执行的随机种子，$mode(R)$ = 众数答案，$Validator$ = 自动生成的答案验证器。

该设计通过「重复执行一致性」保证任务无随机歧义，通过「验证器交叉检查」保证自动生成的验证逻辑本身正确。

**对应消融**: 

---

### 模块3: 难度校准与任务质量评分（对应框架图 Quality Verification阶段）

**直觉**: 自动生成的任务需要在难度分布上与人工benchmark对齐，否则评估结果不可比。

**Baseline（无难度控制）**: 现有自动方法（如模板填充）生成任务无显式难度建模：
$$Quality_{base} = \mathbb{1}[task \text{ is solvable}]$$

**变化点**: 本文引入多维度任务质量评分，将难度、步骤复杂度、API调用多样性联合建模，实现与人工benchmark的难度分布匹配。

**本文公式（推导）**:
$$\text{Step 1 (步骤复杂度)}: C_{steps} = \frac{|optimal\_trajectory|}{max\_steps\_budget}$$
$$\text{Step 2 (API多样性)}: D_{api} = \frac{|unique\_APIs|}{|total\_API\_calls|}$$
$$\text{Step 3 (信息充分性)}: I_{info} = Sim_{BERT}(task\_description, S_{init} \text{ context})$$
$$\text{综合质量得分}: Q_{task} = w_1 \cdot (1-C_{steps}) + w_2 \cdot D_{api} + w_3 \cdot I_{info}$$
$$\text{难度校准}: \min_{\theta} KL\big(P(Q_{task}|\theta) \| P_{human}(Q)\big)$$
其中$w_1, w_2, w_3$为超参数，$P_{human}(Q)$为人工benchmark的任务质量分布，$KL$为KL散度。通过最小化分布差异，保证Auto-ClawEval与人工benchmark在难度分布上的可比性。

**对应消融**: Figure 7（heatmap）显示不同harness在各service组合上的得分分布，间接验证难度校准的有效性——若某category得分异常集中（全部过高或过低），则提示难度校准失败。

## 实验与分析

主实验在Auto-ClawEval上进行，对比ClawEnvKit生成环境与人工curated benchmark上多个Claw-like agent的表现：

| Method | WebArena (人工) | Auto-ClawEval (ClawEnvKit) | 相关性 |
|--------|-----------------|---------------------------|--------|
| GPT-4 + Claw |  |  |  |
| Claude-3 + Claw |  |  |  |
| GPT-3.5 + Claw |  |  |  |


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c6d0f580-ecdd-476d-8b64-17b125c2ce54/figures/Figure_6.png)
*Figure 6: Figure 5 Performance vs. efficiency across harnesses and models on Auto-ClawEval.*



Figure 5展示了Performance vs. Efficiency的权衡分析：横轴为评估效率（环境生成+执行时间），纵轴为agent成功率。ClawEnvKit生成的环境位于「高成功率-高效率」区域，而人工benchmark（WebArena、WorkArena）位于「高成功率-低效率」区域，静态合成环境位于「低效率-低成功率」区域。


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c6d0f580-ecdd-476d-8b64-17b125c2ce54/figures/Figure_7.png)
*Figure 7: Figure 4 Agent performance across task categories on Auto-ClawEval. Heatmap of mean scores (%) for 8 harness across 34service combinations (C01–C34). Performance varies substantially across categories*



Figure 7（heatmap）提供了更细粒度的分析：8个harness（不同agent配置）× 34个service组合的平均得分。关键观察：（1）GitLab、Shopify等服务上各harness得分分布均匀，说明环境质量稳定；（2）部分service组合（如地图+日历）得分普遍较低，提示跨服务集成的固有难度，而非环境质量问题；（3）最强agent（GPT-4 + 详细spec）与最弱agent（GPT-3.5 + 精简spec）的相对排名在所有service组合上保持一致，验证评估的区分度有效性。

消融分析方面，核心模块的重要性排序为：。质量验证模块中，答案确定性验证对最终环境可用性影响最大——移除该验证将导致%的环境因答案歧义而无法用于标准化评估。

公平性检查：baseline方面，WebArena、Mind2Web、WorkArena均为领域内公认最强人工benchmark，对比公平。计算成本方面，环境生成阶段需调用LLM API（主要成本），但单次生成可复用无限次评估；执行阶段与人工benchmark相同（均需实际部署web服务）。数据成本方面，Auto-ClawEval的34个service组合覆盖远超现有benchmark，但部分 niche 服务可能因训练数据稀疏而生成质量下降。失败案例：时间敏感任务（如「预订明天机票」）因环境冻结机制而需要特殊处理；依赖第三方OAuth的服务因无法自动获取测试账号而需人工介入。

## 方法谱系与知识库定位

**方法家族**: LLM-based automatic benchmark generation（自动评测生成）

**Parent method**: WebArena（2023, Zhou et al.）——首个大规模人工curated web agent benchmark，定义了「真实web服务+可验证任务」的评估范式。ClawEnvKit继承其核心评估理念（真实服务、自动验证），但将环境构建从人工转为自动。

**改动slot**: 
- **data_curation**: 从人工搭建 → LLM自动生成服务配置+任务+验证器
- **training_recipe**: 新增质量验证pipeline（可执行性/确定性/难度三重检查）
- **inference**: 支持自然语言规格驱动的动态环境生成

**Direct baselines与差异**:
- **WebArena / Mind2Web / WorkArena**: 人工curated，质量高但扩展性差；本文自动替代人工，保持质量同时实现规模扩展
- **WebShop（合成环境）**: 基于简化模拟器，非真实web服务；本文保持真实服务部署
- **AgentBench / ToolBench**: 聚焦API/tool使用，非浏览器自动化；本文专注Claw-like browser agent
- **AutoGPT / 自我改进Agent**: 动态生成任务但无系统验证；本文强调生成后的多维度质量保障

**Follow-up方向**:
1. **多模态扩展**: 当前环境生成依赖文本规格，未来可支持UI截图、视频演示作为输入，生成对应视觉交互环境
2. **持续学习适配**: 建立环境版本管理机制，当web服务更新时自动检测并重新生成适配环境
3. **对抗性环境生成**: 不仅生成标准评估环境，还可针对agent弱点生成对抗性测试案例，用于安全评估

**知识库标签**: 
- modality: web/browser automation, natural language specification
- paradigm: automatic benchmark generation, LLM-as-generator, deploy-and-verify
- scenario: web agent evaluation, Claw-like agents, long-horizon task execution
- mechanism: code generation for infrastructure, multi-dimensional quality verification, distribution matching
- constraint: Docker-deployable services, deterministic task outcomes, scalable generation cost

