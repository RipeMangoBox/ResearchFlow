---
title: "OpenBench: A New Benchmark and Baseline for Semantic Navigation in Smart Logistics"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/semantic-navigation
  - task/last-mile-delivery
  - long-horizon-evaluation
  - osm-grounding
  - llm-task-planning
  - dataset/OpenBench
  - opensource/full
core_operator: 用住宅区 OSM、多地址自然语言配送任务和指数衰减长期指标，把户外最后一公里语义导航评测从“单次到达”升级为“连续配送稳定性诊断”。
primary_logic: |
  最后一公里住宅区语义配送评测目标 → 构建 small/medium/large 仿真场景与真实校园测试，配套 OSM 和自然语言多地址指令 → 以 SRTP/SR/SPL 与指数衰减的 LSR/LSPL 联合评分，并提供 OPEN 基线系统 → 揭示任务理解、GPS-free 定位、地图完备性与连续执行稳定性的能力边界
claims:
  - "OPEN 在 small/medium/large 仿真环境中的单任务 SR 分别为 100%、100% 和 60%，显著高于 ViNT 与 NoMaD 的 40%/20%/0 水平 [evidence: comparison]"
  - "LSR/LSPL 能区分单次成功与连续配送稳定性：OPEN 在 large 场景单任务 SR 为 60%，但连续任务 LSR/LSPL 为 83.98%/47.97%，说明前序任务权重会改变长期表现判定 [evidence: analysis]"
  - "在线地图更新能提高重复配送效率，三个示例目标的 SPL 分别从 22.44%→29.31%、85.68%→91.09%、34.72%→51.34% [evidence: ablation]"
related_work_position:
  extends: "On Evaluation of Embodied Navigation Agents (Anderson et al. 2018)"
  competes_with: "NoMaD (Sridhar et al. 2023); ViNT (Shah et al. 2023)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Benchmark/arXiv_2025/2025_OpenBench_A_New_Benchmark_and_Baseline_for_Semantic_Navigation_in_Smart_Logistics.pdf
category: Survey_Benchmark
---

# OpenBench: A New Benchmark and Baseline for Semantic Navigation in Smart Logistics

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.09238), [Project](https://ei-nav.github.io/OpenBench/)
> - **Summary**: 论文提出面向智能物流最后一公里的户外语义导航基准 OpenBench，并给出结合 OSM、LLM/VLM 与经典规划的 OPEN 基线，用于评测无需高精预建图的连续配送能力。
> - **Key Performance**: 仿真 small/medium/large 上 OPEN 的单任务 SR 为 **100% / 100% / 60%**，明显高于 ViNT/NoMaD；地图更新让示例任务 SPL 进一步提升 **6.31%–47.87%**。

> [!info] **Agent Summary**
> - **task_path**: 自然语言多地址配送指令 + OSM + RGB/LiDAR感知 -> 户外住宅区连续到门配送导航
> - **bottleneck**: 现有评测偏室内、短程、单目标，无法诊断公开稀疏地图下多地址连续配送中的任务理解、地图缺失与长程漂移耦合问题
> - **mechanism_delta**: 把 OSM 公共地图、LLM 层级地址解析、VLM 全局定位/地图更新与 LSR/LSPL 长期指标打包成统一 benchmark+baseline
> - **evidence_signal**: 三尺度仿真对比 ViNT/NoMaD + 地图更新消融 + 真实校园顺序配送完成
> - **reusable_ops**: [层级地址解析, OSM驱动导航-探索切换]
> - **failure_modes**: [大场景连续任务性能下降, 门牌不可见或OSM缺失时探索成本升高]
> - **open_questions**: [指数衰减长期指标是否贴合真实物流损失, 在动态交通和更大城市区域上是否仍稳定]

## Part I：问题与挑战

这篇论文真正补的，不只是一个更强的导航器，而是一个**更贴近真实最后一公里配送的评测缺口**。

现有语义导航 benchmark 大多有三个偏差：

1. **场景偏差**：多数集中在室内，而不是住宅区、园区这类户外开放环境。
2. **任务偏差**：多数是单目标到达，而不是“一个自然语言订单里包含多个地址、按顺序连续配送”。
3. **地图偏差**：要么假设已有高精地图，要么完全 mapless；但现实配送常见的是只有 **公开但稀疏的 OSM**，建筑级信息有，单元门/门牌级信息往往缺失。

所以这篇论文的**真实瓶颈**是：

> 如何评测一个系统在“公开稀疏地图 + 多地址自然语言任务 + 长程连续执行 + GPS 不稳定”条件下，是否还能稳定完成最后一公里配送。

### 输入/输出接口

- **输入**：
  - 多语言自由文本配送指令
  - 公开 OSM 地图
  - 机载 RGB 相机与 LiDAR
- **输出**：
  - 结构化配送任务序列
  - 顺序到达各住户门口的导航/探索轨迹

### 边界条件

- 不依赖高精预建图
- 主要面向户外住宅/校园配送
- 允许 GPS 不可用或不可靠
- 顺序任务前后有关联，前序失败会拖累后续任务
- 文中实验把“到达目标 10m 内”视为成功

### 为什么现在值得做

因为三个条件刚好同时成熟：

- **最后一公里智能物流**需求快速增长；
- **OSM** 提供了可扩展、低成本的公开地图底座；
- **LLM/VLM** 已足够胜任文本任务解析和开放世界语义识别，使“无需重建高精地图”的方案首次具备实用性。

---

## Part II：方法与洞察

这篇工作其实是“**benchmark + baseline**”双件套：

- **OpenBench**：定义任务、场景和指标，专门测“户外连续配送”
- **OPEN**：给出一个可跑通的参考系统，证明该 benchmark 不是纸上设计

### Benchmark 设计：测什么，为什么能测到

OpenBench 把评测拆成三层：

1. **任务理解层**：  
   用 **SRTP** 测 LLM 是否能把自然语言配送单正确解析成结构化地址任务。

2. **单任务执行层**：  
   用经典 **SR / SPL** 测单次导航是否成功、路径是否高效。

3. **长期连续执行层**：  
   引入 **LSR / LSPL**，对顺序配送使用指数衰减权重。  
   核心含义不是“前面的任务更重要”这么简单，而是：
   - 前序任务失败会改变后续任务的时间与状态
   - 连续配送不是若干独立 episode 的平均数
   - 所以需要一个能显式刻画**级联失败**的指标

数据与环境方面，作者构建了：
- small / medium / large 三种仿真住宅区
- 配套 OSM
- 真实校园环境测试

这样评测覆盖了从语言理解、单次导航到多任务连续配送的完整链路。

### OPEN 基线：如何把粗地图变成可执行配送

OPEN 的系统逻辑可以概括为四步：

1. **LLM 做地址解析与任务排序**  
   - 把自由文本拆成层级地址
   - 二次 prompt 校验，缓解 hallucination
   - 对多地址任务做顺序优化

2. **OSM 分级查询，决定“导航还是探索”**  
   - 若 OSM 已有足够细粒度地址：直接导航
   - 若只有 building 级信息：先导航到已知上层地址，再围绕建筑做局部探索

3. **VLM 做全局定位与地图更新**  
   - 用 MobileSAM + CLIP 做开放世界语义识别
   - 把视觉语义与 OSM 元素对齐，修正全局位姿
   - 识别到新门牌后写回地图，为后续任务积累记忆

4. **经典模块负责局部稳定执行**  
   - LiDAR 里程计做局部状态估计
   - 因子图融合全局先验
   - A* / TEB 做局部可执行路径

它不是端到端替代传统导航，而是把 foundation model 用在**最缺信息的地方**：
- 语言歧义由 LLM 解
- 公开地图缺的语义细节由 VLM 补
- 局部连续控制仍交给经典算法

### 核心直觉

这篇论文最关键的变化不是“换了一个模型”，而是**换了问题分解方式**：

**原来**：  
已知/高精地图 + 单目标导航 → 主要考局部规划与避障

**现在**：  
公开稀疏 OSM + 层级地址 + 多任务连续配送 → 主要考  
1) 任务理解是否正确，  
2) 粗地图能否支撑 coarse-to-fine 导航，  
3) 长程执行中定位漂移和地图缺失是否会累计失效。

其因果链条是：

**自然语言多地址任务**  
→ LLM 把地址结构化并排序  
→ OSM 先给出 building 级粗目标  
→ 缺失的单元门/门牌交给探索模式解决  
→ VLM 周期性做重定位并把新语义写回地图  
→ 后续任务搜索范围更小、漂移更可控、重复配送更高效

也就是说，作者引入的关键“旋钮”是：

> 把“高精地图依赖”改成“公开粗地图 + 在线语义补全”，再用长期指标把这种能力真正测出来。

### 战略权衡

| 设计选择 | 改变了什么约束 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| 用 OSM 替代高精语义地图 | 从重预建图变成公开稀疏地图 | 易部署、跨区域扩展、存储极小 | 门牌/单元级信息常缺失 |
| LLM 解析多地址文本 | 从模板化目标变成真实配送指令 | 支持多任务、人类语言输入 | 受 hallucination 与模型差异影响 |
| 导航/探索双模式 | 从“已知目标点直达”变成“已知到未知的分层求解” | 可处理 OSM 不完备目标 | 探索时间增加，失败点更多 |
| VLM+OSM 全局重定位 | 从纯 odometry / GPS 依赖变成语义对齐 | GPS 弱场景下抑制长程漂移 | 依赖可见语义与识别质量 |
| LSR/LSPL 长期指标 | 从独立 episode 平均分变成连续任务评价 | 能暴露级联失败 | 权重设计是否贴近业务仍需验证 |

---

## Part III：证据与局限

### 关键证据

- **比较信号：仿真三尺度对比**
  - OPEN 在 small / medium / large 上的单任务 SR 为 **100% / 100% / 60%**
  - ViNT 与 NoMaD 分别只有 **40% / 20% / 0** 量级表现
  - 说明在公开地图驱动的户外场景里，纯学习式视觉导航基线泛化明显不足，而“OSM + foundation models + classic planning”的混合范式更稳

- **分析信号：长期指标确实测出了 SR/SPL 看不到的东西**
  - OPEN 在 large 场景单任务 SR 只有 **60%**
  - 但连续任务 **LSR / LSPL = 83.98% / 47.97%**
  - 这不是“指标变好看”，而是说明 benchmark 能区分：
    - 单次是否到达
    - 长期配送里前序任务是否稳定
    - 路线效率是否被后续探索拖垮

- **分析信号：任务理解能力高度依赖 LLM 质量**
  - GPT-4o-mini 的 SRTP 为 **1.0**
  - Claude-3.5 为 **0.97**
  - 而 Gemini-1.5-pro 仅 **0.27**
  - 这揭示 benchmark 不只是测导航，也在测“语言接口是否可靠”

- **消融信号：地图更新对重复配送有效**
  - 加入 map update 后，三个示例目标的 SPL 分别提升 **30.61% / 6.31% / 47.87%**
  - 说明“看过一次就写回 OSM”确实能减少后续重复探索

- **部署成本信号：OSM 表示极轻量**
  - OSM 地图是 **kB 级**
  - 对比点云图是 **百 kB 到 MB 级**
  - NoMaD 式拓扑图是 **几十到几百 MB 级**
  - 这直接支撑了作者“易部署、可扩展”的主张

- **真实世界信号：至少能跑通顺序配送**
  - 在真实校园双目标顺序配送中，OPEN 完成任务
  - ViNT / NoMaD 在首个目标前即碰撞失败
  - 人工遥控基线 SPL 为 **96.1%**，OPEN 轨迹与其接近，但论文未给出更完整的大规模真实世界统计

### 局限性

- **Fails when**: OSM 建筑几何或层级地址错误、门牌被遮挡/不可见、动态交通干扰强、连续任务链很长时，系统会因探索成本或累计误差而掉速；large 场景 SR 降到 60% 已体现这一边界。
- **Assumes**: 依赖可用的 OSM、RGB+LiDAR 传感器、可见建筑/门牌语义；任务规划结果强依赖具体 LLM，最佳结果来自 GPT-4o-mini/Claude 等外部模型；真实世界评测部分依赖 GPS 记录与人工判定；实验算力为 RTX 4060 级别。
- **Not designed for**: 室内导航、密集人车混行的社会导航、车队级调度与时间窗优化、多机器人协同配送，也不是纯端到端 mapless policy 的公平统一评测。

### 可复用组件

- **SRTP + LSR/LSPL**：可迁移到任何“多步、连续、前后依赖”的 embodied benchmark
- **OSM 分级查询 + 导航/探索切换**：适合公开粗地图上的 coarse-to-fine 任务执行
- **OSM + VLM 地图更新机制**：适合重复服务型机器人做“长期记忆”
- **轻量地图表示**：对部署成本敏感的真实机器人系统尤其有价值

## Local PDF reference

![[paperPDFs/Benchmark/arXiv_2025/2025_OpenBench_A_New_Benchmark_and_Baseline_for_Semantic_Navigation_in_Smart_Logistics.pdf]]