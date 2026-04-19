# V6 设计评审：四个核心问题

---

## 问题 1: 打分是否有区分度？

### 真实问题

DiscoveryScore 的 **7 个子分数加权求和** 看似能区分，但存在两个结构性弱点：

#### 弱点 A: "糊中间"问题

同一领域内的论文，DomainMatch 都会在 70-100 之间（关键词命中率高），差异主要靠 SourceSignal 和 GraphProximity 拉开。但大量论文的发现来源是 `s2_reference` (60分) 或 `s2_citation` (50分) — 这两个来源涵盖了一篇论文 80% 的引用/被引用。结果：**大量论文的 DiscoveryScore 挤在 50-70 区间**。

具体模拟（假设研究 "video RL" 领域，一篇 anchor 论文有 50 篇引用）：

```
50 篇引用中的典型分布:
  直接 baseline (3篇): Domain=90, Source=60, Graph=80 → 总分 ~74-78 ✓ 过线
  同方法族 (5篇):      Domain=85, Source=60, Graph=60 → 总分 ~65-70 ← 候选池
  同任务前作 (10篇):   Domain=80, Source=60, Graph=40 → 总分 ~58-63 ← metadata
  跨任务引用 (15篇):   Domain=50, Source=60, Graph=20 → 总分 ~42-48 ← metadata
  背景引用 (17篇):     Domain=30, Source=60, Graph=10 → 总分 ~32-38 ← archive

问题区域: 同方法族(65-70) 和 同任务前作(58-63) 之间只差 5-7 分
```

**这不是灾难性问题** — 候选池的设计就是容纳这些"不确定"论文。真正的质量筛选在 DeepIngestScore 阶段由 ReferenceRoleAgent 完成。

#### 弱点 B: SourceSignal 是"来源类型"，不是"引用重要性"

当前代码中，`_extract_discovery_signals` 把 `candidate.discovery_source` 作为 `source_type` 传给评分引擎。一篇被 method section 引用的直接 baseline 和一篇只在 related work 中提到的论文，如果都通过 S2 references 发现，它们的 `discovery_source` 都是 `s2_reference` (60分)。

**区分它们的信息（"在正文哪里被引用"）只有 ReferenceRoleAgent 才能判断，而 ReferenceRoleAgent 在 shallow ingest 阶段才运行。**

这是设计故意为之：DiscoveryScore 是粗筛（用便宜的元数据），DeepIngestScore 是精筛（用 LLM）。但代价是第一层筛选会漏掉一些重要论文或放进一些不重要论文。

### 现有设计是否可接受

**可接受**，原因：
1. DiscoveryScore 阈值 75 → shallow_ingest。shallow ingest 只花 $0.22/篇 (4 次 LLM)。误放 10 篇不重要论文的成本 = $2.2，可以承受
2. 真正的质量门控在 DeepIngestScore ≥ 88 → deep ingest ($1+/篇)，这时已经有 ReferenceRoleAgent 的精确角色分类
3. 候选池 (60-74) 的论文不花 LLM 成本，只占 DB 空间

### 建议改进（如果需要）

```
改进方案: 在 DiscoveryScore 阶段增加 "引用位置" 信号

当前: discovery_source = "s2_reference" (60分)，无法区分 method 引用和 background 引用

改进: 在 discover_neighborhood 中，S2 API 返回的 citation context 包含 
      "intents" (background/methodology/result_comparison)
      把这个信息传递到 candidate:
        discovery_source = "s2_reference"
        discovery_reason = "methodology"  → SourceSignal 加 10 分
        discovery_reason = "background"   → SourceSignal 不变

这样直接 baseline 可以从 60 提升到 70-80，和 awesome_repo (70) 对齐
```

---

## 问题 2: 引用和 awesome 仓库的检索质量

### 检索盲点分析

| 检索来源 | 覆盖范围 | 盲点 |
|---------|---------|------|
| **arXiv API** | arXiv 上的论文 | 非 arXiv 论文 (AAAI/IJCAI 等只投会议不放 arXiv 的); 搜索是关键词匹配，不是语义搜索; 同义不同词会漏 |
| **S2 references** | 目标论文引用的论文 | 覆盖好，但依赖目标论文作者列的引用是否全面; 有些论文引用不规范或没有 S2 记录 |
| **S2 citations** | 引用目标论文的论文 | 新论文刚发表时被引用为 0; S2 索引有延迟（1-2 周） |
| **S2 recommendations** | 语义相似论文 | 质量依赖 S2 的推荐算法; 可能推荐主题相关但方法无关的论文 |
| **awesome repo** | README 中列出的论文 | **当前代码只做 URL 提取，不判断 README 的结构**; 按年份组织的 repo 会漏掉旧但重要的论文; README 更新不及时 |
| **OpenAlex** | 学术论文元数据 | 覆盖最广但搜索精度不如 S2 |

### 当前代码的具体问题

**问题 A: arXiv 搜索质量低**

```python
# cold_start_service.py line 207-213
resp = await client.get(
    ARXIV_API,
    params={"search_query": f"all:{quote(query)}", "max_results": "20"}
)
```

`all:` 搜索是全文匹配，不是按标题/摘要/关键词分别搜索。"video QA GRPO video" 这样的查询在 arXiv 上会返回很多噪声（任何包含 "video" 的论文都可能命中）。

**改进**:
```python
# 应该用更精确的查询:
params = {
    "search_query": f"ti:{quote(task)} AND abs:{quote(method)}",
    "max_results": "20",
    "sortBy": "relevance",
}
```

**问题 B: awesome repo 解析过于简陋**

当前 `detect_awesome_repo_changes` 的 `_extract_paper_urls` 只用正则提取 URL。但很多 awesome repo 的格式是：

```markdown
- [Paper Title](https://arxiv.org/abs/2501.12345) - Brief description
```

当前代码只提取了 URL，没有提取标题和描述。更好的做法是提取标题+URL+描述，用标题做去重和领域匹配。

**问题 C: 没有用 S2 的 citation intent**

S2 API 在 reference/citation 数据中提供 `intents` 字段（`background`, `methodology`, `result_comparison`），但当前 `discover_neighborhood` 没有利用这个信息。这是免费的引用角色分类，可以直接传给 DiscoveryScore。

**问题 D: 缺少 OpenReview/DBLP 的论文发现**

冷启动只从 arXiv + S2 搜索。但很多重要论文是通过会议 proceeding 发现的（ICLR 2025 accepted papers list）。当前系统有 `openreview_adapter` 和 `dblp_adapter`，但没有在冷启动中用它们做论文发现。

### 检索质量评级

| 来源 | 精度 | 召回率 | 成本 | 当前实现质量 |
|------|------|--------|------|-----------|
| S2 references | 高 | 高 (依赖目标论文) | 免费 | **好** |
| S2 citations | 中 | 中 (有延迟) | 免费 | **好** |
| arXiv search | 低 | 中 | 免费 | **需改进** (查询质量) |
| awesome repo | 中 | 低 (覆盖不全) | 免费 | **需改进** (解析质量) |
| S2 search | 中 | 中 | 免费 | **好** |
| OpenReview | 高 | 低 (只有部分会议) | 免费 | **未接入冷启动** |

---

## 问题 3: 不存入知识库的论文有什么作用？

### 当前状态：L0 论文几乎无用

**这是一个设计缺陷。** 当前 L0 (metadata_only) 论文只存在 `paper_candidates` 表中，有以下问题：

| 能力 | L0 能做到吗？ | 说明 |
|------|-------------|------|
| 防止重复发现 | **能** | 通过 arxiv_id/DOI/title 去重 |
| 被搜索接口检索 | **不能** | `search_service` 只搜 `papers` 表 |
| 在前端显示 | **不能** | 前端只展示 `papers` |
| 作为 citation graph 节点 | **不能** | graph 只连接 `papers` |
| 影响 GraphProximity 评分 | **不能** | GraphProximity 只看已有 anchor 连接 |
| 未来重新激活 | **理论上能** | 但没有触发机制 |

### L0 论文应该发挥的作用

```
1. 引用图谱的"幽灵节点"
   → 即使一篇论文没有深度分析，它仍然是引用网络的一部分
   → 当 Paper A 引用了 L0 论文 X，X 应该在 A 的 related work 中显示
   → X 不需要有 profile，但需要有 title/venue/year 基础信息

2. 候选重新激活
   → 当新论文 B 也引用了 L0 论文 X，X 的 "被引用次数" 增加
   → 当 X 被 3 篇以上 KB 论文引用时，自动提升优先级重新评分
   → 这是"低关注度但高质量论文"的发现机制

3. 领域覆盖率指标
   → "我们的 KB 覆盖了该领域 500 篇论文中的 50 篇深度 + 150 篇浅层"
   → L0 论文提供 "已知但未处理" 的计数

4. 搜索候选结果
   → 用户搜索时，除了 KB 中的论文，还应该展示 "相关但未入库的论文"
   → 标记为 "候选"，允许用户手动提升
```

### 需要的改进

```python
# 1. 在 search_service 中增加候选搜索
async def search_with_candidates(session, query, *, include_candidates=True):
    # 先搜 papers 表
    papers = await search_papers(session, query)
    
    if include_candidates:
        # 再搜 paper_candidates 表 (只搜有 abstract 的 L0/L1)
        candidates = await session.execute(
            select(PaperCandidate).where(
                PaperCandidate.abstract.isnot(None),
                PaperCandidate.status.in_(["discovered", "metadata_resolved", "scored"]),
                # 简单关键词匹配
                func.lower(PaperCandidate.title).contains(query.lower()),
            ).limit(10)
        )
        # 标记为候选，与正式论文区分显示
    
    return {"papers": papers, "candidates": candidates}

# 2. 候选重新激活机制
async def check_reactivation(session, candidate_id: UUID):
    """当 L0 候选被新论文引用时调用"""
    candidate = await session.get(PaperCandidate, candidate_id)
    
    # 统计 KB 中多少篇论文引用了这个候选
    citing_count = await count_citing_papers(session, candidate.arxiv_id)
    
    if citing_count >= 3:
        # 重新评分，可能提升
        await score_candidate(session, candidate_id)
```

---

## 问题 4: 检索到已在库论文时的对比流程

### 当前行为：发现重复 → 静默跳过

```python
# candidate_service.py create_candidate():
existing = await find_duplicate(session, arxiv_id=arxiv_id, ...)
if existing:
    # 只是补充元数据，返回已有记录
    if abstract and not existing.abstract:
        existing.abstract = abstract
    ...
    return existing  # 什么都不触发
```

**这丢失了关键信息**：当 Paper B 引用了已经在 KB 中的 Paper A，系统应该：

```
Paper B 正在分析 → 发现引用了 Paper A (已在 KB)
                      ↓
              应该发生的事情:
              1. 创建 B→A 的引用边 (cites/builds_on/extends)
              2. 更新 A 的 downstream_count
              3. 如果 B 修改了 A 的方法 → 创建 method evolution edge
              4. 如果 A 的 profile 受影响 → staleness++
              5. 如果 B 在新任务上用了 A 的方法 → 更新 A 的 method_applications
              6. 不需要重新分析 A → 但需要"对比分析"
```

### 缺失的"对比流程"

当前系统没有 **"已知论文重遇"** 的专门处理逻辑。这是一个重要的设计缺口。

应该增加的流程：

```
在 shallow_ingest 阶段，ReferenceRoleAgent 分类引用后:

对每篇 role=direct_baseline 的引用:
  ├─ 检查: 该引用是否已在 papers 表中？
  │
  ├─ 如果不在 KB → 递归 import_and_score (当前行为，正确)
  │
  └─ 如果已在 KB → 触发对比流程 (当前缺失):
       │
       ├─ 1. 创建关系边
       │    在 graph_edge_candidates 中创建:
       │      source=当前论文, target=KB中的论文,
       │      relation_type=从 ReferenceRoleAgent 获取 (builds_on/extends/compares_against)
       │    → 后续由 EdgeProfileAgent 生成 one_liner
       │
       ├─ 2. 更新 KB 论文的状态
       │    KB论文.downstream_count += 1
       │    如果 downstream_count >= 3 → is_established_baseline = true
       │    如果 downstream_count >= 5 → 进入 anchor_review
       │
       ├─ 3. 增加 staleness
       │    KB论文的 node_profile.staleness_trigger_count += 1
       │    相关 Method 节点的 staleness += 1
       │    → 达阈值后 Worker 自动刷新 profile
       │
       ├─ 4. 更新方法应用
       │    如果当前论文在新任务上用了 KB 论文的方法:
       │      创建 method_applications 记录
       │      (paper_id=当前论文, method_id=KB论文方法, role=adapted_baseline, task_id=新任务)
       │
       └─ 5. 对比信息注入报告
            在 PaperReportAgent 生成报告时:
            §8 lineage section 自动包含:
              "本文基于 [[P__已知论文]]，改了 reward_function slot..."
            §6 experiment section 自动包含:
              "与 [[P__已知论文]] 相比，在 VideoMME 上提升 3.2%"
```

### 具体代码修改建议

```python
# ingest_workflow.py → shallow_ingest() 中增加:

async def _handle_known_references(self, paper_id, ref_classifications):
    """处理引用了 KB 中已有论文的情况"""
    from backend.models.paper import Paper
    from backend.models.kb import GraphEdgeCandidate
    
    for ref_cls in ref_classifications:
        role = ref_cls.get("role", "background_citation")
        ref_title = ref_cls.get("ref_title")
        
        if role in ("direct_baseline", "method_source", "comparison_baseline"):
            # 检查是否已在 KB 中
            existing_paper = await self.session.execute(
                select(Paper).where(
                    func.lower(Paper.title).contains(ref_title.lower()[:50])
                ).limit(1)
            )
            kb_paper = existing_paper.scalar_one_or_none()
            
            if kb_paper:
                # ── 对比流程，不重新分析 ──
                
                # 1. 创建关系边候选
                edge_candidate = GraphEdgeCandidate(
                    paper_id=paper_id,
                    source_entity_type="paper",
                    source_entity_id=paper_id,
                    target_entity_type="paper",
                    target_entity_id=kb_paper.id,
                    relation_type=self._map_role_to_relation(role),
                    one_liner=ref_cls.get("reason", ""),
                    confidence_score=ref_cls.get("confidence", 0.7),
                    status="candidate",
                )
                self.session.add(edge_candidate)
                
                # 2. 更新下游计数
                if hasattr(kb_paper, 'cited_by_count'):
                    kb_paper.cited_by_count = (kb_paper.cited_by_count or 0) + 1
                
                # 3. staleness++
                from backend.services import node_profile_service
                await node_profile_service.increment_staleness(
                    self.session, "paper", kb_paper.id
                )
                
                # 4. 记录方法应用（如果是不同任务）
                # ... (省略)
    
    @staticmethod
    def _map_role_to_relation(role: str) -> str:
        return {
            "direct_baseline": "builds_on",
            "method_source": "extends",
            "comparison_baseline": "compares_against",
            "dataset_source": "evaluates_on",
        }.get(role, "cites")
```

### 对比 vs 重新分析的决策

```
                    发现引用论文 X
                         │
                ┌────────┴────────┐
                │                 │
          X 不在 KB            X 已在 KB
                │                 │
         import_and_score     对比流程:
         (可能递归分析)       ├─ 创建关系边
                              ├─ 更新下游计数
                              ├─ staleness++
                              ├─ 方法应用记录
                              └─ 对比信息注入报告

         不需要重新分析 X
         只需要建立"当前论文→X"的关系
```

---

## 总结: 4 个问题的修复优先级

| # | 问题 | 当前状态 | 影响 | 修复优先级 |
|---|------|---------|------|----------|
| 1 | 打分区分度 | 可接受，但 SourceSignal 不区分引用重要性 | **中** — 第一层筛选有噪声但不致命 | P2 |
| 2 | 检索质量 | arXiv 查询质量低，awesome 解析简陋，S2 intent 未利用 | **高** — 冷启动论文池质量直接受影响 | P1 |
| 3 | L0 论文无用 | 只用于去重，不可搜索，不参与图谱 | **中** — 浪费已获取的元数据 | P2 |
| 4 | 已知论文重遇无对比 | 发现重复直接跳过，不创建边/不更新计数 | **高** — 图谱无法增量构建关系 | P0 |
