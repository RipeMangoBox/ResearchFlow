# Video QA Cold Start Quality Report

**Date**: 2026-04-20
**Domain**: Video Question Answering (2025-2026)
**Papers**: 15 imported, 12 parsed, 9 deep-analyzed
**Pipeline**: VLM page scan (no GROBID), S2 API fallback

---

## 1. Collection & Enrichment (15/15)

| Metric | Result | Grade |
|--------|--------|-------|
| Papers imported | 15/15 | A |
| Title resolved | 10/15 (5 placeholder due to arXiv 429) | B |
| Abstract filled | 15/15 | A |
| Authors filled | 10/15 (same 5 missing) | B |
| Year extracted | 15/15 | A |
| Venue resolved | 1 CVPR + 3 Open MIND + 11 arXiv | B |
| OpenReview check | Ran for all, found 1 acceptance (CVPR) | B |
| DBLP check | Ran for all | OK |

**Issues**:
- arXiv API rate-limited (429) on 5 papers — titles/authors stayed as placeholders
- S2 API rate-limited — needed 3s sleep between requests
- Venue mostly "arXiv (Cornell University)" — expected for 2026 preprints

---

## 2. L2 Parse Quality (12/15 papers parsed)

| arxiv_id | Sections | Formulas | Fig Caps | Fig Images | Tables | Grade |
|----------|----------|----------|----------|------------|--------|-------|
| 2512.17229 | 8 | 8 | 6 | 6 | 3 | A |
| 2601.02536 | 7 | 6 | 2 | 2 | 11 | A |
| 2602.21137 | 7 | 5 | 26 | 8 | 14 | A |
| 2603.04349 | 6 | 27 | 4 | 3 | 5 | A |
| 2603.09827 | 6 | 6 | 12 | 8 | 5 | A |
| 2603.12533 | 8 | 4 | 21 | 8 | 12 | A |
| 2603.18558 | 8 | 12 | 7 | 4 | 10 | A |
| 2603.18850 | 8 | 27 | 4 | 2 | 7 | A |
| 2603.19481 | 8 | 8 | 4 | 4 | 4 | A |
| 2603.29962 | 6 | 24 | 17 | 4 | 12 | A |
| 2604.01824 | 8 | 7 | 4 | 3 | 5 | A |
| 2604.01966 | 7 | **0** | 11 | 6 | 10 | **C** |

**Average**: 6.7 sections, 11.2 formulas, 10.2 fig captions, 4.8 fig images, 8.2 tables

### Formula Quality (Spot Check)

**2603.04349 (FocusGraph)**: LaTeX 正确
- `\mathbf{E} = [\mathbf{E}_1^{\text{text}}, \mathbf{z}_1^{\text{cap}}, ...]` — 输入序列定义 ✅
- `\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^T \log p_\theta(y_t|y_{<t}, x)` — SFT loss ✅
- `r_s^{t+1} = \frac{c_j^{t+1}}{n_j^t}` — 分数更新规则 ✅

**2603.29962 (SurgTEMP)**: LaTeX 正确
- `E_V : \mathbb{R}^{T \times C \times W \times H} \to \mathbb{R}^{T \times M \times D_V}` — 视觉编码器签名 ✅

**2603.18850 (HORNet)**: LaTeX 正确但偏简单
- `V = \{v_1, v_2, ..., v_T\}`, `F \in \mathbb{R}^{T \times D}` — 基础符号定义

**Issues**:
- 2604.01966 (Ego-Grounding) 公式数=0 — VLM page scan 未检测到公式（可能论文确实公式少或 VLM 遗漏）
- GROBID 全部显示 false — 符合预期（已移除）

### Figure Extraction Quality

- 平均每篇 4.8 张图片，最多 8 张（受 cap 限制）
- Caption 提取平均 10.2 个，某些论文很高（2602.21137 有 26 个 caption）
- 图片数 < caption 数是正常的（PyMuPDF 提取的嵌入图片有限，但文本 caption 可以提取更多）

---

## 3. L3 Skim Quality (12/15 papers skimmed)

| arxiv_id | worth_deep | is_plugin | problem_summary | Grade |
|----------|-----------|-----------|-----------------|-------|
| 2512.17229 | NULL | NULL | NULL | **D** |
| 2601.02536 | NULL | NULL | NULL | **D** |
| 2602.21137 | NULL | NULL | NULL | **D** |
| 2603.04349 | NULL | NULL | NULL | **D** |
| 2603.09827 | true | false | 多具身智能体视频理解 | A |
| 2603.12533 | true | true | 手势指向的第一人称视频QA | A |
| 2603.18558 | true | false | 长视频关键帧选择效率与准确性 | A |
| 2603.18850 | true | false | 视频QA帧选择问题 | A |
| 2603.19481 | true | false | 长视频叙事推理能力不足 | A |
| 2603.29962 | true | false | 手术视频QA时间语义建模 | A |
| 2604.01824 | NULL | NULL | NULL | **D** |
| 2604.01966 | NULL | NULL | NULL | **D** |

**Pass Rate**: 6/12 (50%) 有完整 L3 数据
**Issue**: 早期处理的论文（OOM 之前）L3 数据为空 — skim 返回不完整

---

## 4. L4 Deep Quality (9/15 papers deep-analyzed)

| arxiv_id | Report Len | problem_summary | core_intuition | delta_cards | assertions | Grade |
|----------|-----------|-----------------|----------------|-------------|------------|-------|
| 2512.17229 | 1234 | ✅ 长视频QA上下文限制 | ✅ 模仿人类认知过程 | 1 | 7 | A |
| 2601.02536 | **76** | ❌ empty | ❌ empty | 0 | 0 | **F** |
| 2603.04349 | 1311 | ✅ 长视频MLLM性能退化 | ✅ 帧选择从视觉域到语义域 | 1 | 7 | A |
| 2603.09827 | 1062 | ✅ 多智能体视频理解 | ✅ 共享记忆+智能体检索 | 1 | 7 | A |
| 2603.12533 | **86** | ❌ empty | ❌ empty | 0 | 0 | **F** |
| 2603.18558 | **82** | ❌ empty | ❌ empty | 0 | 0 | **F** |
| 2603.18850 | 1244 | ✅ 视频QA帧选择效率 | ✅ 任务导向帧选择 | 1 | 7 | A |
| 2603.19481 | 1076 | ✅ 叙事推理能力 | ✅ 从顺序到叙事结构化 | 1 | 7 | A |
| 2603.29962 | 1085 | ✅ 手术视频时间语义 | ✅ 文本查询引导+层次记忆 | 1 | 7 | A |

**Full Success**: 6/9 (67%) 有完整 L4 报告 + delta card + assertions
**Partial Failure**: 3/9 (33%) 报告只有标题 — JSON truncation repair 未完全修复

### L4 Report Quality (Spot Check: 2603.04349 FocusGraph)

- **problem_summary**: "长视频问答面临的核心挑战是多模态大语言模型在处理长视频时的性能退化" ✅ 准确
- **core_intuition**: "将帧选择问题从视觉域转移到语义域" ✅ 抓住核心
- **delta_card**: 1 个 ✅
- **assertions**: 7 个 (method/baseline/mechanism 类型) ✅

---

## 5. Venue Resolution Quality

| Paper | Detected Venue | Actual | Grade |
|-------|---------------|--------|-------|
| 2603.12533 | **CVPR** | CVPR 2026 | **A** |
| Others | arXiv / Open MIND | Preprint | OK |

**OpenReview search**: 成功检测到 CVPR acceptance
**DBLP search**: 运行但 2026 论文尚未收录

---

## Summary

| Pipeline Step | Success Rate | Key Issue |
|---------------|-------------|-----------|
| **Import** | 15/15 (100%) | - |
| **Enrich** | 15/15 (100%) | arXiv 429 导致 5 篇 title 缺失 |
| **Download PDF** | 12/15 (80%) | 3 篇下载失败或 parse 阶段 OOM |
| **L2 Parse** | 12/12 (100%) | 1 篇 formula=0 |
| **VLM Formulas** | 11/12 (92%) | 平均 11.2 个/篇，质量高 |
| **L3 Skim** | 6/12 (50%) | 早期 OOM 导致数据不完整 |
| **L4 Deep** | 6/9 (67%) | JSON truncation 导致 3 篇报告为空 |
| **Delta Cards** | 6/9 (67%) | 与 L4 deep 成功率一致 |
| **Venue Resolve** | 1/15 conference detected | 正确检测 CVPR |
