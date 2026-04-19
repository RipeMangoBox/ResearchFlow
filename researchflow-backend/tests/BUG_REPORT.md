# Bug Report — Video QA Cold Start Test

**Date**: 2026-04-20
**Test**: 15 Video QA papers, full pipeline cold start

---

## Critical Bugs (P0)

### BUG-1: API OOM Kill on pipeline/run (FIXED during test)

**Symptom**: API 容器在处理 `pipeline/{id}/run` 时被 OOM kill，curl 返回空响应
**Root Cause**: `pipeline/run` 是同步端点，在单次请求中执行 PDF 下载 + PyMuPDF 解析 + 图片渲染 + 多次 VLM API 调用，峰值内存超过容器限制
**Original limit**: 512MB (太小)
**Fix**: 临时增加到 2048MB
**Status**: ✅ 临时修复。**长期应改为异步**: 触发后返回 job_id，实际处理在 worker 中

### BUG-2: L4 Deep Report 为空 (3/9 papers, JSON truncation)

**Symptom**: `full_report_md` 只有标题 (76-86 chars), problem_summary/method_summary/core_intuition 全空
**Root Cause**: 代理 API (apicursor.com) streaming 模式偶尔截断响应在 ~2800 chars，`_repair_truncated_json()` 修复了部分但不是全部
**Evidence**: 日志显示 `JSON parse failed (len=2821)` 但内容中确实有 problem_summary/method_summary 字段
**Status**: ⚠️ 部分修复。6/9 成功 (67%)，需进一步改进 JSON repair 或换更稳定的 API

---

## Major Bugs (P1)

### BUG-3: L3 Skim 数据为空 (6/12 papers)

**Symptom**: `worth_deep_read`, `is_plugin_patch`, `problem_summary`, `method_summary` 全部为 NULL
**Root Cause**: 早期 pipeline 在 API OOM 前处理的论文，skim 步骤部分完成但结果不完整；状态被更新为 l3_skimmed 但数据为空
**Affected**: 2512.17229, 2601.02536, 2602.21137, 2603.04349, 2604.01824, 2604.01966
**Status**: ⚠️ 需要对这些论文重跑 skim + deep

### BUG-4: 3 篇论文 Parse 失败，停在 downloaded 状态

**Symptom**: papers 2601.07459, 2602.08448, 2603.14953 PDF 下载成功但 parse 阶段失败
**Root Cause**: 未确认 — 可能是 OOM kill 中断了处理，或 PDF 格式问题
**Status**: ⚠️ 需要重跑

### BUG-5: paperAnalysis 目录不存在导致自动导出失败

**Symptom**: `Auto-export to paperAnalysis/ failed: No such file or directory: '../paperAnalysis/VideoQA'`
**Root Cause**: compose 中挂载了 `./paperAnalysis:/paperAnalysis` 但服务器目录不存在
**Fix**: `mkdir -p /opt/researchflow/researchflow-backend/paperAnalysis/VideoQA`
**Status**: ❌ 未修复

---

## Minor Bugs (P2)

### BUG-6: arXiv API 429 Rate Limit 导致 5/15 论文 title 缺失

**Symptom**: 5 篇论文 title 还是 arXiv ID 占位符，authors 也为空
**Root Cause**: 批量 enrich 时 arXiv API 返回 429，当前无退避重试
**Workaround**: 第二次 enrich 时部分补全了
**Status**: P2 — 重新 enrich 即可

### BUG-7: VLM Figure Classify 仍偶尔 400

**Symptom**: `VLM classify+detect failed: Request too large`
**Root Cause**: 虽然限制了 15 candidates + 5 pages，但某些 PDF 图片尺寸大，总 payload 仍超限
**Status**: P2 — 可进一步降低图片分辨率或减少 batch size

### BUG-8: 1 篇论文公式提取为 0

**Symptom**: 2604.01966 (Ego-Grounding) formula_count=0
**Root Cause**: VLM page scan 未检测到公式 — 可能论文确实公式很少，或数学符号密度排序错误
**Status**: P2 — 需要人工核实 PDF

### BUG-9: pipeline/batch 端点 500 错误 (MissingGreenlet)

**Symptom**: `POST /api/v1/pipeline/batch` 返回 Internal Server Error
**Root Cause**: `sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been called` — async 上下文中同步 IO
**Status**: ❌ 未修复

### BUG-10: S2 API recommendations 404

**Symptom**: `Client error '404 Not Found' for url .../recommendations`
**Root Cause**: Semantic Scholar 某些论文没有 recommendations 数据
**Impact**: 不影响主流程，discover_citations 返回空
**Status**: Known limitation, not a bug

---

## Discovered During Test (Fixed)

| # | Bug | Fix | Commit |
|---|-----|-----|--------|
| F1 | API 容器内存 512MB 不够 pipeline/run | 增加到 2048MB | docker-compose.yml |
| F2 | Worker 内存调整 | 1536MB (从 2048 降以腾出给 api) | docker-compose.yml |
| F3 | PyMuPDF 文本含 `\x00` null byte → PG 拒绝写入 | parse_service.py 加 `_clean_text()` | parse_service.py |
| F4 | `prompt_version` varchar(20) 溢出 (formula_page_scan_v1=21 chars) | ALTER TABLE 扩到 varchar(50) | DB migration needed |
| F5 | 代理 API 偶尔返回 Claude 自我介绍而非 completion | 无法修复 (第三方问题)，重试可解决 | - |

---

## Recommendations

1. **P0**: 把 `pipeline/run` 改成异步 (enqueue to worker)
2. **P0**: 改进 JSON repair — 对代理 API 截断更鲁棒
3. **P1**: 添加 arXiv API retry with exponential backoff
4. **P1**: 创建 `paperAnalysis/` 目录结构在启动时自动创建
5. **P2**: VLM figure classify 进一步限制图片尺寸（max 500px width）
