# Paper State Convention

`analysis_log.csv` 中 `state` 列的统一定义。所有 skill 必须遵循此约定。

## 主流程状态

```
Wait → Downloaded → checked
```

| state | 含义 | 由哪个阶段写入 |
|-------|------|--------------|
| `Wait` | 新收集的候选，等待下载 | collect（from-web / from-github-awesome） |
| `Downloaded` | PDF 已下载到 `paperPDFs/`，等待分析 | download（papers-download-from-list） |
| `checked` | 已完成结构化分析，`paperAnalysis/` 中有对应 `.md` | analyze（papers-analyze-pdf） |

## 非主流程状态

| state | 含义 | 说明 |
|-------|------|------|
| `Skip` | 人工筛选后决定不处理 | 不进入主流程，保留在 log 中方便后续回顾 |
| `Missing` | 多次下载尝试后仍无法获取 PDF | 保留在 log 中，后续可手动补充或重试 |

## 规则

1. 每条记录的 state 只能沿主流程单向推进：`Wait → Downloaded → checked`
2. `Skip` 和 `Missing` 可从 `Wait` 转入，不可逆转回 `Wait`（除非用户手动修改）
3. 下载时自动压缩超大 PDF（>20MB）
4. 所有 skill 读取 log 时，只处理对应阶段的 state：
   - download 只处理 `Wait`
   - analyze 只处理 `Downloaded`
   - build-collection-index 只处理 `checked`

## 字段 Fallback 值

| 字段 | Fallback | 说明 |
|------|---------|------|
| venue | `arXiv YYYY` | 未被会议收录但在 arXiv 等公开平台发表，如 `arXiv 2025` |
| project_link_or_github_link | `N/A` | 确认无开源代码或项目页 |
