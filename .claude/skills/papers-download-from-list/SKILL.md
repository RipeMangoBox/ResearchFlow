---
name: papers-download-from-list
description: Uses colocated paper download tools to download, verify, and repair local PDFs according to triage logs (e.g., ICLR_2026.txt) so that `paperPDFs/` stays complete and deduplicated. Use when you have a curated candidate list under `paperAnalysis/` and want all corresponding PDFs downloaded/checked before analysis or collection rebuild.
---

# Paper PDF Download & Repair Tools

## What this skill does

Connects **online triage lists** to **local PDFs** by orchestrating the utilities in the colocated paper download tools directory:

- 从 `paperAnalysis/*.txt`（如 `ICLR_2026.txt`）中读取候选论文列表；
- 下载缺失 PDF 到 `paperPDFs/...`；
- 纠正常见下载错误（错误链接、错下版本）；
- 去重、检查下载完整性，并打标“缺失但待手动处理”的条目。

它是 “**papers-collect-from-web → papers-download-from-list → papers-analyze-pdf → papers-build-collection-index → papers-query-knowledge-base / research-brainstorm-from-kb**” 流水线里，负责 **“下载（Download）+ 修复（Repair）”** 的一环。

## Directory & scripts

脚本目录（相对当前 skill 目录）

- `paper_download_tools/check_paper_downloads.py`
- `paper_download_tools/check_pdfs_against_log.py`
- `paper_download_tools/download_wait_papers.py`
- `paper_download_tools/redownload_correct_pdfs.py`
- `paper_download_tools/fix_wrong_downloads_from_log.py`
- `paper_download_tools/mark_missing_wait.py`
- `paper_download_tools/dedupe_paperpdfs.py`

## Recommended workflow

典型使用顺序（从“候选列表”到“干净的 PDF 集合”）：

1. **前置：有 triage 列表**
   - `paperAnalysis/*.txt` triage 文件
     - `状态 | title | 会议&时间 | paper link | project/github link | 文章分类`

2. **批量下载 Wait 条目**
   - 使用（示例命令）：
   - ```bash
     python3 ".claude/skills/papers-download-from-list/scripts/paper_download_tools/download_wait_papers.py" \
       --log "paperAnalysis/ICLR_2026.txt" \
       --out-root "paperPDFs"
     ```
   - 行为：
     - 读取 `状态=Wait` 的行；
     - 按 `文章分类`/`会议&时间` 推断/创建目标子目录；
     - 下载 PDF 到对应 `paperPDFs/...` 路径；
     - 下载成功后自动压缩超大 PDF（>20MB），使用 Ghostscript `/ebook` 设置（失败则 `/screen`）；
     - 在 log 中更新状态为 `Downloaded`。

3. **检查缺失与坏文件**
   - ```bash
     python3 ".claude/skills/papers-download-from-list/scripts/paper_download_tools/check_paper_downloads.py" \
       --log "paperAnalysis/ICLR_2026.txt" \
       --pdf-root "paperPDFs"
     ```
   - ```bash
     python3 ".claude/skills/papers-download-from-list/scripts/paper_download_tools/check_pdfs_against_log.py" \
       --log "paperAnalysis/ICLR_2026.txt" \
       --pdf-root "paperPDFs"
     ```
   - 行为：
     - 报告 log 中有链接但本地缺 PDF 的条目；
     - 报告已下载但尺寸异常/打不开的 PDF，供后续重下。

4. **重下错误或错配的 PDF**
   - ```bash
     python3 ".claude/skills/papers-download-from-list/scripts/paper_download_tools/fix_wrong_downloads_from_log.py" \
       --log "paperAnalysis/ICLR_2026.txt" \
       --pdf-root "paperPDFs"
     ```
   - ```bash
     python3 ".claude/skills/papers-download-from-list/scripts/paper_download_tools/redownload_correct_pdfs.py" \
       --log "paperAnalysis/ICLR_2026.txt" \
       --pdf-root "paperPDFs"
     ```
   - 行为：
     - 根据 log 中的原始链接 / 修正链接，重新下载有问题的条目；
     - 更新 log 状态，标记为已修复。

5. **标记长期缺失、避免反复尝试**
   - ```bash
     python3 ".claude/skills/papers-download-from-list/scripts/paper_download_tools/mark_missing_wait.py" \
       --log "paperAnalysis/ICLR_2026.txt"
     ```
   - 行为：
     - 对多次尝试后仍拿不到 PDF 的条目，将状态更新为 `Missing`，保留在 log 中方便后续手动补充或重试。

6. **去重与清理**
   - ```bash
     python3 ".claude/skills/papers-download-from-list/scripts/paper_download_tools/dedupe_paperpdfs.py" \
       --root "paperPDFs"
     ```
   - 行为：
     - 查找同一论文的重复下载（按 hash / 文件名 / log 信息）；
     - 合并/保留单份 PDF，并在 log 中统一引用。

## When to use vs other skills

- **爬取（Collect）**：使用 `papers-collect-from-web` 从会议/专题页面抽取候选论文与链接。
- **下载（Download）**：使用本 `papers-download-from-list` skill，将 triage 列表中的链接转为本地 `paperPDFs/`。
- **分析（Analyze）**：使用 `papers-analyze-pdf` 把 PDF 解析为结构化 `paperAnalysis/*.md`。
- **分类与索引（Classify / Index）**：使用 `papers-build-collection-index` 重建 `paperCollection` 索引。
- **整合与查询（Integrate / Query）**：使用 `papers-query-knowledge-base` 在整个库上检索、对比、引用。
- **反思与选题（Reflect / Ideate）**：使用 `research-brainstorm-from-kb` 把以上成果沉淀为 `paperIDEAs/` 里的研究想法与路线。

## Triggers (examples)

- “这份 `ICLR_2026.txt` 里的论文帮我都下好 PDF。”  
- “检查一下哪些候选论文还没成功下载，并修复错误下载。”  
- “清理一下 `paperPDFs/` 里重复或坏掉的 PDF，再继续做分析。”  

