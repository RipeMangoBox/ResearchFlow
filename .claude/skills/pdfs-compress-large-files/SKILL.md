---
name: pdfs-compress-large-files
description: Checks and compresses all PDFs larger than 20 MB under `paperPDFs/` using Ghostscript, preserving high quality. Use when the user asks to compress PDFs, reduce PDF sizes, or prepare paperPDFs for analysis. Runs a colocated script that scans `paperPDFs/`, compresses in-place with /ebook settings (fallback to /screen), and prints a summary report.
---

# Compress Large PDFs in paperPDFs

Scan `paperPDFs/` for PDFs > 20 MB and compress them in-place using Ghostscript.

## Quick Start

```bash
python3 .claude/skills/pdfs-compress-large-files/scripts/compress_large_pdfs.py
```

Optional flags:
```bash
# Dry-run: show what would be compressed without changing files
python3 .claude/skills/pdfs-compress-large-files/scripts/compress_large_pdfs.py --dry-run

# Custom size threshold (default: 20 MB)
python3 .claude/skills/pdfs-compress-large-files/scripts/compress_large_pdfs.py --threshold 15

# Target a specific subdirectory
python3 .claude/skills/pdfs-compress-large-files/scripts/compress_large_pdfs.py --dir paperPDFs/Motion_Generation_Text_Speech_Music_Driven
```

## What the Script Does

1. Finds all `.pdf` files under `paperPDFs/` larger than `--threshold` MB
2. Compresses each with Ghostscript `/ebook` preset (150 dpi images, high text quality)
3. If result is still > threshold, retries with `/screen` preset
4. Replaces the original only if the compressed file is smaller
5. Prints a summary: original size → compressed size, savings %

## Ghostscript Presets Used

| Preset | Image DPI | Use case |
|--------|-----------|----------|
| `/ebook` | 150 dpi | Default — high quality, good compression |
| `/screen` | 72 dpi | Fallback — smaller file, lower image quality |

## Requirements

- `ghostscript` must be installed: `sudo apt install ghostscript` or `brew install ghostscript`
- Python 3.6+

## When to Use

- Before running `papers-analyze-pdf` on a batch with large PDFs
- After downloading new PDFs via `papers-download-from-list`
- Periodically to keep `paperPDFs/` storage lean
