# -*- coding: utf-8 -*-
"""
Apply renames from fuzzy_match_results.txt: rename .md to suggested_md_basename,
rename corresponding PDF to suggested_pdf_basename, and update pdf_ref in .md frontmatter.
By default only processes rows with need_review=N. Use --all to include need_review=Y, or --only-y to process only need_review=Y rows.
Run from repository root: python scripts/apply_fuzzy_rename.py [--dry-run] [--all | --only-y]
"""
import os
import re
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WINERROR_ALREADY_EXISTS = 183
PAPER_ANALYSIS = ROOT / "paperAnalysis"
RESULTS_PATH = PAPER_ANALYSIS / "fuzzy_match_results.txt"


MIN_SAFE_SCORE_FOR_Y = 0.80  # when processing need_review=Y, only apply if score >= this

def parse_results(include_review=False, only_y=False):
    rows = []
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 9:
                continue
            need_review = (parts[6].strip().upper() == "Y")
            if only_y and not need_review:
                continue
            if not only_y and need_review and not include_review:
                continue
            try:
                score = float(parts[5].strip())
            except Exception:
                score = 0.0
            rows.append({
                "log_title": parts[0],
                "matched_md_path": parts[3].strip(),
                "suggested_md_basename": parts[7].strip(),
                "suggested_pdf_basename": parts[8].strip(),
                "score": score,
            })
    return rows


def resolve_pdf_path(pdf_ref, root):
    if not pdf_ref:
        return None
    s = pdf_ref.strip()
    for prefix in (f"{root.name}/", f"{root.name}\\"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    p = root / s.replace("/", "\\")
    return p if p.suffix.lower() == ".pdf" else p


def main():
    dry_run = "--dry-run" in sys.argv
    include_review = "--all" in sys.argv
    only_y = "--only-y" in sys.argv
    rows = parse_results(include_review=include_review, only_y=only_y)
    done_md = []
    done_pdf = []
    skipped = []
    skipped_pdf_exists = []
    errors = []

    for r in rows:
        if only_y and r.get("score", 0) < MIN_SAFE_SCORE_FOR_Y:
            skipped.append((r["log_title"], f"need_review=Y and score {r.get('score', 0):.2f} < {MIN_SAFE_SCORE_FOR_Y} (skip to avoid wrong match)"))
            continue
        raw_path = r["matched_md_path"].strip().replace("/", os.sep)
        md_path = ROOT / raw_path
        if not md_path.is_file():
            alt_path = md_path.parent / r["suggested_md_basename"].strip()
            if alt_path.is_file():
                md_path = alt_path
            else:
                skipped.append((r["log_title"], "md not found: " + r["matched_md_path"]))
                continue
        new_md_name = r["suggested_md_basename"]
        new_pdf_name = r["suggested_pdf_basename"]
        md_dir = md_path.parent
        new_md_path = md_dir / new_md_name

        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            errors.append((r["log_title"], "read md: " + str(e)))
            continue

        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
        if not m:
            errors.append((r["log_title"], "no frontmatter"))
            continue
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception as e:
            errors.append((r["log_title"], "yaml: " + str(e)))
            continue

        pdf_ref = fm.get("pdf_ref") or ""
        pdf_path = resolve_pdf_path(pdf_ref, ROOT) if pdf_ref else None

        if pdf_path and pdf_path.exists():
            pdf_dir = pdf_path.parent
            new_pdf_path = pdf_dir / new_pdf_name
            if new_pdf_path.resolve() != pdf_path.resolve():
                if dry_run:
                    done_pdf.append((str(pdf_path.relative_to(ROOT)), str(new_pdf_path.relative_to(ROOT))))
                else:
                    try:
                        pdf_path.rename(new_pdf_path)
                        done_pdf.append((str(pdf_path.relative_to(ROOT)), str(new_pdf_path.relative_to(ROOT))))
                    except OSError as e:
                        if getattr(e, "winerror", None) == WINERROR_ALREADY_EXISTS or (e.errno == os.errno.EEXIST if hasattr(os, "errno") else False):
                            skipped_pdf_exists.append((r["log_title"], str(new_pdf_path.relative_to(ROOT))))
                        else:
                            errors.append((r["log_title"], "rename pdf: " + str(e)))
                            continue
                    except Exception as e:
                        errors.append((r["log_title"], "rename pdf: " + str(e)))
                        continue
                new_ref = str(new_pdf_path.relative_to(ROOT)).replace("\\", "/")
                fm["pdf_ref"] = new_ref
                new_fm_str = "---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False) + "---"
                body = text[m.end():].lstrip()
                text = new_fm_str + "\n\n" + body

        if new_md_path.resolve() != md_path.resolve():
            if dry_run:
                done_md.append((str(md_path.relative_to(ROOT)), str(new_md_path.relative_to(ROOT))))
            else:
                try:
                    new_md_path.write_text(text, encoding="utf-8")
                    md_path.unlink()
                    done_md.append((str(md_path.relative_to(ROOT)), str(new_md_path.relative_to(ROOT))))
                except Exception as e:
                    errors.append((r["log_title"], "rename md: " + str(e)))

    report = []
    report.append("=== Apply fuzzy rename report ===")
    report.append(f"Processed: {len(rows)}")
    report.append(f"MD renames: {len(done_md)}")
    report.append(f"PDF renames: {len(done_pdf)}")
    report.append(f"Skipped (md not found): {len(skipped)}")
    report.append(f"Skipped (pdf target exists): {len(skipped_pdf_exists)}")
    report.append(f"Errors: {len(errors)}")
    if dry_run:
        report.append("(dry-run: no files changed)")
    report.append("")
    report.append("--- MD renames ---")
    for a, b in done_md:
        report.append(f"  {a} -> {b}")
    report.append("")
    report.append("--- PDF renames ---")
    for a, b in done_pdf:
        report.append(f"  {a} -> {b}")
    report.append("")
    report.append("--- Skipped ---")
    for title, reason in skipped[:20]:
        report.append(f"  {title}: {reason}")
    if len(skipped) > 20:
        report.append(f"  ... and {len(skipped) - 20} more")
    report.append("")
    report.append("--- Skipped (pdf target already exists) ---")
    for title, path in skipped_pdf_exists[:15]:
        report.append(f"  {title}: {path}")
    if len(skipped_pdf_exists) > 15:
        report.append(f"  ... and {len(skipped_pdf_exists) - 15} more")
    report.append("")
    report.append("--- Errors ---")
    for title, reason in errors:
        report.append(f"  {title}: {reason}")

    out_path = PAPER_ANALYSIS / "apply_fuzzy_rename_report.txt"
    result = "\n".join(report)
    out_path.write_text(result, encoding="utf-8")
    print(f"Report: {out_path.relative_to(ROOT)}")
    print(f"MD: {len(done_md)}, PDF: {len(done_pdf)}, Skipped: {len(skipped)}, PDF-exists: {len(skipped_pdf_exists)}, Errors: {len(errors)}")
    if errors:
        for t, r in errors:
            print(f"  ERROR {t}: {r}")


if __name__ == "__main__":
    main()
