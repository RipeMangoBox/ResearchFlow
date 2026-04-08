import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[5]
ANALYSIS_LOG_PATH = REPO_ROOT / "paperAnalysis" / "analysis_log.csv"
PAPER_LIST_PATH = REPO_ROOT / "paperPDFs" / "paper_list.txt"
PDF_ROOT = REPO_ROOT / "paperPDFs"
LOG_HEADER = [
    "state",
    "importance",
    "paper_title",
    "venue",
    "project_link_or_github_link",
    "paper_link",
    "sort",
    "pdf_path",
]


def normalize_category(category: str) -> str:
    if not category:
        return ""
    t = category.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", "_", t).strip("_")
    return t


def normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_venue(venue: str) -> Tuple[str, str]:
    tokens = venue.strip().split()
    if not tokens:
        return "", ""
    year = tokens[-1]
    conf_name = "_".join(tokens[:-1]) if len(tokens) > 1 else tokens[0]
    return conf_name, year


def load_paper_list() -> Dict[Tuple[str, str, str, str], str]:
    mapping: Dict[Tuple[str, str, str, str], str] = {}
    if not PAPER_LIST_PATH.exists():
        raise FileNotFoundError(f"paper_list not found: {PAPER_LIST_PATH}")

    with PAPER_LIST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            category_raw = parts[0].strip()
            conf_year = parts[1].strip()
            filename = parts[2].strip()

            if not filename.lower().endswith(".pdf"):
                continue

            conf_tokens = conf_year.split("_")
            if not conf_tokens:
                continue
            year = conf_tokens[-1]
            conf_name = "_".join(conf_tokens[:-1]) if len(conf_tokens) > 1 else conf_tokens[0]

            name_no_ext = filename[:-4]
            if len(name_no_ext) > 5 and name_no_ext[:4].isdigit() and name_no_ext[4] == "_":
                title_part = name_no_ext[5:]
            else:
                title_part = name_no_ext

            title_norm = normalize_title(title_part.replace("_", " "))
            norm_category = normalize_category(category_raw)
            key = (norm_category, conf_name, year, title_norm)
            rel_path = os.path.join(category_raw, conf_year, filename)
            mapping[key] = rel_path

    return mapping


def load_rows() -> List[List[str]]:
    if not ANALYSIS_LOG_PATH.exists():
        raise FileNotFoundError(f"analysis_log.csv not found: {ANALYSIS_LOG_PATH}")

    with ANALYSIS_LOG_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.reader(f)]

    if not rows:
        return [LOG_HEADER]

    if rows[0][:8] != LOG_HEADER:
        rows.insert(0, LOG_HEADER)

    return rows


def save_rows(rows: List[List[str]]) -> None:
    with ANALYSIS_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            out = (row + [""] * 8)[:8]
            writer.writerow(out)


def update_analysis_log() -> None:
    mapping = load_paper_list()
    rows = load_rows()

    count_checked = 0
    count_wait = 0

    for i in range(1, len(rows)):
        row = (rows[i] + [""] * 8)[:8]
        status, _, title, venue, _, _, category_raw, pdf_path = row

        title = title.strip()
        venue = venue.strip()
        category_raw = category_raw.strip()
        pdf_path = pdf_path.strip()

        rel_pdf_path = ""
        if pdf_path.startswith("paperPDFs/"):
            rel_pdf_path = pdf_path.replace("paperPDFs/", "", 1)
        else:
            conf_name, year = parse_venue(venue)
            if conf_name and year:
                key = (normalize_category(category_raw), conf_name, year, normalize_title(title))
                rel_pdf_path = mapping.get(key, "")

        found = False
        if rel_pdf_path:
            abs_pdf = PDF_ROOT / rel_pdf_path
            found = abs_pdf.exists()

        if not found:
            row[0] = "Wait"
            count_wait += 1
        else:
            row[0] = status.strip() or "checked"
            count_checked += 1

        rows[i] = row

    save_rows(rows)
    print(f"Summary: checked={count_checked}, Wait={count_wait}")


if __name__ == "__main__":
    update_analysis_log()
