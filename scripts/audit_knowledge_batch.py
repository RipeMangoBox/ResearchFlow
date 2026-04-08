import csv
import os
from typing import Dict, List, Tuple


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_ANALYSIS_DIR = os.path.join(ROOT_DIR, "paperAnalysis")
LOG_PATH = os.path.join(PAPER_ANALYSIS_DIR, "analysis_log.csv")


class AnalysisFile:
    def __init__(self, path: str, title: str, content: str, has_abstract: bool,
                 has_part_i: bool, has_part_ii: bool, has_part_iii: bool,
                 has_local_reading: bool, has_metrics_table: bool,
                 has_pdf_ref: bool):
        self.path = path
        self.title = title
        self.content = content
        self.has_abstract = has_abstract
        self.has_part_i = has_part_i
        self.has_part_ii = has_part_ii
        self.has_part_iii = has_part_iii
        self.has_local_reading = has_local_reading
        self.has_metrics_table = has_metrics_table
        self.has_pdf_ref = has_pdf_ref

    @property
    def is_emergent_style_basic_ok(self) -> bool:
        structural_ok = (
            self.has_abstract
            and self.has_part_i
            and self.has_part_ii
            and self.has_part_iii
            and self.has_local_reading
        )
        depth_ok = self.has_metrics_table
        meta_ok = self.has_pdf_ref
        return structural_ok and depth_ok and meta_ok


def cleanup_kb_suffixes() -> List[Tuple[str, str]]:
    renamed: List[Tuple[str, str]] = []
    for root, _dirs, files in os.walk(PAPER_ANALYSIS_DIR):
        for fname in files:
            if not fname.endswith("_KB.md"):
                continue
            old_path = os.path.join(root, fname)
            new_fname = fname[:-6] + ".md"
            new_path = os.path.join(root, new_fname)
            if os.path.exists(new_path):
                continue
            os.rename(old_path, new_path)
            renamed.append((old_path, new_path))
    return renamed


def parse_frontmatter_and_body(text: str) -> Tuple[Dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    frontmatter: Dict[str, str] = {}
    body_lines: List[str] = []
    in_front = True
    for idx, line in enumerate(lines[1:], start=1):
        if in_front and line.strip() == "---":
            body_lines = lines[idx + 1 :]
            break
        if in_front:
            if ":" in line:
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip()
        else:
            body_lines.append(line)

    if not body_lines:
        return frontmatter, text
    return frontmatter, "\n".join(body_lines)


def build_analysis_index() -> Dict[str, AnalysisFile]:
    index: Dict[str, AnalysisFile] = {}
    for root, _dirs, files in os.walk(PAPER_ANALYSIS_DIR):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            if fname in {"paper_analysis_check_task.txt", "analysis_log_updated.txt", "analysis_log.csv"}:
                continue

            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                continue

            front, body = parse_frontmatter_and_body(text)

            raw_title = front.get("title")
            if not raw_title:
                title_line = None
                for line in body.splitlines():
                    if line.startswith("# "):
                        title_line = line[2:].strip()
                        break
                if not title_line:
                    continue
                title = title_line
            else:
                title = raw_title.strip().strip('"').strip("'")

            lower_body = body.lower()

            has_abstract = "[!abstract" in lower_body
            has_part_i = "part i:" in lower_body
            has_part_ii = "part ii:" in lower_body
            has_part_iii = "part iii:" in lower_body
            has_local_reading = "local reading" in lower_body

            has_metrics_table = False
            for line in body.splitlines():
                if "|" in line and any(tok in line.lower() for tok in ["fid", "%", "accuracy", "top-1", "top-3"]):
                    has_metrics_table = True
                    break

            has_pdf_ref = any(
                ln.strip().startswith("pdf_ref:")
                for ln in text.splitlines()
            )

            index[title] = AnalysisFile(
                path=path,
                title=title,
                content=text,
                has_abstract=has_abstract,
                has_part_i=has_part_i,
                has_part_ii=has_part_ii,
                has_part_iii=has_part_iii,
                has_local_reading=has_local_reading,
                has_metrics_table=has_metrics_table,
                has_pdf_ref=has_pdf_ref,
            )
    return index


def load_log_rows() -> List[List[str]]:
    with open(LOG_PATH, "r", encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f)]


def save_log_rows(rows: List[List[str]]) -> None:
    with open(LOG_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def audit_batch(batch_size: int = 30) -> None:
    renamed_files = cleanup_kb_suffixes()
    index = build_analysis_index()
    rows = load_log_rows()

    data_start = 1 if rows and rows[0] and rows[0][0] == "state" else 0

    knowledge_indices: List[int] = []
    for i in range(data_start, len(rows)):
        row = (rows[i] + [""] * 8)[:8]
        status = row[0]
        if status == "Knowledge":
            knowledge_indices.append(i)

    target_indices = knowledge_indices[:batch_size]

    updated_entries: List[Tuple[str, str, str]] = []

    for idx in target_indices:
        row = (rows[idx] + [""] * 8)[:8]
        status, title = row[0], row[2]

        if status != "Knowledge":
            continue

        analysis = index.get(title)

        if analysis is None:
            new_status = "analysis_mismatch"
            row[0] = new_status
            rows[idx] = row
            updated_entries.append((status, new_status, title))
            continue

        if analysis.is_emergent_style_basic_ok:
            continue

        new_status = "analysis_mismatch"
        row[0] = new_status
        rows[idx] = row
        updated_entries.append((status, new_status, title))

    save_log_rows(rows)

    print("=== Paper Analysis Audit Batch Summary ===")
    print(f"Root: {ROOT_DIR}")
    print(f"Log: {LOG_PATH}")
    print()

    print(f"Renamed {_plural(len(renamed_files), 'file')}:")
    if renamed_files:
        for old, new in renamed_files:
            print(f"  - {old} -> {new}")
    else:
        print("  (none)")

    print()
    print(f"Updated {_plural(len(updated_entries), 'log entry')}:")
    if updated_entries:
        for old_status, new_status, title in updated_entries:
            print(f"  - \"{title}\": {old_status} -> {new_status}")
    else:
        print("  (none)")


def _plural(n: int, word: str) -> str:
    return f"{n} {word if n == 1 else word + 's'}"


if __name__ == "__main__":
    audit_batch(batch_size=30)
