import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ANALYSIS_DIR = REPO_ROOT / "paperAnalysis"
LOG_PATH = PAPER_ANALYSIS_DIR / "analysis_log.csv"


PART_PATTERNS = {
    "I": re.compile(r"(?i)\bPart\s*I\b"),
    "II": re.compile(r"(?i)\bPart\s*II\b"),
    "III": re.compile(r"(?i)\bPart\s*III\b"),
}


def _norm_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _extract_md_title(md_text: str, md_path: Path) -> str:
    if md_text.startswith("---"):
        end = md_text.find("\n---", 3)
        if end != -1:
            fm = md_text[3:end]
            m = re.search(r"(?im)^title\s*:\s*(.+)$", fm)
            if m:
                return m.group(1).strip().strip("\"'").strip()
    stem = re.sub(r"^\d{4}_", "", md_path.stem)
    return stem.replace("_", " ").strip()


def _md_missing_parts(md_text: str) -> list[str]:
    missing = []
    for k, rx in PART_PATTERNS.items():
        if not rx.search(md_text):
            missing.append(k)
    return missing


@dataclass
class LogEntry:
    row_idx: int
    row: list[str]

    @property
    def status(self) -> str:
        return self.row[0] if self.row else ""

    @property
    def title(self) -> str:
        return self.row[2] if len(self.row) > 2 else ""

    def set_status(self, status: str) -> None:
        if self.row:
            self.row[0] = status


def _parse_log_rows(rows: list[list[str]]) -> list[LogEntry]:
    entries: list[LogEntry] = []
    data_start = 1 if rows and rows[0] and rows[0][0] == "state" else 0
    for i in range(data_start, len(rows)):
        row = (rows[i] + [""] * 8)[:8]
        if not row[2].strip():
            continue
        entries.append(LogEntry(i, row))
    return entries


def _best_log_match(entries: list[LogEntry], md_title: str) -> tuple[float, LogEntry] | None:
    nmd = _norm_title(md_title)
    best: tuple[float, LogEntry] | None = None
    for e in entries:
        nt = _norm_title(e.title)
        if nmd and (nmd in nt or nt in nmd):
            score = 0.92
        else:
            score = SequenceMatcher(None, nmd, nt).ratio()
        if best is None or score > best[0]:
            best = (score, e)
    if best and best[0] >= 0.78:
        return best
    return None


def main() -> int:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing log file: {LOG_PATH}")

    md_files: list[Path] = []
    for p in PAPER_ANALYSIS_DIR.rglob("*.md"):
        if "emergentmind_paper_analysis" in str(p).lower():
            continue
        md_files.append(p)

    missing_mds: list[tuple[Path, list[str], str]] = []
    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="utf-8", errors="ignore")
        miss = _md_missing_parts(text)
        if miss:
            title_guess = _extract_md_title(text, p)
            missing_mds.append((p, miss, title_guess))

    with LOG_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f)]

    entries = _parse_log_rows(rows)

    updates: list[tuple[int, str, str, str, list[str], str, str, float]] = []
    for md_path, miss, title_guess in missing_mds:
        match = _best_log_match(entries, title_guess)
        if not match:
            continue
        score, entry = match
        old = entry.status
        if old != "Wait":
            entry.set_status("Wait")
            updates.append(
                (
                    entry.row_idx,
                    old,
                    "Wait",
                    str(md_path.relative_to(REPO_ROOT)),
                    miss,
                    title_guess,
                    entry.title,
                    score,
                )
            )

    md_by_best_entry: dict[int, tuple[Path, str]] = {}
    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="utf-8", errors="ignore")
        title_guess = _extract_md_title(text, p)
        match = _best_log_match(entries, title_guess)
        if not match:
            continue
        score, entry = match
        if score < 0.85:
            continue
        md_by_best_entry[entry.row_idx] = (p, text)

    restored = []
    for e in entries:
        if e.status != "Wait":
            continue
        hit = md_by_best_entry.get(e.row_idx)
        if not hit:
            continue
        p, text = hit
        if not _md_missing_parts(text):
            e.set_status("checked")
            restored.append((e.row_idx, str(p.relative_to(REPO_ROOT))))

    if updates or restored:
        for e in entries:
            rows[e.row_idx] = e.row
        with LOG_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    print(f"Scanned md files (excluding emergentmind): {len(md_files)}")
    print(f"MD missing any Part I/II/III: {len(missing_mds)}")
    print(f"Log entries updated to Wait: {len(updates)}")
    print(f"Log entries restored to checked: {len(restored)}")
    for row_idx, old, new, rel, miss, md_title, log_title, score in updates[:60]:
        print(
            f"- row {row_idx+1}: {old} -> {new} | miss {miss} | md: {rel} | match {score:.2f}\n"
            f"  md_title: {md_title}\n"
            f"  log_title: {log_title}"
        )
    if len(updates) > 60:
        print(f"... {len(updates)-60} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
