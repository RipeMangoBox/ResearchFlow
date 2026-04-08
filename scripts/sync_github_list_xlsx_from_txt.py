#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import List


def split_pipe_line(line: str) -> List[str] | None:
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 8:
        return None
    return parts[:8]


def read_txt_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = split_pipe_line(line)
        if not parts:
            continue
        rows.append(parts)
    return rows


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def xlsx_col_name(idx: int) -> str:
    n = idx + 1
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def write_xlsx(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rels = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>
</Relationships>
"""
    content_types = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>
  <Default Extension=\"xml\" ContentType=\"application/xml\"/>
  <Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>
  <Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>
</Types>
"""
    workbook_rels = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>
</Relationships>
"""
    workbook = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
  <sheets>
    <sheet name=\"papers\" sheetId=\"1\" r:id=\"rId1\"/>
  </sheets>
</workbook>
"""

    sheet_lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">",
        "  <sheetData>",
    ]

    for r, row in enumerate(rows, start=1):
        sheet_lines.append(f"    <row r=\"{r}\">")
        for c, value in enumerate(row):
            cell_ref = f"{xlsx_col_name(c)}{r}"
            v = xml_escape(value or "")
            sheet_lines.append(
                f"      <c r=\"{cell_ref}\" t=\"inlineStr\"><is><t>{v}</t></is></c>"
            )
        sheet_lines.append("    </row>")

    sheet_lines.extend(["  </sheetData>", "</worksheet>", ""])
    worksheet = "\n".join(sheet_lines)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync github awesome xlsx from txt list")
    p.add_argument(
        "--txt",
        default="paperAnalysis/human_motion_from_github.txt",
        help="Source txt list with 8 pipe-separated fields",
    )
    p.add_argument(
        "--xlsx",
        default="paperAnalysis/github_awesome_human_motion.xlsx",
        help="Target xlsx output path",
    )
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    txt_path = Path(ns.txt)
    xlsx_path = Path(ns.xlsx)

    if not txt_path.exists():
        raise FileNotFoundError(f"txt not found: {txt_path}")

    rows = read_txt_rows(txt_path)
    header = [
        "state",
        "importance",
        "paper_title",
        "venue",
        "project_link_or_github_link",
        "paper_link",
        "sort",
        "pdf_path",
    ]

    write_xlsx(xlsx_path, [header, *rows])

    print(f"[OK] source txt: {txt_path}")
    print(f"[OK] rows written: {len(rows)}")
    print(f"[OK] output xlsx: {xlsx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
