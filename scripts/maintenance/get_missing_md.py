from __future__ import annotations

from pathlib import Path


def extract_missing_papers(md_log: str = "md_paths.txt", pdf_log: str = "pdf_paths.txt", output_log: str = "to_be_analyzed.txt") -> None:
    if not Path(md_log).exists():
        print(f"❌ 错误: 找不到 {md_log}")
        return

    with open(md_log, "r", encoding="utf-8") as f:
        analyzed_set = {line.strip() for line in f if line.strip()}

    if not Path(pdf_log).exists():
        print(f"❌ 错误: 找不到 {pdf_log}")
        return

    missing_papers: list[str] = []

    with open(pdf_log, "r", encoding="utf-8") as f:
        for line in f:
            clean_path = line.strip()
            if not clean_path:
                continue
            if clean_path not in analyzed_set:
                paper_name = Path(clean_path).name
                missing_papers.append(paper_name)

    with open(output_log, "w", encoding="utf-8") as f:
        for name in missing_papers:
            f.write(name + "\n")

    print("📊 对比完成：")
    print(f"  - 已分析论文 (MD): {len(analyzed_set)}")
    print(f"  - 待分析论文 (仅有 PDF): {len(missing_papers)}")
    print(f"✅ 结果已保存至: {output_log}")


if __name__ == "__main__":
    extract_missing_papers()

