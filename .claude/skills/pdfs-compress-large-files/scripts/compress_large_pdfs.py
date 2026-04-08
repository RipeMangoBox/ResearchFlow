"""
compress_large_pdfs.py
Scan paperPDFs/ for PDFs > threshold MB and compress in-place using Ghostscript.

Usage:
    python3 scripts/compress_large_pdfs.py [--dry-run] [--threshold 20] [--dir paperPDFs/]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIR = REPO_ROOT / "paperPDFs"
DEFAULT_THRESHOLD_MB = 20


def compress_pdf(src: Path, preset: str = "/ebook") -> Path:
    """Compress src PDF with Ghostscript into a temp file. Returns temp path."""
    tmp = Path(tempfile.mktemp(suffix=".pdf"))
    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS={preset}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        "-dColorImageResolution=150",
        "-dGrayImageResolution=150",
        "-dMonoImageResolution=300",
        "-dEmbedAllFonts=true",
        "-dSubsetFonts=true",
        f"-sOutputFile={tmp}",
        str(src),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    return tmp


def fmt_mb(size_bytes: int) -> str:
    return f"{size_bytes / 1_048_576:.1f} MB"


def main():
    parser = argparse.ArgumentParser(description="Compress large PDFs in paperPDFs/")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compressed without changing files")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_MB, help="Size threshold in MB (default: 20)")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Directory to scan (default: paperPDFs/)")
    args = parser.parse_args()

    scan_dir = args.dir if args.dir.is_absolute() else REPO_ROOT / args.dir
    threshold_bytes = int(args.threshold * 1_048_576)

    if not scan_dir.exists():
        print(f"[ERROR] Directory not found: {scan_dir}")
        sys.exit(1)

    # Check ghostscript
    if shutil.which("gs") is None:
        print("[ERROR] Ghostscript (gs) not found. Install with: sudo apt install ghostscript")
        sys.exit(1)

    pdfs = sorted(p for p in scan_dir.rglob("*.pdf") if p.stat().st_size > threshold_bytes)

    if not pdfs:
        print(f"No PDFs larger than {args.threshold} MB found under {scan_dir}")
        return

    print(f"Found {len(pdfs)} PDF(s) larger than {args.threshold} MB\n")

    total_saved = 0
    skipped = []

    for pdf in pdfs:
        orig_size = pdf.stat().st_size
        rel = pdf.relative_to(REPO_ROOT)
        print(f"  {rel}")
        print(f"    Original: {fmt_mb(orig_size)}", end="")

        if args.dry_run:
            print("  [dry-run, skipped]")
            continue

        # Try /ebook first
        tmp = None
        compressed_size = orig_size
        used_preset = None

        for preset in ["/ebook", "/screen"]:
            try:
                tmp = compress_pdf(pdf, preset)
                compressed_size = tmp.stat().st_size
                used_preset = preset
                if compressed_size <= threshold_bytes:
                    break
                # Still too large, try next preset
                tmp.unlink(missing_ok=True)
                tmp = None
            except Exception as exc:
                print(f"\n    [ERROR] gs failed with {preset}: {exc}")
                tmp = None

        if tmp is None or compressed_size >= orig_size:
            print(f"  → could not reduce below {args.threshold} MB, keeping original")
            skipped.append(str(rel))
            if tmp:
                tmp.unlink(missing_ok=True)
            continue

        saved = orig_size - compressed_size
        total_saved += saved
        pct = saved / orig_size * 100
        print(f"  → {fmt_mb(compressed_size)} ({pct:.0f}% saved, preset={used_preset})")

        # Replace original
        shutil.move(str(tmp), str(pdf))

    print(f"\n{'='*50}")
    if args.dry_run:
        print("Dry-run complete. No files were modified.")
    else:
        print(f"Total saved: {fmt_mb(total_saved)}")
        if skipped:
            print(f"Could not compress ({len(skipped)}):")
            for s in skipped:
                print(f"  - {s}")
    print("Done.")


if __name__ == "__main__":
    main()
