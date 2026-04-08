#!/usr/bin/env python3
import re
import sys
import shutil
from pathlib import Path

def normalize(s: str) -> str:
    import unicodedata
    s = s or ''
    s = unicodedata.normalize('NFKD', s)
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_md_mappings(md_text: str):
    # Find occurrences like [TransPhase](url): Full Title, Author
    pattern = re.compile(r"\[([^\]]+)\].*?:\s*([^,\n]+)")
    mappings = {}
    for m in pattern.findall(md_text):
        abbr = m[0].strip()
        full = m[1].strip()
        norm = normalize(full)
        # prefer first occurrence
        if norm not in mappings:
            mappings[norm] = (abbr, full)
    return mappings

def main(md_path: Path, log_path: Path):
    md_text = md_path.read_text(encoding='utf-8')
    mappings = extract_md_mappings(md_text)

    txt = log_path.read_text(encoding='utf-8')
    lines = txt.splitlines()
    out_lines = []
    changed = []
    missing = []

    for i, line in enumerate(lines):
        if ' | ' not in line:
            out_lines.append(line)
            continue
        parts = [p.strip() for p in line.split(' | ')]
        if len(parts) < 2:
            out_lines.append(line)
            continue
        title = parts[1]
        norm_title = normalize(title)

        if norm_title in mappings:
            abbr, full = mappings[norm_title]
            # if bracket already present, skip
            if re.search(re.escape(abbr), title, re.IGNORECASE):
                out_lines.append(line)
                continue
            # if bracket equals full title, skip insertion
            if abbr.strip().lower() == full.strip().lower():
                out_lines.append(line)
                continue
            parts[1] = f'[{abbr}] {full}'
            out_lines.append(' | '.join(parts))
            changed.append((i+1, title, parts[1]))
        else:
            # try fuzzy substring match against mappings
            found = False
            for k, (abbr, full) in mappings.items():
                if k in norm_title or norm_title in k:
                    if re.search(re.escape(abbr), title, re.IGNORECASE):
                        found = True
                        break
                    if abbr.strip().lower() != full.strip().lower():
                        parts[1] = f'[{abbr}] {full}'
                        out_lines.append(' | '.join(parts))
                        changed.append((i+1, title, parts[1]))
                        found = True
                        break
            if not found:
                out_lines.append(line)
                missing.append((i+1, title))

    if changed:
        bak = log_path.with_suffix(log_path.suffix + '.bak')
        shutil.copy2(log_path, bak)
        log_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')

    # Print summary to stdout
    print(f'Processed {len(lines)} lines')
    print(f'Updated {len(changed)} entries')
    if changed:
        for c in changed:
            print(f'Line {c[0]}: "{c[1]}" -> "{c[2]}"')
    if missing:
        print(f'Missing mappings for {len(missing)} entries:')
        for m in missing[:50]:
            print(f'Line {m[0]}: "{m[1]}"')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: update_download_log.py <foruck_md> <download_log.txt>')
        sys.exit(2)
    md = Path(sys.argv[1])
    log = Path(sys.argv[2])
    if not md.exists() or not log.exists():
        print('Input files not found')
        sys.exit(3)
    main(md, log)
