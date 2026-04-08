# -*- coding: utf-8 -*-
import re
import shutil
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "paperPDFs" / "download_log.txt"

def format_title_field(title: str):
    # If already contains a fullwidth colon, assume formatted
    if '：' in title:
        return None
    s = title.strip()
    if not s.startswith('['):
        return None
    # Try to match pattern where URL ends with "):" then title
    m = re.match(r'^\s*\[([^\]]+)\].*?\):\s*(.*)$', s)
    if m:
        abbr = m.group(1).strip()
        rest = m.group(2).strip()
        if rest:
            return f'{abbr}：{rest}'
    # Fallback: bracket then whatever
    m2 = re.match(r'^\s*\[([^\]]+)\]\s*(.*)$', s)
    if m2:
        abbr = m2.group(1).strip()
        rest = m2.group(2).strip()
        return f'{abbr}：{rest}' if rest else f'{abbr}：'
    return None


def main():
    if not LOG_PATH.exists():
        print('download_log.txt not found:', LOG_PATH)
        return
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bak = LOG_PATH.with_suffix(LOG_PATH.suffix + f'.bak.{ts}')
    shutil.copy2(LOG_PATH, bak)
    print('Backup created:', bak)

    lines = LOG_PATH.read_text(encoding='utf-8').splitlines(keepends=True)
    out_lines = []
    changed = 0
    total = 0
    changed_examples = []

    for i, line in enumerate(lines, start=1):
        total += 1
        # split into fields by ' | '
        parts = line.split(' | ')
        if len(parts) >= 2:
            title = parts[1]
            new_title = format_title_field(title)
            if new_title:
                parts[1] = new_title
                new_line = ' | '.join(parts)
                # ensure newline preserved
                if line.endswith('\n') and not new_line.endswith('\n'):
                    new_line += '\n'
                out_lines.append(new_line)
                changed += 1
                if len(changed_examples) < 10:
                    changed_examples.append((i, title.strip(), new_title))
                continue
        out_lines.append(line)

    if changed:
        LOG_PATH.write_text(''.join(out_lines), encoding='utf-8')
    print(f'Processed {total} lines; updated {changed} title(s).')
    if changed_examples:
        print('Examples (line, old -> new):')
        for ex in changed_examples:
            print(ex[0], ' | ', ex[1], ' -> ', ex[2])
    print('Original backed up to', bak)

if __name__ == '__main__':
    main()
