import asyncio
import re
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests
from playwright.async_api import async_playwright

LOG_PATH = Path("paperPDFs/download_log_updated.txt")
ENC = "utf-8"


def safe_name(s: str, maxlen: int = 200) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:maxlen]


def parse_log():
    with open(LOG_PATH, "r", encoding=ENC, errors="ignore") as f:
        lines = f.readlines()
    entries = []
    for idx, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        while len(parts) < 6:
            parts.append("")
        entries.append({
            "idx": idx,
            "parts": parts,
            "status": parts[0],
            "title": parts[1],
            "venue": parts[2],
            "project_url": parts[3],
            "pdf_url": parts[4],
            "topic": parts[5] or "Misc",
        })
    return entries, lines


async def extract_pdf_links(page):
    # return list of absolute pdf urls found on page
    content = await page.content()
    links = re.findall(r'href=["\']([^"\']+\.pdf)["\']', content, flags=re.IGNORECASE)
    js_links = re.findall(r'(https?://[^"\'>]+\.pdf)', content, flags=re.IGNORECASE)
    for l in js_links:
        if l not in links:
            links.append(l)
    # also look for citation_pdf_url meta
    m = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', content, flags=re.IGNORECASE)
    if m:
        if m.group(1) not in links:
            links.insert(0, m.group(1))
    # make absolute
    abs_links = []
    url = page.url
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for l in links:
        if l.startswith("http"):
            abs_links.append(l)
        else:
            if l.startswith('/'):
                abs_links.append(base + l)
            else:
                abs_links.append(base + '/' + l)
    return abs_links


async def try_download_with_playwright(entries, lines):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        results = []
        for e in entries:
            if e['status'].upper() != 'WAIT':
                continue
            title = e['title']
            project = e['project_url']
            pdf = e['pdf_url']
            found = None
            tried_urls = []
            candidates = []
            # try pdf url first
            if pdf:
                try:
                    resp = await page.goto(pdf, wait_until='networkidle', timeout=30000)
                    if resp and resp.headers.get('content-type', '') and 'pdf' in resp.headers.get('content-type', ''):
                        found = pdf
                    else:
                        # try extract from that page
                        cl = await extract_pdf_links(page)
                        candidates += cl
                except Exception:
                    pass
            # try project page
            if not found and project:
                try:
                    await page.goto(project, wait_until='networkidle', timeout=30000)
                    cl = await extract_pdf_links(page)
                    candidates += cl
                except Exception:
                    pass

            # try candidates
            for c in candidates:
                if c in tried_urls:
                    continue
                tried_urls.append(c)
                try:
                    resp = await page.goto(c, wait_until='networkidle', timeout=30000)
                    if resp and 'pdf' in resp.headers.get('content-type', ''):
                        found = c
                        break
                except Exception:
                    continue

            # download found
            if found:
                # build outpath: paperPDFs/<topic>/<venue>/<YEAR_Title>.pdf
                topic = safe_name(e['topic'])
                venue = safe_name(e['venue'] or 'Unknown')
                m = re.search(r"(19|20)\d{2}", e['venue'] or '')
                year = m.group(0) if m else 'unknown'
                folder = Path('paperPDFs') / topic / venue
                folder.mkdir(parents=True, exist_ok=True)
                fname = safe_name(f"{year}_{title}") + '.pdf'
                outpath = folder / fname
                # use requests to download final URL
                try:
                    r = requests.get(found, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=30)
                    ct = r.headers.get('Content-Type','')
                    if r.status_code == 200 and 'pdf' in ct:
                        with open(outpath, 'wb') as wf:
                            for chunk in r.iter_content(1024*32):
                                wf.write(chunk)
                        # update log line
                        parts = e['parts']
                        parts[0] = 'Downloaded'
                        parts[4] = found
                        lines[e['idx']] = ' | '.join(parts) + '\n'
                        results.append((title, True, str(outpath)))
                    else:
                        results.append((title, False, f'non-pdf content-type={ct}'))
                except Exception as ex:
                    results.append((title, False, str(ex)))
            else:
                results.append((title, False, 'no-pdf-found'))

        await browser.close()
    # write back lines
    updated = sum(1 for r in results if r[1])
    if updated:
        bak = str(LOG_PATH) + '.playwright.bak'
        shutil.copy2(LOG_PATH, bak)
        with open(LOG_PATH, 'w', encoding=ENC) as wf:
            wf.writelines(lines)
    return results


def main():
    entries, lines = parse_log()
    # only process WAIT entries
    wait = [e for e in entries if e['status'].upper() == 'WAIT']
    if not wait:
        print('No WAIT entries to process.')
        return
    print(f'Playwright will attempt {len(wait)} entries...')
    res = asyncio.run(try_download_with_playwright(wait, lines))
    succ = [r for r in res if r[1]]
    fail = [r for r in res if not r[1]]
    print(f'Playwright done. Success: {len(succ)}, Fail: {len(fail)}')
    for t,ok,msg in fail:
        print('-', t, '->', msg)


if __name__ == '__main__':
    main()
