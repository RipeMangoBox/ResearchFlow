#!/usr/bin/env python3
"""Auto-download papers listed in download_log_updated.txt.

Features:
- Parse a pipe-separated log file with lines like:
  STATUS | Title | Venue | Year | url1 | url2 | topic
- Detect direct PDF URLs in the fields and optionally download them.
- Safe file naming and folder mapping: paperPDFs/<topic>/<Venue Year>/Author_Year_Title.pdf
- Dry-run mode prints counts and examples without performing downloads.
"""
import argparse
import concurrent.futures
import os
import re
import shutil
import sys
from urllib.parse import urlparse

try:
    import requests
except Exception:
    print("Missing dependency 'requests'. Install with: pip install -r requirements.txt")
    raise

LOG_ENCODING = "utf-8"


def safe_name(s: str, maxlen: int = 200) -> str:
    s = s.strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "", s)
    s = re.sub(r"\s+", "_", s)
    if len(s) > maxlen:
        return s[:maxlen]
    return s


def parse_log(path):
    """
    Parse lines assuming fields:
    status | title | venue | project_url | pdf_url | topic
    Returns list of entries with line index and parts.
    """
    entries = []
    with open(path, "r", encoding=LOG_ENCODING, errors="ignore") as f:
        lines = f.readlines()
    for idx, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        # ensure we have at least 6 parts by padding
        while len(parts) < 6:
            parts.append("")
        status = parts[0]
        title = parts[1]
        venue = parts[2]
        project_url = parts[3]
        pdf_url = parts[4]
        topic = parts[5] or "Misc"

        pdf_candidates = []
        if pdf_url and pdf_url.lower().startswith("http"):
            pdf_candidates.append(pdf_url)
        # also consider project_url if it looks like direct pdf
        if project_url and project_url.lower().startswith("http") and ".pdf" in project_url.lower():
            pdf_candidates.insert(0, project_url)

        entries.append({
            "line_idx": idx,
            "raw_line": line,
            "parts": parts,
            "status": status,
            "title": title,
            "venue": venue,
            "project_url": project_url,
            "pdf_url": pdf_url,
            "pdf_candidates": pdf_candidates,
            "topic": topic,
        })
    return entries, lines


def choose_pdf_candidate(candidates):
    # prefer direct .pdf links
    for u in candidates:
        if ".pdf" in u.lower():
            return u
    return candidates[0] if candidates else None


def download_one(task):
    url = task["url"]
    outpath = task["outpath"]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PaperDownloader/1.0)"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=30)
        ct = r.headers.get("Content-Type", "").lower()
        if r.status_code == 200 and "application/pdf" in ct:
            tmp = outpath + ".part"
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            with open(tmp, "wb") as wf:
                shutil.copyfileobj(r.raw, wf)
            os.replace(tmp, outpath)
            return (True, "downloaded")
        else:
            # save landing page for manual follow-up
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            htmlpath = outpath + ".html"
            with open(htmlpath, "wb") as hf:
                hf.write(r.content)
            meta = {
                "status_code": r.status_code,
                "content_type": ct,
                "url": url,
            }
            with open(outpath + ".meta.txt", "w", encoding="utf-8") as mf:
                mf.write(str(meta))
            if r.status_code in (401, 403):
                return (False, "paywalled")
            return (False, f"non-pdf content-type={ct}")
    except Exception as e:
        return (False, str(e))


def fetch_and_find_pdf(url):
    """Fetch a URL and try to find a direct PDF link in its HTML or return if response is PDF."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PaperDownloader/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        ct = r.headers.get("Content-Type", "").lower()
        if r.status_code == 200 and "application/pdf" in ct:
            return url
        # search for hrefs ending with .pdf
        text = r.text
        matches = re.findall(r'href=["\']([^"\']+\.pdf)["\']', text, flags=re.IGNORECASE)
        # also search for raw https links to PDFs inside JS or attributes
        js_matches = re.findall(r'(https?://[^"\'>]+\.pdf)', text, flags=re.IGNORECASE)
        if js_matches:
            for jm in js_matches:
                if jm not in matches:
                    matches.append(jm)
        if matches:
            # choose first absolute or make absolute
            for m in matches:
                if m.startswith('http'):
                    return m
                else:
                    # build absolute
                    parsed = urlparse(url)
                    base = f"{parsed.scheme}://{parsed.netloc}"
                    return base + m if m.startswith('/') else base + '/' + m
        # try meta tags for citation_pdf_url
        m2 = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
        if m2:
            return m2.group(1)
        # try to extract DOI from meta tags or text
        doi = extract_doi_from_html(text)
        if doi:
            return 'https://doi.org/' + doi
        return None
    except Exception:
        return None


def extract_doi_from_html(text):
    # try meta tag
    m = re.search(r'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # common DOI pattern
    m2 = re.search(r'(10\.\d{4,9}/[^"\'\s<>]+)', text)
    if m2:
        return m2.group(1).strip().rstrip('.')
    return None


def try_doi_resolve(doi):
    """Follow doi.org and try to find a PDF target or return final URL."""
    url = 'https://doi.org/' + doi
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, allow_redirects=True)
        final = r.url
        ct = r.headers.get('Content-Type', '').lower()
        if 'application/pdf' in ct:
            return final
        # for ACM DOI try specific pdf path
        if 'dl.acm.org' in final:
            return 'https://dl.acm.org/doi/pdf/' + doi
        return final
    except Exception:
        return None


def try_ieee_arnumber_from_html(text):
    m = re.search(r'arnumber["=:\s]*"?(\d{5,})"?', text)
    if m:
        return m.group(1)
    return None


def search_arxiv_by_title(title):
    """Query arXiv API for title. Return PDF url if found."""
    q = title.replace('"', '')
    url = 'http://export.arxiv.org/api/query?search_query=ti:%22' + requests.utils.requote_uri(q) + '%22&max_results=1'
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code != 200:
            return None
        text = r.text
        m = re.search(r'<id>https?://arxiv.org/abs/([0-9.]+v?\d*)</id>', text)
        if m:
            aid = m.group(1)
            return f'https://arxiv.org/pdf/{aid}.pdf'
        return None
    except Exception:
        return None


def crossref_search(title):
    """Query Crossref for DOI by title. Return doi string like 10.x/... or None."""
    try:
        url = 'https://api.crossref.org/works?query.title=' + requests.utils.requote_uri(title) + '&rows=1'
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        items = data.get('message', {}).get('items', [])
        if not items:
            return None
        doi = items[0].get('DOI')
        return doi
    except Exception:
        return None


def try_common_paths(base_url):
    """Try common paper paths on the same site (paper.pdf, /static/pdfs/, /papers/)."""
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [
        base + '/paper.pdf',
        base + '/papers/paper.pdf',
        base + '/papers/',
        base + '/static/pdfs/paper.pdf',
        base + '/static/paper.pdf',
        base + '/static/pdfs/',
        base + '/assets/paper.pdf',
    ]
    for c in candidates:
        try:
            r = requests.head(c, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, allow_redirects=True)
            ct = r.headers.get('Content-Type', '').lower()
            if r.status_code == 200 and 'pdf' in ct:
                return c
        except Exception:
            continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="paperPDFs/download_log_updated.txt")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Disable dry-run and perform actions")
    p.add_argument("--download", action="store_true", default=False, help="Perform downloads (otherwise dry-run)")
    args = p.parse_args()

    entries, lines = parse_log(args.log)
    wait_entries = [e for e in entries if e["status"].upper() == "WAIT"]
    downloadable = []
    for e in wait_entries:
        candidate = choose_pdf_candidate(e["pdf_candidates"])
        if candidate:
            # build outpath: paperPDFs/<topic>/<venue>/<YEAR_title>.pdf
            topic = safe_name(e.get("topic") or "Misc")
            venue_raw = (e.get("venue") or "Unknown").strip()
            venue = safe_name(venue_raw)
            # try to extract year from venue (e.g., "NeurIPS 2025")
            m = re.search(r"(19|20)\d{2}", venue_raw)
            year = m.group(0) if m else "unknown"
            folder = os.path.join("paperPDFs", topic, venue)
            # filename = YEAR_Title
            title = e.get('title') or 'paper'
            base = safe_name(f"{year}_{title}")
            filename = base + ".pdf"
            outpath = os.path.join(folder, filename)
            downloadable.append({"entry": e, "url": candidate, "outpath": outpath})

    print(f"Total entries parsed: {len(entries)}")
    print(f"WAIT entries: {len(wait_entries)}")
    print(f"Direct/downloadable candidates found: {len(downloadable)}")
    if len(downloadable) > 0:
        print("Sample candidates:")
        for s in downloadable[:10]:
            print("-", s['entry']['title'], "->", s['url'])

    if args.download and not args.dry_run:
        tasks = []
        for d in downloadable:
            tasks.append({"url": d['url'], "outpath": d['outpath'], "entry": d['entry']})
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download_one, t): t for t in tasks}
            for fut in concurrent.futures.as_completed(futs):
                t = futs[fut]
                ok, note = fut.result()
                results.append((t, ok, note))
                print(t['outpath'], ok, note)

        # Update original log lines for successful downloads
        updated_count = 0
        failed = []
        for t, ok, note in results:
            entry = t['entry']
            if ok:
                idx = entry['line_idx']
                parts = entry['parts']
                parts[0] = 'Downloaded'
                new_line = ' | '.join(parts)
                lines[idx] = new_line + '\n'
                updated_count += 1
            else:
                failed.append((entry, note))

        # Attempt secondary download methods for failed entries
        secondary_results = []
        for entry, reason in failed:
            title = entry.get('title')
            project_url = entry.get('project_url')
            pdf_url = entry.get('pdf_url')
            tried = set()
            found_pdf = None
            # 1) if pdf_url exists but returned HTML, try to extract embedded PDF
            if pdf_url and pdf_url not in tried:
                tried.add(pdf_url)
                candidate = fetch_and_find_pdf(pdf_url)
                if candidate:
                    found_pdf = candidate
            # 2) try project page
            if not found_pdf and project_url and project_url not in tried:
                tried.add(project_url)
                candidate = fetch_and_find_pdf(project_url)
                if candidate:
                    found_pdf = candidate
            # 2.5) try common paths on project site
            if not found_pdf and project_url:
                candidate = try_common_paths(project_url)
                if candidate:
                    found_pdf = candidate
            # 3) try arXiv search by title
            if not found_pdf and title:
                candidate = search_arxiv_by_title(title)
                if candidate:
                    found_pdf = candidate
            # 4) try Crossref DOI and resolve
            if not found_pdf and title:
                doi = crossref_search(title)
                if doi:
                    candidate = try_doi_resolve(doi)
                    if candidate:
                        found_pdf = candidate

            if found_pdf:
                outpath = os.path.join('paperPDFs', safe_name(entry.get('topic') or 'Misc'), safe_name(entry.get('venue') or 'Unknown'), safe_name((re.search(r"(19|20)\\d{2}", entry.get('venue') or '') or ['unknown'])[0] + '_' + entry.get('title')) + '.pdf')
                ok2, note2 = download_one({"url": found_pdf, "outpath": outpath})
                if ok2:
                    idx = entry['line_idx']
                    parts = entry['parts']
                    parts[0] = 'Downloaded'
                    parts[4] = found_pdf
                    lines[idx] = ' | '.join(parts) + '\n'
                    updated_count += 1
                    secondary_results.append((entry, True, found_pdf))
                else:
                    secondary_results.append((entry, False, note2))
            else:
                secondary_results.append((entry, False, reason))

        if updated_count > 0:
            bak = args.log + '.bak'
            shutil.copy2(args.log, bak)
            with open(args.log, 'w', encoding=LOG_ENCODING) as wf:
                wf.writelines(lines)
            print(f"Updated {updated_count} entries in log; backup saved to {bak}")

        # report secondary failures for user's manual follow-up
        sec_failures = [s for s in secondary_results if not s[1]]
        if sec_failures:
            print(f"{len(sec_failures)} entries still failed after secondary attempts. See list below:")
            for e, okf, notef in sec_failures:
                print('-', e.get('title'), '->', notef)
    else:
        print("Dry-run mode (no downloads). To perform downloads: run with --download --no-dry-run")


if __name__ == "__main__":
    main()
