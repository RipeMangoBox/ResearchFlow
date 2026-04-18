"""DBLP adapter for conference acceptance verification.

If a paper appears in DBLP under a conference proceedings entry
(e.g., conf/cvpr/2025), it was accepted at that conference.

DBLP API: https://dblp.org/faq/How+to+use+the+dblp+search+API.html
"""

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DBLPResult:
    """Result of DBLP lookup."""
    dblp_key: str = ""           # e.g., "conf/iclr/SmithJ25"
    title: str = ""
    venue: str = ""              # e.g., "ICLR", "CVPR"
    year: int = 0
    venue_type: str = ""         # "conference" or "journal"
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    url: str = ""                # e.g., "https://dblp.org/rec/conf/iclr/SmithJ25"
    is_conference_accepted: bool = False
    confidence: float = 0.0


# Mapping from DBLP venue keys to canonical venue names
VENUE_KEY_MAP = {
    "conf/cvpr": "CVPR",
    "conf/iccv": "ICCV",
    "conf/eccv": "ECCV",
    "conf/iclr": "ICLR",
    "conf/nips": "NeurIPS",
    "conf/icml": "ICML",
    "conf/aaai": "AAAI",
    "conf/ijcai": "IJCAI",
    "conf/acl": "ACL",
    "conf/emnlp": "EMNLP",
    "conf/naacl": "NAACL",
    "conf/coling": "COLING",
    "conf/sigir": "SIGIR",
    "conf/kdd": "KDD",
    "conf/www": "WWW",
    "conf/mm": "ACM MM",
    "conf/siggraph": "SIGGRAPH",
    "conf/chi": "CHI",
    "conf/interspeech": "Interspeech",
    "conf/icra": "ICRA",
    "conf/iros": "IROS",
    "conf/rss": "RSS",
    "journals/pami": "TPAMI",
    "journals/ijcv": "IJCV",
    "journals/tip": "TIP",
    "journals/corr": "arXiv",
}


class DBLPAdapter:
    """Adapter for DBLP search API."""

    BASE_URL = "https://dblp.org/search/publ/api"

    async def search_paper(
        self,
        title: str,
        max_results: int = 5,
    ) -> list[DBLPResult]:
        """Search DBLP for a paper by title.

        Returns list of matching results with venue information.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                self.BASE_URL,
                params={
                    "q": title[:300],
                    "format": "json",
                    "h": max_results,
                },
            )

            if resp.status_code != 200:
                logger.warning(f"DBLP search failed: {resp.status_code}")
                return []

            data = resp.json()
            hits = data.get("result", {}).get("hits", {}).get("hit", [])

            results = []
            for hit in hits:
                info = hit.get("info", {})
                result = self._parse_hit(info)
                if result:
                    results.append(result)

            return results

    async def verify_conference_acceptance(
        self,
        title: str,
        expected_venue: str = "",
    ) -> DBLPResult | None:
        """Check if a paper was accepted at a conference via DBLP.

        If the paper appears under conf/* in DBLP, it was accepted.
        """
        results = await self.search_paper(title)

        if not results:
            return None

        # Find best match
        title_norm = self._normalize(title)

        for result in results:
            if self._normalize(result.title) == title_norm:
                # Exact match
                if result.is_conference_accepted:
                    result.confidence = 0.95
                else:
                    result.confidence = 0.8
                return result

        # Partial match — take best conference result
        for result in results:
            if result.is_conference_accepted:
                # Check title similarity
                if title_norm[:40] in self._normalize(result.title):
                    result.confidence = 0.7
                    return result

        # Return first result even if not conference
        if results:
            results[0].confidence = 0.5
            return results[0]

        return None

    async def get_author_publications(
        self,
        author_name: str,
        max_results: int = 30,
    ) -> list[DBLPResult]:
        """Search DBLP for an author's publications."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://dblp.org/search/author/api",
                params={
                    "q": author_name,
                    "format": "json",
                    "h": 5,
                },
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            hits = data.get("result", {}).get("hits", {}).get("hit", [])

            if not hits:
                return []

            # Get first author's publication page
            author_url = hits[0].get("info", {}).get("url", "")
            if not author_url:
                return []

            # Fetch publications
            resp2 = await client.get(
                f"{author_url}.xml",
                params={"format": "json"},
            )

            if resp2.status_code != 200:
                return []

            # Parse publications (simplified)
            return []  # TODO: parse author XML

    async def get_venue_papers(
        self,
        venue_key: str,
        year: int,
    ) -> list[DBLPResult]:
        """Get all papers from a DBLP venue/year.

        venue_key: e.g., "conf/cvpr" or "conf/iclr"
        """
        async with httpx.AsyncClient(timeout=30) as client:
            # DBLP venue proceedings URL pattern
            toc_url = f"https://dblp.org/db/{venue_key}/{venue_key.split('/')[-1]}{year}.html"

            resp = await client.get(
                self.BASE_URL,
                params={
                    "q": f"venue:{venue_key.split('/')[-1]}: year:{year}:",
                    "format": "json",
                    "h": 1000,
                },
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            hits = data.get("result", {}).get("hits", {}).get("hit", [])

            results = []
            for hit in hits:
                info = hit.get("info", {})
                result = self._parse_hit(info)
                if result and result.year == year:
                    results.append(result)

            return results

    def _parse_hit(self, info: dict) -> DBLPResult | None:
        """Parse a single DBLP search hit."""
        if not info:
            return None

        result = DBLPResult()
        result.title = info.get("title", "").rstrip(".")
        result.dblp_key = info.get("key", "")
        result.url = info.get("url", "")
        result.doi = info.get("doi", "")

        # Parse year
        year_str = info.get("year", "")
        if year_str:
            try:
                result.year = int(year_str)
            except (ValueError, TypeError):
                pass

        # Parse authors
        authors_data = info.get("authors", {}).get("author", [])
        if isinstance(authors_data, dict):
            authors_data = [authors_data]
        for a in authors_data:
            if isinstance(a, dict):
                result.authors.append(a.get("text", a.get("@text", "")))
            elif isinstance(a, str):
                result.authors.append(a)

        # Determine venue from key
        if result.dblp_key:
            for prefix, venue_name in VENUE_KEY_MAP.items():
                if result.dblp_key.startswith(prefix):
                    result.venue = venue_name
                    result.venue_type = "conference" if prefix.startswith("conf/") else "journal"
                    result.is_conference_accepted = prefix.startswith("conf/") and venue_name != "arXiv"
                    break

        # Fallback: parse venue from info
        if not result.venue:
            venue_info = info.get("venue", "")
            if isinstance(venue_info, list):
                venue_info = venue_info[0] if venue_info else ""
            result.venue = venue_info
            result.venue_type = info.get("type", "")
            result.is_conference_accepted = (
                result.venue_type in ("Conference and Workshop Papers", "inproceedings")
                and result.venue.lower() not in ("arxiv", "corr")
            )

        return result

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
