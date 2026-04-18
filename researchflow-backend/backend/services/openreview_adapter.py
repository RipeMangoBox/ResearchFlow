"""OpenReview adapter for conference acceptance detection.

Core objects in OpenReview are notes. Submissions, reviews, comments,
official responses, and decisions are all notes under different invitations.

For ICLR/NeurIPS/workshop tracks:
  Use invitation/forum/replyto to get decision notes.

Requires: pip install openreview-py
"""

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# Known venue IDs for major conferences
KNOWN_VENUES = {
    "ICLR": {
        2024: "ICLR.cc/2024/Conference",
        2025: "ICLR.cc/2025/Conference",
        2026: "ICLR.cc/2026/Conference",
    },
    "NeurIPS": {
        2024: "NeurIPS.cc/2024/Conference",
        2025: "NeurIPS.cc/2025/Conference",
    },
    "ICML": {
        2025: "ICML.cc/2025/Conference",
    },
}


@dataclass
class VenueDecision:
    """Result of venue acceptance check."""
    venue: str = ""
    year: int = 0
    acceptance_status: str = "unknown"
    # accepted / rejected / withdrawn / desk_rejected / under_review / unknown
    acceptance_type: str = ""
    # oral / poster / spotlight / workshop / unknown
    review_scores: list[dict] = field(default_factory=list)
    # [{reviewer, score, confidence}]
    avg_score: float = 0.0
    forum_url: str = ""
    source: str = "openreview"
    confidence: float = 0.0


class OpenReviewAdapter:
    """Adapter for OpenReview API.

    Uses the REST API directly (no openreview-py dependency required,
    but compatible if installed).
    """

    BASE_URL = "https://api2.openreview.net"

    def __init__(self, username: str = "", password: str = ""):
        self.username = username
        self.password = password
        self._token: str | None = None

    async def _get_token(self) -> str | None:
        """Authenticate and get API token (optional, needed for some queries)."""
        if self._token:
            return self._token
        if not self.username or not self.password:
            return None

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.BASE_URL}/login",
                json={"id": self.username, "password": self.password},
            )
            if resp.status_code == 200:
                self._token = resp.json().get("token")
                return self._token
        return None

    async def _get_headers(self) -> dict:
        token = await self._get_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    async def match_paper(
        self,
        title: str,
        authors: list[str] | None = None,
        venue: str = "",
        year: int = 0,
    ) -> VenueDecision | None:
        """Match a paper by title to an OpenReview submission.

        Returns VenueDecision with acceptance status, type, and review scores.
        """
        # Try to find the paper via title search
        headers = await self._get_headers()

        async with httpx.AsyncClient(timeout=30) as client:
            # Search for the paper
            params = {
                "term": title[:200],
                "type": "terms",
                "content": "all",
                "source": "forum",
                "limit": 5,
            }
            resp = await client.get(
                f"{self.BASE_URL}/notes/search",
                params=params,
                headers=headers,
            )

            if resp.status_code != 200:
                logger.warning(f"OpenReview search failed: {resp.status_code}")
                return None

            data = resp.json()
            notes = data.get("notes", [])
            if not notes:
                return None

            # Find best title match
            best_match = self._find_best_match(notes, title, authors)
            if not best_match:
                return None

            forum_id = best_match.get("forum", best_match.get("id", ""))
            return await self._get_decision(client, headers, forum_id, best_match)

    async def get_venue_decisions(
        self,
        venue_id: str,
        limit: int = 100,
    ) -> list[VenueDecision]:
        """Get all decisions for a venue (e.g., ICLR.cc/2025/Conference).

        Returns list of VenueDecision for accepted papers.
        """
        headers = await self._get_headers()
        decisions = []

        async with httpx.AsyncClient(timeout=60) as client:
            # Get decision notes
            invitation = f"{venue_id}/-/Decision"
            offset = 0

            while True:
                resp = await client.get(
                    f"{self.BASE_URL}/notes",
                    params={
                        "invitation": invitation,
                        "offset": offset,
                        "limit": min(limit - len(decisions), 100),
                    },
                    headers=headers,
                )

                if resp.status_code != 200:
                    break

                data = resp.json()
                notes = data.get("notes", [])
                if not notes:
                    break

                for note in notes:
                    content = note.get("content", {})
                    decision_text = ""
                    if isinstance(content.get("decision"), dict):
                        decision_text = content["decision"].get("value", "")
                    elif isinstance(content.get("decision"), str):
                        decision_text = content["decision"]

                    status, acc_type = self._parse_decision(decision_text)

                    decisions.append(VenueDecision(
                        venue=venue_id.split("/")[0],
                        acceptance_status=status,
                        acceptance_type=acc_type,
                        forum_url=f"https://openreview.net/forum?id={note.get('forum', '')}",
                        source="openreview",
                        confidence=0.95,
                    ))

                offset += len(notes)
                if len(decisions) >= limit or len(notes) < 100:
                    break

        return decisions

    async def _get_decision(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        forum_id: str,
        submission_note: dict,
    ) -> VenueDecision:
        """Get decision and reviews for a specific forum."""
        result = VenueDecision(
            forum_url=f"https://openreview.net/forum?id={forum_id}",
            source="openreview",
        )

        # Extract venue from invitation
        invitation = submission_note.get("invitation", "")
        if "/" in invitation:
            parts = invitation.split("/")
            result.venue = parts[0] if parts else ""

        # Get all replies to this forum (decisions, reviews)
        resp = await client.get(
            f"{self.BASE_URL}/notes",
            params={"forum": forum_id},
            headers=headers,
        )

        if resp.status_code != 200:
            return result

        replies = resp.json().get("notes", [])

        for reply in replies:
            inv = reply.get("invitation", "")
            content = reply.get("content", {})

            # Decision note
            if "Decision" in inv or "decision" in inv.lower():
                decision_text = ""
                if isinstance(content.get("decision"), dict):
                    decision_text = content["decision"].get("value", "")
                elif isinstance(content.get("decision"), str):
                    decision_text = content["decision"]

                status, acc_type = self._parse_decision(decision_text)
                result.acceptance_status = status
                result.acceptance_type = acc_type
                result.confidence = 0.95

            # Review notes
            if "Official_Review" in inv or "Review" in inv:
                score = None
                if isinstance(content.get("rating"), dict):
                    score_str = content["rating"].get("value", "")
                    score = self._extract_score(score_str)
                elif isinstance(content.get("rating"), str):
                    score = self._extract_score(content["rating"])
                elif isinstance(content.get("soundness"), dict):
                    score = self._extract_score(content["soundness"].get("value", ""))

                confidence = None
                if isinstance(content.get("confidence"), dict):
                    confidence = self._extract_score(content["confidence"].get("value", ""))
                elif isinstance(content.get("confidence"), str):
                    confidence = self._extract_score(content["confidence"])

                if score is not None:
                    result.review_scores.append({
                        "score": score,
                        "confidence": confidence,
                    })

        # Calculate average score
        if result.review_scores:
            scores = [r["score"] for r in result.review_scores if r["score"] is not None]
            if scores:
                result.avg_score = sum(scores) / len(scores)

        return result

    def _find_best_match(
        self,
        notes: list[dict],
        title: str,
        authors: list[str] | None,
    ) -> dict | None:
        """Find best matching note by title similarity."""
        title_lower = title.lower().strip()

        for note in notes:
            content = note.get("content", {})
            note_title = ""
            if isinstance(content.get("title"), dict):
                note_title = content["title"].get("value", "")
            elif isinstance(content.get("title"), str):
                note_title = content["title"]

            if not note_title:
                continue

            # Simple exact match (case-insensitive, strip punctuation)
            if self._normalize(note_title) == self._normalize(title):
                return note

        # Fallback: partial match
        for note in notes:
            content = note.get("content", {})
            note_title = ""
            if isinstance(content.get("title"), dict):
                note_title = content["title"].get("value", "")
            elif isinstance(content.get("title"), str):
                note_title = content["title"]

            if note_title and self._normalize(title)[:50] in self._normalize(note_title):
                return note

        return None

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def _parse_decision(decision_text: str) -> tuple[str, str]:
        """Parse decision text into (status, type)."""
        text = decision_text.lower().strip()

        if "reject" in text:
            return "rejected", ""
        if "withdraw" in text:
            return "withdrawn", ""
        if "desk reject" in text:
            return "desk_rejected", ""

        if "accept" in text:
            status = "accepted"
            if "oral" in text:
                return status, "oral"
            if "spotlight" in text:
                return status, "spotlight"
            if "poster" in text:
                return status, "poster"
            if "workshop" in text:
                return status, "workshop"
            return status, "poster"  # default accepted type

        return "unknown", ""

    @staticmethod
    def _extract_score(text: str) -> float | None:
        """Extract numeric score from review text like '8: Strong Accept'."""
        import re
        if not text:
            return None
        match = re.match(r'(\d+\.?\d*)', str(text).strip())
        if match:
            return float(match.group(1))
        return None
