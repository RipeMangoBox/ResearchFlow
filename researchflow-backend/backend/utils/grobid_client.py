"""GROBID client for structured academic PDF parsing.

Extracts: title, authors, affiliations, abstract, sections,
references (structured), figure/table captions via TEI XML.

Requires GROBID server running (Docker: lfoppiano/grobid:0.8.1).
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

TEI_NS = "{http://www.tei-c.org/ns/1.0}"


@dataclass
class GrobidAuthor:
    name: str = ""
    given_name: str = ""
    surname: str = ""
    affiliation: str = ""
    email: str = ""
    orcid: str = ""


@dataclass
class GrobidReference:
    """A single parsed reference from the bibliography."""
    ref_id: str = ""           # e.g., "b0", "b1"
    title: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    year: str = ""
    doi: str = ""
    arxiv_id: str = ""
    volume: str = ""
    pages: str = ""
    publisher: str = ""
    raw_text: str = ""


@dataclass
class GrobidFormula:
    """A formula detected by GROBID."""
    text: str = ""           # Raw text content from GROBID
    label: str = ""          # e.g., "(1)", "(2)"
    coords: str = ""         # GROBID coordinates: "page,x,y,w,h" format
    page: int = -1
    bbox: list[float] = field(default_factory=list)  # [x0, y0, x1, y1] in PDF points


@dataclass
class GrobidResult:
    """Result of GROBID fulltext parsing."""
    title: str = ""
    authors: list[GrobidAuthor] = field(default_factory=list)
    abstract: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    references: list[GrobidReference] = field(default_factory=list)
    figure_captions: list[dict] = field(default_factory=list)
    table_captions: list[dict] = field(default_factory=list)
    formulas: list[GrobidFormula] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    raw_tei_xml: str = ""


class GrobidClient:
    """Client for GROBID REST API.

    Usage:
        client = GrobidClient("http://localhost:8070")
        result = await client.parse_fulltext("/path/to/paper.pdf")
    """

    def __init__(self, base_url: str = "http://localhost:8070", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def is_alive(self) -> bool:
        """Check if GROBID server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/isalive")
                return resp.status_code == 200
        except Exception:
            return False

    async def parse_fulltext(self, pdf_path: str | Path) -> GrobidResult:
        """Parse a PDF with GROBID fulltext endpoint.

        Returns structured GrobidResult with title, authors, affiliations,
        abstract, sections, references, figure/table captions.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(pdf_path, "rb") as f:
                resp = await client.post(
                    f"{self.base_url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, f, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "consolidateCitations": "1",
                        "includeRawAffiliations": "1",
                        "includeRawCitations": "1",
                        "teiCoordinates": "formula",  # Request formula coordinates
                    },
                )

            if resp.status_code != 200:
                logger.error(f"GROBID returned {resp.status_code}: {resp.text[:200]}")
                return GrobidResult()

            tei_xml = resp.text
            return self._parse_tei(tei_xml)

    async def parse_header(self, pdf_path: str | Path) -> GrobidResult:
        """Parse only the header (faster, for metadata extraction)."""
        pdf_path = Path(pdf_path)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(pdf_path, "rb") as f:
                resp = await client.post(
                    f"{self.base_url}/api/processHeaderDocument",
                    files={"input": (pdf_path.name, f, "application/pdf")},
                    data={"consolidateHeader": "1"},
                )

            if resp.status_code != 200:
                return GrobidResult()

            return self._parse_tei(resp.text)

    async def parse_references(self, pdf_path: str | Path) -> list[GrobidReference]:
        """Parse only references from a PDF."""
        pdf_path = Path(pdf_path)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(pdf_path, "rb") as f:
                resp = await client.post(
                    f"{self.base_url}/api/processReferences",
                    files={"input": (pdf_path.name, f, "application/pdf")},
                    data={"consolidateCitations": "1"},
                )

            if resp.status_code != 200:
                return []

            return self._parse_tei(resp.text).references

    def _parse_tei(self, tei_xml: str) -> GrobidResult:
        """Parse TEI XML output from GROBID into structured result."""
        result = GrobidResult(raw_tei_xml=tei_xml)

        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            logger.error(f"Failed to parse TEI XML: {e}")
            return result

        # Title
        title_el = root.find(f".//{TEI_NS}titleStmt/{TEI_NS}title[@type='main']")
        if title_el is None:
            title_el = root.find(f".//{TEI_NS}titleStmt/{TEI_NS}title")
        if title_el is not None and title_el.text:
            result.title = title_el.text.strip()

        # Authors + affiliations
        result.authors = self._parse_authors(root)

        # Abstract
        abstract_el = root.find(f".//{TEI_NS}profileDesc/{TEI_NS}abstract")
        if abstract_el is not None:
            result.abstract = _get_all_text(abstract_el).strip()

        # Keywords
        keywords_el = root.find(f".//{TEI_NS}profileDesc/{TEI_NS}textClass/{TEI_NS}keywords")
        if keywords_el is not None:
            for term in keywords_el.findall(f".//{TEI_NS}term"):
                if term.text and term.text.strip():
                    result.keywords.append(term.text.strip())

        # Body sections
        body = root.find(f".//{TEI_NS}body")
        if body is not None:
            result.sections = self._parse_body_sections(body)

        # References
        back = root.find(f".//{TEI_NS}back")
        if back is not None:
            result.references = self._parse_references(back)

        # Formulas with coordinates
        result.formulas = self._parse_formulas(root)

        # Figure captions
        for fig in root.findall(f".//{TEI_NS}figure"):
            fig_type = fig.get("type", "figure")
            label_el = fig.find(f"{TEI_NS}label")
            desc_el = fig.find(f"{TEI_NS}figDesc")

            label = label_el.text.strip() if label_el is not None and label_el.text else ""
            caption = _get_all_text(desc_el).strip() if desc_el is not None else ""

            if caption and len(caption) > 10:
                if fig_type == "table":
                    result.table_captions.append({
                        "label": label,
                        "caption": caption[:500],
                    })
                else:
                    result.figure_captions.append({
                        "label": label,
                        "caption": caption[:500],
                    })

        return result

    def _parse_authors(self, root: ET.Element) -> list[GrobidAuthor]:
        """Parse author elements from TEI header."""
        authors = []
        for author_el in root.findall(
            f".//{TEI_NS}fileDesc/{TEI_NS}sourceDesc/{TEI_NS}biblStruct/"
            f"{TEI_NS}analytic/{TEI_NS}author"
        ):
            author = GrobidAuthor()

            # Name
            persname = author_el.find(f"{TEI_NS}persName")
            if persname is not None:
                given = persname.find(f"{TEI_NS}forename")
                surname = persname.find(f"{TEI_NS}surname")
                if given is not None and given.text:
                    author.given_name = given.text.strip()
                if surname is not None and surname.text:
                    author.surname = surname.text.strip()
                author.name = f"{author.given_name} {author.surname}".strip()

            # Affiliation
            affil = author_el.find(f"{TEI_NS}affiliation")
            if affil is not None:
                org_parts = []
                for org in affil.findall(f"{TEI_NS}orgName"):
                    if org.text:
                        org_parts.append(org.text.strip())
                author.affiliation = ", ".join(org_parts)

            # Email
            email_el = author_el.find(f"{TEI_NS}email")
            if email_el is not None and email_el.text:
                author.email = email_el.text.strip()

            # ORCID (idno type="ORCID")
            for idno in author_el.findall(f"{TEI_NS}idno"):
                if idno.get("type") == "ORCID" and idno.text:
                    author.orcid = idno.text.strip()

            if author.name:
                authors.append(author)

        return authors

    def _parse_formulas(self, root: ET.Element) -> list[GrobidFormula]:
        """Parse formula elements from TEI, including coordinates."""
        formulas = []
        for formula_el in root.findall(f".//{TEI_NS}formula"):
            formula = GrobidFormula()
            formula.text = _get_all_text(formula_el).strip()

            # Label: e.g., "(1)", "(2)"
            label_el = formula_el.find(f"{TEI_NS}label")
            if label_el is not None and label_el.text:
                formula.label = label_el.text.strip()

            # Coordinates: GROBID outputs coords="page,x,y,w,h" attribute
            coords = formula_el.get("coords", "")
            if coords:
                formula.coords = coords
                try:
                    parts = coords.split(",")
                    if len(parts) >= 5:
                        page = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        formula.page = page - 1  # GROBID uses 1-based pages
                        formula.bbox = [x, y, x + w, y + h]
                except (ValueError, IndexError):
                    pass

            if formula.text and len(formula.text) > 3:
                formulas.append(formula)

        return formulas

    def _parse_body_sections(self, body: ET.Element) -> dict[str, str]:
        """Parse body divisions into named sections."""
        sections: dict[str, str] = {}

        for div in body.findall(f"{TEI_NS}div"):
            head = div.find(f"{TEI_NS}head")
            if head is not None and head.text:
                section_name = head.text.strip()
                # Get all paragraphs in this div
                paragraphs = []
                for p in div.findall(f"{TEI_NS}p"):
                    text = _get_all_text(p).strip()
                    if text:
                        paragraphs.append(text)
                if paragraphs:
                    key = _normalize_section_name(section_name)
                    content = "\n\n".join(paragraphs)
                    if key in sections:
                        sections[key] += "\n\n" + content
                    else:
                        sections[key] = content
            else:
                # Unnamed div — append to previous or "body"
                paragraphs = []
                for p in div.findall(f"{TEI_NS}p"):
                    text = _get_all_text(p).strip()
                    if text:
                        paragraphs.append(text)
                if paragraphs:
                    key = "body"
                    content = "\n\n".join(paragraphs)
                    sections[key] = sections.get(key, "") + "\n\n" + content

        return sections

    def _parse_references(self, back: ET.Element) -> list[GrobidReference]:
        """Parse bibliography entries from TEI back matter."""
        refs = []
        for bib in back.findall(f".//{TEI_NS}biblStruct"):
            ref = GrobidReference()
            ref.ref_id = bib.get(f"{{{TEI_NS.strip('{}')}}}id", bib.get("xml:id", ""))

            # Title
            analytic = bib.find(f"{TEI_NS}analytic")
            if analytic is not None:
                title_el = analytic.find(f"{TEI_NS}title")
                if title_el is not None:
                    ref.title = _get_all_text(title_el).strip()

                # Authors
                for author in analytic.findall(f"{TEI_NS}author"):
                    persname = author.find(f"{TEI_NS}persName")
                    if persname is not None:
                        given = persname.find(f"{TEI_NS}forename")
                        surname = persname.find(f"{TEI_NS}surname")
                        name_parts = []
                        if given is not None and given.text:
                            name_parts.append(given.text.strip())
                        if surname is not None and surname.text:
                            name_parts.append(surname.text.strip())
                        if name_parts:
                            ref.authors.append(" ".join(name_parts))

            # Monograph info (venue, year, volume, pages)
            monogr = bib.find(f"{TEI_NS}monogr")
            if monogr is not None:
                venue_el = monogr.find(f"{TEI_NS}title")
                if venue_el is not None and venue_el.text:
                    ref.venue = venue_el.text.strip()

                imprint = monogr.find(f"{TEI_NS}imprint")
                if imprint is not None:
                    date_el = imprint.find(f"{TEI_NS}date")
                    if date_el is not None:
                        ref.year = date_el.get("when", date_el.text or "").strip()[:4]

                    vol = imprint.find(f"{TEI_NS}biblScope[@unit='volume']")
                    if vol is not None and vol.text:
                        ref.volume = vol.text.strip()

                    page = imprint.find(f"{TEI_NS}biblScope[@unit='page']")
                    if page is not None:
                        fr = page.get("from", "")
                        to = page.get("to", "")
                        ref.pages = f"{fr}-{to}" if fr and to else (page.text or "").strip()

                    pub = imprint.find(f"{TEI_NS}publisher")
                    if pub is not None and pub.text:
                        ref.publisher = pub.text.strip()

            # DOI
            for idno in bib.findall(f".//{TEI_NS}idno"):
                id_type = idno.get("type", "").lower()
                if id_type == "doi" and idno.text:
                    ref.doi = idno.text.strip()
                elif id_type == "arxiv" and idno.text:
                    ref.arxiv_id = idno.text.strip()

            # Raw text (note element)
            note = bib.find(f"{TEI_NS}note[@type='raw_reference']")
            if note is not None:
                ref.raw_text = _get_all_text(note).strip()

            if ref.title or ref.raw_text:
                refs.append(ref)

        return refs


def _get_all_text(element: ET.Element | None) -> str:
    """Recursively get all text content from an XML element."""
    if element is None:
        return ""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_get_all_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)


def _normalize_section_name(name: str) -> str:
    """Normalize section header to a canonical key."""
    name_lower = name.lower().strip()
    # Remove leading numbers: "1. Introduction" -> "introduction"
    import re
    name_lower = re.sub(r"^\d+\.?\d*\.?\s*", "", name_lower)

    mapping = {
        "introduction": "introduction",
        "related work": "related_work",
        "related works": "related_work",
        "prior work": "related_work",
        "background": "background",
        "preliminary": "preliminary",
        "preliminaries": "preliminary",
        "method": "method",
        "methods": "method",
        "approach": "method",
        "proposed method": "method",
        "our method": "method",
        "methodology": "method",
        "framework": "method",
        "model": "method",
        "architecture": "method",
        "experiment": "experiments",
        "experiments": "experiments",
        "evaluation": "experiments",
        "results": "experiments",
        "experimental results": "experiments",
        "ablation": "ablation",
        "ablation study": "ablation",
        "analysis": "analysis",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "discussion": "discussion",
        "limitation": "limitations",
        "limitations": "limitations",
        "future work": "future_work",
        "appendix": "appendix",
        "supplementary": "appendix",
    }

    return mapping.get(name_lower, name_lower.replace(" ", "_")[:50])
