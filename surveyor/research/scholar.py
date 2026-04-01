"""Paper discovery via OpenAlex and Semantic Scholar APIs.

OpenAlex: completely free, no API key, excellent coverage, 10 req/sec.
Semantic Scholar: free tier, 100 req/5min, good for citation graphs.
Unpaywall: free, resolves open-access fulltext URLs from DOIs.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"

# Polite pool — OpenAlex gives faster responses if you identify yourself
OPENALEX_EMAIL = "surveyor@ecomancy.org"
USER_AGENT = "hooke-surveyor/0.1 (https://ecomancy.org; mailto:surveyor@ecomancy.org)"

HEADERS = {"User-Agent": USER_AGENT}


@dataclass
class Paper:
    """Minimal representation of a discovered paper."""

    doi: Optional[str] = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    source: str = "openalex"  # openalex | semantic_scholar
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: int = 0
    topics: list[str] = field(default_factory=list)
    is_open_access: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


# ── OpenAlex ─────────────────────────────────────────────────────────────────


def search_openalex(
    query: str,
    max_results: int = 25,
    from_year: Optional[int] = None,
    sort: str = "relevance_score:desc",
) -> list[Paper]:
    """Search OpenAlex for papers matching a query.

    Args:
        query: Search terms (supports boolean: AND, OR, NOT)
        max_results: Number of results (max 200 per page)
        from_year: Only papers published from this year onwards
        sort: Sort order (relevance_score:desc, cited_by_count:desc, publication_date:desc)
    """
    params = {
        "search": query,
        "per_page": min(max_results, 200),
        "sort": sort,
        "mailto": OPENALEX_EMAIL,
    }

    filters = ["type:article"]
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    params["filter"] = ",".join(filters)

    resp = requests.get(f"{OPENALEX_BASE}/works", params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    papers = []
    for work in data.get("results", []):
        doi = work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None

        authors = []
        for authorship in work.get("authorships", [])[:5]:
            name = authorship.get("author", {}).get("display_name")
            if name:
                authors.append(name)

        oa = work.get("open_access", {})
        best_oa = work.get("best_oa_location", {}) or {}

        topics = []
        for topic in work.get("topics", [])[:3]:
            topics.append(topic.get("display_name", ""))

        papers.append(Paper(
            doi=doi,
            title=work.get("title", ""),
            authors=authors,
            year=work.get("publication_year"),
            abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
            source="openalex",
            openalex_id=work.get("id", "").replace("https://openalex.org/", ""),
            url=work.get("doi") or work.get("id"),
            pdf_url=best_oa.get("pdf_url"),
            citation_count=work.get("cited_by_count", 0),
            topics=topics,
            is_open_access=oa.get("is_oa", False),
        ))

    logger.info(f"OpenAlex: {len(papers)} results for '{query}'")
    return papers


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


# ── Semantic Scholar ─────────────────────────────────────────────────────────


def search_semantic_scholar(
    query: str,
    max_results: int = 20,
    from_year: Optional[int] = None,
) -> list[Paper]:
    """Search Semantic Scholar for papers.

    Rate limit: 100 requests per 5 minutes on free tier.
    """
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,authors,year,abstract,externalIds,citationCount,isOpenAccess,openAccessPdf,url",
    }
    if from_year:
        params["year"] = f"{from_year}-"

    resp = requests.get(f"{S2_BASE}/paper/search", params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    papers = []
    for item in data.get("data", []):
        ext_ids = item.get("externalIds", {}) or {}
        oa_pdf = item.get("openAccessPdf", {}) or {}

        papers.append(Paper(
            doi=ext_ids.get("DOI"),
            title=item.get("title", ""),
            authors=[a.get("name", "") for a in (item.get("authors") or [])[:5]],
            year=item.get("year"),
            abstract=item.get("abstract"),
            source="semantic_scholar",
            s2_id=item.get("paperId"),
            url=item.get("url"),
            pdf_url=oa_pdf.get("url"),
            citation_count=item.get("citationCount", 0),
            is_open_access=item.get("isOpenAccess", False),
        ))

    logger.info(f"Semantic Scholar: {len(papers)} results for '{query}'")
    return papers


# ── Citation graph ───────────────────────────────────────────────────────────


def get_references(paper_id: str, source: str = "semantic_scholar") -> list[Paper]:
    """Get papers cited by a given paper (follow the citation trail)."""
    if source == "semantic_scholar" and paper_id:
        resp = requests.get(
            f"{S2_BASE}/paper/{paper_id}/references",
            params={"fields": "title,authors,year,externalIds,citationCount,isOpenAccess", "limit": 50},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        papers = []
        for ref in resp.json().get("data", []):
            cited = ref.get("citedPaper", {})
            if not cited or not cited.get("title"):
                continue
            ext_ids = cited.get("externalIds", {}) or {}
            papers.append(Paper(
                doi=ext_ids.get("DOI"),
                title=cited.get("title", ""),
                authors=[a.get("name", "") for a in (cited.get("authors") or [])[:3]],
                year=cited.get("year"),
                source="semantic_scholar",
                s2_id=cited.get("paperId"),
                citation_count=cited.get("citationCount", 0),
                is_open_access=cited.get("isOpenAccess", False),
            ))
        return papers
    return []


# ── Bulk search across topics ────────────────────────────────────────────────


def search_topics(
    topics_file: str | Path,
    max_per_topic: int = 15,
    from_year: int = 2015,
) -> dict[str, list[Paper]]:
    """Search across all topics defined in a topics.json file.

    Returns {topic_id: [papers]}.
    """
    with open(topics_file) as f:
        topics = json.load(f)

    results = {}
    for topic in topics.get("topics", []):
        tid = topic["id"]
        for query in topic.get("queries", []):
            logger.info(f"Searching [{tid}]: {query}")
            papers = search_openalex(query, max_results=max_per_topic, from_year=from_year)
            results.setdefault(tid, []).extend(papers)
            time.sleep(0.2)  # polite rate limiting

    # Deduplicate by DOI
    for tid in results:
        seen = set()
        unique = []
        for p in results[tid]:
            key = p.doi or p.title
            if key not in seen:
                seen.add(key)
                unique.append(p)
        results[tid] = unique

    total = sum(len(v) for v in results.values())
    logger.info(f"Total: {total} papers across {len(results)} topics")
    return results
