"""Paper index — tracks discovered, analyzed, and extracted papers."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

INDEX_FILE = "knowledge/paper_index.json"


def _load_index(index_path: str | Path = INDEX_FILE) -> dict:
    path = Path(index_path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"papers": [], "stats": {"total": 0, "analyzed": 0, "with_techniques": 0}}


def _save_index(index: dict, index_path: str | Path = INDEX_FILE):
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


def add_paper(paper_dict: dict, index_path: str | Path = INDEX_FILE) -> bool:
    """Add a discovered paper to the index. Returns True if new."""
    index = _load_index(index_path)

    key = paper_dict.get("doi") or paper_dict.get("title", "")
    existing_keys = {p.get("doi") or p.get("title", "") for p in index["papers"]}

    if key in existing_keys:
        return False

    paper_dict.setdefault("status", "discovered")  # discovered | queued | analyzed | extracted
    paper_dict.setdefault("techniques_extracted", 0)
    paper_dict.setdefault("species_insights_extracted", 0)

    index["papers"].append(paper_dict)
    index["stats"]["total"] = len(index["papers"])
    _save_index(index, index_path)
    return True


def mark_analyzed(doi_or_title: str, techniques: int = 0, species_insights: int = 0,
                  index_path: str | Path = INDEX_FILE):
    """Mark a paper as analyzed with extraction counts."""
    index = _load_index(index_path)

    for paper in index["papers"]:
        key = paper.get("doi") or paper.get("title", "")
        if key == doi_or_title:
            paper["status"] = "extracted" if (techniques + species_insights) > 0 else "analyzed"
            paper["techniques_extracted"] = techniques
            paper["species_insights_extracted"] = species_insights
            break

    index["stats"]["analyzed"] = sum(1 for p in index["papers"] if p.get("status") in ("analyzed", "extracted"))
    index["stats"]["with_techniques"] = sum(1 for p in index["papers"] if p.get("techniques_extracted", 0) > 0)
    _save_index(index, index_path)


def get_unanalyzed(max_results: int = 10, index_path: str | Path = INDEX_FILE) -> list[dict]:
    """Get papers that haven't been analyzed yet, sorted by citation count."""
    index = _load_index(index_path)
    unanalyzed = [p for p in index["papers"] if p.get("status") in ("discovered", "queued")]
    unanalyzed.sort(key=lambda p: p.get("citation_count", 0), reverse=True)
    return unanalyzed[:max_results]


def get_stats(index_path: str | Path = INDEX_FILE) -> dict:
    """Get summary statistics of the knowledge base."""
    index = _load_index(index_path)
    stats = index.get("stats", {})

    # Count by topic
    topic_counts = {}
    for paper in index["papers"]:
        for topic in paper.get("topics", []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    stats["by_topic"] = topic_counts
    stats["by_status"] = {}
    for paper in index["papers"]:
        s = paper.get("status", "discovered")
        stats["by_status"][s] = stats["by_status"].get(s, 0) + 1

    return stats


def search_index(query: str, index_path: str | Path = INDEX_FILE) -> list[dict]:
    """Simple keyword search across paper titles and abstracts."""
    index = _load_index(index_path)
    query_lower = query.lower()
    results = []

    for paper in index["papers"]:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        if query_lower in text:
            results.append(paper)

    return results
