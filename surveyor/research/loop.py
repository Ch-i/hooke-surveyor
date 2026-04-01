"""Research loop — discovers papers across all topics and queues for analysis.

The loop runs in phases:
1. DISCOVER: Search OpenAlex for each topic, add new papers to index
2. PRIORITIZE: Sort unanalyzed papers by citation count × bioregion relevance
3. ANALYZE: Queue top papers for OpenClaw to read and extract
4. FOLLOW: Chase citation trails from high-value papers
5. REPEAT: Sleep and search again (new papers appear daily)
"""

import json
import logging
import time
from pathlib import Path

from .scholar import search_openalex, get_references, Paper
from .index import add_paper, get_unanalyzed, get_stats

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path("knowledge")
TOPICS_FILE = KNOWLEDGE_DIR / "topics.json"
LOOP_STATE_FILE = KNOWLEDGE_DIR / ".research_state.json"


def discover_phase(from_year: int = 2015, max_per_query: int = 20) -> int:
    """Search all topics, add new papers to index. Returns count of new papers."""
    with open(TOPICS_FILE) as f:
        topics = json.load(f)

    new_count = 0
    for topic in topics.get("topics", []):
        for query in topic.get("queries", []):
            logger.info(f"[{topic['id']}] Searching: {query}")
            try:
                papers = search_openalex(query, max_results=max_per_query, from_year=from_year)
            except Exception as e:
                logger.error(f"Search failed: {e}")
                continue

            for paper in papers:
                paper_dict = paper.to_dict()
                paper_dict["research_topic"] = topic["id"]
                if add_paper(paper_dict):
                    new_count += 1

            time.sleep(0.3)  # rate limiting

    logger.info(f"Discover phase complete: {new_count} new papers")
    return new_count


def follow_phase(max_papers: int = 5) -> int:
    """Follow citation trails from high-value analyzed papers."""
    # Get recently analyzed papers with good extractions
    from .index import _load_index
    index = _load_index()

    high_value = [
        p for p in index["papers"]
        if p.get("techniques_extracted", 0) > 0
        and p.get("s2_id")
    ]
    high_value.sort(key=lambda p: p.get("citation_count", 0), reverse=True)

    new_count = 0
    for paper in high_value[:max_papers]:
        logger.info(f"Following citations from: {paper.get('title', '')[:60]}")
        try:
            refs = get_references(paper["s2_id"])
        except Exception as e:
            logger.error(f"Citation follow failed: {e}")
            continue

        for ref in refs[:10]:
            ref_dict = ref.to_dict()
            ref_dict["discovered_via"] = paper.get("doi") or paper.get("title")
            if add_paper(ref_dict):
                new_count += 1

        time.sleep(1.0)  # S2 rate limit

    logger.info(f"Follow phase complete: {new_count} new papers from citations")
    return new_count


def prioritize_queue() -> list[dict]:
    """Return top unanalyzed papers sorted by expected value.

    Score = citation_count × bioregion_relevance × topic_weight
    """
    with open(TOPICS_FILE) as f:
        topics = json.load(f)
    topic_weights = {t["id"]: t.get("weight", 1.0) for t in topics.get("topics", [])}

    unanalyzed = get_unanalyzed(max_results=50)

    for paper in unanalyzed:
        citations = paper.get("citation_count", 0)
        topic = paper.get("research_topic", "")
        weight = topic_weights.get(topic, 1.0)

        # Boost open-access (we can read fulltext)
        oa_bonus = 1.5 if paper.get("is_open_access") else 1.0

        # Boost recent papers
        year = paper.get("year", 2020)
        recency = 1.0 + max(0, year - 2015) * 0.05

        paper["priority_score"] = citations * weight * oa_bonus * recency

    unanalyzed.sort(key=lambda p: p.get("priority_score", 0), reverse=True)
    return unanalyzed


def run_cycle(from_year: int = 2015) -> dict:
    """Run one full research cycle: discover → follow → prioritize.

    Returns summary stats.
    """
    new_discovered = discover_phase(from_year=from_year)
    new_followed = follow_phase()
    queue = prioritize_queue()
    stats = get_stats()

    summary = {
        "new_discovered": new_discovered,
        "new_from_citations": new_followed,
        "queue_length": len(queue),
        "top_5_queue": [
            {"title": p.get("title", "")[:80], "citations": p.get("citation_count", 0)}
            for p in queue[:5]
        ],
        "total_papers": stats.get("total", 0),
        "analyzed": stats.get("analyzed", 0),
        "with_techniques": stats.get("with_techniques", 0),
    }

    logger.info(f"Research cycle: {json.dumps(summary, indent=2)}")
    return summary
