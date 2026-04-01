"""Knowledge extraction from papers.

This module prepares papers for OpenClaw to analyze. Rather than calling
a separate LLM API, the /research command guides OpenClaw through reading
each paper and extracting structured knowledge — OpenClaw IS the expert.

The extraction schema defines what to look for in each paper.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Technique:
    """An extracted ecological technique or method from a paper."""

    name: str
    description: str
    category: str  # planting | soil | water | habitat | carbon | management
    species: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)  # soil_type, climate, slope, etc.
    outcomes: dict = field(default_factory=dict)  # carbon_gain, biodiversity_increase, etc.
    timeframe_years: Optional[int] = None
    source_doi: Optional[str] = None
    source_title: Optional[str] = None
    confidence: str = "medium"  # low | medium | high (based on study design)
    bioregion_match: float = 0.0  # 0-1, how applicable to Dorset maritime

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class SpeciesInsight:
    """Species-specific knowledge extracted from a paper."""

    species_id: str  # matches species_db/species.json
    insight: str
    category: str  # growth | competition | symbiosis | soil | climate | management
    quantitative: dict = field(default_factory=dict)  # measured values
    source_doi: Optional[str] = None
    applicable_to_hooke: bool = True

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


# ── Extraction prompts (for OpenClaw to use) ─────────────────────────────────

EXTRACTION_SCHEMA = """
When analyzing a paper, extract the following into structured JSON:

1. **Techniques**: Specific methods that could be applied at Hooke Park
   - name, description, category (planting/soil/water/habitat/carbon/management)
   - species involved, site conditions required
   - measured outcomes (with units and timeframes)
   - how applicable to Dorset maritime climate (50.79°N, chalk/clay, 1000mm rain)

2. **Species insights**: Per-species knowledge relevant to our 43-species DB
   - growth rates, competition dynamics, symbiotic relationships
   - quantitative measurements (with sample sizes when available)
   - management implications

3. **Design principles**: Higher-level patterns
   - guild structures, succession pathways, planting densities
   - spatial arrangements (Miyawaki, syntropic rows, keyhole, etc.)
   - temporal sequences (what to plant when and why)

4. **Quantitative results**: Numbers we can use in our models
   - carbon sequestration rates (tCO2/ha/yr by species/system)
   - survival rates by species and method
   - soil health improvements (organic matter, pH, CEC changes)
   - biodiversity metrics (species richness, Shannon index)

Rate each extraction for:
- **Confidence**: low (observational/single site), medium (replicated), high (meta-analysis/long-term)
- **Bioregion match**: 0-1 how applicable to Hooke Park (Dorset, 50.79°N, maritime, chalk/clay)
"""

ANALYSIS_PROMPT_TEMPLATE = """
Read this paper and extract knowledge for the Hooke Park forest management system.

**Paper**: {title}
**Authors**: {authors}
**Year**: {year}
**DOI**: {doi}

**Abstract**:
{abstract}

{fulltext_section}

Apply the extraction schema. Focus on:
- Techniques applicable to UK maritime climate
- Species in our database: oak, beech, birch, alder, scots pine, douglas fir, hazel, holly, etc.
- Quantitative results we can parameterize our models with
- Syntropic/permaculture/regenerative design principles

Write extracted knowledge as JSON following the Technique and SpeciesInsight schemas.
Save to knowledge/techniques/ and update the paper index.
"""


def prepare_paper_for_analysis(paper_dict: dict) -> str:
    """Format a paper for OpenClaw to analyze.

    Returns the analysis prompt as a string.
    """
    fulltext = ""
    if paper_dict.get("fulltext_path"):
        fulltext = f"**Full text available at**: {paper_dict['fulltext_path']}\nRead the full text for detailed extraction."
    else:
        fulltext = "No full text available — extract what you can from the abstract."

    return ANALYSIS_PROMPT_TEMPLATE.format(
        title=paper_dict.get("title", "Unknown"),
        authors=", ".join(paper_dict.get("authors", [])),
        year=paper_dict.get("year", ""),
        doi=paper_dict.get("doi", ""),
        abstract=paper_dict.get("abstract", "No abstract available."),
        fulltext_section=fulltext,
    )


def save_technique(technique: Technique, knowledge_dir: str | Path):
    """Save an extracted technique to the knowledge base."""
    knowledge_dir = Path(knowledge_dir)
    techniques_dir = knowledge_dir / "techniques"
    techniques_dir.mkdir(parents=True, exist_ok=True)

    # Filename from technique name
    slug = technique.name.lower().replace(" ", "_").replace("/", "_")[:60]
    path = techniques_dir / f"{slug}.json"

    # Append if file exists (multiple sources for same technique)
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(technique.to_dict())
        else:
            existing = [existing, technique.to_dict()]
        data = existing
    else:
        data = technique.to_dict()

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved technique: {technique.name} → {path}")


def save_species_insight(insight: SpeciesInsight, knowledge_dir: str | Path):
    """Save species-specific knowledge to the knowledge base."""
    knowledge_dir = Path(knowledge_dir)
    species_dir = knowledge_dir / "species"
    species_dir.mkdir(parents=True, exist_ok=True)

    path = species_dir / f"{insight.species_id}.json"

    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(insight.to_dict())
        else:
            existing = [existing, insight.to_dict()]
        data = existing
    else:
        data = [insight.to_dict()]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved species insight: {insight.species_id} → {path}")
