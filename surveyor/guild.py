"""Guild compatibility scoring — 5-component weighted matrix."""

import json
import logging

import numpy as np
import geopandas as gpd

from .config import SPECIES_DB_PATH

logger = logging.getLogger(__name__)

_species_db: dict | None = None

STRATA_ORDER = ["ground", "shrub", "understory", "canopy", "emergent"]
SUCCESSION_ORDER = {"pioneer": 0, "secondary": 1, "climax": 2}
DEPTH_ORDER = {"shallow": 0, "medium": 1, "deep": 2}

# Component weights
W_STRATUM = 0.25
W_NITROGEN = 0.25
W_SUCCESSION = 0.20
W_ROOT = 0.10
W_EXPLICIT = 0.20


def _load_species_db() -> dict:
    global _species_db
    if _species_db is None:
        with open(SPECIES_DB_PATH) as f:
            species_list = json.load(f)
        _species_db = {s["id"]: s for s in species_list}
        logger.info(f"Loaded species DB: {len(_species_db)} species")
    return _species_db


def compatibility_score(sp_a: str, sp_b: str) -> float:
    """Compute guild compatibility between two species.

    Returns score in [-1.0, +1.0]:
        > 0  = beneficial (symbiotic), plant together
        = 0  = neutral
        < 0  = antagonistic, avoid adjacency
    """
    db = _load_species_db()
    if sp_a not in db or sp_b not in db:
        return 0.0

    a, b = db[sp_a], db[sp_b]

    # 1. Stratum diversity
    sa = STRATA_ORDER.index(a["stratum"]) if a["stratum"] in STRATA_ORDER else 2
    sb = STRATA_ORDER.index(b["stratum"]) if b["stratum"] in STRATA_ORDER else 2
    diff = abs(sa - sb)
    stratum = -0.3 if diff == 0 else (0.5 if diff == 1 else 1.0)

    # 2. Nitrogen symbiosis
    roles = {a.get("nitrogen_role", "neutral"), b.get("nitrogen_role", "neutral")}
    if "fixer" in roles and "heavy_feeder" in roles:
        nitrogen = 1.0
    elif "fixer" in roles and "light_feeder" in roles:
        nitrogen = 0.5
    elif "fixer" in roles:
        nitrogen = 0.3
    else:
        nitrogen = 0.0

    # 3. Succession mixing
    sa_s = SUCCESSION_ORDER.get(a.get("succession", "secondary"), 1)
    sb_s = SUCCESSION_ORDER.get(b.get("succession", "secondary"), 1)
    succ_diff = abs(sa_s - sb_s)
    succession = 0.0 if succ_diff == 0 else (0.7 if succ_diff == 1 else 1.0)

    # 4. Root depth
    ra = DEPTH_ORDER.get(a.get("root_depth", "medium"), 1)
    rb = DEPTH_ORDER.get(b.get("root_depth", "medium"), 1)
    root_diff = abs(ra - rb)
    root = -0.2 if root_diff == 0 else (0.4 if root_diff == 1 else 0.8)

    # 5. Explicit relationships
    explicit = 0.0
    if sp_b in a.get("antagonists", []) or sp_a in b.get("antagonists", []):
        explicit = -1.0
    elif sp_b in a.get("companions", []) or sp_a in b.get("companions", []):
        explicit = 1.0

    return round(
        W_STRATUM * stratum + W_NITROGEN * nitrogen + W_SUCCESSION * succession
        + W_ROOT * root + W_EXPLICIT * explicit,
        3,
    )


def compute_guild_scores(trees: gpd.GeoDataFrame, graph: dict):
    """Compute mean guild compatibility for each tree with its neighbours.

    Writes 'guild_score' column onto trees.
    """
    db = _load_species_db()
    scores = []

    for _, row in trees.iterrows():
        h = row["h3_13"]
        sp = row.get("species_detected")

        if not sp or sp not in db or h not in graph:
            scores.append(np.nan)
            continue

        nb_scores = []
        for nb in graph[h]:
            nb_sp = nb.get("species")
            if nb_sp and nb_sp != "empty" and nb_sp in db:
                nb_scores.append(compatibility_score(sp, nb_sp))

        scores.append(round(np.mean(nb_scores), 3) if nb_scores else np.nan)

    trees["guild_score"] = scores
    valid = sum(1 for s in scores if not (isinstance(s, float) and np.isnan(s)))
    logger.info(f"Guild scores computed for {valid}/{len(trees)} trees")
