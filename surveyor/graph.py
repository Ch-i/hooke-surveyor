"""H3 neighbourhood graph — k-ring(1) for each tree hex."""

import logging

import h3
import geopandas as gpd

logger = logging.getLogger(__name__)


def build_neighbourhood_graph(trees: gpd.GeoDataFrame) -> dict:
    """Build k-ring(1) neighbourhood graph for all trees.

    Returns dict mapping each h3_13 → list of 6 neighbour dicts:
        {"h3": str, "height_m": float, "species": str | None}

    Also writes 'neighbours' and 'neighbour_species' columns onto trees.
    """
    logger.info(f"Building neighbourhood graph for {len(trees)} trees")

    h3_to_idx = {h: i for i, h in enumerate(trees["h3_13"])}
    has_species = "species_detected" in trees.columns
    has_height = "height_m" in trees.columns

    graph = {}
    neighbours_col = []
    neighbour_species_col = []

    for _, row in trees.iterrows():
        h = row["h3_13"]
        ring = h3.k_ring(h, 1) - {h}

        neighbours = []
        for nb_h3 in sorted(ring):
            if nb_h3 in h3_to_idx:
                nb_row = trees.iloc[h3_to_idx[nb_h3]]
                neighbours.append({
                    "h3": nb_h3,
                    "height_m": float(nb_row["height_m"]) if has_height else 0.0,
                    "species": nb_row["species_detected"] if has_species else None,
                })
            else:
                neighbours.append({"h3": nb_h3, "height_m": 0.0, "species": "empty"})

        graph[h] = neighbours
        neighbours_col.append([n["h3"] for n in neighbours])
        neighbour_species_col.append([n["species"] for n in neighbours])

    trees["neighbours"] = neighbours_col
    trees["neighbour_species"] = neighbour_species_col

    logger.info(f"Graph built: {len(graph)} nodes, {sum(len(v) for v in graph.values())} edges")
    return graph
