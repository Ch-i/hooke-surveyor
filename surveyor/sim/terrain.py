"""Terrain data interpolation — fills missing slope, TWI, solar, NDVI.

Uses KD-tree nearest-neighbour from records that have data to those that don't.
IDW (inverse distance weighting) from k=3 nearest donors, max 100m radius.
Fallback to site median when no donor in range.
"""

import logging
from collections import defaultdict
from typing import Optional

import h3
import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# Hooke Park site medians (from observed data)
SITE_MEDIANS = {
    "slope_deg": 7.4,
    "twi": 2.0,
    "solar_par_kwh": 989.5,
    "ndvi": 0.7,
}

VALID_RANGES = {
    "slope_deg": (0, 70),
    "twi": (-1, 15),
    "solar_par_kwh": (100, 1400),
    "ndvi": (-1, 1),
}

# Approximate WGS84 conversion at Hooke Park latitude (~50.8N)
_M_PER_DEG_LAT = 111_000.0
_M_PER_DEG_LON = 70_000.0  # cos(50.8) * 111000 ≈ 70000

CONIFER_SPECIES = frozenset({
    "scots_pine",
    "douglas_fir",
    "monterey_cypress",
    "western_hemlock",
    "coast_redwood",
    "swamp_cypress",
})


def _to_xy(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to approximate local metres for KD-tree distances."""
    return lat * _M_PER_DEG_LAT, lon * _M_PER_DEG_LON


def _clamp(value: float, field: str) -> float:
    """Clamp interpolated value to valid physical range."""
    lo, hi = VALID_RANGES[field]
    return max(lo, min(hi, value))


def fill_terrain(
    records: list[dict],
    fields: list[str] = ("slope_deg", "twi", "solar_par_kwh", "ndvi"),
    max_distance_m: float = 100.0,
    k_neighbours: int = 3,
    cog_dir: str = None,
) -> list[dict]:
    """Fill missing terrain values — COG raster sampling with IDW fallback.

    Priority:
    1. Sample directly from full-site COGs (slope, TWI, CHM at 0.8m resolution)
    2. Fall back to KD-tree IDW from existing snapshot values
    3. Last resort: site median

    The COGs on ll0odog at data/260227/cog/ cover ~72% of trees.
    Combined with existing snapshot data, coverage reaches 95%+.
    """
    import os
    from pathlib import Path

    # Auto-detect COG directory
    if cog_dir is None:
        candidates = [
            "/mnt/c/Users/Aiapaec/loki-Hooke/hooke-surveyor/data/260227/cog",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "260227", "cog"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                cog_dir = c
                break

    # COG field mapping: field_name -> (cog_filename, band, scale_factor)
    COG_MAP = {
        "slope_deg": ("slope_deg_cog.tif", 1, 1.0),
        "twi": ("twi_cog.tif", 1, 1.0),
        "chm": ("chm_0.5m_cog.tif", 1, 1.0),
    }

    # Phase 1: Sample from COGs (fastest, highest quality)
    cog_sampled = {f: 0 for f in fields}
    if cog_dir and os.path.isdir(cog_dir):
        try:
            import rasterio
            from pyproj import Transformer

            for field_name in fields:
                if field_name not in COG_MAP:
                    continue
                cog_file, band, scale = COG_MAP[field_name]
                cog_path = os.path.join(cog_dir, cog_file)
                if not os.path.exists(cog_path):
                    continue

                with rasterio.open(cog_path) as src:
                    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    data = src.read(band)

                    # Vectorized transform for all records
                    lons = np.array([r.get("lon", 0) for r in records])
                    lats = np.array([r.get("lat", 0) for r in records])
                    xs, ys = transformer.transform(lons, lats)

                    for i, r in enumerate(records):
                        if r.get(field_name) is not None:
                            continue  # already has data from snapshot
                        try:
                            row, col = rasterio.transform.rowcol(src.transform, xs[i], ys[i])
                            if 0 <= row < src.height and 0 <= col < src.width:
                                val = float(data[row, col])
                                if not np.isnan(val) and val != src.nodata:
                                    r[field_name] = round(val * scale, 2)
                                    cog_sampled[field_name] += 1
                        except (IndexError, ValueError):
                            pass

                logger.info(f"{field_name}: sampled {cog_sampled[field_name]} from COG {cog_file}")
        except ImportError:
            logger.warning("rasterio/pyproj not available — skipping COG sampling")

    # Phase 2: KD-tree IDW for remaining gaps
    for field_name in fields:
        missing_indices = [i for i, r in enumerate(records) if r.get(field_name) is None]
        if not missing_indices:
            continue

        # Build donor pool from records that now have data (original + COG-sampled)
        donors = [(r["lat"], r["lon"], r[field_name]) for r in records
                   if r.get(field_name) is not None and r.get("lat") and r.get("lon")]

        if len(donors) < 3:
            # Not enough donors — use site median
            median = SITE_MEDIANS.get(field_name, 0)
            for i in missing_indices:
                records[i][field_name] = median
            logger.warning(f"{field_name}: {len(donors)} donors, filling {len(missing_indices)} with median {median}")
            continue

        # Build KD-tree from donors
        donor_coords = np.array([(d[0] * 111000, d[1] * 70000) for d in donors])  # approx meters
        donor_values = np.array([d[2] for d in donors])
        tree = KDTree(donor_coords)

        max_dist_m = max_distance_m
        idw_count = 0
        median_count = 0

        for i in missing_indices:
            r = records[i]
            pt = np.array([r["lat"] * 111000, r["lon"] * 70000])
            dists, indices = tree.query(pt, k=min(k_neighbours, len(donors)))

            if np.isscalar(dists):
                dists = np.array([dists])
                indices = np.array([indices])

            # Filter to within max distance
            mask = dists < max_dist_m
            if mask.any():
                d = dists[mask]
                v = donor_values[indices[mask]]
                d = np.maximum(d, 0.1)  # avoid division by zero
                weights = 1.0 / d
                value = float(np.sum(v * weights) / np.sum(weights))
                # Clamp to valid range
                lo, hi = VALID_RANGES.get(field_name, (-1e9, 1e9))
                value = max(lo, min(hi, value))
                r[field_name] = round(value, 2)
                idw_count += 1
            else:
                r[field_name] = SITE_MEDIANS.get(field_name, 0)
                median_count += 1

        total_before = len(records) - len(missing_indices)
        total_after = sum(1 for r in records if r.get(field_name) is not None)
        logger.info(f"{field_name}: COG={cog_sampled.get(field_name,0)}, IDW={idw_count}, median={median_count} "
                    f"({total_before}->{total_after}/{len(records)}, {total_after/len(records)*100:.1f}%)")

    # Mark interpolated records
    for r in records:
        r["_terrain_interpolated"] = True

    return records

    # Pre-compute XY coords for every record (metres)
    all_xy = np.array([_to_xy(r["lat"], r["lon"]) for r in records])

    for fld in fields:
        # Partition into donors (have value) and targets (missing)
        donor_idx = []
        target_idx = []
        for i, r in enumerate(records):
            val = r.get(fld)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                donor_idx.append(i)
            else:
                target_idx.append(i)

        n_donors = len(donor_idx)
        n_targets = len(target_idx)

        if n_targets == 0:
            logger.info("%s: all %d records already populated", fld, n_total)
            continue

        if n_donors == 0:
            # No donors at all — fill everything with site median
            median = SITE_MEDIANS[fld]
            logger.warning(
                "%s: zero donors, filling %d records with site median %.2f",
                fld, n_targets, median,
            )
            for i in target_idx:
                records[i][fld] = median
                records[i]["_terrain_interpolated"] = True
            continue

        # Build KD-tree from donor positions
        donor_xy = all_xy[donor_idx]
        donor_vals = np.array([records[i][fld] for i in donor_idx], dtype=np.float64)
        tree = KDTree(donor_xy)

        # Query targets
        target_xy = all_xy[target_idx]
        k_actual = min(k_neighbours, n_donors)
        distances, indices = tree.query(target_xy, k=k_actual)

        # query returns 1-D arrays when k=1 — normalise to 2-D
        if k_actual == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        filled_idw = 0
        filled_median = 0

        for j, rec_i in enumerate(target_idx):
            dists = distances[j]
            idxs = indices[j]

            # Keep only donors within max_distance_m
            mask = dists <= max_distance_m
            if np.any(mask):
                d = dists[mask]
                v = donor_vals[idxs[mask]]

                # Guard against zero distance (co-located records)
                zero_mask = d == 0.0
                if np.any(zero_mask):
                    # Use straight average of co-located donors
                    value = float(np.mean(v[zero_mask]))
                else:
                    # IDW: value = sum(v_i / d_i) / sum(1 / d_i)
                    weights = 1.0 / d
                    value = float(np.dot(v, weights) / np.sum(weights))

                value = _clamp(value, fld)
                filled_idw += 1
            else:
                value = SITE_MEDIANS[fld]
                filled_median += 1

            records[rec_i][fld] = round(value, 4)
            records[rec_i]["_terrain_interpolated"] = True

        logger.info(
            "%s: filled %d/%d missing — %d IDW, %d median fallback "
            "(%.1f%% coverage before, 100%% after)",
            fld,
            n_targets,
            n_total,
            filled_idw,
            filled_median,
            100.0 * n_donors / n_total,
        )

    return records


def compute_block_features(
    records: list[dict],
) -> dict[str, dict]:
    """Compute res-11 block-level aggregate features.

    For each h3_11 parent:
      conifer_fraction, height_cv, empty_fraction,
      mean_height, mean_twi, mean_slope, tree_count

    Returns {h3_11: {features...}}
    """
    blocks: dict[str, list[dict]] = defaultdict(list)

    for r in records:
        h3_13 = r.get("h3_13")
        if not h3_13:
            continue
        parent = h3.cell_to_parent(h3_13, 11)
        blocks[parent].append(r)

    result: dict[str, dict] = {}

    for block_id, members in blocks.items():
        n = len(members)

        # Heights
        heights = [
            r["height_m"]
            for r in members
            if r.get("height_m") is not None
        ]
        mean_height = float(np.mean(heights)) if heights else 0.0
        height_cv = (
            float(np.std(heights) / np.mean(heights))
            if heights and np.mean(heights) > 0
            else 0.0
        )

        # Species breakdown
        species_list = [r.get("species_detected") for r in members]
        n_conifer = sum(
            1 for s in species_list
            if s is not None and s in CONIFER_SPECIES
        )
        n_empty = sum(1 for s in species_list if s is None or s == "empty")
        conifer_fraction = n_conifer / n if n > 0 else 0.0
        empty_fraction = n_empty / n if n > 0 else 0.0

        # Terrain aggregates (use filled values)
        twi_vals = [
            r["twi"]
            for r in members
            if r.get("twi") is not None
        ]
        slope_vals = [
            r["slope_deg"]
            for r in members
            if r.get("slope_deg") is not None
        ]

        mean_twi = float(np.mean(twi_vals)) if twi_vals else SITE_MEDIANS["twi"]
        mean_slope = float(np.mean(slope_vals)) if slope_vals else SITE_MEDIANS["slope_deg"]

        result[block_id] = {
            "tree_count": n,
            "mean_height": round(mean_height, 2),
            "height_cv": round(height_cv, 4),
            "conifer_fraction": round(conifer_fraction, 4),
            "empty_fraction": round(empty_fraction, 4),
            "mean_twi": round(mean_twi, 2),
            "mean_slope": round(mean_slope, 2),
        }

    logger.info(
        "Computed block features for %d res-11 blocks from %d records",
        len(result),
        len(records),
    )

    return result
