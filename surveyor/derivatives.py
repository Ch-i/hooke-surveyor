"""Temporal derivatives — height/NDVI velocity and acceleration from multi-scan data."""

import logging

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


def compute_temporal_derivatives(trees: gpd.GeoDataFrame, scan_ids: list[str]):
    """Compute per-tree temporal derivatives from multi-scan measurements.

    Adds columns: growth_m_yr, growth_accel, ndvi_trend
    """
    from .config import SCANS

    # Collect scan epochs and height columns
    epochs = []
    height_cols = []
    for sid in scan_ids:
        col = f"height_{sid}"
        if col in trees.columns:
            date = pd.Timestamp(SCANS[sid]["date"])
            epochs.append(date.year + date.day_of_year / 365.25)
            height_cols.append(col)

    if len(epochs) < 2:
        logger.warning(f"Only {len(epochs)} height columns found — need 2+ for derivatives")
        return

    epochs = np.array(epochs)
    heights = trees[height_cols].values.astype(np.float64)
    dt_total = epochs[-1] - epochs[0]

    # Height velocity (m/yr) — simple delta over total span
    if dt_total > 0:
        trees["growth_m_yr"] = np.round((heights[:, -1] - heights[:, 0]) / dt_total, 4)
    logger.info(f"Height velocity computed from {len(height_cols)} scans over {dt_total:.2f} yr")

    # Height acceleration (m/yr²) — needs 3+ scans, quadratic fit
    if len(epochs) >= 3:
        t = epochs - epochs[0]  # normalize to start at 0
        accels = np.full(len(trees), np.nan)

        for i in range(len(trees)):
            h = heights[i]
            valid = ~np.isnan(h)
            if valid.sum() >= 3:
                coeffs = np.polyfit(t[valid], h[valid], 2)
                accels[i] = round(2 * coeffs[0], 4)

        trees["growth_accel"] = accels
        logger.info("Height acceleration computed (quadratic fit)")

    # NDVI trend — if per-scan NDVI columns exist
    ndvi_cols = [f"ndvi_{sid}" for sid in scan_ids if f"ndvi_{sid}" in trees.columns]
    if len(ndvi_cols) >= 2:
        ndvi_epochs = np.array([
            epochs[i] for i, sid in enumerate(scan_ids) if f"ndvi_{sid}" in trees.columns
        ])
        ndvi_vals = trees[ndvi_cols].values.astype(np.float64)
        ndvi_dt = ndvi_epochs[-1] - ndvi_epochs[0]

        if ndvi_dt > 0:
            trees["ndvi_trend"] = np.round(
                (ndvi_vals[:, -1] - ndvi_vals[:, 0]) / ndvi_dt, 4
            )
            logger.info(f"NDVI trend computed from {len(ndvi_cols)} epochs")
