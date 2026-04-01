"""Risk scoring — drought, structural, loss risk per tree."""

import logging

import numpy as np
import geopandas as gpd

logger = logging.getLogger(__name__)


def compute_risk_scores(trees: gpd.GeoDataFrame):
    """Compute per-tree risk scores from available signals.

    Writes columns: risk_drought, risk_structural, risk_loss, risk_overall
    """
    n = len(trees)

    # ── Drought risk (0–1) ───────────────────────────────────────────────
    # Low NDVI + declining trend + dry site (low TWI)
    drought = np.zeros(n)

    if "ndvi" in trees.columns:
        ndvi = trees["ndvi"].fillna(0.5).values
        drought += np.clip(1.0 - ndvi / 0.8, 0, 1) * 0.4

    if "ndvi_trend" in trees.columns:
        trend = trees["ndvi_trend"].fillna(0).values
        drought += np.clip(-trend / 0.1, 0, 1) * 0.3

    if "twi" in trees.columns:
        twi = trees["twi"].fillna(8).values
        drought += np.clip(1.0 - twi / 12, 0, 1) * 0.3

    trees["risk_drought"] = np.round(np.clip(drought, 0, 1), 3)

    # ── Structural risk (0–1) ────────────────────────────────────────────
    # Decelerating growth + steep slope
    structural = np.zeros(n)

    if "growth_accel" in trees.columns:
        accel = trees["growth_accel"].fillna(0).values
        structural += np.clip(-accel / 0.1, 0, 1) * 0.5

    if "slope_deg" in trees.columns:
        slope = trees["slope_deg"].fillna(0).values
        structural += np.clip(slope / 40, 0, 1) * 0.3

    if "crown_area_m2" in trees.columns:
        # TODO: crown contraction rate needs temporal crown data
        pass

    trees["risk_structural"] = np.round(np.clip(structural, 0, 1), 3)

    # ── Loss risk (0–1) ─────────────────────────────────────────────────
    # Already lost or observed in few scans
    loss = np.zeros(n)

    if "status" in trees.columns:
        loss[trees["status"] == "lost"] = 0.9
        loss[trees["status"] == "recruited"] = 0.3

    if "n_scans" in trees.columns:
        single_scan = trees["n_scans"] == 1
        loss[single_scan] = np.maximum(loss[single_scan], 0.4)

    trees["risk_loss"] = np.round(loss, 3)

    # ── Overall (weighted) ───────────────────────────────────────────────
    trees["risk_overall"] = np.round(
        0.40 * trees["risk_drought"]
        + 0.35 * trees["risk_structural"]
        + 0.25 * trees["risk_loss"],
        3,
    )

    logger.info(
        f"Risk scores: drought μ={trees['risk_drought'].mean():.3f}, "
        f"structural μ={trees['risk_structural'].mean():.3f}, "
        f"loss μ={trees['risk_loss'].mean():.3f}, "
        f"overall μ={trees['risk_overall'].mean():.3f}"
    )
