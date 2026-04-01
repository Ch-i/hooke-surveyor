"""Silva crown segmentation with CUDA-accelerated CHM normalization.

Replaces the R-based lidR silva2016 that OOMs on CHM normalization.
Uses cupy for GPU array ops, falls back to numpy/scipy when no CUDA.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from scipy.ndimage import label, maximum_filter, watershed_ift
import geopandas as gpd
from shapely.geometry import shape

logger = logging.getLogger(__name__)

# ── CUDA availability ────────────────────────────────────────────────────────

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian

    HAS_CUDA = True
    logger.info(f"CUDA available: {cp.cuda.runtime.getDeviceCount()} GPU(s)")
except ImportError:
    HAS_CUDA = False
    logger.warning("cupy not available — falling back to CPU")


# ── CHM normalization ────────────────────────────────────────────────────────


def normalize_chm(
    dsm_path: str, dtm_path: str, output_path: Optional[str] = None
) -> tuple[np.ndarray, dict]:
    """Compute CHM = DSM - DTM, optionally on GPU.

    This is the step where lidR OOMs — R's normalize_height() loads the
    full point cloud. We operate on the raster directly.

    Returns (chm_array, rasterio_profile).
    """
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype(np.float32)

    if HAS_CUDA:
        logger.info("Normalizing CHM on GPU")
        dsm_gpu = cp.asarray(dsm)
        dtm_gpu = cp.asarray(dtm)
        chm_gpu = cp.clip(dsm_gpu - dtm_gpu, 0, 60)
        chm = cp.asnumpy(chm_gpu)
        del dsm_gpu, dtm_gpu, chm_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        logger.info("Normalizing CHM on CPU")
        chm = np.clip(dsm - dtm, 0, 60)

    if output_path:
        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1, nodata=-9999)
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(chm, 1)
        logger.info(f"CHM written to {output_path}")

    return chm, {"profile": profile, "transform": transform, "crs": crs}


# ── Treetop detection ────────────────────────────────────────────────────────

# Silva2016 variable window: height bands → search window size
HEIGHT_BANDS = [
    (2, 8, 3),      # small trees: 3×3
    (8, 15, 5),     # medium: 5×5
    (15, 25, 7),    # tall: 7×7
    (25, 60, 9),    # very tall: 9×9
]


def detect_treetops(
    chm: np.ndarray, min_height: float = 2.0, smooth_sigma: float = 1.0
) -> np.ndarray:
    """Find local maxima in smoothed CHM as candidate treetops.

    Uses variable window size based on height (taller trees → wider crowns),
    matching the silva2016 approach.
    """
    if HAS_CUDA:
        chm_smooth = cp.asnumpy(gpu_gaussian(cp.asarray(chm), sigma=smooth_sigma))
    else:
        chm_smooth = ndimage.gaussian_filter(chm, sigma=smooth_sigma)

    treetops = np.zeros_like(chm, dtype=bool)

    for h_min, h_max, win_size in HEIGHT_BANDS:
        mask = (chm_smooth >= h_min) & (chm_smooth < h_max)
        if not mask.any():
            continue
        local_max = maximum_filter(chm_smooth, size=win_size)
        band_tops = (chm_smooth == local_max) & mask & (chm_smooth >= min_height)
        treetops |= band_tops

    n_tops = int(treetops.sum())
    logger.info(f"Detected {n_tops} treetop candidates (min_height={min_height}m)")
    return treetops


# ── Crown segmentation ───────────────────────────────────────────────────────


def segment_crowns(
    chm: np.ndarray, treetops: np.ndarray, min_height: float = 2.0
) -> tuple[np.ndarray, int]:
    """Marker-controlled watershed segmentation (silva2016 style).

    Each treetop seeds a watershed basin. Crown boundaries form where basins meet.
    """
    markers, n_trees = label(treetops)
    logger.info(f"Segmenting {n_trees} crowns via watershed")

    # Invert CHM: watershed finds basins, we want peaks
    chm_inv = chm.max() - chm
    chm_inv[chm < min_height] = chm_inv.max()  # barrier at ground level

    crowns = watershed_ift(chm_inv.astype(np.int32), markers.astype(np.int32))
    crowns[chm < min_height] = 0  # mask out ground

    return crowns, n_trees


# ── Vectorization ────────────────────────────────────────────────────────────


def crowns_to_geodataframe(
    crowns: np.ndarray, chm: np.ndarray, transform, crs
) -> gpd.GeoDataFrame:
    """Convert crown label raster to GeoDataFrame with per-crown metrics."""
    pixel_area = abs(transform.a * transform.e)
    records = []

    for geom_dict, crown_id in shapes(crowns.astype(np.int32), transform=transform):
        if crown_id == 0:
            continue

        mask = crowns == int(crown_id)
        crown_heights = chm[mask]

        records.append({
            "geometry": shape(geom_dict),
            "tree_id": int(crown_id),
            "height_m": round(float(np.max(crown_heights)), 2),
            "height_mean_m": round(float(np.mean(crown_heights)), 2),
            "crown_area_m2": round(float(np.sum(mask) * pixel_area), 2),
        })

    gdf = gpd.GeoDataFrame(records, crs=crs)
    logger.info(f"Vectorized {len(gdf)} crown polygons")
    return gdf


# ── Full pipeline ────────────────────────────────────────────────────────────


def run_silva(
    dsm_path: str,
    dtm_path: str,
    output_dir: str,
    min_height: float = 2.0,
    smooth_sigma: float = 1.0,
) -> gpd.GeoDataFrame:
    """Full silva2016-style pipeline: normalize → detect → segment → vectorize.

    Args:
        dsm_path: Path to DSM raster (EPSG:27700)
        dtm_path: Path to DTM raster (EPSG:27700)
        output_dir: Directory for outputs (crowns.gpkg, tree_tops.gpkg, chm_normalized.tif)
        min_height: Minimum height (m) to count as a tree
        smooth_sigma: Gaussian smoothing sigma for treetop detection

    Returns:
        GeoDataFrame of crown polygons with tree_id, height_m, crown_area_m2
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normalize CHM
    chm_path = str(output_dir / "chm_normalized.tif")
    chm, meta = normalize_chm(dsm_path, dtm_path, chm_path)
    logger.info(f"CHM shape: {chm.shape}, range: [{chm.min():.1f}, {chm.max():.1f}] m")

    # 2. Detect treetops
    treetops = detect_treetops(chm, min_height=min_height, smooth_sigma=smooth_sigma)

    # 3. Segment crowns
    crown_raster, n_trees = segment_crowns(chm, treetops, min_height=min_height)

    # 4. Vectorize crowns
    crowns_gdf = crowns_to_geodataframe(crown_raster, chm, meta["transform"], meta["crs"])

    # 5. Export crown polygons
    crowns_path = output_dir / "crowns.gpkg"
    crowns_gdf.to_file(crowns_path, driver="GPKG")

    # 6. Export treetop points (centroids of crowns)
    tops_gdf = crowns_gdf.copy()
    tops_gdf["geometry"] = tops_gdf.geometry.centroid
    tops_gdf.to_file(output_dir / "tree_tops.gpkg", driver="GPKG")

    logger.info(
        f"Silva complete: {len(crowns_gdf)} crowns, "
        f"mean height {crowns_gdf['height_m'].mean():.1f}m, "
        f"mean area {crowns_gdf['crown_area_m2'].mean():.1f}m²"
    )
    return crowns_gdf
