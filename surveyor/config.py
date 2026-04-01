"""Central configuration — GCS paths, scan definitions, thresholds, constants."""

from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).parent.parent
SPECIES_DB_PATH = REPO_ROOT / "species_db" / "species.json"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

# ── GCS ──────────────────────────────────────────────────────────────────────

GCS_BUCKET = "cybernetichooke-aebb4.firebasestorage.app"
GCS_PROJECT = "cybernetichooke-aebb4"


def gcs_url(path: str) -> str:
    """Firebase Storage download URL for a GCS object."""
    return f"https://firebasestorage.googleapis.com/v0/b/{GCS_BUCKET}/o/{quote(path, safe='')}?alt=media"


# ── Scans ────────────────────────────────────────────────────────────────────

SCANS = {
    "240913": {
        "season": "autumn",
        "year": 2024,
        "date": "2024-09-13",
        "sensor": "ZenmuseL2",
        "label": "Autumn 2024",
    },
    "250322": {
        "season": "spring",
        "year": 2025,
        "date": "2025-03-22",
        "sensor": "ZenmuseL2",
        "label": "Early Spring 2025",
    },
    "250525": {
        "season": "spring",
        "year": 2025,
        "date": "2025-05-25",
        "sensor": "ZenmuseL2",
        "label": "Spring 2025",
    },
    "260227": {
        "season": "winter",
        "year": 2026,
        "date": "2026-02-27",
        "sensor": "ZenmuseL2+Mavic3M",
        "label": "Winter 2026",
    },
}

# Scans included in the multiscan tree matching (have tree data)
MULTISCAN_IDS = ["240913", "250525", "260227"]


def scan_prefix(scan_id: str) -> str:
    """GCS path prefix for a scan."""
    s = SCANS[scan_id]
    return f"scans/{scan_id}_{s['season']}_{s['year']}"


# ── Raster products ─────────────────────────────────────────────────────────

RASTER_PRODUCTS = {
    "lidar": [
        "chm_0.5m", "dsm_0.5m", "dtm_1m", "slope_deg", "aspect_deg",
        "hillshade", "twi", "canopy_cover_pct", "ground_penetration_pct",
        "point_density", "intensity_mean",
    ],
    "spectral": ["ndvi", "ndre", "gndvi", "lci", "osavi"],
    "structure": [
        "stem_density_ha", "basal_area_m2_ha",
        "canopy_height_diversity", "subcanopy_light",
    ],
    "solar": [
        "annual_radiation_kwh", "summer_radiation_kwh",
        "winter_radiation_kwh", "light_availability",
    ],
    "drainage": ["flow_accumulation", "flow_direction", "wetness_classes"],
}

# Fields to sample from rasters → snapshot field name : (category, product)
RASTER_SAMPLE_MAP = {
    "solar_par_kwh": ("solar", "annual_radiation_kwh"),
    "twi": ("lidar", "twi"),
    "slope_deg": ("lidar", "slope_deg"),
    "ndvi": ("spectral", "ndvi"),
    "canopy_cover_pct": ("lidar", "canopy_cover_pct"),
    "flow_accumulation": ("drainage", "flow_accumulation"),
}

# ── Anomaly thresholds ──────────────────────────────────────────────────────

ANOMALY_THRESHOLDS = {
    "ndvi_drop_sigma": 2.0,
    "growth_decel_threshold": -0.05,  # m/yr²
    "crown_contract_pct_yr": 10,
    "vci_drought": 0.35,
}

# ── H3 resolutions ──────────────────────────────────────────────────────────

H3_RES_TREE = 13     # ~1.2 m — individual tree
H3_RES_VIS = 12      # ~150 m² — visualization
H3_RES_BLOCK = 11    # ~25 m — management block

# ── CRS ──────────────────────────────────────────────────────────────────────

CRS_PIPELINE = "EPSG:27700"  # British National Grid
CRS_DISPLAY = "EPSG:4326"    # WGS84
CRS_WEB = "EPSG:3857"        # Web Mercator (COGs on GCS)
