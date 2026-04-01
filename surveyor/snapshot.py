"""Tree snapshot builder — merges all data sources into one record per H3 res-13 hex."""

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from .config import (
    SCANS, MULTISCAN_IDS, DATA_DIR, OUTPUT_DIR,
    scan_prefix, RASTER_SAMPLE_MAP, CRS_PIPELINE, CRS_DISPLAY,
)
from .derivatives import compute_temporal_derivatives
from .graph import build_neighbourhood_graph
from .guild import compute_guild_scores
from .risk import compute_risk_scores

logger = logging.getLogger(__name__)


class TreeSnapshot:
    """Builds the per-tree snapshot at H3 res-13 resolution.

    Workflow:
        load_base()        → 87,935 trees from h3_trees_all.geojson
        join_crowns()      → crown_area_m2, species_detected
        join_dbh()         → dbh_cm, dbh_confidence
        join_carbon()      → carbon_tco2
        sample_rasters()   → solar_par_kwh, twi, slope_deg, ndvi, ...
        compute_*()        → growth derivatives, neighbours, guild scores, risk
        export()           → {scan_id}_trees_res13.json
    """

    def __init__(
        self,
        scan_id: str,
        output_dir: str | Path = OUTPUT_DIR,
        data_dir: str | Path = DATA_DIR,
    ):
        self.scan_id = scan_id
        self.scan = SCANS[scan_id]
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trees: Optional[gpd.GeoDataFrame] = None
        self.graph: Optional[dict] = None

    # ── Full build ───────────────────────────────────────────────────────

    def build(self) -> Path:
        """Full snapshot build: load → enrich → compute → export."""
        logger.info(f"Building snapshot for scan {self.scan_id}")

        self.load_base()
        self.join_crowns()
        self.join_dbh()
        self.join_carbon()
        self.sample_rasters()

        compute_temporal_derivatives(self.trees, MULTISCAN_IDS)
        self.graph = build_neighbourhood_graph(self.trees)
        compute_guild_scores(self.trees, self.graph)
        compute_risk_scores(self.trees)

        path = self.export()
        logger.info(f"Snapshot complete: {path} ({len(self.trees)} trees)")
        return path

    # ── Patch (incremental) ──────────────────────────────────────────────

    def patch(self, fields: list[str]) -> Path:
        """Partial update — only recompute specified fields."""
        self.load_base()

        existing = self._load_existing_snapshot()
        if existing is not None:
            for col in existing.columns:
                if col not in self.trees.columns and col not in fields:
                    self.trees[col] = existing[col].values

        handlers = {
            "ndvi": self.sample_rasters,
            "crowns": self.join_crowns,
            "dbh": self.join_dbh,
            "carbon": self.join_carbon,
            "derivatives": lambda: compute_temporal_derivatives(self.trees, MULTISCAN_IDS),
            "risk": lambda: compute_risk_scores(self.trees),
        }

        for field in fields:
            if field in handlers:
                handlers[field]()
            else:
                logger.warning(f"Unknown patch field: {field}")

        return self.export()

    # ── Data loading ─────────────────────────────────────────────────────

    def load_base(self):
        """Load the multiscan H3 tree GeoJSON (87,935 trees)."""
        base_path = self.data_dir / "multiscan" / "h3_trees_all.geojson"
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base tree GeoJSON not found at {base_path}. "
                f"Download: scans/multiscan/h3_trees_all.geojson from GCS"
            )

        logger.info(f"Loading base: {base_path}")
        self.trees = gpd.read_file(base_path)
        logger.info(f"Loaded {len(self.trees)} trees with columns: {list(self.trees.columns)}")

    def join_crowns(self):
        """Join crown polygon metrics from crowns.gpkg or crowns_species.gpkg."""
        crowns_path = self._find_scan_file(
            "lidar/vectors/crowns_species.gpkg",
            "lidar/vectors/crowns.gpkg",
        )
        if crowns_path is None:
            logger.warning(f"No crown data for {self.scan_id}")
            return

        logger.info(f"Joining crowns: {crowns_path}")
        crowns = gpd.read_file(crowns_path)

        if crowns.crs and crowns.crs.to_epsg() == 27700:
            crowns = crowns.to_crs(epsg=4326)

        # Spatial join: nearest crown to each tree
        join_cols = ["geometry"]
        if "crown_area_m2" in crowns.columns:
            join_cols.append("crown_area_m2")
        if "height_m" in crowns.columns:
            join_cols.append("height_m")

        joined = gpd.sjoin_nearest(
            self.trees[["h3_13", "geometry"]],
            crowns[join_cols],
            how="left",
            max_distance=2.0,
        )

        if "crown_area_m2" in joined.columns:
            self.trees["crown_area_m2"] = joined["crown_area_m2"].values

        if "species" in crowns.columns:
            sp_joined = gpd.sjoin_nearest(
                self.trees[["h3_13", "geometry"]],
                crowns[["geometry", "species"]],
                how="left",
                max_distance=2.0,
            )
            self.trees["species_detected"] = sp_joined["species"].values

    def join_dbh(self):
        """Join DBH estimates from per-scan analysis JSON."""
        dbh_path = self._find_scan_file("analysis/dbh_estimates.json")
        if dbh_path is None:
            logger.warning(f"No DBH data for {self.scan_id}")
            return

        logger.info(f"Joining DBH: {dbh_path}")
        with open(dbh_path) as f:
            dbh_data = json.load(f)

        if not isinstance(dbh_data, list):
            return

        dbh_df = pd.DataFrame(dbh_data)

        if "dbh_m" in dbh_df.columns:
            dbh_df["dbh_cm"] = dbh_df["dbh_m"] * 100

        # Match by spatial proximity if coordinates available, else by index
        if "lat" in dbh_df.columns and "lon" in dbh_df.columns:
            dbh_gdf = gpd.GeoDataFrame(
                dbh_df,
                geometry=gpd.points_from_xy(dbh_df["lon"], dbh_df["lat"]),
                crs="EPSG:4326",
            )
            joined = gpd.sjoin_nearest(
                self.trees[["h3_13", "geometry"]],
                dbh_gdf[["geometry", "dbh_cm"]],
                how="left",
                max_distance=2.0,
            )
            self.trees["dbh_cm"] = joined["dbh_cm"].values
        elif len(dbh_df) == len(self.trees) and "dbh_cm" in dbh_df.columns:
            self.trees["dbh_cm"] = dbh_df["dbh_cm"].values
            if "confidence" in dbh_df.columns:
                self.trees["dbh_confidence"] = dbh_df["confidence"].values

    def join_carbon(self):
        """Join carbon estimates from per-scan analysis JSON."""
        carbon_path = self._find_scan_file("analysis/carbon_estimates.json")
        if carbon_path is None:
            logger.warning(f"No carbon data for {self.scan_id}")
            return

        logger.info(f"Joining carbon: {carbon_path}")
        with open(carbon_path) as f:
            carbon_data = json.load(f)

        if not isinstance(carbon_data, list):
            return

        carbon_df = pd.DataFrame(carbon_data)

        if "carbon_kg" in carbon_df.columns:
            carbon_df["carbon_tco2"] = carbon_df["carbon_kg"] / 1000 * 3.667

        if len(carbon_df) == len(self.trees) and "carbon_tco2" in carbon_df.columns:
            self.trees["carbon_tco2"] = carbon_df["carbon_tco2"].values

    def sample_rasters(self):
        """Sample raster values at each tree location (BNG coordinates)."""
        transformer = Transformer.from_crs(CRS_DISPLAY, CRS_PIPELINE, always_xy=True)

        # Pre-compute BNG coordinates for all trees
        bng_coords = [
            transformer.transform(row.geometry.x, row.geometry.y)
            for _, row in self.trees.iterrows()
        ]

        for field, (category, product) in RASTER_SAMPLE_MAP.items():
            raster_path = self._find_scan_file(f"{category}/{product}_cog.tif")
            if raster_path is None:
                continue

            logger.info(f"Sampling {field} from {raster_path.name}")
            try:
                with rasterio.open(raster_path) as src:
                    values = np.array([v[0] for v in src.sample(bng_coords)])
                    self.trees[field] = np.where(values == src.nodata, np.nan, values)
            except Exception as e:
                logger.error(f"Failed to sample {field}: {e}")

    # ── Export ───────────────────────────────────────────────────────────

    def export(self) -> Path:
        """Export snapshot as compact JSON (one record per tree)."""
        out_path = self.output_dir / f"{self.scan_id}_trees_res13.json"

        records = []
        for _, row in self.trees.iterrows():
            rec = row.drop("geometry").to_dict()
            rec["lat"] = round(row.geometry.y, 6)
            rec["lon"] = round(row.geometry.x, 6)
            rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
            records.append(rec)

        with open(out_path, "w") as f:
            json.dump(records, f, separators=(",", ":"))

        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info(f"Exported {len(records)} trees → {out_path} ({size_mb:.1f} MB)")
        return out_path

    def upload(self):
        """Upload snapshot + graph to GCS."""
        from .gcs import upload_to_gcs

        snapshot_path = self.output_dir / f"{self.scan_id}_trees_res13.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not built: {snapshot_path}")

        upload_to_gcs(snapshot_path, f"agent/snapshots/{self.scan_id}_trees_res13.json")

        if self.graph:
            graph_path = self.output_dir / f"{self.scan_id}_graph.json"
            with open(graph_path, "w") as f:
                json.dump(self.graph, f, separators=(",", ":"))
            upload_to_gcs(graph_path, f"agent/snapshots/{self.scan_id}_graph.json")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _find_scan_file(self, *relative_paths: str) -> Optional[Path]:
        """Try multiple relative paths under the scan data directory."""
        s = self.scan
        prefixes = [
            f"{self.scan_id}_{s['season']}_{s['year']}",
            self.scan_id,
        ]
        for rel in relative_paths:
            for prefix in prefixes:
                path = self.data_dir / prefix / rel
                if path.exists():
                    return path
        return None

    def _load_existing_snapshot(self) -> Optional[pd.DataFrame]:
        """Load existing snapshot for incremental patching."""
        path = self.output_dir / f"{self.scan_id}_trees_res13.json"
        if not path.exists():
            return None
        with open(path) as f:
            return pd.DataFrame(json.load(f))
