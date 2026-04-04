"""Verification engine — validates planting scheme against ecological rules."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import h3

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of all verification checks on a simulation run."""
    passed: bool = True
    checks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "summary": f"{sum(1 for c in self.checks if c['passed'])}/{len(self.checks)} checks passed",
        }


# The 7 management classes from STRATEGY.md
ALL_CLASSES = {"A", "B", "C", "D", "E", "F", "G"}


def verify_simulation(
    checkpoint_states: dict,  # {year: {h3: cell_state}}
    classifications: dict,    # {h3_11: BlockClassification}
    species_db: dict,
) -> VerificationResult:
    """Run all verification checks on simulation output.

    Checks:
    1. Existing tree protection: < 5% mortality for trees h > 5m
    2. Species-class compatibility: species match their class palette
    3. Succession visibility: species richness increases over time
    4. Mortality rate: < 10% per year site-wide
    5. Class distribution: all 7 classes represented in output

    Parameters
    ----------
    checkpoint_states : dict
        {year: {h3_13: cell_state_dict}} where cell_state_dict has keys
        like 's' (species), 'h' (height), 'hp' (health), etc.
        Year 0 must be present as the baseline.
    classifications : dict
        {h3_11: BlockClassification} from classify_all_blocks.
    species_db : dict
        {species_id: {...}} species parameters.

    Returns
    -------
    VerificationResult
        .passed is False if any critical check fails.
    """
    result = VerificationResult()

    # Import palette for species-class compatibility check
    from surveyor.sim.planting import CLASS_SPECIES_PALETTES

    # Build a lookup: h3_13 -> management_class via h3_11 parent
    cell_to_class = {}
    for h3_11, block_cls in classifications.items():
        cls_label = (
            block_cls.management_class
            if hasattr(block_cls, "management_class")
            else block_cls
        )
        cell_to_class[h3_11] = cls_label

    def _get_class(h3_13: str) -> Optional[str]:
        """Resolve management class for a res-13 cell."""
        try:
            parent = h3.cell_to_parent(h3_13, 11)
            return cell_to_class.get(parent)
        except Exception:
            return None

    sorted_years = sorted(checkpoint_states.keys())

    # ──────────────────────────────────────────────────────────────────────
    # CHECK 1: Existing tree protection — < 5% mortality for trees h > 5m
    # ──────────────────────────────────────────────────────────────────────
    check1_name = "existing_tree_protection"
    if len(sorted_years) >= 2:
        yr0 = sorted_years[0]
        yr_last = sorted_years[-1]
        state0 = checkpoint_states[yr0]
        state_last = checkpoint_states[yr_last]

        # Count established trees (h > 5m) at year 0
        established_at_start = set()
        for h3_id, cell in state0.items():
            height = cell.get("h", 0) if isinstance(cell, dict) else getattr(cell, "height_m", 0)
            species = cell.get("s") if isinstance(cell, dict) else getattr(cell, "species", None)
            if species and height > 5.0:
                established_at_start.add(h3_id)

        if established_at_start:
            # How many of those are dead/empty at end?
            lost = 0
            for h3_id in established_at_start:
                end_cell = state_last.get(h3_id)
                if end_cell is None:
                    lost += 1
                    continue
                end_species = (
                    end_cell.get("s") if isinstance(end_cell, dict)
                    else getattr(end_cell, "species", None)
                )
                end_health = (
                    end_cell.get("hp", 0) if isinstance(end_cell, dict)
                    else getattr(end_cell, "health", 0)
                )
                if not end_species or end_health <= 0:
                    lost += 1

            mortality_pct = (lost / len(established_at_start)) * 100
            passed = mortality_pct < 5.0
            result.checks.append({
                "name": check1_name,
                "passed": passed,
                "detail": (
                    f"{lost}/{len(established_at_start)} established trees (h>5m) lost "
                    f"({mortality_pct:.1f}% mortality, threshold <5%)"
                ),
            })
            if not passed:
                result.passed = False
        else:
            result.checks.append({
                "name": check1_name,
                "passed": True,
                "detail": "No established trees (h>5m) found at year 0 — check skipped",
            })
    else:
        result.checks.append({
            "name": check1_name,
            "passed": True,
            "detail": "Insufficient checkpoints (need >= 2) — check skipped",
        })

    # ──────────────────────────────────────────────────────────────────────
    # CHECK 2: Species-class compatibility — planted species match palette
    # ──────────────────────────────────────────────────────────────────────
    check2_name = "species_class_compatibility"
    if len(sorted_years) >= 2:
        yr0 = sorted_years[0]
        state0 = checkpoint_states[yr0]

        # Collect species that are NEW (not in year 0) across all later checkpoints
        baseline_species = {}
        for h3_id, cell in state0.items():
            sp = cell.get("s") if isinstance(cell, dict) else getattr(cell, "species", None)
            if sp:
                baseline_species[h3_id] = sp

        total_new = 0
        violations = 0
        violation_examples = []

        for yr in sorted_years[1:]:
            state = checkpoint_states[yr]
            for h3_id, cell in state.items():
                sp = cell.get("s") if isinstance(cell, dict) else getattr(cell, "species", None)
                if not sp:
                    continue
                # Only check newly planted species (not in baseline or changed species)
                if h3_id in baseline_species and baseline_species[h3_id] == sp:
                    continue

                total_new += 1
                mgmt_class = _get_class(h3_id)
                if mgmt_class and mgmt_class in CLASS_SPECIES_PALETTES:
                    palette = CLASS_SPECIES_PALETTES[mgmt_class]
                    allowed = set()
                    for spp_list in palette.values():
                        allowed.update(spp_list)
                    if sp not in allowed:
                        violations += 1
                        if len(violation_examples) < 5:
                            violation_examples.append(
                                f"{sp} in class {mgmt_class} ({h3_id[:16]}...)"
                            )

        if total_new > 0:
            violation_pct = (violations / total_new) * 100
            passed = violation_pct < 15.0  # allow some flexibility for fallback species
            detail = (
                f"{violations}/{total_new} new plantings outside class palette "
                f"({violation_pct:.1f}%, threshold <15%)"
            )
            if violation_examples:
                detail += f". Examples: {'; '.join(violation_examples)}"
        else:
            passed = True
            detail = "No new plantings detected — check skipped"

        result.checks.append({
            "name": check2_name,
            "passed": passed,
            "detail": detail,
        })
        if not passed:
            result.passed = False
    else:
        result.checks.append({
            "name": check2_name,
            "passed": True,
            "detail": "Insufficient checkpoints — check skipped",
        })

    # ──────────────────────────────────────────────────────────────────────
    # CHECK 3: Succession visibility — species richness increases over time
    # ──────────────────────────────────────────────────────────────────────
    check3_name = "succession_visibility"
    if len(sorted_years) >= 2:
        yr_first = sorted_years[0]
        yr_last = sorted_years[-1]

        def _richness(state: dict) -> int:
            species_set = set()
            for cell in state.values():
                sp = cell.get("s") if isinstance(cell, dict) else getattr(cell, "species", None)
                if sp:
                    species_set.add(sp)
            return len(species_set)

        richness_start = _richness(checkpoint_states[yr_first])
        richness_end = _richness(checkpoint_states[yr_last])

        passed = richness_end >= richness_start
        result.checks.append({
            "name": check3_name,
            "passed": passed,
            "detail": (
                f"Species richness: {richness_start} (year {yr_first}) -> "
                f"{richness_end} (year {yr_last}). "
                f"{'Increased/maintained' if passed else 'DECREASED — succession failed'}"
            ),
        })
        if not passed:
            result.passed = False
    else:
        result.checks.append({
            "name": check3_name,
            "passed": True,
            "detail": "Insufficient checkpoints — check skipped",
        })

    # ──────────────────────────────────────────────────────────────────────
    # CHECK 4: Mortality rate — < 10% per year site-wide
    # ──────────────────────────────────────────────────────────────────────
    check4_name = "annual_mortality_rate"
    max_annual_mortality = 0.0
    worst_year_pair = None
    check4_passed = True

    for i in range(1, len(sorted_years)):
        yr_prev = sorted_years[i - 1]
        yr_curr = sorted_years[i]
        state_prev = checkpoint_states[yr_prev]
        state_curr = checkpoint_states[yr_curr]

        # Count alive in each
        alive_prev = 0
        for cell in state_prev.values():
            sp = cell.get("s") if isinstance(cell, dict) else getattr(cell, "species", None)
            hp = cell.get("hp", 0) if isinstance(cell, dict) else getattr(cell, "health", 0)
            if sp and hp > 0:
                alive_prev += 1

        alive_curr = 0
        for cell in state_curr.values():
            sp = cell.get("s") if isinstance(cell, dict) else getattr(cell, "species", None)
            hp = cell.get("hp", 0) if isinstance(cell, dict) else getattr(cell, "health", 0)
            if sp and hp > 0:
                alive_curr += 1

        if alive_prev > 0:
            # Net deaths (ignoring new births for mortality calc)
            # Deaths = alive_prev - (alive_curr - new_births), but we approximate
            # conservatively by just looking at net loss
            year_span = yr_curr - yr_prev
            if year_span <= 0:
                continue
            net_loss = max(0, alive_prev - alive_curr)
            annual_mortality = (net_loss / alive_prev) / year_span * 100

            if annual_mortality > max_annual_mortality:
                max_annual_mortality = annual_mortality
                worst_year_pair = (yr_prev, yr_curr)

    if worst_year_pair:
        check4_passed = max_annual_mortality < 10.0
        result.checks.append({
            "name": check4_name,
            "passed": check4_passed,
            "detail": (
                f"Peak annual mortality: {max_annual_mortality:.2f}% "
                f"(years {worst_year_pair[0]}-{worst_year_pair[1]}, threshold <10%)"
            ),
        })
        if not check4_passed:
            result.passed = False
    else:
        result.checks.append({
            "name": check4_name,
            "passed": True,
            "detail": "No year-over-year comparison possible — check skipped",
        })

    # ──────────────────────────────────────────────────────────────────────
    # CHECK 5: Class distribution — all 7 classes represented in output
    # ──────────────────────────────────────────────────────────────────────
    check5_name = "class_distribution"
    observed_classes = set()
    for h3_11, block_cls in classifications.items():
        cls_label = (
            block_cls.management_class
            if hasattr(block_cls, "management_class")
            else block_cls
        )
        observed_classes.add(cls_label)

    missing = ALL_CLASSES - observed_classes
    check5_passed = len(missing) == 0
    if check5_passed:
        # Count per class
        class_counts = Counter()
        for block_cls in classifications.values():
            cls_label = (
                block_cls.management_class
                if hasattr(block_cls, "management_class")
                else block_cls
            )
            class_counts[cls_label] += 1
        dist_str = ", ".join(f"{c}={class_counts[c]}" for c in sorted(class_counts))
        detail = f"All 7 classes represented: {dist_str}"
    else:
        detail = (
            f"Missing classes: {sorted(missing)}. "
            f"Present: {sorted(observed_classes)}. "
            f"Site may lack terrain variety for missing classes (non-critical)."
        )

    result.checks.append({
        "name": check5_name,
        "passed": check5_passed,
        "detail": detail,
    })
    # Class distribution is informational — missing a class is not critical
    # for small sites, so we do NOT set result.passed = False here.

    # ── Final summary ────────────────────────────────────────────────────
    n_passed = sum(1 for c in result.checks if c["passed"])
    n_total = len(result.checks)
    logger.info(
        "Verification: %d/%d checks passed (overall: %s)",
        n_passed, n_total, "PASS" if result.passed else "FAIL",
    )
    for check in result.checks:
        level = logging.INFO if check["passed"] else logging.WARNING
        logger.log(level, "  [%s] %s: %s",
                   "PASS" if check["passed"] else "FAIL",
                   check["name"], check["detail"])

    return result
