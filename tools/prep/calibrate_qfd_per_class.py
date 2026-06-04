#!/usr/bin/env python3
"""
calibrate_qfd_per_class.py — Per-receptor-class Bayesian optimisation of QFD weights.

Classifies each receptor in a validation set into one of five categories:
  metalloenzyme | kinase | nuclear_receptor | gpcr | other

Then uses scikit-optimize (skopt) Gaussian-process Bayesian optimisation to
find the QFD weight triple (w_elec, w_desolv, w_info) that maximises top-1
RMSD < 2 Å success rate for each class separately.

The result is written to qfd_class_weights.json, which can be loaded by
QFD-enabled docking scripts to select per-class weights at runtime.

Requirements:
  pip install scikit-optimize numpy

Usage:
  python3 calibrate_qfd_per_class.py \\
      --results_dir /path/to/redock_results \\
      [--rmsd_cutoff 2.0] [--top_n 5] [--n_calls 50] \\
      [--out_json qfd_class_weights.json]

Expected layout of results_dir:
  results_dir/
    1ABC/
      receptor.pdbqt     (or 1ABC_receptor.pdbqt)
      scores.tsv         (pose_id  vina_score  qfd_elec  qfd_desolv  qfd_info  rmsd)
    2XYZ/ ...

scores.tsv must have columns: pose_id, vina_score, qfd_elec, qfd_desolv, qfd_info, rmsd
These represent the raw per-term QFD contributions already precomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Real
    _SKOPT_AVAILABLE = True
except ImportError:
    _SKOPT_AVAILABLE = False
    warnings.warn(
        "scikit-optimize not found — falling back to grid search. "
        "Install with: pip install scikit-optimize",
        ImportWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Receptor classification
# ---------------------------------------------------------------------------

# Keywords in PDB HEADER / COMPND records that signal receptor class.
# Checked case-insensitively.
_CLASS_KEYWORDS: dict[str, list[str]] = {
    "metalloenzyme": [
        "zinc", "copper", "carbonic anhydrase", "metalloprotease",
        "metallo", "matrix metalloprotein", "histone deacetylase", "hdac",
        "thermolysin", "carboxypeptidase",
    ],
    "kinase": [
        "kinase", "phosphotransfer", "phosphorylase", "tyrosine receptor",
    ],
    "nuclear_receptor": [
        "nuclear receptor", "androgen receptor", "estrogen receptor",
        "glucocorticoid receptor", "thyroid receptor", "ppar",
        "retinoic acid receptor", "vitamin d receptor",
    ],
    "gpcr": [
        "g protein-coupled", "gpcr", "adrenergic receptor", "adenosine receptor",
        "dopamine receptor", "opioid receptor", "serotonin receptor",
        "rhodopsin",
    ],
}

# PDBQT metal atom types — presence implies metalloenzyme override
_METAL_TYPES: frozenset[str] = frozenset({
    "Zn", "ZN", "Cu", "CU", "Fe", "FE", "Co", "CO",
    "Mn", "MN", "Mg", "MG", "Ni", "NI", "Mo", "MO",
})


@dataclass(frozen=True)
class ReceptorFeatures:
    pdb_id: str
    header_text: str   # raw HEADER + COMPND lines for keyword matching
    has_metal: bool    # HETATM metal detected in PDBQT


def _read_receptor_features(receptor_path: Path, pdb_id: str) -> ReceptorFeatures:
    text = receptor_path.read_text(errors="ignore")
    header_lines: list[str] = []
    has_metal = False
    for line in text.splitlines():
        if line.startswith(("HEADER", "COMPND", "REMARK")):
            header_lines.append(line.lower())
        if line.startswith(("ATOM", "HETATM")):
            ad4 = line[77:].strip().split()[0] if len(line) > 77 else ""
            if ad4 in _METAL_TYPES:
                has_metal = True
    return ReceptorFeatures(
        pdb_id=pdb_id,
        header_text=" ".join(header_lines),
        has_metal=has_metal,
    )


def classify_receptor(feat: ReceptorFeatures) -> str:
    """
    Returns one of: metalloenzyme | kinase | nuclear_receptor | gpcr | other.

    Metal presence overrides keyword-based class to metalloenzyme because the
    QFD weight calibration for metals is the highest-value case.
    """
    if feat.has_metal:
        return "metalloenzyme"
    text = feat.header_text
    for cls, keywords in _CLASS_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return cls
    return "other"


# ---------------------------------------------------------------------------
# scores.tsv parsing
# ---------------------------------------------------------------------------

@dataclass
class PoseRecord:
    target: str
    pose_id: int
    vina_score: float
    qfd_elec: float
    qfd_desolv: float
    qfd_info: float
    rmsd: float


def _parse_scores_tsv(path: Path, pdb_id: str) -> list[PoseRecord]:
    records: list[PoseRecord] = []
    lines = path.read_text(errors="ignore").splitlines()
    if not lines:
        return records
    header = [h.lower() for h in lines[0].split("\t")]
    try:
        idx_pose = header.index("pose_id")
        idx_vina = header.index("vina_score")
        idx_elec = header.index("qfd_elec")
        idx_des  = header.index("qfd_desolv")
        idx_info = header.index("qfd_info")
        idx_rmsd = header.index("rmsd")
    except ValueError as exc:
        warnings.warn(f"{path}: missing column — {exc}", RuntimeWarning, stacklevel=2)
        return records

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(idx_pose, idx_vina, idx_elec, idx_des, idx_info, idx_rmsd):
            continue
        try:
            records.append(PoseRecord(
                target=pdb_id,
                pose_id=int(parts[idx_pose]),
                vina_score=float(parts[idx_vina]),
                qfd_elec=float(parts[idx_elec]),
                qfd_desolv=float(parts[idx_des]),
                qfd_info=float(parts[idx_info]),
                rmsd=float(parts[idx_rmsd]),
            ))
        except ValueError:
            continue
    return records


# ---------------------------------------------------------------------------
# Success-rate objective
# ---------------------------------------------------------------------------

def _composite_score(
    records: list[PoseRecord],
    w_elec: float,
    w_desolv: float,
    w_info: float,
) -> float:
    """
    Return the composite score for a single pose given weight triple.
    Lower = better (mirrors Vina convention: more negative = better binding).
    """
    return (
        records[0].vina_score
        + w_elec   * records[0].qfd_elec
        + w_desolv * records[0].qfd_desolv
        + w_info   * records[0].qfd_info
    )


def success_rate_for_weights(
    targets: dict[str, list[PoseRecord]],
    w_elec: float,
    w_desolv: float,
    w_info: float,
    rmsd_cutoff: float = 2.0,
    top_n: int = 5,
) -> float:
    """
    For each target, rerank top_n poses by composite score and check if rank-1
    has RMSD < rmsd_cutoff.  Returns fraction of targets that succeed.
    """
    n_success = 0
    for pdb_id, records in targets.items():
        candidates = sorted(
            records[:top_n],
            key=lambda r: (
                r.vina_score
                + w_elec * r.qfd_elec
                + w_desolv * r.qfd_desolv
                + w_info * r.qfd_info
            ),
        )
        if candidates and candidates[0].rmsd < rmsd_cutoff:
            n_success += 1
    return n_success / len(targets) if targets else 0.0


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

_WEIGHT_SPACE = [
    Real(-0.2, 0.5, name="w_elec"),
    Real(-0.2, 0.5, name="w_desolv"),
    Real(-0.1, 0.3, name="w_info"),
]

_GRID_POINTS = 5  # per axis for fallback grid search → 5^3 = 125 evals


def _bayesian_optimise(
    targets: dict[str, list[PoseRecord]],
    rmsd_cutoff: float,
    top_n: int,
    n_calls: int,
) -> tuple[float, float, float]:
    """Run skopt GP minimise, return best (w_elec, w_desolv, w_info)."""

    def objective(params: list[float]) -> float:
        w_e, w_d, w_i = params
        return -success_rate_for_weights(targets, w_e, w_d, w_i, rmsd_cutoff, top_n)

    result = gp_minimize(
        objective,
        _WEIGHT_SPACE,
        n_calls=n_calls,
        random_state=42,
        noise=1e-10,
        n_initial_points=max(10, n_calls // 5),
    )
    w_e, w_d, w_i = result.x
    return float(w_e), float(w_d), float(w_i)


def _grid_search_optimise(
    targets: dict[str, list[PoseRecord]],
    rmsd_cutoff: float,
    top_n: int,
) -> tuple[float, float, float]:
    """Fallback grid search when skopt is unavailable."""
    lo, hi, n = -0.1, 0.4, _GRID_POINTS
    best_rate = -1.0
    best_w = (0.05, 0.05, 0.02)
    vals = np.linspace(lo, hi, n)
    for w_e in vals:
        for w_d in vals:
            for w_i in np.linspace(-0.05, 0.2, n):
                r = success_rate_for_weights(targets, float(w_e), float(w_d), float(w_i), rmsd_cutoff, top_n)
                if r > best_rate:
                    best_rate = r
                    best_w = (float(w_e), float(w_d), float(w_i))
    return best_w


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

_RECEPTOR_NAMES = [
    "receptor.pdbqt",
    "{pdb_id}_receptor.pdbqt",
    "{pdb_id}.pdbqt",
    "receptor_prep.pdbqt",
]


def _find_receptor(target_dir: Path, pdb_id: str) -> Optional[Path]:
    for tmpl in _RECEPTOR_NAMES:
        candidate = target_dir / tmpl.format(pdb_id=pdb_id)
        if candidate.exists():
            return candidate
    # broader glob
    hits = list(target_dir.glob("*receptor*.pdbqt"))
    return hits[0] if hits else None


def run_calibration(
    results_dir: Path,
    rmsd_cutoff: float = 2.0,
    top_n: int = 5,
    n_calls: int = 50,
    out_json: Path = Path("qfd_class_weights.json"),
) -> dict[str, dict]:
    """
    Main calibration entry point.

    Returns per-class weight dict (also written to out_json).
    """
    # --- Collect targets ---
    by_class: dict[str, dict[str, list[PoseRecord]]] = {
        "metalloenzyme": {},
        "kinase": {},
        "nuclear_receptor": {},
        "gpcr": {},
        "other": {},
    }
    n_skipped = 0
    n_loaded = 0

    for target_dir in sorted(results_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        pdb_id = target_dir.name
        scores_tsv = target_dir / "scores.tsv"
        if not scores_tsv.exists():
            n_skipped += 1
            continue

        records = _parse_scores_tsv(scores_tsv, pdb_id)
        if not records:
            n_skipped += 1
            continue

        receptor_path = _find_receptor(target_dir, pdb_id)
        if receptor_path:
            feat = _read_receptor_features(receptor_path, pdb_id)
        else:
            # No receptor PDBQT — classify as "other"
            feat = ReceptorFeatures(pdb_id=pdb_id, header_text="", has_metal=False)

        cls = classify_receptor(feat)
        by_class[cls][pdb_id] = records
        n_loaded += 1

    print(f"Loaded {n_loaded} targets ({n_skipped} skipped).")
    for cls, tgts in by_class.items():
        print(f"  {cls}: {len(tgts)} targets")

    # --- Default weights (used when class has < 10 targets) ---
    defaults = {"w_elec": 0.05, "w_desolv": 0.05, "w_info": 0.02}

    # --- Optimise per class ---
    output: dict[str, dict] = {}

    for cls, targets in by_class.items():
        if len(targets) < 10:
            print(f"\n[{cls}] Too few targets ({len(targets)}) — using defaults.")
            baseline = success_rate_for_weights(
                targets,
                defaults["w_elec"], defaults["w_desolv"], defaults["w_info"],
                rmsd_cutoff, top_n,
            ) if targets else 0.0
            output[cls] = {
                **defaults,
                "n_targets": len(targets),
                "success_rate": round(baseline, 4),
                "method": "default",
            }
            continue

        print(f"\n[{cls}] Optimising over {len(targets)} targets …")
        baseline = success_rate_for_weights(
            targets,
            defaults["w_elec"], defaults["w_desolv"], defaults["w_info"],
            rmsd_cutoff, top_n,
        )
        print(f"  Baseline (default weights): {baseline:.3f}")

        if _SKOPT_AVAILABLE:
            w_e, w_d, w_i = _bayesian_optimise(targets, rmsd_cutoff, top_n, n_calls)
            method = "bayesian"
        else:
            w_e, w_d, w_i = _grid_search_optimise(targets, rmsd_cutoff, top_n)
            method = "grid_search"

        best_rate = success_rate_for_weights(targets, w_e, w_d, w_i, rmsd_cutoff, top_n)
        print(f"  Best weights: w_elec={w_e:.4f}  w_desolv={w_d:.4f}  w_info={w_i:.4f}")
        print(f"  Success rate: {best_rate:.3f}  (baseline {baseline:.3f})")

        output[cls] = {
            "w_elec": round(w_e, 6),
            "w_desolv": round(w_d, 6),
            "w_info": round(w_i, 6),
            "n_targets": len(targets),
            "success_rate": round(best_rate, 4),
            "baseline_rate": round(baseline, 4),
            "method": method,
        }

    out_json.write_text(json.dumps(output, indent=2))
    print(f"\nWrote per-class weights → {out_json}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-receptor-class Bayesian optimisation of QFD scoring weights"
    )
    p.add_argument("--results_dir", required=True, type=Path,
                   help="Directory with one sub-folder per redocked target")
    p.add_argument("--rmsd_cutoff", type=float, default=2.0,
                   help="RMSD threshold for success (default: 2.0 Å)")
    p.add_argument("--top_n", type=int, default=5,
                   help="Number of top poses to consider (default: 5)")
    p.add_argument("--n_calls", type=int, default=50,
                   help="Number of GP evaluations per class (skopt, default: 50)")
    p.add_argument("--out_json", type=Path, default=Path("qfd_class_weights.json"),
                   help="Output JSON file (default: qfd_class_weights.json)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.results_dir.is_dir():
        print(f"ERROR: results_dir not found: {args.results_dir}", file=sys.stderr)
        return 1
    run_calibration(
        results_dir=args.results_dir,
        rmsd_cutoff=args.rmsd_cutoff,
        top_n=args.top_n,
        n_calls=args.n_calls,
        out_json=args.out_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
