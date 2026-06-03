#!/usr/bin/env python3
"""
calibrate_qfd_weights.py — Optimise QFD scoring weights on a validation set.

Runs a grid search over (QFD_ELEC_WEIGHT, QFD_DESOLV_WEIGHT, QFD_INFO_WEIGHT)
on a set of PDBbind redocking results and reports the combination that
maximises top-1 RMSD < 2 Å success rate.

This does NOT rerun docking — it reads pre-computed result DLGs / score files
and reranks poses using candidate weight combinations.

Usage:
  python3 calibrate_qfd_weights.py --results_dir /path/to/redock_results \\
      [--rmsd_cutoff 2.0] [--top_n 5]

Expected layout of results_dir (one sub-dir per target):
  results_dir/
    1ABC/
      1ABC_out.pdbqt       (or *_out.sdf / docked.pdbqt)
      1ABC_native.pdbqt    (crystal pose)
      scores.tsv           (columns: pose_id  vina_score  qfd_delta_g  rmsd)
    2XYZ/ ...

If scores.tsv is absent for a target, that target is skipped.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass
class PoseRecord:
    target:    str
    pose_id:   int
    vina_e:    float
    qfd_dg:    float   # free-energy blend from kernel (already contains QFD contribution)
    rmsd:      float


class WeightCombo(NamedTuple):
    w_vina:   float
    w_qfd:    float
    w_elec:   float
    w_desolv: float
    w_info:   float


def _load_scores(results_dir: str) -> dict[str, list[PoseRecord]]:
    """Load per-target score files."""
    targets = {}
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        tsv = os.path.join(entry.path, 'scores.tsv')
        if not os.path.exists(tsv):
            continue
        poses = []
        with open(tsv) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                try:
                    poses.append(PoseRecord(
                        target   = entry.name,
                        pose_id  = int(row.get('pose_id', 0)),
                        vina_e   = float(row['vina_score']),
                        qfd_dg   = float(row.get('qfd_dg', row['vina_score'])),
                        rmsd     = float(row['rmsd']),
                    ))
                except (KeyError, ValueError):
                    continue
        if poses:
            targets[entry.name] = poses
    return targets


def _rank_norm(values: list[float]) -> list[float]:
    """Map values to [0,1] rank order (0 = best = lowest value)."""
    n = len(values)
    if n <= 1:
        return [0.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    result = [0.0] * n
    for rank, idx in enumerate(order):
        result[idx] = rank / (n - 1)
    return result


def _evaluate(targets: dict, w_vina: float, w_qfd: float,
               rmsd_cutoff: float, top_n: int) -> float:
    """Return top-1 success rate for given weights."""
    success = 0
    total = 0
    for tname, poses in targets.items():
        if not poses:
            continue
        vina_norms = _rank_norm([p.vina_e  for p in poses])
        qfd_norms  = _rank_norm([p.qfd_dg  for p in poses])
        combined = [w_vina * vn + w_qfd * qn
                    for vn, qn in zip(vina_norms, qfd_norms)]
        # Sort by combined score ascending (0 = best rank)
        sorted_poses = [poses[i] for i in sorted(range(len(poses)), key=lambda i: combined[i])]
        top = sorted_poses[:top_n]
        if any(p.rmsd < rmsd_cutoff for p in top):
            success += 1
        total += 1
    return success / total if total > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results_dir', required=True, help='Directory of per-target results')
    ap.add_argument('--rmsd_cutoff', type=float, default=2.0, help='RMSD success threshold (Å)')
    ap.add_argument('--top_n',  type=int, default=1, help='Top-N poses to check for success')
    ap.add_argument('--steps',  type=int, default=5,
                    help='Grid steps per weight axis (higher = finer search)')
    args = ap.parse_args()

    targets = _load_scores(args.results_dir)
    if not targets:
        sys.exit(f"No valid scores.tsv files found under {args.results_dir!r}")
    print(f"Loaded {len(targets)} targets from {args.results_dir}")

    # Grid search: w_vina + w_qfd = 1.0
    n     = args.steps
    best_rate  = -1.0
    best_combo = None
    results    = []

    for vi in range(n + 1):
        w_v = vi / n
        w_q = 1.0 - w_v
        rate = _evaluate(targets, w_v, w_q, args.rmsd_cutoff, args.top_n)
        results.append((rate, w_v, w_q))
        if rate > best_rate:
            best_rate  = rate
            best_combo = (w_v, w_q)

    results.sort(reverse=True)

    print(f"\nTop-5 weight combinations (top-{args.top_n}, RMSD < {args.rmsd_cutoff} Å):")
    print(f"{'w_vina':>8}  {'w_qfd':>8}  {'success%':>10}")
    print("-" * 32)
    for rate, wv, wq in results[:5]:
        print(f"  {wv:6.3f}    {wq:6.3f}    {rate*100:8.1f}%")

    print(f"\nBest: w_vina={best_combo[0]:.3f}  w_qfd={best_combo[1]:.3f}"
          f"  →  {best_rate*100:.1f}% top-{args.top_n} success")
    print(f"\nTo apply: edit kernel2.h lines QFD_ELEC_WEIGHT / QFD_DESOLV_WEIGHT / QFD_INFO_WEIGHT")
    print(f"  Then rebuild and delete Kernel2_Opt.bin to recompile the GPU kernel.")


if __name__ == '__main__':
    main()
