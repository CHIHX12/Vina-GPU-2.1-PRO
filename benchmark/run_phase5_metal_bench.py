#!/usr/bin/env python3
"""
run_phase5_metal_bench.py — QFD Phase 5 (pipi + LS metal) benchmark.

Tests 19 metal-enzyme targets, 3 repeats each (57 trials total).
Compares:
  - Baseline: VINA_LS_METAL_WEIGHT=0.0 (Vina/AD4Zn only, no LS)
  - Phase5:   VINA_LS_METAL_WEIGHT=0.3 (C++ LS metal rescoring active)

For each target:
  1. Generate QFD grids (esp/desolv/infomap/pipi) in per-target dir
  2. Run docking N_REPEAT times for each condition
  3. Extract top-1 RMSD vs crystal ligand
  4. Report hit rate (top-1 RMSD < 2.0 Å)

Usage:
    python3 benchmark/run_phase5_metal_bench.py
    python3 benchmark/run_phase5_metal_bench.py --no_qfd_grids   # LS metal only, no grids
    python3 benchmark/run_phase5_metal_bench.py --n_repeat 1     # quick single-run
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
VINA_DIR  = Path("/home/cycheng/Vina-GPU-2.1")
VINA_BIN  = VINA_DIR / "AutoDock-Vina-GPU-2-1"
PREP_QFD  = VINA_DIR / "tools/prep/prep_qfd_grids.py"
METAL_DIR = Path("/home/cycheng/LigandScope/data/Metal_enzymes")
OUTBASE   = VINA_DIR / "benchmark" / "phase5_metal"
PYTHON    = sys.executable

# ── Targets ───────────────────────────────────────────────────────────────────
TARGETS: List[str] = [
    "1A42", "1BNN", "1G52", "1GKC", "1JAQ", "1MMQ",
    "1O86", "1OQ5", "1UZE", "1YDB",
    "2C6N", "2G1M", "2OVX", "2W0D", "3HS4",
    "3L2U", "3P5A", "3S3M",
]

LIG_CODES: Dict[str, str] = {
    "1A42": "BZU", "1BNN": "AL1", "1G52": "F2B", "1GKC": "NFH", "1JAQ": "01S",
    "1MMQ": "RRS", "1O86": "LPR", "1OQ5": "CEL", "1UZE": "EAL", "1YDB": "AZM",
    "2C6N": "LPR", "2G1M": "4HG", "2OVX": "4MR", "2W0D": "CGS", "3HS4": "AZM",
    "3L2U": "ELV", "3P5A": "IT2", "3S3M": "DLU",
}

# Box centers (from run_metal_validation.sh)
CX: Dict[str, float] = {
    "1A42": -4.14, "1BNN": -3.98, "1G52": -4.32, "1GKC":  65.61,
    "1JAQ":  27.22, "1MMQ":  49.47, "1O86":  40.55, "1OQ5":  17.48,
    "1UZE":  40.46, "1YDB":  -5.31, "2C6N": -25.58, "2G1M":  40.06,
    "2OVX":  24.78, "2W0D": -13.37, "3HS4":  -5.42, "3L2U": -38.42,
    "3P5A":  -5.52, "3S3M": -39.41,
}
CY: Dict[str, float] = {
    "1A42":   5.18, "1BNN":   4.93, "1G52":   5.57, "1GKC":  31.08,
    "1JAQ":  58.79, "1MMQ": -37.76, "1O86":  32.80, "1OQ5":   6.49,
    "1UZE":  35.43, "1YDB":   3.25, "2C6N": -17.16, "2G1M":  19.58,
    "2OVX":   8.44, "2W0D":  24.70, "3HS4":   3.10, "3L2U":  33.31,
    "3P5A":   2.44, "3S3M":  32.60,
}
CZ: Dict[str, float] = {
    "1A42":  14.52, "1BNN":  15.26, "1G52":  14.51, "1GKC": 117.84,
    "1JAQ":  51.81, "1MMQ":  47.08, "1O86":  47.29, "1OQ5":  12.98,
    "1UZE":  47.14, "1YDB":  15.59, "2C6N": -33.71, "2G1M":  11.70,
    "2OVX":  50.41, "2W0D": -24.88, "3HS4":  15.10, "3L2U": -21.30,
    "3P5A":  15.19, "3S3M": -20.08,
}
BOX = 25.0   # Å per side


# ── RMSD helpers ──────────────────────────────────────────────────────────────
def _pdbqt_heavy(path: Path, model: int = 1) -> List[Tuple[float, float, float]]:
    coords: List[Tuple[float, float, float]] = []
    has_model = False
    cur_model = 0
    in_target = False
    with path.open(errors="ignore") as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "MODEL":
                has_model = True
                cur_model += 1
                in_target = cur_model == model
                continue
            if rec == "ENDMDL":
                if in_target:
                    break
                in_target = False
                continue
            if (in_target or (not has_model and model == 1)) and rec in ("ATOM", "HETATM"):
                ad4 = line[77:79].strip() if len(line) > 77 else ""
                if ad4 in ("H", "HD", "HS"):
                    continue
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except (ValueError, IndexError):
                    pass
    return coords


def idx_rmsd(
    ref: List[Tuple[float, float, float]],
    mob: List[Tuple[float, float, float]],
) -> Optional[float]:
    n = min(len(ref), len(mob))
    if n < 3:
        return None
    return math.sqrt(
        sum(
            (ref[i][0] - mob[i][0]) ** 2
            + (ref[i][1] - mob[i][1]) ** 2
            + (ref[i][2] - mob[i][2]) ** 2
            for i in range(n)
        )
        / n
    )


def top1_rmsd(out_pdbqt: Path, crystal: Path) -> Optional[float]:
    ref = _pdbqt_heavy(crystal)
    mob = _pdbqt_heavy(out_pdbqt, model=1)
    return idx_rmsd(ref, mob)


def best_rmsd(out_pdbqt: Path, crystal: Path, n_modes: int = 9) -> Optional[float]:
    ref = _pdbqt_heavy(crystal)
    best: Optional[float] = None
    for m in range(1, n_modes + 1):
        mob = _pdbqt_heavy(out_pdbqt, model=m)
        r = idx_rmsd(ref, mob)
        if r is not None and (best is None or r < best):
            best = r
    return best


# ── Grid generation ───────────────────────────────────────────────────────────
def generate_qfd_grids(target: str, grid_dir: Path) -> bool:
    """Generate QFD grids (no water grid — no PDB available). Returns True on success."""
    rec = METAL_DIR / target / f"{target}_receptor.pdbqt"
    if not rec.exists():
        print(f"    [WARN] receptor not found: {rec}")
        return False

    grid_dir.mkdir(parents=True, exist_ok=True)
    esp_out = grid_dir / "qfd_esp.bin"
    if esp_out.exists():
        return True   # already generated

    cmd = [
        PYTHON, str(PREP_QFD),
        "--receptor",   str(rec),
        "--center_x",   str(CX[target]),
        "--center_y",   str(CY[target]),
        "--center_z",   str(CZ[target]),
        "--size_x",     str(BOX),
        "--size_y",     str(BOX),
        "--size_z",     str(BOX),
        "--output_dir", str(grid_dir),
        "--no_water",    # skip water grid — no crystal PDB available
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not esp_out.exists():
        print(f"    [WARN] grid generation failed for {target}: {r.stderr[:200]}")
        return False
    return True


# ── Docking ───────────────────────────────────────────────────────────────────
def run_docking(
    target: str,
    out_pdbqt: Path,
    ls_metal_weight: float,
    run_cwd: Optional[Path] = None,
    ad4zn: bool = True,
) -> bool:
    """Run Vina-GPU from run_cwd (QFD grids must be present there). Returns True on success."""
    rec = METAL_DIR / target / f"{target}_receptor.pdbqt"
    lig_code = LIG_CODES[target]
    lig = METAL_DIR / target / f"{target}_{lig_code}.pdbqt"

    if not rec.exists() or not lig.exists():
        return False

    cmd = [
        str(VINA_BIN),
        "--receptor",           str(rec),
        "--ligand",             str(lig),
        "--out",                str(out_pdbqt),
        "--opencl_binary_path", str(VINA_DIR),
        "--center_x",           str(CX[target]),
        "--center_y",           str(CY[target]),
        "--center_z",           str(CZ[target]),
        "--size_x",             str(BOX),
        "--size_y",             str(BOX),
        "--size_z",             str(BOX),
        "--thread",             "8000",
        "--search_depth",       "20",
        "--num_modes",          "9",
        "--gpu_id",             "0",
    ]
    if ad4zn:
        cmd += ["--ad4zn"]

    env = {
        **os.environ,
        "VINA_GPU_HOME": str(VINA_DIR),
        "VINA_LS_METAL_WEIGHT": str(ls_metal_weight),
    }

    cwd = str(run_cwd) if run_cwd else str(VINA_DIR)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       cwd=cwd, env=env)
    return out_pdbqt.exists()


# ── Benchmark ─────────────────────────────────────────────────────────────────
def run_benchmark(
    n_repeat: int,
    use_qfd_grids: bool,
    ls_weights: Tuple[float, ...],
) -> None:
    OUTBASE.mkdir(parents=True, exist_ok=True)

    # Per-target grid dirs (shared across conditions)
    grid_dirs: Dict[str, Optional[Path]] = {}
    if use_qfd_grids:
        print("\n── Generating QFD grids ─────────────────────────────────────────────")
        for t in TARGETS:
            gdir = OUTBASE / "grids" / t
            ok = generate_qfd_grids(t, gdir)
            grid_dirs[t] = gdir if ok else None
            status = "OK" if ok else "FAIL"
            print(f"  {t}: {status}")
    else:
        for t in TARGETS:
            grid_dirs[t] = None

    # Results: {condition_label: {target: [top1_rmsd, ...]}}
    results: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for w in ls_weights:
        label = f"LS={w:.2f}"
        results[label] = {t: [] for t in TARGETS}

    print(f"\n── Docking ({n_repeat}× per target per condition) ──────────────────────")
    for rep in range(1, n_repeat + 1):
        print(f"\n  Run {rep}/{n_repeat}")
        for t in TARGETS:
            gdir = grid_dirs[t]
            print(f"    {t} ... ", end="", flush=True)
            timings: List[str] = []
            for w in ls_weights:
                label = f"LS={w:.2f}"
                out_dir = OUTBASE / label.replace("=", "") / t
                out_dir.mkdir(parents=True, exist_ok=True)
                out_pdbqt = out_dir / f"out_rep{rep}.pdbqt"

                ok = run_docking(t, out_pdbqt, ls_metal_weight=w, run_cwd=gdir)
                if ok:
                    rmsd = top1_rmsd(out_pdbqt, METAL_DIR / t / f"{t}_{LIG_CODES[t]}.pdbqt")
                    results[label][t].append(rmsd)
                    r_str = f"{rmsd:.3f}" if rmsd is not None else "N/A"
                else:
                    results[label][t].append(None)
                    r_str = "FAIL"
                timings.append(f"{label}→{r_str}")
            print("  ".join(timings))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Phase 5 Metal Benchmark Results                                ║")
    print("╠══════════════════════════════════════════════════════════════════╣")

    labels = [f"LS={w:.2f}" for w in ls_weights]
    header = f"{'Target':<8}" + "".join(f"  {'RMSD' if n_repeat == 1 else 'Top1avg':>8}" for _ in labels)
    print(f"  {header}")
    print(f"  {'':8}" + "".join(f"  {lb:>8}" for lb in labels))
    print(f"  {'-' * (8 + 10 * len(labels))}")

    hits: Dict[str, int] = {lb: 0 for lb in labels}
    trials: Dict[str, int] = {lb: 0 for lb in labels}

    for t in TARGETS:
        row = f"  {t:<8}"
        for lb in labels:
            rmsds = [r for r in results[lb][t] if r is not None]
            trials[lb] += len(rmsds)
            h = sum(1 for r in rmsds if r < 2.0)
            hits[lb] += h
            if rmsds:
                avg = sum(rmsds) / len(rmsds)
                mark = "*" if all(r < 2.0 for r in rmsds) else ("~" if h > 0 else " ")
                row += f"  {avg:>7.3f}{mark}"
            else:
                row += f"  {'FAIL':>8}"
        print(row)

    print(f"  {'-' * (8 + 10 * len(labels))}")
    hit_row = f"  {'HITS':<8}"
    for lb in labels:
        hit_row += f"  {hits[lb]:>7}/{trials[lb]}"
    print(hit_row)

    pct_row = f"  {'%':8}"
    for lb in labels:
        if trials[lb] > 0:
            pct_row += f"  {100.0 * hits[lb] / trials[lb]:>7.1f}%"
    print(pct_row)

    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Reference baselines:                                           ║")
    print("║    Baseline Vina (no QFD):  20/54 = 37.0%                      ║")
    print("║    QFD v5 (no Phase 5):     22/54 = 40.7%  (+3.7 pp)           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Per-target detail
    print("\n── Per-target detail ───────────────────────────────────────────────")
    for lb in labels:
        print(f"\n  {lb}:")
        for t in TARGETS:
            rmsds = [r for r in results[lb][t] if r is not None]
            if rmsds:
                hits_t = sum(1 for r in rmsds if r < 2.0)
                avg = sum(rmsds) / len(rmsds)
                mn = min(rmsds)
                status = "PASS" if hits_t == n_repeat else ("RECOV" if hits_t > 0 else "FAIL")
                print(f"    {t:<6}  {status:<5}  avg={avg:.3f}  best={mn:.3f}  "
                      f"hits={hits_t}/{len(rmsds)}")
            else:
                print(f"    {t:<6}  ERROR")


def main() -> None:
    parser = argparse.ArgumentParser(description="QFD Phase 5 metal benchmark")
    parser.add_argument("--n_repeat", type=int, default=3,
                        help="Number of repeat docking runs per target (default: 3)")
    parser.add_argument("--no_qfd_grids", action="store_true",
                        help="Skip QFD grid generation (test LS metal only)")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Only run baseline (LS=0.0), no Phase5 comparison")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="LS metal weights to test (default: 0.0 0.3)")
    args = parser.parse_args()

    use_qfd = not args.no_qfd_grids
    if args.weights is not None:
        weights = tuple(args.weights)
    elif args.baseline_only:
        weights = (0.0,)
    else:
        weights = (0.0, 0.3)

    mode = "QFD Phase5 (pipi+LS)" if use_qfd else "LS-metal-only (no QFD grids)"
    print(f"QFD Phase 5 Metal Benchmark")
    print(f"  Mode:     {mode}")
    print(f"  Repeats:  {args.n_repeat}")
    print(f"  Weights:  {weights}")
    print(f"  Targets:  {len(TARGETS)}")
    print(f"  Total:    {len(TARGETS) * args.n_repeat * len(weights)} docking runs")
    print(f"  Binary:   {VINA_BIN}")
    print(f"  Output:   {OUTBASE}")

    run_benchmark(n_repeat=args.n_repeat, use_qfd_grids=use_qfd, ls_weights=weights)


if __name__ == "__main__":
    main()
