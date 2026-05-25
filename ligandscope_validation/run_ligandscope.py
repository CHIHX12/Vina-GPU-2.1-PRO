#!/usr/bin/env python3
"""
run_ligandscope.py — Self-docking validation on 12 LigandScope targets.

For each target:
  1. Compute binding box from crystal ligand centroid (+ 20 Å padding)
  2. Run GPU Vina (via Singularity SIF)
  3. Compute heavy-atom RMSD between best pose(s) and crystal ligand
  4. Output comparison table

Usage
-----
    python3 run_ligandscope.py [--pdb 2ZV2 4AG8]
    python3 run_ligandscope.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent          # ligandscope_validation/
REPO_DIR      = SCRIPT_DIR.parent             # repo root
LS_VAL_DIR    = SCRIPT_DIR                    # pdbqt files live here
SIF_PATH      = REPO_DIR / 'autodock-vina-gpu.sif'
OUT_DIR       = SCRIPT_DIR / 'results'

SYSTEMS: Dict[str, List[str]] = {
    'Kinases':                     ['2ZV2', '4AG8', '5L2S'],
    'Nuclear_receptors':           ['1ERR', '1T7R', '4R06'],
    'Ion_channels':                ['4F8H', '5EK0', '6D6T'],
    'G_protein-coupled_receptors': ['4YAY', '5MZJ', '6IIU'],
}

# ── Docking parameters ───────────────────────────────────────────────────────
BOX_SIZE    = 20.0   # Å each side
GPU_THREAD  = 8000
GPU_DEPTH   = 20
GPU_SEED    = 42
CPU_EXHAUSTIVE = 32
CPU_NMODES     = 10
CPU_SEED       = 42


# ── PDBQT helpers ─────────────────────────────────────────────────────────────
def read_coords(pdbqt: Path) -> List[Tuple[float, float, float, str]]:
    """Return list of (x, y, z, element) for all heavy atoms (no H)."""
    atoms: List[Tuple[float, float, float, str]] = []
    with pdbqt.open() as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ('ATOM', 'HETATM'):
                continue
            elem = line[77:79].strip() if len(line) > 77 else ''
            if not elem:
                elem = line[12:16].strip()[:1]
            if elem.upper() in ('H', 'HD'):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append((x, y, z, elem))
            except ValueError:
                pass
    return atoms


def centroid(atoms: List[Tuple[float, float, float, str]]) -> Tuple[float, float, float]:
    n = len(atoms)
    return (
        sum(a[0] for a in atoms) / n,
        sum(a[1] for a in atoms) / n,
        sum(a[2] for a in atoms) / n,
    )


def rmsd(
    ref: List[Tuple[float, float, float, str]],
    mob: List[Tuple[float, float, float, str]],
) -> float:
    """Minimum-assignment heavy-atom RMSD (symmetric matching by element)."""
    # Simple approach: match atoms positionally (same order assumption for crystal vs docked)
    # Fall back to nearest-neighbour if sizes differ
    if len(ref) == 0 or len(mob) == 0:
        return float('inf')

    if len(ref) == len(mob):
        sq = sum(
            (r[0]-m[0])**2 + (r[1]-m[1])**2 + (r[2]-m[2])**2
            for r, m in zip(ref, mob)
        )
        return math.sqrt(sq / len(ref))

    # Different atom count: use centroid-based nearest-neighbour
    used = set()
    sq_sum = 0.0
    count = 0
    for r in ref:
        best_d = float('inf')
        best_j = -1
        for j, m in enumerate(mob):
            if j in used:
                continue
            d = (r[0]-m[0])**2 + (r[1]-m[1])**2 + (r[2]-m[2])**2
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            sq_sum += best_d
            count += 1
    return math.sqrt(sq_sum / count) if count else float('inf')


def parse_poses(out_pdbqt: Path) -> List[List[Tuple[float, float, float, str]]]:
    """Split multi-model PDBQT into list of pose atom lists."""
    poses: List[List[Tuple[float, float, float, str]]] = []
    current: List[Tuple[float, float, float, str]] = []
    with out_pdbqt.open() as f:
        for line in f:
            if line.startswith('MODEL'):
                current = []
            elif line.startswith('ENDMDL'):
                if current:
                    poses.append(current)
                    current = []
            else:
                rec = line[:6].strip()
                if rec not in ('ATOM', 'HETATM'):
                    continue
                elem = line[77:79].strip() if len(line) > 77 else ''
                if not elem:
                    elem = line[12:16].strip()[:1]
                if elem.upper() in ('H', 'HD'):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    current.append((x, y, z, elem))
                except ValueError:
                    pass
    if current:  # single-model file without MODEL/ENDMDL
        poses.append(current)
    return poses


def best_rmsd(
    crystal_atoms: List[Tuple[float, float, float, str]],
    poses: List[List[Tuple[float, float, float, str]]],
) -> Tuple[float, float]:
    """Return (top1_rmsd, best_rmsd) across all poses."""
    if not poses:
        return float('inf'), float('inf')
    rmsds = [rmsd(crystal_atoms, p) for p in poses]
    return rmsds[0], min(rmsds)


# ── Docking runners ──────────────────────────────────────────────────────────
def run_gpu(
    receptor: Path,
    ligand: Path,
    out_pdbqt: Path,
    cx: float, cy: float, cz: float,
    dry_run: bool = False,
) -> bool:
    """Run AutoDock-Vina-GPU via Singularity."""
    config = out_pdbqt.with_suffix('.cfg')
    log    = out_pdbqt.with_suffix('.log')

    config.write_text(
        f"receptor = {receptor}\n"
        f"ligand   = {ligand}\n"
        f"out      = {out_pdbqt}\n"
        f"center_x = {cx:.3f}\n"
        f"center_y = {cy:.3f}\n"
        f"center_z = {cz:.3f}\n"
        f"size_x   = {BOX_SIZE}\n"
        f"size_y   = {BOX_SIZE}\n"
        f"size_z   = {BOX_SIZE}\n"
        f"thread      = {GPU_THREAD}\n"
        f"search_depth = {GPU_DEPTH}\n"
        f"seed        = {GPU_SEED}\n"
    )

    cmd = [
        'singularity', 'exec', '--nv', str(SIF_PATH),
        '/opt/vina-gpu/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1',
        '--config', str(config),
    ]

    if dry_run:
        print(f"  [DRY-RUN GPU] {' '.join(cmd)}")
        return False

    t0 = time.time()
    with log.open('w') as lf:
        ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
    elapsed = time.time() - t0
    print(f"  GPU done in {elapsed:.1f}s (exit {ret.returncode})")
    return ret.returncode == 0 and out_pdbqt.exists()


def run_cpu(
    receptor: Path,
    ligand: Path,
    out_pdbqt: Path,
    cx: float, cy: float, cz: float,
    dry_run: bool = False,
) -> bool:
    """Run CPU AutoDock Vina (system binary)."""
    log = out_pdbqt.with_suffix('.log')

    cmd = [
        'vina',
        '--receptor', str(receptor),
        '--ligand',   str(ligand),
        '--out',      str(out_pdbqt),
        '--center_x', f'{cx:.3f}',
        '--center_y', f'{cy:.3f}',
        '--center_z', f'{cz:.3f}',
        '--size_x',   str(BOX_SIZE),
        '--size_y',   str(BOX_SIZE),
        '--size_z',   str(BOX_SIZE),
        '--exhaustiveness', str(CPU_EXHAUSTIVE),
        '--num_modes',      str(CPU_NMODES),
        '--seed',           str(CPU_SEED),
    ]

    if dry_run:
        print(f"  [DRY-RUN CPU] {' '.join(cmd)}")
        return False

    t0 = time.time()
    with log.open('w') as lf:
        ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=600)
    elapsed = time.time() - t0
    print(f"  CPU done in {elapsed:.1f}s (exit {ret.returncode})")
    return ret.returncode == 0 and out_pdbqt.exists()


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description='LigandScope GPU vs CPU benchmark')
    ap.add_argument('--gpu-only',  action='store_true')
    ap.add_argument('--cpu-only',  action='store_true')
    ap.add_argument('--pdb',       nargs='+', metavar='PDB')
    ap.add_argument('--dry-run',   action='store_true')
    args = ap.parse_args()

    run_gpu_flag = not args.cpu_only
    run_cpu_flag = not args.gpu_only

    # Build target list
    targets: List[Tuple[str, str]] = []  # (class, pdb)
    for cls, pdbs in SYSTEMS.items():
        for pdb in pdbs:
            if args.pdb and pdb not in args.pdb:
                continue
            targets.append((cls, pdb))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    for cls, pdb in targets:
        print(f"\n{'='*60}")
        print(f"  {pdb}  [{cls}]")
        print(f"{'='*60}")

        receptor = LS_VAL_DIR / 'pdbqt_receptor' / f'{pdb}_pocket.pdbqt'
        ligand   = LS_VAL_DIR / 'pdbqt_ligand'   / f'{pdb}_ligand.pdbqt'

        if not receptor.exists():
            print(f"  ERROR: receptor not found: {receptor}")
            continue
        if not ligand.exists():
            print(f"  ERROR: ligand not found: {ligand}")
            continue

        crystal_atoms = read_coords(ligand)
        if not crystal_atoms:
            print(f"  ERROR: no atoms parsed from crystal ligand")
            continue

        cx, cy, cz = centroid(crystal_atoms)
        print(f"  Box centre: ({cx:.2f}, {cy:.2f}, {cz:.2f})  heavy atoms: {len(crystal_atoms)}")

        pdb_out = OUT_DIR / pdb
        pdb_out.mkdir(exist_ok=True)

        row: dict = {
            'PDB': pdb,
            'Class': cls,
            'CrystalAtoms': len(crystal_atoms),
        }

        # --- GPU ---
        if run_gpu_flag:
            gpu_out = pdb_out / f'{pdb}_gpu_out.pdbqt'
            print(f"  Running GPU docking …")
            ok = run_gpu(receptor, ligand, gpu_out, cx, cy, cz, dry_run=args.dry_run)
            if ok:
                poses = parse_poses(gpu_out)
                t1, tb = best_rmsd(crystal_atoms, poses)
                print(f"  GPU poses={len(poses)}  top1={t1:.2f}Å  best={tb:.2f}Å")
                row['GPU_poses']      = len(poses)
                row['GPU_top1_RMSD']  = round(t1, 3)
                row['GPU_best_RMSD']  = round(tb, 3)
                row['GPU_top1_ok']    = t1 < 2.0
                row['GPU_best_ok']    = tb < 2.0
            else:
                print(f"  GPU FAILED")
                row['GPU_poses']      = 0
                row['GPU_top1_RMSD']  = 'FAILED'
                row['GPU_best_RMSD']  = 'FAILED'
                row['GPU_top1_ok']    = False
                row['GPU_best_ok']    = False

        # --- CPU ---
        if run_cpu_flag:
            cpu_out = pdb_out / f'{pdb}_cpu_out.pdbqt'
            print(f"  Running CPU docking …")
            ok = run_cpu(receptor, ligand, cpu_out, cx, cy, cz, dry_run=args.dry_run)
            if ok:
                poses = parse_poses(cpu_out)
                t1, tb = best_rmsd(crystal_atoms, poses)
                print(f"  CPU poses={len(poses)}  top1={t1:.2f}Å  best={tb:.2f}Å")
                row['CPU_poses']      = len(poses)
                row['CPU_top1_RMSD']  = round(t1, 3)
                row['CPU_best_RMSD']  = round(tb, 3)
                row['CPU_top1_ok']    = t1 < 2.0
                row['CPU_best_ok']    = tb < 2.0
            else:
                print(f"  CPU FAILED")
                row['CPU_poses']      = 0
                row['CPU_top1_RMSD']  = 'FAILED'
                row['CPU_best_RMSD']  = 'FAILED'
                row['CPU_top1_ok']    = False
                row['CPU_best_ok']    = False

        rows.append(row)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("LIGANDSCOPE BENCHMARK RESULTS")
    print(f"{'='*100}")

    hdr = f"{'PDB':<8} {'Class':<35} {'CrysAtoms':>9} | "
    if run_gpu_flag:
        hdr += f"{'GPU top1':>9} {'GPU best':>9} {'top1<2Å':>8} {'best<2Å':>8} | "
    if run_cpu_flag:
        hdr += f"{'CPU top1':>9} {'CPU best':>9} {'top1<2Å':>8} {'best<2Å':>8}"
    print(hdr)
    print('-' * len(hdr))

    for row in rows:
        line = f"{row['PDB']:<8} {row['Class']:<35} {row['CrystalAtoms']:>9} | "
        if run_gpu_flag:
            gt1 = f"{row.get('GPU_top1_RMSD', '-')}"
            gb  = f"{row.get('GPU_best_RMSD', '-')}"
            go1 = '✓' if row.get('GPU_top1_ok') else '✗'
            gob = '✓' if row.get('GPU_best_ok') else '✗'
            line += f"{gt1:>9} {gb:>9} {go1:>8} {gob:>8} | "
        if run_cpu_flag:
            ct1 = f"{row.get('CPU_top1_RMSD', '-')}"
            cb  = f"{row.get('CPU_best_RMSD', '-')}"
            co1 = '✓' if row.get('CPU_top1_ok') else '✗'
            cob = '✓' if row.get('CPU_best_ok') else '✗'
            line += f"{ct1:>9} {cb:>9} {co1:>8} {cob:>8}"
        print(line)

    print()
    if run_gpu_flag:
        gpu_top1_ok = sum(1 for r in rows if r.get('GPU_top1_ok'))
        gpu_best_ok = sum(1 for r in rows if r.get('GPU_best_ok'))
        n = len(rows)
        print(f"GPU  Top-1 <2Å: {gpu_top1_ok}/{n} ({100*gpu_top1_ok//n if n else 0}%)   "
              f"Best <2Å: {gpu_best_ok}/{n} ({100*gpu_best_ok//n if n else 0}%)")
    if run_cpu_flag:
        cpu_top1_ok = sum(1 for r in rows if r.get('CPU_top1_ok'))
        cpu_best_ok = sum(1 for r in rows if r.get('CPU_best_ok'))
        n = len(rows)
        print(f"CPU  Top-1 <2Å: {cpu_top1_ok}/{n} ({100*cpu_top1_ok//n if n else 0}%)   "
              f"Best <2Å: {cpu_best_ok}/{n} ({100*cpu_best_ok//n if n else 0}%)")

    # ── TSV output ──────────────────────────────────────────────────────────
    tsv = SCRIPT_DIR / 'ligandscope_results.tsv'
    if rows and not args.dry_run:
        cols = list(rows[0].keys())
        with tsv.open('w') as f:
            f.write('\t'.join(cols) + '\n')
            for row in rows:
                f.write('\t'.join(str(row.get(c, '')) for c in cols) + '\n')
        print(f"\nResults saved → {tsv}")


if __name__ == '__main__':
    main()
