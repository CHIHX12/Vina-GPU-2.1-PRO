#!/usr/bin/env python3
"""
xtb_rerank.py — xTB GFN2-based QM/MM reranking for metal-containing targets.

For targets with metal cofactors (Zn, Cu, Fe, Co, Mn, Mg, Ca, Ni, …), the
classical Vina force field systematically mis-ranks poses because metal-
coordination energy is poorly represented.  This script:

  1. Reads the top-N docked poses from a Vina output PDBQT.
  2. Detects whether the receptor contains a metal ion.
  3. For metal targets: extracts each pose + a ~8 Å metal-shell of receptor
     atoms to a temporary XYZ file, then calls xTB (GFN2-xTB) to compute the
     single-point energy of the complex.  The pose is re-ranked by ΔE_xTB.
  4. For non-metal targets: falls back to plain Vina rank (no-op mode).
  5. Writes a ranked TSV and (optionally) re-orders the output PDBQT.

Requirements:
  - xtb (>= 6.6) on PATH  (https://github.com/grimme-lab/xtb)
  - RDKit (for XYZ extraction, optional fallback to OpenBabel if absent)
  - numpy

Usage:
  python3 xtb_rerank.py --receptor receptor.pdbqt --docked docked_out.pdbqt \\
      --top_n 5 --metal_shell 8.0 --out_tsv reranked.tsv \\
      [--reorder_pdbqt reranked_out.pdbqt]

Output TSV columns:
  pose_id  vina_rank  vina_affinity  xtb_energy_au  xtb_rank  delta_xtb
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Metal detection constants
# ---------------------------------------------------------------------------

# AD4 / PDBQT atom types that indicate metal coordination sites
METAL_TYPES: frozenset[str] = frozenset({
    "Zn", "ZN", "Cu", "CU", "Fe", "FE", "Co", "CO",
    "Mn", "MN", "Mg", "MG", "Ca", "CA", "Ni", "NI",
    "Mo", "MO", "W",  "WW", "V",  "VV", "Cr", "CR",
    "Na", "NA", "K",  "KK", "Hg", "HG", "Cd", "CD",
})

# Periodic table: symbol → atomic number (for XYZ header)
_SYM2NUM: dict[str, int] = {
    "H": 1,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "P": 15, "S": 16,
    "Cl": 17,"Br": 35,"I": 53, "Zn": 30,"Cu": 29,"Fe": 26,"Co": 27,
    "Mn": 25,"Mg": 12,"Ca": 20,"Ni": 28,"Mo": 42,"W": 74, "V": 23,
    "Cr": 24,"Na": 11,"K": 19, "Hg": 80,"Cd": 48,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    element: str
    x: float
    y: float
    z: float


@dataclass
class Pose:
    pose_id: int          # 1-indexed, matches Vina "mode N"
    vina_affinity: float  # kcal/mol
    atoms: list[Atom] = field(default_factory=list)


@dataclass
class RankedPose:
    pose_id: int
    vina_rank: int
    vina_affinity: float
    xtb_energy_au: float   # NaN if not computed
    xtb_rank: int          # -1 if not computed
    delta_xtb: float       # xtb_energy - reference (lowest xTB pose), NaN if not computed


# ---------------------------------------------------------------------------
# PDBQT parsing
# ---------------------------------------------------------------------------

def _pdbqt_element(record: str) -> str:
    """Return element symbol from PDBQT ATOM/HETATM line."""
    # AD4 type is in cols 78-79 (0-based 77-78)
    ad4 = record[77:].strip().split()[0] if len(record) > 77 else ""
    # Map common AD4 types to element symbols
    _map = {
        "C": "C", "A": "C", "N": "N", "NA": "N", "OA": "O", "O": "O",
        "SA": "S", "S": "S", "HD": "H", "H": "H", "F": "F", "P": "P",
        "CL": "Cl", "BR": "Br", "I": "I",
    }
    sym = _map.get(ad4.upper(), "")
    if not sym:
        # fall back to PDB element column (cols 76-77)
        elem_col = record[76:78].strip() if len(record) >= 78 else ""
        sym = elem_col.capitalize() if elem_col else ad4.capitalize()
    return sym or "X"


def parse_pdbqt_receptor(path: Path) -> list[Atom]:
    """Read receptor PDBQT into a list of Atom objects."""
    atoms: list[Atom] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        elem = _pdbqt_element(line)
        atoms.append(Atom(elem, x, y, z))
    return atoms


def parse_pdbqt_docked(path: Path) -> list[Pose]:
    """Parse a multi-model Vina output PDBQT into Pose objects."""
    poses: list[Pose] = []
    current_pose: Optional[Pose] = None
    mode_re = re.compile(r"MODEL\s+(\d+)")
    affinity_re = re.compile(r"REMARK VINA RESULT:\s+([-\d.]+)")

    for line in path.read_text(errors="ignore").splitlines():
        m = mode_re.match(line)
        if m:
            current_pose = Pose(pose_id=int(m.group(1)), vina_affinity=0.0)
            poses.append(current_pose)
            continue
        if current_pose is None:
            continue
        m2 = affinity_re.match(line)
        if m2:
            current_pose.vina_affinity = float(m2.group(1))
            continue
        if line.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            elem = _pdbqt_element(line)
            current_pose.atoms.append(Atom(elem, x, y, z))

    return poses


# ---------------------------------------------------------------------------
# Metal detection
# ---------------------------------------------------------------------------

def find_metal_atoms(receptor_atoms: list[Atom]) -> list[Atom]:
    """Return receptor atoms whose element is a known metal."""
    return [a for a in receptor_atoms if a.element in METAL_TYPES]


def has_metal(receptor_atoms: list[Atom]) -> bool:
    return bool(find_metal_atoms(receptor_atoms))


# ---------------------------------------------------------------------------
# Shell extraction
# ---------------------------------------------------------------------------

def _dist(a: Atom, b: Atom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def receptor_shell(
    receptor_atoms: list[Atom],
    metal_atoms: list[Atom],
    cutoff_ang: float,
) -> list[Atom]:
    """Return receptor atoms within cutoff_ang of any metal atom."""
    shell: list[Atom] = []
    for ra in receptor_atoms:
        if any(_dist(ra, m) <= cutoff_ang for m in metal_atoms):
            shell.append(ra)
    return shell


# ---------------------------------------------------------------------------
# XYZ writing
# ---------------------------------------------------------------------------

def write_xyz(atoms: list[Atom], path: Path, comment: str = "") -> None:
    lines = [str(len(atoms)), comment]
    for a in atoms:
        lines.append(f"{a.element:3s}  {a.x:12.6f}  {a.y:12.6f}  {a.z:12.6f}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# xTB invocation
# ---------------------------------------------------------------------------

def run_xtb_sp(xyz_path: Path, charge: int = 0, mult: int = 1) -> float:
    """
    Run a GFN2-xTB single-point calculation.
    Returns the total energy in atomic units (Hartree), or NaN on failure.
    """
    xtb_bin = shutil.which("xtb")
    if xtb_bin is None:
        raise RuntimeError(
            "xtb binary not found on PATH.  Install from https://github.com/grimme-lab/xtb"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Copy input
        dest_xyz = tmpdir_path / "mol.xyz"
        shutil.copy(xyz_path, dest_xyz)

        cmd = [
            xtb_bin, "mol.xyz",
            "--gfn", "2",
            "--sp",
            "--chrg", str(charge),
            "--uhf", str(mult - 1),  # xTB uses n_unpaired_electrons
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return math.nan

        # Parse "TOTAL ENERGY" from stdout
        energy_re = re.compile(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh")
        for line in result.stdout.splitlines():
            m = energy_re.search(line)
            if m:
                return float(m.group(1))

        return math.nan


# ---------------------------------------------------------------------------
# Reranking logic
# ---------------------------------------------------------------------------

def rerank_top_n(
    receptor_path: Path,
    docked_path: Path,
    top_n: int = 5,
    metal_shell_cutoff: float = 8.0,
) -> list[RankedPose]:
    """
    Core reranking function.

    Returns a list of RankedPose sorted by xtb_rank (or vina_rank for non-metal).
    """
    receptor_atoms = parse_pdbqt_receptor(receptor_path)
    metal_atoms = find_metal_atoms(receptor_atoms)
    is_metal_target = bool(metal_atoms)

    poses = parse_pdbqt_docked(docked_path)[:top_n]

    results: list[RankedPose] = []

    if not is_metal_target:
        # Non-metal: return Vina ranking with NaN xTB fields
        for vrank, pose in enumerate(poses, start=1):
            results.append(RankedPose(
                pose_id=pose.pose_id,
                vina_rank=vrank,
                vina_affinity=pose.vina_affinity,
                xtb_energy_au=math.nan,
                xtb_rank=-1,
                delta_xtb=math.nan,
            ))
        return results

    # Metal target: build shell and run xTB for each pose
    shell_atoms = receptor_shell(receptor_atoms, metal_atoms, metal_shell_cutoff)

    xtb_energies: list[float] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for i, pose in enumerate(poses):
            combined = shell_atoms + pose.atoms
            xyz_path = tmpdir_path / f"pose_{i+1}.xyz"
            write_xyz(combined, xyz_path, comment=f"pose {pose.pose_id}")
            e = run_xtb_sp(xyz_path)
            xtb_energies.append(e)

    # Sort by xTB energy (lower = better); NaN poses go last
    indexed = list(zip(range(len(poses)), xtb_energies))
    indexed_valid = [(i, e) for i, e in indexed if not math.isnan(e)]
    indexed_nan = [(i, e) for i, e in indexed if math.isnan(e)]
    indexed_sorted = sorted(indexed_valid, key=lambda x: x[1]) + indexed_nan

    ref_energy = indexed_sorted[0][1] if indexed_valid else math.nan

    # Build RankedPose list in xTB order
    xtb_rank_map: dict[int, int] = {}
    for xtb_rank, (orig_idx, _) in enumerate(indexed_sorted, start=1):
        xtb_rank_map[orig_idx] = xtb_rank

    for vrank, (orig_idx, pose) in enumerate(
        [(idx, poses[idx]) for idx in range(len(poses))], start=1
    ):
        e = xtb_energies[orig_idx]
        results.append(RankedPose(
            pose_id=pose.pose_id,
            vina_rank=vrank,
            vina_affinity=pose.vina_affinity,
            xtb_energy_au=e,
            xtb_rank=xtb_rank_map[orig_idx],
            delta_xtb=e - ref_energy if not math.isnan(e) else math.nan,
        ))

    # Re-sort results by xtb_rank for output
    results.sort(key=lambda r: (r.xtb_rank if r.xtb_rank > 0 else 9999, r.vina_rank))
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_tsv(ranked: list[RankedPose], out_path: Path) -> None:
    header = "pose_id\tvina_rank\tvina_affinity\txtb_energy_au\txtb_rank\tdelta_xtb\n"
    rows: list[str] = [header]
    for r in ranked:
        xtb_e = f"{r.xtb_energy_au:.8f}" if not math.isnan(r.xtb_energy_au) else "NA"
        delta = f"{r.delta_xtb:.6f}" if not math.isnan(r.delta_xtb) else "NA"
        rank_str = str(r.xtb_rank) if r.xtb_rank > 0 else "NA"
        rows.append(
            f"{r.pose_id}\t{r.vina_rank}\t{r.vina_affinity:.3f}\t"
            f"{xtb_e}\t{rank_str}\t{delta}\n"
        )
    out_path.write_text("".join(rows))


def reorder_pdbqt(
    docked_path: Path,
    ranked: list[RankedPose],
    out_path: Path,
) -> None:
    """
    Write a new PDBQT with poses in xtb_rank order (or vina_rank if no xTB).
    """
    # Parse original models as raw text blocks
    models: dict[int, list[str]] = {}
    current_id: Optional[int] = None
    current_lines: list[str] = []
    mode_re = re.compile(r"MODEL\s+(\d+)")

    for line in docked_path.read_text(errors="ignore").splitlines(keepends=True):
        m = mode_re.match(line)
        if m:
            if current_id is not None:
                models[current_id] = current_lines
            current_id = int(m.group(1))
            current_lines = [line]
        elif line.strip() == "ENDMDL" and current_id is not None:
            current_lines.append(line)
            models[current_id] = current_lines
            current_id = None
            current_lines = []
        elif current_id is not None:
            current_lines.append(line)

    ordered: list[str] = []
    for new_idx, r in enumerate(ranked, start=1):
        block = models.get(r.pose_id, [])
        if block:
            # Update MODEL number to new rank
            block[0] = f"MODEL     {new_idx}\n"
            ordered.extend(block)

    out_path.write_text("".join(ordered))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="xTB GFN2 single-point reranking for metal-containing docking targets"
    )
    p.add_argument("--receptor", required=True, type=Path,
                   help="Receptor PDBQT (must contain HETATM metal lines for activation)")
    p.add_argument("--docked", required=True, type=Path,
                   help="Vina multi-model output PDBQT")
    p.add_argument("--top_n", type=int, default=5,
                   help="Number of poses to rerank (default: 5)")
    p.add_argument("--metal_shell", type=float, default=8.0,
                   help="Angstrom radius around metal for receptor shell (default: 8.0)")
    p.add_argument("--out_tsv", type=Path, default=Path("xtb_reranked.tsv"),
                   help="Output TSV file")
    p.add_argument("--reorder_pdbqt", type=Path, default=None,
                   help="If given, write reordered PDBQT to this path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.receptor.exists():
        print(f"ERROR: receptor not found: {args.receptor}", file=sys.stderr)
        return 1
    if not args.docked.exists():
        print(f"ERROR: docked poses not found: {args.docked}", file=sys.stderr)
        return 1

    ranked = rerank_top_n(
        receptor_path=args.receptor,
        docked_path=args.docked,
        top_n=args.top_n,
        metal_shell_cutoff=args.metal_shell,
    )

    write_tsv(ranked, args.out_tsv)
    print(f"Wrote reranked TSV → {args.out_tsv}")

    if args.reorder_pdbqt:
        reorder_pdbqt(args.docked, ranked, args.reorder_pdbqt)
        print(f"Wrote reordered PDBQT → {args.reorder_pdbqt}")

    # Summary
    metal_active = any(not math.isnan(r.xtb_energy_au) for r in ranked)
    print(f"\nxTB active: {metal_active}")
    if metal_active:
        for r in ranked:
            flag = "*" if r.xtb_rank == 1 else " "
            e_str = f"{r.xtb_energy_au:.6f} Eh" if not math.isnan(r.xtb_energy_au) else "   NA      "
            print(
                f"  {flag} pose {r.pose_id:2d}  vina_rank={r.vina_rank}"
                f"  xtb_rank={r.xtb_rank:2d}  E={e_str}"
                f"  ΔE={r.delta_xtb*627.5:.1f} kcal/mol" if not math.isnan(r.delta_xtb) else ""
            )
    else:
        print("  (non-metal target — Vina ranking retained)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
