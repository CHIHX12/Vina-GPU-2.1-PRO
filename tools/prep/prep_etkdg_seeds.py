#!/usr/bin/env python3
"""
prep_etkdg_seeds.py — Generate diverse ETKDG starting conformers for docking.

Standard Vina-GPU starts all MC trajectories from uniformly random positions and
torsions inside the box.  This wastes search budget on high-energy conformations.
ETKDG (RDKit ETKDGv3) generates low-strain conformers whose torsion profiles match
the Cambridge Structural Database.  Using them as starting seeds dramatically reduces
the number of wasted MC steps.

Workflow:
  1. Generate N ETKDG conformers from a SMILES string or SDF file.
  2. Place each conformer at the box centre with a random orientation.
  3. Write them to PDBQT files ready for ensemble_dock.py.

Usage:
  python3 prep_etkdg_seeds.py --input ligand.sdf \\
      --n_seeds 10 --output_dir ./seeds/

  python3 prep_etkdg_seeds.py --smiles "CCO" --name ethanol \\
      --n_seeds 5 --output_dir ./seeds/ \\
      --center_x 0 --center_y 0 --center_z 0

Output:
  ./seeds/seed_01.pdbqt … seed_N.pdbqt
  ./seeds/seed_list.txt   (one path per line for ensemble_dock.py --seed_list)

Requirements:
  pip install rdkit-pypi  (or conda install -c conda-forge rdkit)
  meeko (optional, for higher-quality PDBQT conversion)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    _RDKIT = True
except ImportError:
    _RDKIT = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mol(input_path: Optional[Path], smiles: Optional[str],
              name: str = "ligand") -> "Chem.Mol":
    if not _RDKIT:
        raise ImportError("RDKit not found. Install with: conda install -c conda-forge rdkit")
    mol = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol.SetProp("_Name", name)
    elif input_path:
        suffix = input_path.suffix.lower()
        if suffix in (".sdf", ".mol"):
            suppl = Chem.SDMolSupplier(str(input_path), removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(input_path), removeHs=False)
        else:
            raise ValueError(f"Unsupported input format: {suffix}")
        if mol is None:
            raise ValueError(f"Failed to load molecule from {input_path}")
    else:
        raise ValueError("Either --input or --smiles must be specified")
    return mol


def generate_etkdg_conformers(mol: "Chem.Mol", n: int, seed: int = 42) -> list[int]:
    """
    Generate up to n ETKDG conformers. Returns list of conformer IDs.
    Uses ETKDGv3 (best stereo/torsion coverage, RDKit ≥ 2020.09).
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0  # use all available
    params.pruneRmsThresh = 0.5  # prune similar conformers
    params.useRandomCoords = True

    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n, params=params)
    if len(cids) == 0:
        # fallback: random coords
        params2 = AllChem.ETKDGv3()
        params2.randomSeed = seed + 1
        params2.useRandomCoords = True
        params2.pruneRmsThresh = -1.0
        cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n, params=params2)

    # MMFF minimisation of each conformer
    for cid in cids:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, confId=cid, maxIters=200)
        except Exception:
            pass

    return mol_h, list(cids)


def _translate_to_center(mol_h: "Chem.Mol", cid: int,
                          cx: float, cy: float, cz: float) -> None:
    """Translate conformer centroid to (cx, cy, cz)."""
    conf = mol_h.GetConformer(cid)
    positions = conf.GetPositions()
    centroid = positions.mean(axis=0)
    delta = [cx - centroid[0], cy - centroid[1], cz - centroid[2]]
    from rdkit.Chem import Geometry
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, Geometry.Point3D(
            pos[0] + delta[0], pos[1] + delta[1], pos[2] + delta[2]
        ))


def _apply_random_rotation(mol_h: "Chem.Mol", cid: int, rng: random.Random) -> None:
    """Apply a random rotation around the centroid."""
    import numpy as np
    conf = mol_h.GetConformer(cid)
    positions = np.array(conf.GetPositions())
    centroid = positions.mean(axis=0)
    positions -= centroid

    # Random rotation matrix via quaternion
    u = [rng.random() for _ in range(3)]
    q = [
        math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1]),
        math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1]),
        math.sqrt(u[0])     * math.sin(2 * math.pi * u[2]),
        math.sqrt(u[0])     * math.cos(2 * math.pi * u[2]),
    ]
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    positions = (R @ positions.T).T + centroid

    from rdkit.Chem import Geometry
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, Geometry.Point3D(float(pos[0]), float(pos[1]), float(pos[2])))


def _write_pdbqt(mol_h: "Chem.Mol", cid: int, out_path: Path, mol_name: str) -> bool:
    """
    Write conformer as PDBQT.  Tries meeko first; falls back to Chem.MolToPDBBlock
    with manual PDBQT formatting (no partial charges — set by Vina at runtime).
    """
    # --- Try meeko ---
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol_h, conformer_id=cid)
        if mol_setups:
            pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
            out_path.write_text(pdbqt_str)
            return True
    except Exception:
        pass

    # --- Fallback: bare-bones PDBQT from PDB block ---
    try:
        pdb_block = Chem.MolToPDBBlock(mol_h, confId=cid)
        if not pdb_block:
            return False
        lines = ["MODEL     1\n", f"REMARK  Name = {mol_name}\n"]
        for line in pdb_block.splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            element   = line[76:78].strip() if len(line) >= 78 else atom_name[:2]
            # Convert element to simple AD4 type (no partial charges)
            ad4 = {"C": "C", "N": "N", "O": "OA", "S": "SA", "P": "P",
                   "F": "F", "CL": "CL", "BR": "BR", "I": "I", "H": "HD"}.get(
                   element.upper(), element.upper()[:2])
            pdbqt_line = line[:79].ljust(79) + f"  0.000 {ad4}\n"
            lines.append(pdbqt_line.replace("HETATM", "ATOM  "))
        lines.append("ENDMDL\n")
        out_path.write_text("".join(lines))
        return True
    except Exception as e:
        print(f"  WARNING: fallback PDBQT write failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate ETKDG seed conformers for ensemble docking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Input SDF/MOL/MOL2 file")
    src.add_argument("--smiles", type=str, help="Input SMILES string")
    p.add_argument("--name",       default="ligand", help="Molecule name [default: ligand]")
    p.add_argument("--n_seeds",    type=int, default=10,
                   help="Number of ETKDG conformers to generate [default: 10]")
    p.add_argument("--output_dir", type=Path, default=Path("."),
                   help="Output directory [default: .]")
    p.add_argument("--center_x",   type=float, default=0.0,
                   help="Box centre X — conformers translated here [default: 0]")
    p.add_argument("--center_y",   type=float, default=0.0,
                   help="Box centre Y [default: 0]")
    p.add_argument("--center_z",   type=float, default=0.0,
                   help="Box centre Z [default: 0]")
    p.add_argument("--rotate",     action="store_true",
                   help="Apply random orientations to each conformer after centering")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for ETKDG and rotation [default: 42]")
    return p.parse_args(argv)


def main(argv=None) -> int:
    if not _RDKIT:
        print("ERROR: RDKit not installed.  Run: conda install -c conda-forge rdkit",
              file=sys.stderr)
        return 1

    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mol = _load_mol(args.input, args.smiles, args.name)
    print(f"Loaded molecule: {mol.GetNumAtoms()} heavy atoms + H")

    mol_h, cids = generate_etkdg_conformers(mol, args.n_seeds, seed=args.seed)
    n_generated = len(cids)
    print(f"Generated {n_generated} ETKDG conformers (requested {args.n_seeds})")

    rng = random.Random(args.seed)
    written: list[Path] = []
    for idx, cid in enumerate(cids):
        _translate_to_center(mol_h, cid, args.center_x, args.center_y, args.center_z)
        if args.rotate:
            _apply_random_rotation(mol_h, cid, rng)

        out_path = args.output_dir / f"seed_{idx+1:02d}.pdbqt"
        ok = _write_pdbqt(mol_h, cid, out_path, args.name)
        if ok:
            written.append(out_path)
            print(f"  Wrote {out_path}")
        else:
            print(f"  FAILED to write seed {idx+1}", file=sys.stderr)

    # Write seed list file for ensemble_dock.py
    seed_list = args.output_dir / "seed_list.txt"
    seed_list.write_text("\n".join(str(p) for p in written) + "\n")
    print(f"\nSeed list: {seed_list}")
    print(f"Usage: python3 tools/ensemble_dock.py --seed_list {seed_list} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
