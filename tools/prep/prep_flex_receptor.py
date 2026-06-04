#!/usr/bin/env python3
"""
prep_flex_receptor.py — Identify and prepare flexible receptor side chains.

Finds residues whose side-chain heavy atoms lie within a cutoff distance of
the docking box centre.  For each such residue, identifies the rotatable
side-chain bonds and splits the receptor into:

  <stem>_rigid.pdbqt  — backbone + all non-flexible residues (Vina --receptor)
  <stem>_flex.pdbqt   — flexible side chains in AutoDock ROOT/BRANCH format
                         (Vina --flex)

The flex PDBQT ROOT is the Cα anchor; BRANCH/ENDBRANCH blocks cover the full
rotatable side chain from Cβ outward.

Usage:
  python3 prep_flex_receptor.py \\
      --receptor receptor.pdbqt \\
      --center_x X --center_y Y --center_z Z \\
      [--cutoff 5.0] [--output_dir .] [--residues ARG42A,HIS64A]

Options:
  --receptor     receptor PDBQT (already protonated, with partial charges)
  --center_x/y/z docking box centre (Å)
  --cutoff       max distance from box centre to include a residue (Å) [5.0]
  --output_dir   where to write the two output files [.]
  --residues     comma-separated list of residues to force-include, e.g.
                 ARG42A (3-letter resname + resnum + chain).  Overrides
                 distance filter.
  --no_h         strip hydrogen atoms from the flex PDBQT (Vina reads them
                 fine but some pipelines prefer H-less flex files)

Example:
  python3 prep_flex_receptor.py \\
      --receptor 1A42_receptor.pdbqt \\
      --center_x -4.14 --center_y 5.18 --center_z 14.52 \\
      --cutoff 5.0 --output_dir .

Requirements:
  Only standard library (math, re, argparse, os, sys).

References:
  AutoDock Vina manual — flexible side chains:
  https://vina.scripps.edu/manual/#flex
  Morris et al. J. Comput. Chem. 30, 2785-2791 (2009)
"""

import argparse
import math
import os
import sys
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Residue side-chain torsion trees
#
# Maps each residue name → ordered list of (parent_atom, child_atom) pairs
# that define the rotatable bonds, in depth-first order from Cα outward.
# Atoms not reachable from these bonds (ring atoms, methyls, etc.) are
# placed in the INNERMOST branch that contains their parent.
#
# Naming follows PDB/PDBQT standard atom names.
# ---------------------------------------------------------------------------

# Side-chain atoms per residue type in topological order (CA first, then BFS)
_SC_ATOMS: dict[str, list[str]] = {
    'SER': ['CA', 'CB', 'OG'],
    'THR': ['CA', 'CB', 'OG1', 'CG2'],
    'CYS': ['CA', 'CB', 'SG'],
    'VAL': ['CA', 'CB', 'CG1', 'CG2'],
    'LEU': ['CA', 'CB', 'CG', 'CD1', 'CD2'],
    'ILE': ['CA', 'CB', 'CG1', 'CG2', 'CD1'],
    'MET': ['CA', 'CB', 'CG', 'SD', 'CE'],
    'PHE': ['CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TYR': ['CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'TRP': ['CA', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'HIS': ['CA', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ASN': ['CA', 'CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['CA', 'CB', 'CG', 'OD1', 'OD2'],
    'GLN': ['CA', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['CA', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'LYS': ['CA', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'ARG': ['CA', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
}

# For each residue type, the sequence of BRANCH anchor pairs (parent, child)
# in the order they should appear as BRANCH statements.  Derived from the
# side-chain topology above.
_TORSIONS: dict[str, list[tuple[str, str]]] = {
    'SER': [('CA', 'CB')],
    'THR': [('CA', 'CB')],
    'CYS': [('CA', 'CB')],
    'VAL': [('CA', 'CB')],
    'LEU': [('CA', 'CB'), ('CB', 'CG')],
    'ILE': [('CA', 'CB'), ('CB', 'CG1')],
    'MET': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'SD')],
    'PHE': [('CA', 'CB'), ('CB', 'CG')],
    'TYR': [('CA', 'CB'), ('CB', 'CG')],
    'TRP': [('CA', 'CB'), ('CB', 'CG')],
    'HIS': [('CA', 'CB'), ('CB', 'CG')],
    'ASN': [('CA', 'CB'), ('CB', 'CG')],
    'ASP': [('CA', 'CB'), ('CB', 'CG')],
    'GLN': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD')],
    'GLU': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD')],
    'LYS': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE')],
    'ARG': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE')],
}

# Residues with no rotatable side-chain bonds — always rigid
_RIGID_ONLY = frozenset(['GLY', 'ALA', 'PRO'])

# Backbone atom names that always stay in the rigid receptor
_BACKBONE = frozenset(['N', 'CA', 'C', 'O', 'OXT', 'H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3'])


# ---------------------------------------------------------------------------
# PDBQT atom representation
# ---------------------------------------------------------------------------

class Atom(NamedTuple):
    record:    str    # 'ATOM' or 'HETATM'
    serial:    int
    name:      str    # atom name, stripped
    altloc:    str
    resname:   str
    chain:     str
    resseq:    int
    icode:     str
    x:         float
    y:         float
    z:         float
    occupancy: float
    bfactor:   float
    charge:    str    # partial charge field (+0.000 etc.)
    atype:     str    # AutoDock atom type (last column)
    raw:       str    # original line (for passthrough in rigid)


def _parse_pdbqt(path: str) -> list[Atom]:
    atoms = []
    with open(path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ('ATOM', 'HETATM'):
                continue
            try:
                serial   = int(line[6:11])
                name     = line[12:16].strip()
                altloc   = line[16]
                resname  = line[17:20].strip()
                chain    = line[21]
                resseq   = int(line[22:26])
                icode    = line[26]
                x        = float(line[30:38])
                y        = float(line[38:46])
                z        = float(line[46:54])
                occupancy = float(line[54:60]) if len(line) > 54 else 1.0
                bfactor   = float(line[60:66]) if len(line) > 60 else 0.0
                charge   = line[70:76].strip() if len(line) > 70 else '+0.000'
                atype    = line[77:].split()[0].strip() if len(line) > 77 else name
            except (ValueError, IndexError):
                continue
            atoms.append(Atom(rec, serial, name, altloc, resname, chain, resseq,
                               icode, x, y, z, occupancy, bfactor, charge, atype,
                               line.rstrip()))
    return atoms


def _group_by_residue(atoms: list[Atom]) -> dict:
    """
    Returns {(chain, resseq, icode, resname): [Atom, ...], ...} ordered dict.
    """
    groups: dict = {}
    for a in atoms:
        key = (a.chain, a.resseq, a.icode, a.resname)
        groups.setdefault(key, []).append(a)
    return groups


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _dist2_to_point(a: Atom, cx: float, cy: float, cz: float) -> float:
    return (a.x - cx)**2 + (a.y - cy)**2 + (a.z - cz)**2


# ---------------------------------------------------------------------------
# Flex residue selection
# ---------------------------------------------------------------------------

def _is_flexible(resname: str) -> bool:
    return resname in _TORSIONS and resname not in _RIGID_ONLY


def _residue_min_dist(res_atoms: list[Atom], cx: float, cy: float, cz: float) -> float:
    """Minimum distance from any heavy side-chain atom to the box centre."""
    min_d2 = math.inf
    for a in res_atoms:
        if a.name in _BACKBONE:
            continue
        if a.atype.startswith('H') or a.atype == 'HD':
            continue
        d2 = _dist2_to_point(a, cx, cy, cz)
        if d2 < min_d2:
            min_d2 = d2
    return math.sqrt(min_d2) if min_d2 < math.inf else math.inf


def _parse_forced_residues(spec: str) -> set[tuple]:
    """
    Parse 'ARG42A,HIS64A' → {('ARG', 42, 'A'), ...}
    Accepts resname + resnum + optional chain.
    """
    out = set()
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        # 3-letter resname at start
        if len(tok) < 4:
            print(f"  Warning: cannot parse residue spec '{tok}' — skipping.")
            continue
        resname = tok[:3].upper()
        rest = tok[3:]
        # extract number and optional chain
        import re
        m = re.match(r'(-?\d+)([A-Za-z]?)$', rest)
        if not m:
            print(f"  Warning: cannot parse residue spec '{tok}' — skipping.")
            continue
        resnum = int(m.group(1))
        chain  = m.group(2).upper() if m.group(2) else ''
        out.add((resname, resnum, chain))
    return out


# ---------------------------------------------------------------------------
# Flex PDBQT writer
# ---------------------------------------------------------------------------

def _format_atom_line(serial: int, a: Atom) -> str:
    """Re-format a PDBQT ATOM line with a new serial number."""
    name_field = f' {a.name:<3}' if len(a.name) < 4 else a.name[:4]
    line = (
        f'{a.record:<6}{serial:>5} {name_field}{a.altloc}'
        f'{a.resname:<3} {a.chain}{a.resseq:>4}{a.icode}   '
        f'{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}'
        f'{a.occupancy:>6.2f}{a.bfactor:>6.2f}    '
        f'{a.charge:>6} {a.atype}'
    )
    return line


def _write_flex_residue(f, res_atoms: list[Atom], resname: str, chain: str,
                        resseq: int, icode: str, base_serial: int,
                        strip_h: bool) -> int:
    """
    Write one flexible residue block to file f.
    Returns the next available serial number.
    """
    torsions = _TORSIONS.get(resname, [])
    if not torsions:
        return base_serial

    # Index atoms by name for easy lookup
    by_name: dict[str, Atom] = {}
    for a in res_atoms:
        if a.name not in by_name:
            by_name[a.name] = a

    resid_str = f'{resname} {chain} {resseq}{icode.strip()}'
    f.write(f'BEGIN_RES {resid_str}\n')

    serial = base_serial
    name_to_serial: dict[str, int] = {}

    # ROOT: Cα anchor
    ca = by_name.get('CA')
    if ca is None:
        f.write(f'END_RES {resid_str}\n')
        return base_serial

    f.write('ROOT\n')
    name_to_serial['CA'] = serial
    f.write(_format_atom_line(serial, ca) + '\n')
    serial += 1
    f.write('ENDROOT\n')

    # Build BRANCH tree from torsion list.
    # Each (parent, child) pair opens a branch; atoms between consecutive
    # torsions (and after the last) go into the deepest open branch.
    def atoms_between(parent: str, next_branch_child: str | None) -> list[Atom]:
        """
        Returns atoms that belong to the BRANCH opened by (parent→child_of_branch)
        but are NOT themselves branch anchors for deeper branches.
        Uses the _SC_ATOMS ordering: the subtree of atoms between two torsion
        points on the linear chain.
        """
        sc_order = _SC_ATOMS.get(resname, [])
        try:
            p_idx = sc_order.index(parent)
        except ValueError:
            return []
        if next_branch_child is not None:
            try:
                n_idx = sc_order.index(next_branch_child)
            except ValueError:
                n_idx = len(sc_order)
        else:
            n_idx = len(sc_order)
        result = []
        for nm in sc_order[p_idx + 1:n_idx]:
            a = by_name.get(nm)
            if a is None:
                continue
            if strip_h and (a.atype.startswith('H') or a.atype == 'HD'):
                continue
            result.append(a)
        return result

    # Open all branches (depth-first, linear chain)
    for i, (parent, child) in enumerate(torsions):
        p_ser = name_to_serial.get(parent, base_serial - 1)
        # write BRANCH header (child serial will be next)
        child_ser = serial
        name_to_serial[child] = child_ser

        # next torsion's child (or None if this is the last)
        next_child = torsions[i + 1][1] if i + 1 < len(torsions) else None

        f.write(f'BRANCH {p_ser:>3} {child_ser:>3}\n')

        # Atoms that go in this level of the branch (child + atoms before next branch)
        for a in atoms_between(parent, next_child):
            if strip_h and (a.atype.startswith('H') or a.atype == 'HD'):
                continue
            name_to_serial[a.name] = serial
            f.write(_format_atom_line(serial, a) + '\n')
            serial += 1

    # Close branches in reverse order
    for i in range(len(torsions) - 1, -1, -1):
        parent, child = torsions[i]
        p_ser = name_to_serial.get(parent, base_serial - 1)
        c_ser = name_to_serial.get(child, base_serial)
        f.write(f'ENDBRANCH {p_ser:>3} {c_ser:>3}\n')

    f.write(f'END_RES {resid_str}\n')
    return serial


# ---------------------------------------------------------------------------
# Main split logic
# ---------------------------------------------------------------------------

def split_receptor(receptor_path: str,
                   cx: float, cy: float, cz: float,
                   cutoff: float,
                   forced_residues: set,
                   output_dir: str,
                   strip_h: bool = False) -> tuple[list, list]:
    """
    Parse receptor, find flexible residues, write rigid + flex PDBQT.
    Returns (flex_residue_keys, skipped_residue_keys).
    """
    atoms = _parse_pdbqt(receptor_path)
    if not atoms:
        sys.exit(f'ERROR: no ATOM/HETATM records found in {receptor_path}')

    groups = _group_by_residue(atoms)

    stem = os.path.splitext(os.path.basename(receptor_path))[0]
    # Strip trailing "_receptor" suffix if present for cleaner output names
    if stem.endswith('_receptor'):
        stem = stem[:-9]

    rigid_path = os.path.join(output_dir, f'{stem}_rigid.pdbqt')
    flex_path  = os.path.join(output_dir, f'{stem}_flex.pdbqt')

    flex_keys: list    = []
    skipped_keys: list = []

    cutoff2 = cutoff ** 2

    for key, res_atoms in groups.items():
        chain, resseq, icode, resname = key
        if not _is_flexible(resname):
            continue

        # Check forced list
        in_forced = False
        for (fn, fi, fc) in forced_residues:
            if fn == resname and fi == resseq and (fc == '' or fc == chain):
                in_forced = True
                break

        if in_forced:
            flex_keys.append(key)
            continue

        # Distance check: any heavy side-chain atom within cutoff
        min_d = _residue_min_dist(res_atoms, cx, cy, cz)
        if min_d <= cutoff:
            flex_keys.append(key)

    print(f'  {len(flex_keys)} flexible residue(s) identified:')
    for key in flex_keys:
        chain, resseq, icode, resname = key
        print(f'    {resname} {chain}{resseq}{icode.strip()}  '
              f'({len(_TORSIONS.get(resname, []))} torsion(s))')

    # --- Build set of (chain, resseq, icode) for flex residues ---
    flex_set = {(k[0], k[1], k[2]) for k in flex_keys}

    # --- Write rigid PDBQT ---
    os.makedirs(output_dir, exist_ok=True)
    with open(rigid_path, 'w') as rf:
        rf.write(f'REMARK  RIGID receptor — flexible residues removed\n')
        rf.write(f'REMARK  Source: {receptor_path}\n')
        for key, res_atoms in groups.items():
            chain, resseq, icode, resname = key
            if (chain, resseq, icode) in flex_set:
                # Keep ONLY backbone atoms for flexible residues
                for a in res_atoms:
                    if a.name in _BACKBONE:
                        rf.write(a.raw + '\n')
            else:
                for a in res_atoms:
                    if strip_h and (a.atype.startswith('H') or a.atype == 'HD'):
                        continue
                    rf.write(a.raw + '\n')

    # --- Write flex PDBQT ---
    serial = 1
    with open(flex_path, 'w') as ff:
        ff.write(f'REMARK  FLEX side chains — source: {receptor_path}\n')
        ff.write(f'REMARK  {len(flex_keys)} flexible residue(s)\n')
        for key in flex_keys:
            chain, resseq, icode, resname = key
            res_atoms = groups[key]
            serial = _write_flex_residue(
                ff, res_atoms, resname, chain, resseq, icode,
                serial, strip_h
            )

    print(f'  Rigid receptor → {rigid_path}')
    print(f'  Flex side chains → {flex_path}')
    return flex_keys, skipped_keys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--receptor', required=True,
                    help='Receptor PDBQT (with partial charges)')
    ap.add_argument('--center_x', type=float, required=True)
    ap.add_argument('--center_y', type=float, required=True)
    ap.add_argument('--center_z', type=float, required=True)
    ap.add_argument('--cutoff', type=float, default=5.0,
                    help='Max distance from box centre to include a residue (Å) [5.0]')
    ap.add_argument('--output_dir', default='.',
                    help='Output directory [.]')
    ap.add_argument('--residues', default='',
                    help='Comma-separated list of residues to force-include, '
                         'e.g. ARG42A,HIS64A (resname+resnum+optional_chain)')
    ap.add_argument('--no_h', action='store_true',
                    help='Strip hydrogen atoms from the flex PDBQT')
    args = ap.parse_args()

    print(f'[prep_flex_receptor] Receptor: {args.receptor}')
    print(f'  Box centre: ({args.center_x:.2f}, {args.center_y:.2f}, {args.center_z:.2f})')
    print(f'  Cutoff: {args.cutoff} Å')

    forced = _parse_forced_residues(args.residues) if args.residues else set()
    if forced:
        print(f'  Force-including: {forced}')

    flex_keys, _ = split_receptor(
        args.receptor,
        args.center_x, args.center_y, args.center_z,
        args.cutoff, forced,
        args.output_dir,
        strip_h=args.no_h,
    )

    if not flex_keys:
        print('  No flexible residues found within cutoff.  '
              'Increase --cutoff or specify --residues.')
        sys.exit(0)

    stem = os.path.splitext(os.path.basename(args.receptor))[0]
    if stem.endswith('_receptor'):
        stem = stem[:-9]
    print()
    print('Run Vina with:')
    print(f'  --receptor {os.path.join(args.output_dir, stem + "_rigid.pdbqt")}')
    print(f'  --flex     {os.path.join(args.output_dir, stem + "_flex.pdbqt")}')


if __name__ == '__main__':
    main()
