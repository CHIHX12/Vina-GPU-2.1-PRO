#!/usr/bin/env python3
"""Diagnostic: energy decomposition for ETr≠best failing cases.

For each target, compute full energy breakdown per pose and show why
the crystal-closest pose does not win ETr=1.
"""
import sys
sys.path.insert(0, '/home/cycheng/LigandScope/scripts')
import math
from pathlib import Path
from rank_poses import (
    split_vina_poses, parse_protein, detect_rings,
    compute_pose_energy, assign_consensus_ranks,
    _center_of_mass, ligand_adaptive_radius
)


def centroid_dist(atoms_a, atoms_b):
    """Heavy-atom centroid distance between two atom lists."""
    heavy_a = [a for a in atoms_a if a.element not in ('H', 'HD')]
    heavy_b = [a for a in atoms_b if a.element not in ('H', 'HD')]
    if not heavy_a or not heavy_b:
        return 999.0
    cx = sum(a.x for a in heavy_a) / len(heavy_a)
    cy = sum(a.y for a in heavy_a) / len(heavy_a)
    cz = sum(a.z for a in heavy_a) / len(heavy_a)
    dx = sum(b.x for b in heavy_b) / len(heavy_b) - cx
    dy = sum(b.y for b in heavy_b) / len(heavy_b) - cy
    dz = sum(b.z for b in heavy_b) / len(heavy_b) - cz
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def analyze(name, receptor_path, poses_path, crystal_path=None):
    """Run full analysis for one target."""
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")

    poses = split_vina_poses(Path(poses_path))
    print(f"  Poses found: {len(poses)}")

    lig0 = poses[0].atoms
    adapt_r = ligand_adaptive_radius(lig0)
    com = _center_of_mass(lig0)
    prot_atoms = parse_protein(Path(receptor_path), adapt_r, com, lig_atoms=lig0)
    prot_rings = detect_rings(prot_atoms)
    print(f"  Protein atoms: {len(prot_atoms)}  rings: {len(prot_rings)}")

    # Check metals in receptor
    metal_elems = [a.element for a in prot_atoms if a.element.upper() in
                   {'ZN', 'MG', 'FE', 'MN', 'CA', 'MO', 'CU', 'NI', 'CO'}]
    print(f"  Metals in receptor: {metal_elems}")

    # Load crystal reference if provided
    crystal_atoms = None
    if crystal_path and Path(crystal_path).exists():
        from rank_poses import Pose, _parse_atom_line
        catoms = []
        with open(crystal_path) as fh:
            for line in fh:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    a = _parse_atom_line(line)
                    if a:
                        catoms.append(a)
        crystal_atoms = catoms
        print(f"  Crystal atoms: {len(crystal_atoms)}")

    # Compute energies
    results = []
    for pose in poses:
        pe = compute_pose_energy(pose, prot_atoms, prot_rings)
        pe.vina_rank = pose.index  # placeholder
        results.append(pe)

    assign_consensus_ranks(results, poses)

    # Compute distances to crystal
    if crystal_atoms:
        dists = [centroid_dist(p.atoms, crystal_atoms) for p in poses]
    else:
        # Use centroid of pose1 as reference (approximate)
        ref = poses[0].atoms
        dists = [centroid_dist(p.atoms, ref) for p in poses]

    # Sort by distance to find best
    best_idx = min(range(len(dists)), key=lambda i: dists[i])
    best_dist = dists[best_idx]
    best_pose = poses[best_idx]

    # Find ETr=1
    etr1_pe = min(results, key=lambda r: r.e_total)
    etr1_idx = etr1_pe.pose_index - 1  # 0-based

    print(f"\n  Crystal-closest pose: Pose {best_pose.index}  "
          f"dist={best_dist:.3f}Å  ETr={etr1_pe.etotal_rank}")
    print(f"  ETr=1 pose:           Pose {etr1_pe.pose_index}  "
          f"dist={dists[etr1_idx]:.3f}Å  e_total={etr1_pe.e_total:.3f}")

    if best_pose.index == etr1_pe.pose_index:
        print(f"  STATUS: ✓ CORRECT — ETr=1 = crystal-closest")
    else:
        best_pe = results[best_idx]
        print(f"\n  STATUS: ✗ WRONG — ETr=1 is NOT crystal-closest")
        print(f"\n  Energy comparison (crystal-closest vs ETr=1):")
        print(f"  {'Term':<12}  {'Crystal-close':>14}  {'ETr=1':>14}  {'Diff':>10}")
        print(f"  {'-'*54}")
        terms = ['e_hb_bb', 'e_hb_sc', 'e_vdw', 'e_elec', 'e_pipi',
                 'e_hphob', 'e_chpi', 'e_cpi', 'e_xbond', 'e_metal']
        for t in terms:
            v_best = getattr(best_pe, t)
            v_etr = getattr(etr1_pe, t)
            diff = v_best - v_etr
            flag = " ← best wins" if diff < -0.1 else (" ← ETr=1 wins" if diff > 0.1 else "")
            print(f"  {t:<12} {v_best:>14.4f}  {v_etr:>14.4f}  {diff:>10.4f}{flag}")
        print(f"  {'e_total':<12} {best_pe.e_total:>14.4f}  {etr1_pe.e_total:>14.4f}  "
              f"{best_pe.e_total - etr1_pe.e_total:>10.4f}  ← gap (positive = ETr=1 wins)")

    # Full table
    print(f"\n  All poses:")
    print(f"  {'Pose':>5}  {'Dist':>7}  {'VR':>4}  {'ER':>4}  {'e_total':>10}  "
          f"{'e_hb':>8}  {'e_vdw':>8}  {'e_hphob':>8}  {'e_metal':>8}")
    print(f"  {'-'*75}")
    for i, (pe, dist) in enumerate(zip(results, dists)):
        star = " ★" if pe.pose_index == etr1_pe.pose_index else (
               " ●" if poses[i].index == best_pose.index else "")
        print(f"  {pe.pose_index:>5}  {dist:>7.3f}  {pe.vina_rank:>4}  "
              f"{pe.etotal_rank:>4}  {pe.e_total:>10.3f}  "
              f"{pe.e_hb:>8.3f}  {pe.e_vdw:>8.3f}  {pe.e_hphob:>8.3f}  "
              f"{pe.e_metal:>8.3f}{star}")

    return best_dist, dists[etr1_idx], best_pose.index == etr1_pe.pose_index


BASE_MV = '/home/cycheng/Vina-GPU-2.1/metal_validation'
BASE_LS = '/home/cycheng/Vina-GPU-2.1/ligandscope_results'
BASE_MO = '/home/cycheng/Vina-GPU-2.1/mo_validation'

# Find crystal ligand PDBQTs for metal_validation
import os

def find_crystal(pdb_id, lig_code):
    """Try to locate crystal ligand PDBQT."""
    candidates = [
        f"{BASE_MV}/pdbqt_ligand/{pdb_id}_{lig_code}.pdbqt",
        f"{BASE_MV}/pdbqt_ligand/{lig_code}.pdbqt",
        f"{BASE_MV}/ligands/{lig_code}.pdbqt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def find_xo_crystal(pdb_id, lig_code):
    """Try to locate XO crystal ligand PDBQT."""
    candidates = [
        f"{BASE_MO}/pdbqt_ligand/{pdb_id}_{lig_code}.pdbqt",
        f"{BASE_MO}/pdbqt_ligand/{lig_code}.pdbqt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# === 1. Metal validation cases with ETr≠best ===
cases_metal = [
    ('1BNN (CAII/Zn — gap 1.55Å)',
     f'{BASE_MV}/pdbqt_receptor/1BNN_receptor.pdbqt',
     f'{BASE_MV}/results/1BNN/AL1_out.pdbqt',
     '1BNN', 'AL1'),
    ('1A42 (CAII/Zn — gap 0.37Å)',
     f'{BASE_MV}/pdbqt_receptor/1A42_receptor.pdbqt',
     f'{BASE_MV}/results/1A42/BZU_out.pdbqt',
     '1A42', 'BZU'),
    ('1YDB (CAII/Zn — gap 0.22Å)',
     f'{BASE_MV}/pdbqt_receptor/1YDB_receptor.pdbqt',
     f'{BASE_MV}/results/1YDB/AZM_out.pdbqt',
     '1YDB', 'AZM'),
]

for label, rec, poses_file, pdb, lig in cases_metal:
    crystal = find_crystal(pdb, lig)
    if not Path(poses_file).exists():
        print(f"\n[SKIP] {label}: poses not found at {poses_file}")
        continue
    analyze(label, rec, poses_file, crystal)

# === 2. XO cases ===
cases_xo = [
    ('3NVZ/I3A (XO — e_metal=0)',
     f'{BASE_MO}/pdbqt_receptor/3NVZ_receptor.pdbqt',
     f'{BASE_MO}/results/3NVZ_redock2/3NVZ_I3A_out.pdbqt',
     f'{BASE_MO}/pdbqt_ligand/3NVZ_I3A.pdbqt'),
    ('3NRZ/HPA (XO — e_metal=0)',
     f'{BASE_MO}/pdbqt_receptor/3NRZ_receptor.pdbqt',
     f'{BASE_MO}/results/3NRZ_redock2/3NRZ_HPA_out.pdbqt',
     f'{BASE_MO}/pdbqt_ligand/3NRZ_HPA.pdbqt'),
]

for label, rec, poses_file, crystal_file in cases_xo:
    if not Path(poses_file).exists():
        print(f"\n[SKIP] {label}: poses not found at {poses_file}")
        continue
    if not Path(rec).exists():
        print(f"\n[SKIP] {label}: receptor not found at {rec}")
        continue
    crystal = crystal_file if Path(crystal_file).exists() else None
    analyze(label, rec, poses_file, crystal)

# === 3. LigandScope non-metal: 4R06 ===
if Path(f'{BASE_LS}/4R06/4R06_gpu_out.pdbqt').exists():
    # Find receptor
    rec4r06 = None
    for p in [
        f'{BASE_LS}/4R06/receptor.pdbqt',
        f'{BASE_LS}/pdbqt_receptor/4R06_receptor.pdbqt',
        f'{BASE_MV}/pdbqt_receptor/4R06_receptor.pdbqt',
    ]:
        if Path(p).exists():
            rec4r06 = p
            break
    if rec4r06:
        analyze('4R06 (nuclear receptor — gap 0.74Å)', rec4r06,
                f'{BASE_LS}/4R06/4R06_gpu_out.pdbqt')

print("\n\nDone.")
